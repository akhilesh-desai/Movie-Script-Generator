import sys
import os
import argparse
import logging
import warnings
import json
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, asdict
import torch
import numpy as np
import pandas as pd
import wandb
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (get_linear_schedule_with_warmup, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel)
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

os.environ["HUGGING_FACE_HUB_TOKEN"] = "HF TOKEN"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration with type hints and documentation"""
    model_name: str = "meta-llama/Llama-2-7b-hf"  # Base model to use
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_length: int = 1024
    num_epochs: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    lora_rank: int = 32
    lora_alpha: int = 64
    dropout: float = 0.1
    mixed_precision: bool = True
    seed: int = 42
    patience: int = 5
    max_grad_norm: float = 1.0
    cache_clear_frequency: int = 5

def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return TrainingConfig(**config_dict)

class ScriptDataset(Dataset):
    """Custom dataset for movie scripts"""
    def __init__(
        self,
        scripts_dir: str,
        genre_csv_path: str,
        tokenizer,
        max_length: int
    ):
        self.scripts_dir = scripts_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load genre information
        self.genre_df = pd.read_csv(genre_csv_path)
        
        # Get all script files
        self.script_files = [
            f for f in os.listdir(scripts_dir)
            if f.endswith('.txt')
        ]
        
        # Create genre mapping
        self.genre_mapping = {
            row['movie_name']: row['genre']
            for _, row in self.genre_df.iterrows()
        }

    def __len__(self) -> int:
        return len(self.script_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        script_file = self.script_files[idx]
        
        # Load script content
        with open(os.path.join(self.scripts_dir, script_file), 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get genre
        genre = self.genre_mapping.get(script_file, 'unknown')
        
        # Create prompt with genre
        prompt = f"Genre: {genre}\n\nScript:\n{content}"
        
        # Tokenize
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Prepare labels (shift input_ids right)
        labels = encodings['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': encodings['input_ids'][0],
            'attention_mask': encodings['attention_mask'][0],
            'labels': labels[0]
        }

def prepare_improved_model(config: TrainingConfig) -> PreTrainedModel:
    """
    Prepare the model with LoRA configuration and proper initialization
    """
    try:
        # Load base model and tokenizer
        logger.info(f"Loading base model: {config.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.mixed_precision else torch.float32,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=config.dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA configuration
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Attach tokenizer to model for convenience
        model.tokenizer = tokenizer
        
        return model
        
    except Exception as e:
        logger.error(f"Error preparing model: {str(e)}")
        raise

def create_dataloaders(
    scripts_dir: str,
    genre_csv_path: str,
    tokenizer,
    batch_size: int,
    max_length: int,
    train_split: float = 0.9
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    """
    # Create dataset
    dataset = ScriptDataset(scripts_dir, genre_csv_path, tokenizer, max_length)
    
    # Calculate split sizes
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader

class ModelCheckpointing:
    """Handles model checkpointing and saving"""
    def __init__(self, output_dir: str, model_name: str):
        self.output_dir = output_dir
        self.model_name = model_name
        self.best_metric = float('inf')
        os.makedirs(output_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, scheduler, epoch: int,
                       metric: float, is_best: bool = False) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metric': metric
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.output_dir, f'{self.model_name}_checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if needed
        if is_best:
            best_path = os.path.join(self.output_dir, f'{self.model_name}_best.pt')
            torch.save(checkpoint, best_path)
            self.best_metric = metric
            logger.info(f"New best model saved with metric: {metric:.4f}")

class MetricsTracker:
    """Tracks and computes training metrics"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.accuracies = []
        self.perplexities = []

    def update(self, loss: float, accuracy: float, perplexity: float):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.perplexities.append(perplexity)

    def get_averages(self) -> Dict[str, float]:
        return {
            'loss': np.mean(self.losses),
            'accuracy': np.mean(self.accuracies),
            'perplexity': np.mean(self.perplexities)
        }

def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, 
                   pad_token_id: int) -> Dict[str, float]:
    """Compute training metrics with improved error handling"""
    try:
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            mask = (labels != pad_token_id)

            # Accuracy
            accuracy = ((predictions == labels) & mask).float().sum() / mask.sum()

            # Perplexity with stable computation
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=pad_token_id,
                reduction='mean'
            )
            perplexity = torch.exp(torch.clamp(loss, max=20))  # Prevent explosion

            return {
                "accuracy": accuracy.item(),
                "perplexity": perplexity.item()
            }
    except Exception as e:
        logger.error(f"Error computing metrics: {str(e)}")
        return {"accuracy": 0.0, "perplexity": float('inf')}

class Trainer:
    """Main trainer class with improved organization and features"""
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config: TrainingConfig,
        output_dir: str
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize training components
        self.setup_training_components()
        
        # Initialize tracking
        self.checkpointing = ModelCheckpointing(output_dir, "movie_script_model")
        self.metrics_tracker = MetricsTracker()
        self.scaler = GradScaler() if config.mixed_precision else None

    def setup_training_components(self):
        """Initialize optimizer and scheduler"""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        num_training_steps = self.config.num_epochs * len(self.train_loader)
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with improved error handling and mixed precision"""
        self.model.train()
        self.metrics_tracker.reset()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Mixed precision training
                with autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs.loss / self.config.gradient_accumulation_steps

                # Backward pass with gradient scaling
                if self.config.mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    if self.config.mixed_precision:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                        
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                # Compute metrics
                metrics = compute_metrics(
                    outputs.logits,
                    batch['labels'],
                    self.model.tokenizer.pad_token_id
                )
                metrics['loss'] = loss.item() * self.config.gradient_accumulation_steps
                
                self.metrics_tracker.update(**metrics)
                
                # Update progress bar
                pbar.set_postfix(metrics)
                
                # Log to wandb
                wandb.log({
                    'train/batch_loss': metrics['loss'],
                    'train/batch_accuracy': metrics['accuracy'],
                    'train/batch_perplexity': metrics['perplexity'],
                    'train/learning_rate': self.scheduler.get_last_lr()[0]
                })
                
                # Clear cache periodically
                if batch_idx % self.config.cache_clear_frequency == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                logger.error(f"Error in training step: {str(e)}")
                continue
        
        return self.metrics_tracker.get_averages()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set"""
        self.model.eval()
        self.metrics_tracker.reset()
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            try:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                metrics = compute_metrics(
                    outputs.logits,
                    batch['labels'],
                    self.model.tokenizer.pad_token_id
                )
                metrics['loss'] = outputs.loss.item()
                
                self.metrics_tracker.update(**metrics)
                
            except Exception as e:
                logger.error(f"Error in validation step: {str(e)}")
                continue
        
        return self.metrics_tracker.get_averages()

    def train(self):
        """Main training loop with improved monitoring and early stopping"""
        best_val_perplexity = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            try:
                # Training epoch
                train_metrics = self.train_epoch(epoch)
                
                # Validation
                val_metrics = self.evaluate()
                
                # Logging
                logger.info(
                    f"Epoch {epoch + 1} - Train loss: {train_metrics['loss']:.4f}, "
                    f"Val loss: {val_metrics['loss']:.4f}, "
                    f"Val perplexity: {val_metrics['perplexity']:.4f}"
                )
                
                # WandB logging
                wandb.log({
                    'train/epoch_loss': train_metrics['loss'],
                    'train/epoch_accuracy': train_metrics['accuracy'],
                    'train/epoch_perplexity': train_metrics['perplexity'],
                    'val/loss': val_metrics['loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/perplexity': val_metrics['perplexity'],
                    'epoch': epoch + 1
                })
                
                # Model checkpointing
                is_best = val_metrics['perplexity'] < best_val_perplexity
                self.checkpointing.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_metrics['perplexity'],
                    is_best
                )
                
                # Early stopping
                if is_best:
                    best_val_perplexity = val_metrics['perplexity']
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.patience:
                    logger.info("Early stopping triggered")
                    break
                    
            except Exception as e:
                logger.error(f"Error in epoch {epoch + 1}: {str(e)}")
                continue

def main():
    """Main function with improved argument parsing and error handling"""
    parser = argparse.ArgumentParser(description="Train movie script generator")
    parser.add_argument("--scripts_dir", required=True, help="Directory containing script files")
    parser.add_argument("--genre_csv", required=True, help="Path to genre CSV file")
    parser.add_argument("--output_dir", required=True, help="Output directory for model checkpoints")
    parser.add_argument("--config", type=str, help="Path to custom config JSON")
    parser.add_argument("--wandb_project", type=str, default="movie-script-generator", 
                       help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, help="WandB entity/username")
    
    args = parser.parse_args()
    
    try:
        # Load config
        config = TrainingConfig()
        if args.config:
            config = load_config(args.config)
            
        # Set random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
            
        # Initialize wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "architecture": "LLaMA-LoRA",
                "dataset": args.scripts_dir,
                **asdict(config)
            }
        )
        
        # Create model and move to GPU
        logger.info("Initializing model...")
        model = prepare_improved_model(config)
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(
            scripts_dir=args.scripts_dir,
            genre_csv_path=args.genre_csv,
            tokenizer=model.tokenizer,
            batch_size=config.batch_size,
            max_length=config.max_length,
            train_split=0.9
        )
        
        # Initialize trainer
        logger.info("Setting up trainer...")
        trainer = Trainer(model, train_loader, val_loader, config, args.output_dir)
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()