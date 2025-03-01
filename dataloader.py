# dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieScriptDataset(Dataset):
    def __init__(
        self,
        scripts_dir: str,
        genre_csv_path: str,
        tokenizer,
        max_length: int = 1024
    ):
        self.scripts_dir = scripts_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load genre mappings
        self.genre_df = pd.read_csv(genre_csv_path)
        self.genre_df.set_index('movie_name', inplace=True)
        
        # Get list of script files and filter for .txt files
        self.script_files = [
            f for f in os.listdir(scripts_dir) 
            if f.endswith('.txt') and os.path.getsize(os.path.join(scripts_dir, f)) > 0
        ]
        logger.info(f"Found {len(self.script_files)} valid script files")
    
    def __len__(self) -> int:
        return len(self.script_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            script_file = self.script_files[idx]
            movie_name = os.path.splitext(script_file)[0]
            
            # Get genre safely
            try:
                genre = self.genre_df.loc[movie_name, 'genre']
            except KeyError:
                genre = "Unknown"
                logger.warning(f"No genre found for {movie_name}, using 'Unknown'")
            
            # Read script
            with open(os.path.join(self.scripts_dir, script_file), 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                logger.warning(f"Empty content in {script_file}")
                # Return a simple placeholder for empty content
                content = "[GENRE]Unknown[/GENRE]\n[SCENE]Empty Scene[/SCENE]"
            
            # Tokenize with error handling
            try:
                encoding = self.tokenizer(
                    content,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Remove the batch dimension added by return_tensors="pt"
                return {
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'labels': encoding['input_ids'].squeeze(0)
                }
                
            except Exception as e:
                logger.error(f"Tokenization error for {script_file}: {str(e)}")
                # Return a simple encoded placeholder
                placeholder = "[ERROR] Failed to process script"
                encoding = self.tokenizer(
                    placeholder,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                return {
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'labels': encoding['input_ids'].squeeze(0)
                }
                
        except Exception as e:
            logger.error(f"Error processing item {idx} ({script_file if 'script_file' in locals() else 'unknown'}): {str(e)}")
            # Return a minimal valid tensor in case of any error
            placeholder = self.tokenizer(
                "[ERROR]",
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return {
                'input_ids': placeholder['input_ids'].squeeze(0),
                'attention_mask': placeholder['attention_mask'].squeeze(0),
                'labels': placeholder['input_ids'].squeeze(0)
            }

def create_dataloaders(
    scripts_dir: str,
    genre_csv_path: str,
    tokenizer,
    batch_size: int = 4,
    max_length: int = 1024,
    train_split: float = 0.8,
    num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders with error handling"""
    try:
        # Create dataset
        full_dataset = MovieScriptDataset(
            scripts_dir=scripts_dir,
            genre_csv_path=genre_csv_path,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        # Calculate split sizes
        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        # Create splits
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size]
        )
        
        # Create dataloaders with error handling
        def collate_fn(batch):
            # Filter out any None values that might have occurred from errors
            batch = [b for b in batch if b is not None]
            if not batch:
                logger.error("Empty batch encountered")
                # Return minimal valid batch
                return {
                    'input_ids': torch.zeros((1, 1), dtype=torch.long),
                    'attention_mask': torch.zeros((1, 1), dtype=torch.long),
                    'labels': torch.zeros((1, 1), dtype=torch.long)
                }
            
            # Stack tensors
            return {
                'input_ids': torch.stack([item['input_ids'] for item in batch]),
                'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
                'labels': torch.stack([item['labels'] for item in batch])
            }
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            persistent_workers=True if num_workers > 0 else False
        )
        
        logger.info(f"Created dataloaders - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"Error creating dataloaders: {str(e)}")
        raise

def validate_dataset(scripts_dir: str, genre_csv_path: str) -> bool:
    """Validate dataset files and structure before creating dataloaders"""
    try:
        # Check if directories exist
        if not os.path.exists(scripts_dir):
            logger.error(f"Scripts directory does not exist: {scripts_dir}")
            return False
            
        if not os.path.exists(genre_csv_path):
            logger.error(f"Genre CSV file does not exist: {genre_csv_path}")
            return False
            
        # Check if there are any script files
        script_files = [f for f in os.listdir(scripts_dir) if f.endswith('.txt')]
        if not script_files:
            logger.error(f"No .txt files found in scripts directory: {scripts_dir}")
            return False
            
        # Validate genre CSV structure
        try:
            genre_df = pd.read_csv(genre_csv_path)
            required_columns = {'movie_name', 'genre'}
            if not all(col in genre_df.columns for col in required_columns):
                logger.error(f"Genre CSV missing required columns: {required_columns}")
                return False
        except Exception as e:
            logger.error(f"Error reading genre CSV: {str(e)}")
            return False
            
        # Validate script files
        total_files = len(script_files)
        valid_files = 0
        for script_file in script_files:
            try:
                with open(os.path.join(scripts_dir, script_file), 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                if content:
                    valid_files += 1
            except Exception:
                continue
                
        if valid_files == 0:
            logger.error("No valid script files found")
            return False
            
        logger.info(f"Dataset validation passed: {valid_files}/{total_files} valid files")
        return True
        
    except Exception as e:
        logger.error(f"Error during dataset validation: {str(e)}")
        return False

class DatasetStatistics:
    """Class to collect and report dataset statistics"""
    
    @staticmethod
    def analyze_dataset(scripts_dir: str, genre_csv_path: str) -> Dict:
        """Analyze dataset and return statistics"""
        stats = {
            'total_files': 0,
            'valid_files': 0,
            'total_tokens': 0,
            'genres': {},
            'avg_file_size': 0,
            'errors': []
        }
        
        try:
            # Load genre data
            genre_df = pd.read_csv(genre_csv_path)
            genre_dict = dict(zip(genre_df['movie_name'], genre_df['genre']))
            
            # Analyze files
            script_files = [f for f in os.listdir(scripts_dir) if f.endswith('.txt')]
            stats['total_files'] = len(script_files)
            
            total_size = 0
            for script_file in script_files:
                try:
                    file_path = os.path.join(scripts_dir, script_file)
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    if content:
                        stats['valid_files'] += 1
                        movie_name = os.path.splitext(script_file)[0]
                        genre = genre_dict.get(movie_name, 'Unknown')
                        stats['genres'][genre] = stats['genres'].get(genre, 0) + 1
                        
                except Exception as e:
                    stats['errors'].append(f"Error processing {script_file}: {str(e)}")
                    
            if stats['total_files'] > 0:
                stats['avg_file_size'] = total_size / stats['total_files']
                
            return stats
            
        except Exception as e:
            logger.error(f"Error analyzing dataset: {str(e)}")
            return stats

# Helper function to create dataloaders with validation
def create_validated_dataloaders(*args, **kwargs):
    """Create dataloaders with dataset validation"""
    if not validate_dataset(kwargs['scripts_dir'], kwargs['genre_csv_path']):
        raise ValueError("Dataset validation failed")
        
    # Collect and log dataset statistics
    stats = DatasetStatistics.analyze_dataset(
        kwargs['scripts_dir'],
        kwargs['genre_csv_path']
    )
    logger.info(f"Dataset statistics: {stats}")
    
    return create_dataloaders(*args, **kwargs)