import os
import pandas as pd
import argparse
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScriptPreprocessor:
    def parse_script_file(self, file_path: str) -> Dict:
        """Parse a movie script file with specific tag handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        
        # Initialize containers
        metadata = {}
        scenes = []
        current_scene = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Split the line into tag and content
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
                
            tag, content = parts[0].strip(), parts[1].strip()
            
            if tag == 'M':  # Metadata
                metadata['title'] = content
            elif tag == 'S':  # Scene heading
                current_scene = {
                    'heading': content,
                    'content': []
                }
                scenes.append(current_scene)
            elif tag == 'N' and current_scene is not None:  # Action/Scene description
                current_scene['content'].append(('action', content))
            elif tag == 'C':  # Character
                if current_scene is not None:
                    current_scene['content'].append(('character', content))
            elif tag == 'D':  # Dialogue
                if current_scene is not None:
                    current_scene['content'].append(('dialogue', content))
            # Skip E (dialogue metadata) and T (transition) tags as they're not needed in the output format
        
        return {
            'metadata': metadata,
            'scenes': scenes
        }

    def format_script(self, parsed_script: Dict, genre: str) -> str:
        """Format parsed script into the required output format"""
        formatted = f"[GENRE]{genre}[/GENRE]\n\n"
        
        for scene in parsed_script['scenes']:
            # Add scene heading
            formatted += f"[SCENE]{scene['heading']}[/SCENE]\n"
            
            # Add scene content
            for content_type, content in scene['content']:
                if content_type == 'action':
                    formatted += f"[ACTION]{content}[/ACTION]\n"
                elif content_type == 'character':
                    formatted += f"[CHARACTER]{content}[/CHARACTER]\n"
                elif content_type == 'dialogue':
                    formatted += f"[DIALOGUE]{content}[/DIALOGUE]\n"
            
            formatted += "\n"
        
        return formatted

def preprocess_scripts(
    input_dir: str,
    output_dir: str,
    genre_csv_path: str
):
    """Preprocess all scripts"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load genre data
    genre_df = pd.read_csv(genre_csv_path)
    genre_dict = dict(zip(genre_df['movie_name'], genre_df['genre']))
    
    preprocessor = ScriptPreprocessor()
    
    # Process each script file
    for filename in os.listdir(input_dir):
        if not filename.endswith('.txt'):
            continue
        
        movie_name = os.path.splitext(filename)[0]
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Parse script
            parsed_script = preprocessor.parse_script_file(input_path)
            
            # Get genre
            if movie_name not in genre_dict:
                logger.warning(f"No genre found for {movie_name}, using 'Unknown'")
                genre = "Unknown"
            else:
                genre = genre_dict[movie_name]
            
            # Format script
            formatted_script = preprocessor.format_script(parsed_script, genre)
            
            # Save processed script
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_script)
            
            logger.info(f"Successfully processed {filename}")
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess movie scripts")
    parser.add_argument("--input_dir", required=True, help="Input directory containing raw script files")
    parser.add_argument("--output_dir", required=True, help="Output directory for processed scripts")
    parser.add_argument("--genre_csv", required=True, help="Path to genre CSV file")
    
    args = parser.parse_args()
    
    preprocess_scripts(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        genre_csv_path=args.genre_csv
    )

if __name__ == "__main__":
    main()