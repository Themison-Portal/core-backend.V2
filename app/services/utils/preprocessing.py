import unicodedata
import re
from typing import Optional

def preprocess_text(text: str, clean_whitespace: bool = True) -> str:
    """
    Preprocess text for embedding and storage
    
    Args:
        text: Text to preprocess
        clean_whitespace: Whether to clean redundant whitespace
        
    Returns:
        Preprocessed text
    """
    # Normalize unicode characters
    text = unicodedata.normalize('NFC', text)
    
    # Remove null bytes and other control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Standardize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    if clean_whitespace:
        # Remove redundant whitespace while preserving paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Standardize paragraph breaks
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'\t', ' ', text)  # Tabs to spaces
        text = text.strip()  # Remove leading/trailing whitespace
    
    return text