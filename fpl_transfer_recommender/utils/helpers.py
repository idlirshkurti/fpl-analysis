"""
Helper functions for FPL Transfer Recommender.

This module provides utility functions that are used across different parts of the application.
"""

import logging
import pickle
import pandas as pd
import numpy as np
import os
import json
import datetime
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Constants for logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def setup_logging(name: str = __name__, level: str = LOG_LEVEL) -> logging.Logger:
    """
    Set up a logger with the specified name and level.
    
    Args:
        name: Logger name
        level: Logging level (INFO, DEBUG, etc.)
        
    Returns:
        logging.Logger: Configured logger
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Check if the logger already has handlers to avoid duplicate messages
    if not logger.handlers:
        # Create console handler and set level
        handler = logging.StreamHandler()
        handler.setLevel(numeric_level)
        
        # Create formatter
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
    
    return logger

def save_data_to_csv(data: pd.DataFrame, filename: Union[str, Path], index: bool = False) -> bool:
    """
    Save a DataFrame to a CSV file.
    
    Args:
        data: DataFrame to save
        filename: Target filename or path
        index: Whether to save the DataFrame index
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger = setup_logging()
    
    try:
        # Create directory if it doesn't exist
        if isinstance(filename, str):
            file_path = Path(filename)
        else:
            file_path = filename
            
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save DataFrame
        data.to_csv(file_path, index=index)
        logger.info(f"Data saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving data to CSV: {e}")
        return False

def load_data_from_csv(filename: Union[str, Path]) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV file.
    
    Args:
        filename: Source filename or path
        
    Returns:
        pd.DataFrame: Loaded DataFrame or empty DataFrame if error
    """
    logger = setup_logging()
    
    try:
        # Check if file exists
        if isinstance(filename, str):
            file_path = Path(filename)
        else:
            file_path = filename
            
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()
        
        # Load DataFrame
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded from {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data from CSV: {e}")
        return pd.DataFrame()

def save_pickle(data: Any, filename: Union[str, Path]) -> bool:
    """
    Save data to a pickle file.
    
    Args:
        data: Data to save
        filename: Target filename or path
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger = setup_logging()
    
    try:
        # Create directory if it doesn't exist
        if isinstance(filename, str):
            file_path = Path(filename)
        else:
            file_path = filename
            
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save data
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"Data saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving data to pickle: {e}")
        return False

def load_pickle(filename: Union[str, Path]) -> Any:
    """
    Load data from a pickle file.
    
    Args:
        filename: Source filename or path
        
    Returns:
        Any: Loaded data or None if error
    """
    logger = setup_logging()
    
    try:
        # Check if file exists
        if isinstance(filename, str):
            file_path = Path(filename)
        else:
            file_path = filename
            
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None
        
        # Load data
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        logger.info(f"Data loaded from {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading data from pickle: {e}")
        return None

def cache_data(key: str, data: Any, cache_dir: Path, expiry_seconds: int = 3600) -> bool:
    """
    Cache data with a key for later retrieval.
    
    Args:
        key: Cache key for identifying the data
        data: Data to cache
        cache_dir: Directory to store cached data
        expiry_seconds: Cache expiry time in seconds
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger = setup_logging()
    
    try:
        # Create cache directory if it doesn't exist
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a hash of the key for the filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        cache_file = cache_dir / f"{key_hash}.pkl"
        
        # Create cache entry with metadata
        cache_entry = {
            'data': data,
            'timestamp': datetime.datetime.now(),
            'expiry': datetime.datetime.now() + datetime.timedelta(seconds=expiry_seconds),
            'key': key
        }
        
        # Save to cache
        return save_pickle(cache_entry, cache_file)
        
    except Exception as e:
        logger.error(f"Error caching data: {e}")
        return False

def get_cached_data(key: str, cache_dir: Path) -> Tuple[bool, Any]:
    """
    Retrieve cached data if it exists and is not expired.
    
    Args:
        key: Cache key for identifying the data
        cache_dir: Directory where cached data is stored
        
    Returns:
        Tuple[bool, Any]: (success, data) where success indicates if cached data was found
                         and not expired, and data is the cached data or None
    """
    logger = setup_logging()
    
    try:
        # Create a hash of the key for the filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        cache_file = cache_dir / f"{key_hash}.pkl"
        
        # Check if cache file exists
        if not cache_file.exists():
            return False, None
        
        # Load cache entry
        cache_entry = load_pickle(cache_file)
        if cache_entry is None:
            return False, None
        
        # Check if expired
        if datetime.datetime.now() > cache_entry.get('expiry'):
            logger.info(f"Cache for key '{key}' has expired")
            return False, None
        
        logger.info(f"Retrieved data from cache for key '{key}'")
        return True, cache_entry.get('data')
        
    except Exception as e:
        logger.error(f"Error retrieving cached data: {e}")
        return False, None

def format_timestamp(dt: Optional[datetime.datetime] = None) -> str:
    """
    Format a timestamp for file naming and display.
    
    Args:
        dt: Datetime object to format (default: current time)
        
    Returns:
        str: Formatted timestamp string
    """
    if dt is None:
        dt = datetime.datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S")

def create_directory_if_not_exists(directory: Union[str, Path]) -> bool:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
        
    Returns:
        bool: True if successful or directory already exists, False on error
    """
    logger = setup_logging()
    
    try:
        if isinstance(directory, str):
            dir_path = Path(directory)
        else:
            dir_path = directory
            
        dir_path.mkdir(parents=True, exist_ok=True)
        return True
        
    except Exception as e:
        logger.error(f"Error creating directory {directory}: {e}")
        return False

def load_json(filename: Union[str, Path]) -> Dict:
    """
    Load data from a JSON file.
    
    Args:
        filename: Source filename or path
        
    Returns:
        Dict: Loaded JSON data or empty dict if error
    """
    logger = setup_logging()
    
    try:
        # Check if file exists
        if isinstance(filename, str):
            file_path = Path(filename)
        else:
            file_path = filename
            
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return {}
        
        # Load JSON
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        logger.info(f"JSON data loaded from {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Error loading JSON data: {e}")
        return {}

def save_json(data: Dict, filename: Union[str, Path], indent: int = 4) -> bool:
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filename: Target filename or path
        indent: Indentation for JSON formatting
        
    Returns:
        bool: True if successful, False otherwise
    """
    logger = setup_logging()
    
    try:
        # Create directory if it doesn't exist
        if isinstance(filename, str):
            file_path = Path(filename)
        else:
            file_path = filename
            
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
            
        logger.info(f"JSON data saved to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving JSON data: {e}")
        return False

