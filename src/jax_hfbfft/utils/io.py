"""
I/O utility functions for JAX-HFBFFT.

This module provides input/output operations for configuration files,
wavefunction data, and results.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def read_yaml(file_path: str) -> Dict[str, Any]:
    """
    Read a YAML configuration file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary with parsed YAML content
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file contains invalid YAML
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    if not isinstance(config, dict):
        raise ValueError(f"The file '{file_path}' does not contain valid YAML data.")
    
    return config


def write_yaml(file_path: str, data: Dict[str, Any], **kwargs):
    """
    Write data to a YAML file.
    
    Args:
        file_path: Path to the output file
        data: Dictionary to write
        **kwargs: Additional arguments passed to yaml.dump
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    default_kwargs = {
        'default_flow_style': False,
        'sort_keys': False,
    }
    default_kwargs.update(kwargs)
    
    with open(path, 'w') as f:
        yaml.dump(data, f, **default_kwargs)


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
