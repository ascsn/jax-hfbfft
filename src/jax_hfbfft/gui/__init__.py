"""
GUI module for JAX-HFBFFT.

This module provides a web-based graphical user interface for running
HFB calculations interactively. The same interface can be used in a
web browser or wrapped in Electron for desktop use.

Usage:
    # From CLI
    hfbfft gui
    hfbfft gui --port 8080
    
    # From Python
    from jax_hfbfft.gui import start_server
    start_server(port=8080)
"""

from jax_hfbfft.gui.server import create_app, start_server

__all__ = ["create_app", "start_server"]
