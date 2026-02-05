"""
Setup script for jax-hfbfft with custom build hooks.

This setup.py enables building the frontend during pip install.
"""

import os
import subprocess
import sys
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class BuildWithFrontend(_build_py):
    """
    Custom build command that builds the frontend before Python package.
    
    This runs `npm install && npm run build` in the gui-frontend directory
    and copies the built assets to src/jax_hfbfft/gui/frontend/.
    """
    
    def run(self):
        """Run the build process."""
        # Check if frontend build should be skipped
        skip_frontend = os.environ.get('SKIP_FRONTEND_BUILD', '').lower() in ('1', 'true', 'yes')
        if skip_frontend:
            print("SKIP_FRONTEND_BUILD is set, skipping frontend build")
            super().run()
            return
        
        # Get the project root directory
        project_root = Path(__file__).parent
        frontend_dir = project_root / "gui-frontend"
        frontend_dist = frontend_dir / "dist"
        target_dir = project_root / "src" / "jax_hfbfft" / "gui" / "frontend"
        
        # Check if frontend directory exists
        if not frontend_dir.exists():
            print("Warning: gui-frontend directory not found, skipping frontend build")
            super().run()
            return
        
        # Check if npm is available
        try:
            subprocess.run(
                ["npm", "--version"],
                check=True,
                capture_output=True,
                timeout=5
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            print("Warning: npm not found, skipping frontend build")
            print("To build the GUI, install Node.js and run: cd gui-frontend && npm install && npm run build:copy")
            super().run()
            return
        
        # Build the frontend
        print("=" * 60)
        print("Building web GUI frontend...")
        print("=" * 60)
        
        try:
            # Install dependencies
            print("Installing frontend dependencies...")
            subprocess.run(
                ["npm", "install"],
                cwd=frontend_dir,
                check=True,
                timeout=300  # 5 minutes for npm install
            )
            
            # Build frontend
            print("Building frontend assets...")
            subprocess.run(
                ["npm", "run", "build"],
                cwd=frontend_dir,
                check=True,
                timeout=120  # 2 minutes for build
            )
            
            # Copy built assets to package
            if frontend_dist.exists():
                print(f"Copying frontend assets to {target_dir}...")
                
                # Create target directory if it doesn't exist
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy all files from dist to target
                import shutil
                for item in frontend_dist.iterdir():
                    target = target_dir / item.name
                    if item.is_dir():
                        if target.exists():
                            shutil.rmtree(target)
                        shutil.copytree(item, target)
                    else:
                        shutil.copy2(item, target)
                
                print("Frontend build complete!")
            else:
                print(f"Warning: Frontend dist directory not found at {frontend_dist}")
                
        except subprocess.CalledProcessError as e:
            print(f"Error building frontend: {e}", file=sys.stderr)
            print("The package will be installed without the GUI frontend.", file=sys.stderr)
            print("To build manually, run: cd gui-frontend && npm install && npm run build:copy", file=sys.stderr)
        except subprocess.TimeoutExpired as e:
            print(f"Frontend build timed out: {e}", file=sys.stderr)
            print("The package will be installed without the GUI frontend.", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error during frontend build: {e}", file=sys.stderr)
        
        # Continue with normal Python package build
        print("=" * 60)
        print("Building Python package...")
        print("=" * 60)
        super().run()


# All configuration is in pyproject.toml
# This file just provides the custom build command
setup(
    cmdclass={
        'build_py': BuildWithFrontend,
    },
)
