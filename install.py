#!/usr/bin/env python3
"""Installation script for the pose extraction system."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    
    return result


def check_uv_installed() -> bool:
    """Check if uv is installed."""
    try:
        run_command("uv --version", check=False)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_uv():
    """Install uv if not already installed."""
    if check_uv_installed():
        print("uv is already installed")
        return
    
    print("Installing uv...")
    if sys.platform == "win32":
        run_command("powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
    else:
        run_command("curl -LsSf https://astral.sh/uv/install.sh | sh")
    
    print("Please restart your terminal or run 'source ~/.bashrc' to use uv")


def setup_environment():
    """Set up the Python environment."""
    print("Setting up Python environment...")
    
    # Create virtual environment
    run_command("uv venv")
    
    # Install dependencies
    run_command("uv pip install -e .")
    
    # Install development dependencies
    run_command("uv pip install -e '.[dev]'")


def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    
    directories = [
        "data/video",
        "data/poses", 
        "data/embeddings",
        "data/analysis",
        "models",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")


def run_tests():
    """Run basic tests to verify installation."""
    print("Running tests...")
    
    try:
        run_command("python tests/test_imports.py")
        print("✓ All tests passed!")
    except subprocess.CalledProcessError:
        print("✗ Some tests failed. Please check the error messages above.")
        return False
    
    return True


def main():
    """Main installation function."""
    print("Pose Extraction System - Installation")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Install uv if needed
    install_uv()
    
    # Set up environment
    setup_environment()
    
    # Create directories
    create_directories()
    
    # Run tests
    if run_tests():
        print("\n" + "=" * 50)
        print("Installation completed successfully!")
        print("\nNext steps:")
        print("1. Place your dance videos in data/video/")
        print("2. Run: python -m pose_extraction.main --input-dir data/video")
        print("3. Check the examples/ directory for usage examples")
        print("4. Read the documentation in documents/")
    else:
        print("\nInstallation completed with warnings.")
        print("Please check the error messages above.")


if __name__ == "__main__":
    main() 