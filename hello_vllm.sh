#!/bin/bash

# Simple vLLM Hello World Script
# Sets up environment with uv and runs a basic inference test

set -e  # Exit on error

echo "=== vLLM Hello World Setup ==="
echo

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    # uv will automatically use Python version from pyproject.toml (>=3.9,<3.13)
    # It will download Python if needed
    uv venv --python 3.11
fi

# Install dependencies (uv automatically uses .venv)
echo "Installing dependencies with uv..."

# First install torch and other dependencies
uv pip install torch transformers accelerate

# Install vLLM in editable mode from current directory
echo "Installing vLLM in editable mode (Python changes will be reflected immediately)..."
echo "Note: First install may take time to build C++ extensions..."
uv pip install -e .

# Run the test (uv automatically uses .venv)
echo
echo "Running inference test..."
echo "Command: uv run python test_inference.py"
echo "="*50
# Add verbose flag to uv to see what it's doing
uv run -v python test_inference.py

echo
echo "Setup complete! You can now:"
echo "  - Modify test_inference.py to try different prompts"
echo "  - Run it again with: uv run python test_inference.py"
echo "  - Try a different model by changing the 'model' parameter"