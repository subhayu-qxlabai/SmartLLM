#!/bin/bash

# Usage: ./setup.sh [environment_name]
#   - environment_name: Optional. If provided, creates a virtual environment with the specified name.
#                      If not provided, installs dependencies without creating a virtual environment.

# Python version
PYTHON_VERSION="3.11"

# Check if the argument is provided
if [ $# -gt 0 ]; then
    # Use the first argument as the environment name
    env_name="$1"
    
    # Create virtual environment with the provided name and specified Python version
    python"$PYTHON_VERSION" -m venv "$env_name"
    
    # Activate the virtual environment
    source "$env_name"/bin/activate
fi

# Always install requirements
python"$PYTHON_VERSION" -m pip install -r requirements.txt
