#!/bin/bash

ENV_NAME="simopt"
YML_FILE="environment.yml"

echo "Checking for Conda installation..."
if ! command -v conda &> /dev/null; then
    echo "Conda not found! Please install Miniconda or Anaconda first."
    exit 1
fi

# Ensure Conda is initialized
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if environment exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' exists. Updating..."
    conda env update --name "$ENV_NAME" --file "$YML_FILE" --prune
else
    echo "Creating new environment '$ENV_NAME'..."
    conda env create -f "$YML_FILE"
fi

# Activate the Conda environment
echo "Activating environment..."
source activate "$ENV_NAME"

# Ensure activation persists by checking Ruby installation
if ! command -v ruby &> /dev/null; then
    echo "Ruby installation not found in Conda environment. Please check installation."
    exit 1
fi

# Install datafarming gem
echo "Installing Ruby 'datafarming' gem..."
gem install datafarming -v 1.4

echo "Setup complete! Run: conda activate $ENV_NAME"
