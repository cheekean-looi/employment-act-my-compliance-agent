#!/bin/bash
# Activate conda environment (creates from environment.yml if needed)

source ~/miniforge3/etc/profile.d/conda.sh

# Check if environment exists, create if not
if ! conda env list | grep -q "faiss-env"; then
    echo "🔨 Creating conda environment from environment.yml..."
    conda env create -f environment.yml
    
    # Activate and fix NumPy version conflict
    conda activate faiss-env
    echo "🔧 Fixing NumPy version for FAISS compatibility..."
    pip install "numpy==1.26.4" --force-reinstall --quiet
    echo "✅ Environment setup complete"
else
    conda activate faiss-env
    echo "✅ Environment 'faiss-env' activated"
fi