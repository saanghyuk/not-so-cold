#!/bin/bash

# Multitouch Attribution Analysis - Setup Script

echo "========================================"
echo "Setting up Multitouch Analysis Project"
echo "========================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8 or later"
    exit 1
fi

echo "✓ Python version: $(python3 --version)"
echo ""

# Create virtual environment (optional but recommended)
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"
echo ""

# Print next steps
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "  1. Start Jupyter:"
echo "     jupyter notebook"
echo ""
echo "  2. Open: multitouch.ipynb"
echo ""
echo "  3. Run all cells to generate visualizations"
echo ""
echo "========================================"
