#!/bin/bash
# Quick setup script for Linux/Mac

echo "🔧 Setting up Time Series Reconstruction Framework"
echo "=================================================="

# Check if venv exists
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists."
    read -p "Do you want to recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing old venv..."
        rm -rf venv
    else
        echo "❌ Aborted."
        exit 1
    fi
fi

# Create venv
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate venv
echo "✅ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo "=================================================="
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""
echo "Quick start:"
echo "  1. Edit config.yaml to configure your experiment"
echo "  2. python degrade_datasets.py"
echo "  3. python reconstruct_datasets.py"
echo "  4. python calculate_mad.py"
echo "  5. streamlit run visualize_mad_comparison.py"
echo ""

