#!/bin/bash
# Quick setup script for Linux/Mac

echo "🔧 Setting up Time Series Reconstruction Framework"
echo "=================================================="

# Check if experiment exists
if [ -d "experiment" ]; then
    echo "⚠️  Virtual environment already exists."
    read -p "Do you want to recreate it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing old experiment..."
        rm -rf experiment
    else
        echo "❌ Aborted."
        exit 1
    fi
fi

# Create experiment
echo "📦 Creating virtual environment..."
python3 -m venv experiment

# Activate experiment
echo "✅ Activating virtual environment..."
source experiment/bin/activate

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
echo "  source experiment/bin/activate"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo ""
echo "Quick start:"
echo "  1. Edit config.yaml to configure your experiment"
echo "  2. python src/1_clean_datasets.py"
echo "  3. python src/2_degrade_datasets.py"
echo "  4. python src/3_reconstruct_datasets.py"
echo "  5. python src/4_calculate_mad.py"
echo "  6. streamlit run src/5_visualize_mad_comparison.py"
echo ""

