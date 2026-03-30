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
echo "Quick start (reconstruction pipeline — or run: make help):"
echo "  1. Edit config/config.yaml"
echo "  2. python src/1_clean_datasets.py          # clean / validate"
echo "  3. python src/2_create_split.py            # train / test split"
echo "  4. python src/3_degrade_datasets.py        # missingness on train"
echo "  # optional: python src/optimization/optimize_sd_hyperparams.py"
echo "  5. python src/4_reconstruct_datasets.py    # reconstruct"
echo "  6. python src/5_calculate_reconstruction_error.py  # error metrics CSV"
echo "  7. streamlit run src/6_visualize_reconstruction_error.py  # dashboard"
echo ""

