# Quick setup script for Windows (PowerShell)

Write-Host "🔧 Setting up Time Series Reconstruction Framework" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Check if experiment exists
if (Test-Path "experiment") {
    Write-Host "⚠️  Virtual environment already exists." -ForegroundColor Yellow
    $response = Read-Host "Do you want to recreate it? (y/n)"
    if ($response -ne "y") {
        Write-Host "❌ Aborted." -ForegroundColor Red
        exit 1
    }
    Write-Host "🗑️  Removing old experiment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force experiment
}

# Create experiment
Write-Host "📦 Creating virtual environment..." -ForegroundColor Cyan
python -m venv experiment

# Activate experiment
Write-Host "✅ Activating virtual environment..." -ForegroundColor Green
& .\experiment\Scripts\Activate.ps1

# Upgrade pip
Write-Host "⬆️  Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

# Install dependencies
Write-Host "📚 Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment in the future, run:" -ForegroundColor Yellow
Write-Host "  .\experiment\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To deactivate when done:" -ForegroundColor Yellow
Write-Host "  deactivate" -ForegroundColor White
Write-Host ""
Write-Host "Quick start (reconstruction pipeline — or run: make help):" -ForegroundColor Yellow
Write-Host "  1. Edit config/config.yaml" -ForegroundColor White
Write-Host "  2. python src/1_clean_datasets.py          # clean / validate" -ForegroundColor White
Write-Host "  3. python src/2_create_split.py            # train / test split" -ForegroundColor White
Write-Host "  4. python src/3_degrade_datasets.py        # missingness on train" -ForegroundColor White
Write-Host "  # optional: python src/optimization/optimize_sd_hyperparams.py" -ForegroundColor DarkGray
Write-Host "  5. python src/4_reconstruct_datasets.py    # reconstruct" -ForegroundColor White
Write-Host "  6. python src/5_calculate_reconstruction_error.py  # error metrics CSV" -ForegroundColor White
Write-Host "  7. streamlit run src/6_visualize_reconstruction_error.py  # dashboard" -ForegroundColor White
Write-Host ""

