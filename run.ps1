# PowerShell script for Windows users
# Alternative to Makefile

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [string[]]$Datasets = @("all"),
    [string[]]$Techniques = @("MCAR", "MAR", "MNAR"),
    [double[]]$Rates = @(0.02, 0.05, 0.10),
    [int]$Iterations = 5,
    [string[]]$Models = @("interpolate_linear"),
    [int]$Seed = 42
)

function Show-Help {
    Write-Host "Time Series Reconstruction Framework" -ForegroundColor Cyan
    Write-Host "=====================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Available commands:" -ForegroundColor Yellow
    Write-Host "  .\run.ps1 degrade          - Degrade source datasets"
    Write-Host "  .\run.ps1 reconstruct      - Reconstruct degraded datasets"
    Write-Host "  .\run.ps1 calculate        - Calculate reconstruction errors"
    Write-Host "  .\run.ps1 visualize        - Launch visualization dashboard"
    Write-Host "  .\run.ps1 pipeline         - Run complete pipeline"
    Write-Host "  .\run.ps1 clean            - Clean generated data"
    Write-Host "  .\run.ps1 install          - Install required packages"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Yellow
    Write-Host "  .\run.ps1 degrade -Techniques MCAR,MAR -Rates 0.05,0.10 -Iterations 3"
    Write-Host "  .\run.ps1 reconstruct -Models interpolate_linear,knn"
    Write-Host "  .\run.ps1 pipeline"
}

function Install-Dependencies {
    Write-Host "Installing required packages..." -ForegroundColor Cyan
    python -m pip install --upgrade pip
    python -m pip install pandas numpy scipy scikit-learn statsmodels
    python -m pip install torch torchvision diffusers transformers pillow
    python -m pip install streamlit plotly
    Write-Host "✓ Installation complete" -ForegroundColor Green
}

function Invoke-Degrade {
    Write-Host "Degrading datasets..." -ForegroundColor Cyan
    
    $datasetsStr = $Datasets -join " "
    $techniquesStr = $Techniques -join " "
    $ratesStr = $Rates -join " "
    
    python degrade_datasets.py `
        --datasets $datasetsStr `
        --techniques $techniquesStr `
        --rates $ratesStr `
        --iterations $Iterations `
        --seed $Seed
    
    Write-Host "✓ Degradation complete" -ForegroundColor Green
}

function Invoke-Reconstruct {
    Write-Host "Reconstructing datasets..." -ForegroundColor Cyan
    
    $modelsStr = $Models -join " "
    
    python reconstruct_datasets.py --models $modelsStr
    
    Write-Host "✓ Reconstruction complete" -ForegroundColor Green
}

function Invoke-Calculate {
    Write-Host "Calculating reconstruction errors..." -ForegroundColor Cyan
    
    python calculate_differences.py
    
    Write-Host "✓ Calculation complete" -ForegroundColor Green
}

function Invoke-Visualize {
    Write-Host "Launching visualization dashboard..." -ForegroundColor Cyan
    Write-Host "Open your browser at http://localhost:8501" -ForegroundColor Yellow
    
    streamlit run visualization.py
}

function Invoke-Pipeline {
    Invoke-Degrade
    Invoke-Reconstruct
    Invoke-Calculate
    
    Write-Host ""
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "✓ PIPELINE COMPLETE" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    Write-Host "To visualize results, run: .\run.ps1 visualize" -ForegroundColor Yellow
}

function Invoke-Clean {
    Write-Host "Cleaning generated data..." -ForegroundColor Cyan
    
    if (Test-Path "data\2_missing_data") {
        Remove-Item "data\2_missing_data\*.csv" -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path "data\3_fixed_data") {
        Remove-Item "data\3_fixed_data\*.csv" -Force -ErrorAction SilentlyContinue
    }
    
    Write-Host "✓ Clean complete" -ForegroundColor Green
}

function Invoke-CleanAll {
    Invoke-Clean
    
    Write-Host "Cleaning all data including results..." -ForegroundColor Cyan
    
    if (Test-Path "experiments_results") {
        Remove-Item "experiments_results\*.csv" -Force -ErrorAction SilentlyContinue
    }
    
    Write-Host "✓ Clean all complete" -ForegroundColor Green
}

# Main command router
switch ($Command.ToLower()) {
    "help" { Show-Help }
    "install" { Install-Dependencies }
    "degrade" { Invoke-Degrade }
    "reconstruct" { Invoke-Reconstruct }
    "calculate" { Invoke-Calculate }
    "visualize" { Invoke-Visualize }
    "pipeline" { Invoke-Pipeline }
    "clean" { Invoke-Clean }
    "clean-all" { Invoke-CleanAll }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host ""
        Show-Help
    }
}

