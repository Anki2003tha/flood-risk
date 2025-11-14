# PowerShell helper to run the demo training in a virtual environment
param(
    [int]$epochs = 1,
    [int]$batchSize = 4
)

Write-Host "Activating virtual environment (if present)"
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
}

$cmd = "python src\train.py --demo --epochs $epochs --batch-size $batchSize --model-out model_demo.h5"
Write-Host "Running: $cmd"
Invoke-Expression $cmd