# PowerShell helper: activate venv and run Streamlit dashboard
$venv = "$PSScriptRoot\..\.venv\Scripts\Activate.ps1"
if (Test-Path $venv) {
    . $venv
} else {
    Write-Host "Virtualenv activate script not found at $venv. Activate your venv manually." -ForegroundColor Yellow
}
python -m pip install --upgrade pip
python -m pip install streamlit folium streamlit-folium matplotlib pandas pillow
python -m streamlit run "$(Resolve-Path "$PSScriptRoot\..\smart_dashboard.py")"
