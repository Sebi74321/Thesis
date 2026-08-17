param([string]$EnvironmentPath = ".venv-thesis-cpu")

$ErrorActionPreference = "Stop"
py -3.10 -m venv $EnvironmentPath
$python = Join-Path $EnvironmentPath "Scripts\python.exe"
& $python -m pip install --upgrade pip setuptools wheel
& $python -m pip install -r requirements-base.txt
& $python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& $python -c "import seaborn, torch; print('Torch:', torch.__version__); print('Seaborn:', seaborn.__version__)"
Write-Host "CPU environment ready. Activate with: $EnvironmentPath\Scripts\Activate.ps1"
