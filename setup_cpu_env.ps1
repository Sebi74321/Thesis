param(
    [string]$EnvironmentPath = ".venv-thesis310-cpu",
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"
$repoRoot = $PSScriptRoot
if (-not $LogPath) {
    $LogPath = Join-Path $repoRoot "setup_thesis310_cpu.log"
}

Start-Transcript -Path $LogPath -Append
try {
    Write-Host "Starting Python 3.10 CPU thesis environment setup..."
    Get-Date

    $baseVersion = & py -3.10 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($baseVersion.Trim() -ne "3.10") {
        throw "CPU setup requires Python 3.10; py -3.10 reported $baseVersion"
    }

    if (-not (Test-Path -LiteralPath $EnvironmentPath)) {
        Write-Host "Creating environment: $EnvironmentPath"
        & py -3.10 -m venv $EnvironmentPath
    }
    else {
        Write-Host "Environment already exists: $EnvironmentPath"
    }

    $environmentRoot = (Resolve-Path -LiteralPath $EnvironmentPath).Path
    $python = Join-Path $environmentRoot "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python)) {
        throw "Python executable not found in environment: $python"
    }

    & $python --version
    & $python -m pip install --upgrade pip==24.2 setuptools==75.1.0 wheel==0.44.0
    & $python -m pip install -r (Join-Path $repoRoot "requirements-base.txt")
    & $python -m pip install `
        torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 `
        --index-url https://download.pytorch.org/whl/cpu
    & $python -m pip install notebook==7.2.2 jupyterlab==4.2.5 ipykernel==6.29.5

    $kernelName = "thesis310-cpu"
    $kernelRoot = Join-Path $env:APPDATA "jupyter\kernels"
    $kernelDirectory = Join-Path $kernelRoot $kernelName
    if (Test-Path -LiteralPath $kernelDirectory) {
        Remove-Item -LiteralPath $kernelDirectory -Recurse -Force
    }
    & $python -m ipykernel install --user `
        --name $kernelName `
        --display-name "Python 3.10 Thesis CPU"

    & $python -c @"
from importlib.metadata import version
import matplotlib, numpy, pandas, scipy, seaborn, shap, sklearn, torch
print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("SciPy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("SHAP:", shap.__version__)
print("dython:", version("dython"))
print("tqdm:", version("tqdm"))
print("Matplotlib:", matplotlib.__version__)
print("Seaborn:", seaborn.__version__)
print("pytest:", version("pytest"))
print("JupyterLab:", version("jupyterlab"))
print("Notebook:", version("notebook"))
print("ipykernel:", version("ipykernel"))
"@

    Write-Host ""
    Write-Host "CPU environment ready."
    Write-Host "Log saved to: $LogPath"
    Write-Host "Use Jupyter kernel: Python 3.10 Thesis CPU"
    Write-Host "Activate with: $environmentRoot\Scripts\Activate.ps1"
}
finally {
    Stop-Transcript
}
