Param(
    [ValidateSet("CU128", "CU126", "CU124", "CPU")]
    [string]$Device = "CU128",
    [string]$EnvName = "xtts-es",
    [switch]$UseCurrentPython
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Write-Step($Message) {
    Write-Host "[xtts-spanish-app]" -ForegroundColor Cyan -NoNewline
    Write-Host " $Message"
}

function Invoke-CondaCommand {
    param(
        [string[]]$Arguments,
        [string]$FailureMessage
    )

    & $script:CondaExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw $FailureMessage
    }
}

function Get-CondaExecutable {
    $candidates = @()

    $condaCommand = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCommand) {
        $candidates += $condaCommand.Source
    }

    $candidates += @(
        (Join-Path $env:USERPROFILE "miniconda3\\condabin\\conda.bat"),
        (Join-Path $env:USERPROFILE "anaconda3\\condabin\\conda.bat"),
        "C:\\Miniconda3\\condabin\\conda.bat",
        "C:\\Anaconda3\\condabin\\conda.bat"
    )

    return $candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}

function Get-CondaEnvPython {
    param(
        [string]$EnvironmentName
    )

    if (-not $script:CondaExe) {
        return $null
    }

    $condaRoot = Split-Path (Split-Path $script:CondaExe -Parent) -Parent
    $pythonPath = Join-Path $condaRoot "envs\\$EnvironmentName\\python.exe"
    if (Test-Path $pythonPath) {
        return $pythonPath
    }

    return $null
}

function Invoke-InstallCommand {
    param(
        [string[]]$Arguments
    )

    if ($script:UseConda) {
        & $script:PythonExe -m pip @Arguments
    } else {
        & $script:PythonExe -m pip @Arguments
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Fallo ejecutando pip con argumentos: $($Arguments -join ' ')"
    }
}

$torchPackages = "torch torchvision torchaudio"
$torchIndexUrl = switch ($Device) {
    "CU128" { "https://download.pytorch.org/whl/cu128" }
    "CU126" { "https://download.pytorch.org/whl/cu126" }
    "CU124" { "https://download.pytorch.org/whl/cu124" }
    "CPU" { "https://download.pytorch.org/whl/cpu" }
}

$script:UseConda = $false
$script:PythonExe = $null
$script:CondaExe = Get-CondaExecutable
if (-not $UseCurrentPython -and $script:CondaExe) {
    $script:UseConda = $true
}

if ($script:UseConda) {
    Write-Step "Creando o reutilizando entorno conda '$EnvName'..."
    $envExists = (& $script:CondaExe env list) -match "^\s*$EnvName\s"
    if (-not $envExists) {
        Invoke-CondaCommand `
            -Arguments @("create", "-n", $EnvName, "python=3.10", "-y") `
            -FailureMessage (
                "No se pudo crear el entorno conda '$EnvName'. " +
                "Si acabas de instalar Miniconda, acepta primero los Terms of Service con 'conda tos accept ...' y vuelve a ejecutar este script."
            )
    }
    $script:PythonExe = Get-CondaEnvPython -EnvironmentName $EnvName
    if (-not $script:PythonExe) {
        throw "No se encontro python.exe dentro del entorno conda '$EnvName'."
    }
} else {
    $candidatePython = @(
        (Join-Path $PSScriptRoot "..\\.venv\\Scripts\\python.exe"),
        (Join-Path $PSScriptRoot ".venv\\Scripts\\python.exe")
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1

    if ($candidatePython) {
        $script:PythonExe = (Resolve-Path $candidatePython).Path
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $script:PythonExe = (Get-Command python).Source
    } else {
        throw "No se encontro ni conda ni python en PATH. Instala Python 3.10+ o activa tu .venv antes de ejecutar este script."
    }

    $pythonVersion = (& $script:PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
    Write-Step "Conda no disponible. Se instalara en Python: $script:PythonExe"
    Write-Step "Version detectada: $pythonVersion"

    $versionParts = $pythonVersion.Split(".")
    if ($versionParts.Length -ge 2) {
        $majorVersion = [int]$versionParts[0]
        $minorVersion = [int]$versionParts[1]
        if ($majorVersion -eq 3 -and $minorVersion -ge 12) {
            throw (
                "Este proyecto usa Coqui TTS 0.22.0, que requiere Python 3.9-3.11. " +
                "Se detecto Python $pythonVersion. Usa Conda (recomendado) o crea un entorno con Python 3.10/3.11."
            )
        }
    }
}

Write-Step "Actualizando pip, setuptools y wheel..."
Invoke-InstallCommand -Arguments @("install", "--upgrade", "pip", "wheel", "setuptools<81")

Write-Step "Instalando PyTorch para $Device..."
Invoke-InstallCommand -Arguments @("install", "--upgrade", "--force-reinstall", "torch", "torchvision", "torchaudio", "--index-url", $torchIndexUrl)

Write-Step "Instalando dependencias de la app..."
Invoke-InstallCommand -Arguments @("install", "--upgrade", "-r", "requirements.txt")

Write-Step "Instalacion completada."
Write-Host "Arranque sugerido:" -ForegroundColor Green
if ($script:UseConda) {
    Write-Host "  .\run.ps1 -EnvName $EnvName"
} else {
    Write-Host "  .\run.ps1"
}
