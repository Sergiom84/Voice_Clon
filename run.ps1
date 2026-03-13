Param(
    [string]$EnvName = "xtts-es",
    [int]$Port = 7860
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

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
        [string]$EnvironmentName,
        [string]$CondaExecutable
    )

    if (-not $CondaExecutable) {
        return $null
    }

    $condaRoot = Split-Path (Split-Path $CondaExecutable -Parent) -Parent
    $pythonPath = Join-Path $condaRoot "envs\\$EnvironmentName\\python.exe"
    if (Test-Path $pythonPath) {
        return $pythonPath
    }

    return $null
}

$condaExe = Get-CondaExecutable
if ($condaExe) {
    $condaPython = Get-CondaEnvPython -EnvironmentName $EnvName -CondaExecutable $condaExe
    if ($condaPython) {
        $env:COQUI_TOS_AGREED = "1"
        & $condaExe run --no-capture-output -n $EnvName python app.py --port $Port
        exit $LASTEXITCODE
    }
}

$candidatePython = @(
    (Join-Path $PSScriptRoot "..\\.venv\\Scripts\\python.exe"),
    (Join-Path $PSScriptRoot ".venv\\Scripts\\python.exe")
) | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($candidatePython) {
    $pythonExe = (Resolve-Path $candidatePython).Path
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = (Get-Command python).Source
} else {
    throw "No se encontro ni conda ni python en PATH. Activa tu .venv o instala Miniconda antes de arrancar la app."
}

$pythonVersion = (& $pythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
$versionParts = $pythonVersion.Split(".")
if ($versionParts.Length -ge 2) {
    $majorVersion = [int]$versionParts[0]
    $minorVersion = [int]$versionParts[1]
    if ($majorVersion -eq 3 -and $minorVersion -ge 12) {
        throw (
            "No se puede arrancar XTTS con Python $pythonVersion en modo local sin Conda. " +
            "Coqui TTS 0.22.0 requiere Python 3.9-3.11. " +
            "Instala y usa Miniconda (.\install.ps1) o ejecuta con un entorno Python 3.10/3.11."
        )
    }
}

& $pythonExe app.py --port $Port
