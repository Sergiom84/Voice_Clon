Param(
    [string]$EnvName = "xtts-es"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Get-CondaExecutable {
    $candidates = @(
        (Get-Command conda -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
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

    $condaRoot = Split-Path (Split-Path $CondaExecutable -Parent) -Parent
    $pythonPath = Join-Path $condaRoot "envs\\$EnvironmentName\\python.exe"
    if (Test-Path $pythonPath) {
        return $pythonPath
    }

    throw "No se encontro python.exe dentro del entorno conda '$EnvironmentName'."
}

$condaExe = Get-CondaExecutable
if (-not $condaExe) {
    throw "No se encontro conda. Instala Miniconda o asegurate de que el entorno '$EnvName' existe."
}

$pythonExe = Get-CondaEnvPython -EnvironmentName $EnvName -CondaExecutable $condaExe
$env:COQUI_TOS_AGREED = "1"

& $condaExe run --no-capture-output -n $EnvName python -c "from xtts_spanish_app.backend import XTTSVoiceBackend; XTTSVoiceBackend().load_model(); print('Modelo XTTS preparado correctamente.')"
exit $LASTEXITCODE
