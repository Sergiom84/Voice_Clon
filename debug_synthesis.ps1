Param(
    [string]$EnvName = "xtts-es",
    [Parameter(Mandatory = $true)]
    [string]$ReferenceAudioPath,
    [string]$Text = "Hola, esta es una prueba corta de clonacion de voz en espanol."
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

Write-Host "[xtts-spanish-app] Ejecutando diagnostico de sintesis..." -ForegroundColor Cyan
Write-Host "Audio: $ReferenceAudioPath"
Write-Host "Texto: $Text"

$pythonSnippet = @"
from xtts_spanish_app.backend import XTTSVoiceBackend
from xtts_spanish_app.service import SpanishVoiceService

audio_path = r'''$ReferenceAudioPath'''
text = r'''$Text'''

backend = XTTSVoiceBackend()
service = SpanishVoiceService(backend=backend)

print("Runtime antes de sintetizar:", service.describe_runtime())
output_path = service.synthesize_spanish(audio_path, text)
print("Sintesis completada:", output_path)
print("Runtime despues de sintetizar:", service.describe_runtime())
"@

$tempScript = Join-Path $env:TEMP "xtts_debug_synthesis.py"
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = if ($previousPythonPath) { "$PSScriptRoot;$previousPythonPath" } else { $PSScriptRoot }
Set-Content -Path $tempScript -Value $pythonSnippet -Encoding Ascii

try {
    & $condaExe run --no-capture-output -n $EnvName python $tempScript
    exit $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Remove-Item $tempScript -ErrorAction SilentlyContinue
}
