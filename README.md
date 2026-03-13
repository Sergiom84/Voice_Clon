# XTTS Spanish App

Aplicacion local para clonado de voz en espanol con `XTTS-v2`.

## Alcance v1

- Clonado por audio de referencia en espanol.
- UI local con Gradio.
- Entrada:
  - `audio_referencia`: WAV o MP3
  - `texto_es`: texto libre
- Salida:
  - `audio_generado`: WAV a 24 kHz

La v1 no implementa fine-tuning. El proyecto deja preparado el contrato de carpetas para una fase 2 sin rehacer la UI.

## Estructura

- `app.py`: punto de entrada de la UI.
- `xtts_spanish_app/backend.py`: backend XTTS y contrato `VoiceBackend`.
- `xtts_spanish_app/service.py`: validacion, segmentacion y guardado del audio.
- `training_data/`: contrato de datos para una futura fase de entrenamiento.
- `outputs/`: audios generados.

## Instalacion en Windows

1. Instala Miniconda o Anaconda y abre PowerShell.
   Si no quieres usar Conda, activa antes tu `.venv`; el instalador ahora puede usar el Python activo.
   Importante: `TTS==0.22.0` solo soporta Python `3.9-3.11` (no `3.12+`).
2. Ve a esta carpeta:

```powershell
cd C:\Users\sergi\Desktop\Aplicaciones\Voice_Clon\xtts_spanish_app
```

3. Ejecuta el instalador:

```powershell
.\install.ps1 -Device CU128
```

Si quieres forzar la instalacion en el Python activo aunque tengas Conda:

```powershell
.\install.ps1 -Device CU128 -UseCurrentPython
```

Opciones validas para `-Device`:

- `CU126`: PyTorch con CUDA 12.6
- `CU128`: PyTorch con CUDA 12.8
- `CU124`: PyTorch con CUDA 12.4
- `CPU`: ejecucion por CPU

Para GPUs recientes de la serie RTX 50, usa `CU128`.
Si ya instalaste antes con `CU126`, vuelve a ejecutar `.\install.ps1 -Device CU128`: el script ahora fuerza la reinstalacion de PyTorch para cambiar de build CUDA.

## Arranque

```powershell
.\run.ps1
```

El script arranca la UI en `http://127.0.0.1:7860` usando Conda si existe, o el Python activo si no existe.

## Preparacion inicial del modelo XTTS

La primera vez, XTTS necesita descargar el modelo y dejar constancia de aceptacion de la licencia CPML de Coqui.

Ejecuta una vez:

```powershell
.\prepare_model.ps1 -EnvName xtts-es
```

Despues, arranca de nuevo la app:

```powershell
.\run.ps1 -EnvName xtts-es
```

## Uso

1. Sube un `WAV` o `MP3` limpio.
   Si dura mas de `30` segundos, la app recorta automaticamente un tramo util antes de sintetizar.
2. Escribe el texto en espanol.
3. Pulsa `Generar audio`.
4. Reproduce el resultado o descargalo.
   El estado mostrara tambien el runtime real de inferencia (`XTTS-v2 en cuda` o `XTTS-v2 en cpu`).

## Contrato para una futura fase 2

La carpeta `training_data/` queda reservada para este formato:

```text
training_data/
  <speaker_id>/
    raw/
    processed/
    metadata.json
```

En v2, la UI podria anadir una segunda ruta de backend basada en fine-tuning sin cambiar la interfaz principal de inferencia.

## Verificacion rapida

Los tests incluidos validan:

- limpieza y segmentacion de texto
- validacion del audio WAV
- guardado de audios sin sobrescritura

Ejecuta:

```powershell
python -m unittest discover -s tests -v
```

## Notas

- La app fuerza `language="es"` en todas las inferencias.
- El backend intenta cargar XTTS al arrancar. Si faltan dependencias, la UI sigue levantando pero muestra el error de entorno.
- Si el PyTorch instalado no soporta la GPU detectada, la app cae automaticamente a CPU y deja un aviso en la cabecera.
- Esta implementacion asume aceptable la `Coqui Public Model License` para uso de prototipo/personal.
