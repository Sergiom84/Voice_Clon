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
Esta fase sigue siendo 100% local con XTTS. No integra ElevenLabs.

## Estructura

- `app.py`: punto de entrada de la UI.
- `xtts_spanish_app/backend.py`: backend XTTS y contrato `VoiceBackend`.
- `xtts_spanish_app/service.py`: validacion, preparacion de referencias, jobs por lotes y guardado del audio.
- `training_data/`: contrato de datos para una futura fase de entrenamiento.
- `outputs/jobs/<job_id>/`: artefactos de cada generacion (`segments/`, `references/`, `final.wav`, `manifest.json`).

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
   Si dura mas de `30` segundos, la app prepara automaticamente `3` referencias WAV internas de `10` segundos, una por cada tercio del audio original, escogiendo en cada tercio la zona con mas energia.
2. Escribe el texto en espanol.
3. Pulsa `Generar audio`.
4. Reproduce el resultado o descargalo.
   El estado mostrara tambien el runtime real de inferencia (`XTTS-v2 en cuda` o `XTTS-v2 en cpu`).

## Sintesis larga

- La app ya no concatena todo el audio final en memoria.
- El texto se normaliza, se separa por parrafos y se agrupa en bloques de sintesis de hasta `500` caracteres.
- Cada bloque se renderiza a un WAV intermedio dentro de `outputs/jobs/<job_id>/segments/`.
- Al terminar, la app ensambla todos los bloques por streaming en `outputs/jobs/<job_id>/final.wav`.
- La UI mantiene una sola accion, pero informa del progreso: `Preparando referencia`, `Sintetizando bloque X/Y`, `Ensamblando audio final`, `Listo`.
- Si falla un bloque, el job se detiene, conserva los artefactos ya generados y deja el detalle en `manifest.json`.

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
- chunking para sintesis larga
- validacion del audio WAV y MP3
- preparacion automatica de multiples referencias
- escritura de jobs y manifiestos sin depender de una concatenacion global en memoria
- conservacion de artefactos si falla un bloque intermedio

Ejecuta:

```powershell
python -m unittest discover -s tests -v
```

## Mejoras de consistencia (v1.1)

Para evitar problemas de pérdida de foco (hablar en otros idiomas, seseo, inconsistencias), se han implementado:

### Semilla fija para reproducibilidad
- Cada síntesis usa una semilla fija (`DEFAULT_SYNTHESIS_SEED = 42`)
- Cada bloque usa una semilla derivada: `seed + índice_del_bloque`
- Esto permite resultados reproducibles entre ejecuciones

### Chunks más pequeños
- `MAX_SYNTHESIS_CHARS = 300` (reducido desde 500)
- Bloques más cortos = menor probabilidad de alucinaciones

### Referencia única consolidada
- `DEFAULT_REFERENCE_EXCERPT_COUNT = 1` (reducido desde 3)
- `DEFAULT_REFERENCE_EXCERPT_SECONDS = 12` (aumentado desde 10)
- Una sola referencia de12 segundos del mejor segmento del audio original

### Detección de problemas y re-síntesis automática
- Cada segmento se analiza tras generarse
- Métricas: velocidad de habla, proporción de silencio, varianza de energía
- Si se detectan problemas, se re-sintetiza automáticamente (hasta 3 intentos)
- La UI muestra avisos de calidad y segmentos re-sintetizados

## Notas

- La app fuerza `language="es"` en todas las inferencias.
- Cuando la referencia es larga, XTTS recibe una unica referencia preparada del mejor segmento.
- El backend intenta cargar XTTS al arrancar. Si faltan dependencias, la UI sigue levantando pero muestra el error de entorno.
- Si el PyTorch instalado no soporta la GPU detectada, la app cae automaticamente a CPU y deja un aviso en la cabecera.
- Esta implementacion asume aceptable la `Coqui Public Model License` para uso de prototipo/personal.
