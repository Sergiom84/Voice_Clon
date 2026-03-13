# XTTS Spanish App

Aplicacion local para clonado de voz en espanol con `XTTS-v2`.

## Alcance v1

- Clonado por audio de referencia en espanol.
- UI local con Gradio.
- Entrada:
  - `audio_referencia`: WAV o MP3
  - `texto_es`: texto libre
  - `perfil_referencia`: `Auto`, `Corto (1-3 min)`, `Largo (5-30 min)`
  - `modo_fidelidad`: `Normal` o `Máxima`
  - `speaker_profile` opcional para reutilizar/adaptar una voz entre jobs
- Salida:
  - `audio_generado`: WAV a 24 kHz

La v1 sigue siendo zero-shot (sin fine-tuning obligatorio), pero ahora deja preparado un pipeline de adaptacion por `speaker_profile` (segmentacion + transcripcion pendiente) para una fase de entrenamiento.
Esta fase es 100% local con XTTS. No integra ElevenLabs.

## Estructura

- `app.py`: punto de entrada de la UI.
- `xtts_spanish_app/backend.py`: backend XTTS y contrato `VoiceBackend`.
- `xtts_spanish_app/service.py`: validacion, seleccion de perfil, ranking de candidatos por similitud y jobs por lotes.
- `xtts_spanish_app/reference_profiles.py`: construccion de referencia consolidada por perfil corto/largo.
- `xtts_spanish_app/speaker_similarity.py`: embedding y similitud coseno de speaker para ranking.
- `xtts_spanish_app/speaker_profiles.py`: persistencia de perfiles, dataset segmentado y estado para futura adaptacion.
- `training_data/speaker_profiles/<nombre>/`: referencia consolidada + manifest + dataset segmentado.
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
   Tambien puedes dejar el campo vacio y usar un `speaker_profile` ya guardado.
2. Escribe el texto en espanol.
3. Elige perfil de referencia:
   - `Auto`: decide segun duracion.
   - `Corto (1-3 min)`: referencia consolidada de ~18 s.
   - `Largo (5-30 min)`: referencia consolidada de ~24 s (inicio/medio/final).
4. Elige modo de fidelidad:
   - `Normal`: sintesis estable con verificacion de calidad.
   - `Máxima`: 3 candidatos por bloque + ranking por calidad y similitud de speaker.
5. Opcional: define un `speaker_profile` y activa guardar para dejar listo dataset de adaptacion.
6. Pulsa `Generar audio`.
7. Reproduce el resultado o descargalo.
   El estado mostrara tambien el runtime real de inferencia (`XTTS-v2 en cuda` o `XTTS-v2 en cpu`).

## Sintesis larga

- La app ya no concatena todo el audio final en memoria.
- El texto se normaliza, se separa por parrafos y se agrupa en bloques de sintesis de hasta `300` caracteres.
- Cada bloque se renderiza a un WAV intermedio dentro de `outputs/jobs/<job_id>/segments/`.
- Al terminar, la app ensambla todos los bloques por streaming en `outputs/jobs/<job_id>/final.wav`.
- La UI mantiene una sola accion, pero informa del progreso: `Preparando referencia`, `Sintetizando bloque X/Y`, `Ensamblando audio final`, `Listo`.
- Si falla un bloque, el job se detiene, conserva los artefactos ya generados y deja el detalle en `manifest.json`.
- El `manifest.json` ahora incluye `reference_profile`, `fidelity_mode`, `selected_seed`, `speaker_similarity`, `candidate_count` y avisos de similitud.

## Speaker profile y adaptacion (fase preparada)

Cuando guardas un `speaker_profile`, la app crea:

```text
training_data/
  speaker_profiles/
    <nombre>/
      manifest.json
      reference.wav
      dataset_manifest.json
      dataset_segments/
        segment_0001.wav
        ...
```

`dataset_manifest.json` deja lista la segmentacion y marca `transcription_status: pending` para conectar despues el entrenamiento.
Recomendacion de datos para adaptacion:
- minimo util: `>=10` minutos
- ideal: `15-30` minutos limpios y consistentes

## Verificacion rapida

Los tests incluidos validan:

- limpieza y segmentacion de texto
- chunking para sintesis larga
- validacion del audio WAV y MP3
- seleccion auto de perfil corto/largo y referencia consolidada
- seleccion determinista del mejor candidato en modo `Máxima`
- escritura de jobs y manifiestos sin depender de una concatenacion global en memoria
- conservacion de artefactos si falla un bloque intermedio

Ejecuta:

```powershell
conda run --no-capture-output -n xtts-es python -m unittest discover -s tests -v
```

## Mejoras de consistencia y fidelidad (v1.2)

Para evitar problemas de pérdida de foco (hablar en otros idiomas, seseo, inconsistencias), se han implementado:

### Semilla fija para reproducibilidad
- Cada síntesis usa una semilla fija (`DEFAULT_SYNTHESIS_SEED = 42`)
- Cada bloque usa una semilla derivada: `seed + índice_del_bloque`
- Esto permite resultados reproducibles entre ejecuciones

### Chunks más pequeños
- `MAX_SYNTHESIS_CHARS = 300` (reducido desde 500)
- Bloques más cortos = menor probabilidad de alucinaciones

### Perfil corto/largo con referencia consolidada
- `Auto`: selecciona perfil segun duracion de entrada
- `Corto`: ~18 segundos consolidados
- `Largo`: ~24 segundos consolidados (inicio/medio/final)
- Se usa una referencia final unificada para reducir deriva de timbre/prosodia entre bloques

### Detección de problemas y re-síntesis
- Cada segmento se analiza tras generarse
- Métricas: velocidad de habla, proporción de silencio, varianza de energía
- En modo `Normal`, si hay problemas se re-sintetiza (hasta 3 intentos)
- La UI muestra avisos de calidad y segmentos re-sintetizados

### Modo Máxima fidelidad (nuevo)
- Genera `3` candidatos por bloque con seeds deterministas
- Calcula similitud de speaker (coseno sobre embedding acústico)
- Selecciona el mejor candidato por score compuesto calidad/similitud
- Expone similitud media y minima por job, con avisos de umbral

## Notas

- La app fuerza `language="es"` en todas las inferencias.
- La clonacion zero-shot mejora bastante con esta estrategia, pero no garantiza identidad 100%.
- Los `speaker_profile` preparados permiten reutilizar voz y dejar lista la futura ruta de adaptacion/entrenamiento.
- El backend intenta cargar XTTS al arrancar. Si faltan dependencias, la UI sigue levantando pero muestra el error de entorno.
- Si el PyTorch instalado no soporta la GPU detectada, la app cae automaticamente a CPU y deja un aviso en la cabecera.
- Esta implementacion asume aceptable la `Coqui Public Model License` para uso de prototipo/personal.
