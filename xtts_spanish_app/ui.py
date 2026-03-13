from __future__ import annotations

from typing import Any

from .service import SpanishVoiceService
from .validation import ValidationError


def build_app(service: SpanishVoiceService, startup_error: str | None = None) -> Any:
    try:
        import gradio as gr
    except ModuleNotFoundError as exc:  # pragma: no cover - depende del entorno
        raise RuntimeError("Gradio no esta instalado. Ejecuta .\\install.ps1 antes de iniciar la app.") from exc

    runtime_status = service.describe_runtime()
    startup_message = (
        f"Backend listo: {runtime_status}"
        if startup_error is None
        else (
            "La UI ha arrancado, pero XTTS no se pudo precargar. "
            f"Corrige el entorno y vuelve a probar. Detalle: {startup_error}"
        )
    )

    def generate_audio(reference_audio_path: str | None, text_es: str) -> tuple[str | None, str | None, str]:
        try:
            output_path = service.synthesize_spanish(reference_audio_path=reference_audio_path, text_es=text_es)
            runtime_status = service.describe_runtime()
            status_message = f"Audio generado correctamente. Runtime: {runtime_status}."
            operation_note = service.get_last_operation_note()
            if operation_note:
                status_message = f"{status_message} {operation_note}"
            return output_path, output_path, status_message
        except ValidationError as exc:
            return None, None, str(exc)
        except Exception as exc:  # pragma: no cover - depende del entorno/modelo
            return None, None, f"Error de sintesis: {exc}"

    def mark_generation_started() -> str:
        return "Generando audio..."

    with gr.Blocks(title="XTTS Spanish App", analytics_enabled=False) as app:
        gr.Markdown("# XTTS Spanish App")
        gr.Markdown(
            "Clonado de voz en espanol con XTTS-v2. "
            "Sube un WAV o MP3 limpio y escribe el texto que quieres sintetizar. "
            "Si el audio es largo, la app recorta automaticamente un tramo util para clonar la voz."
        )
        status_box = gr.Markdown(startup_message)

        with gr.Row():
            reference_audio = gr.File(
                label="Archivo de referencia (WAV o MP3)",
                type="filepath",
                file_types=[".wav", ".mp3"],
                file_count="single",
            )
            text_es = gr.Textbox(
                label="Texto en espanol",
                lines=10,
                max_lines=16,
                placeholder="Escribe aqui el texto que quieres convertir en audio...",
            )

        gr.Markdown(
            "La referencia se sube como archivo para evitar dependencias externas de previsualizacion en Windows."
        )

        generate_button = gr.Button("Generar audio", variant="primary")
        generated_audio = gr.Audio(label="Audio generado", type="filepath", interactive=False)
        generated_file = gr.File(label="Descargar WAV", interactive=False)

        generate_button.click(
            fn=mark_generation_started,
            outputs=[status_box],
            queue=False,
        ).then(
            fn=generate_audio,
            inputs=[reference_audio, text_es],
            outputs=[generated_audio, generated_file, status_box],
        )

    return app
