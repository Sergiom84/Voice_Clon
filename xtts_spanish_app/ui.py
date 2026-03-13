from __future__ import annotations

from typing import Any

from .settings import (
    FIDELITY_MODE_MAXIMUM,
    REFERENCE_PROFILE_AUTO,
)
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

    def generate_audio(
        reference_audio_path: str | None,
        text_es: str,
        reference_profile: str,
        fidelity_mode: str,
        speaker_profile_name: str,
        save_speaker_profile: bool,
        progress=gr.Progress(track_tqdm=False),
    ) -> tuple[str | None, str | None, str]:
        latest_status = "Preparando referencia..."

        def update_progress(status: str, step: int, total_steps: int) -> None:
            nonlocal latest_status
            latest_status = status
            progress(step / max(total_steps, 1), desc=status)

        try:
            result = service.synthesize_spanish(
                reference_audio_path=reference_audio_path,
                text_es=text_es,
                reference_profile=reference_profile,
                fidelity_mode=fidelity_mode,
                speaker_profile_name=speaker_profile_name,
                save_speaker_profile=save_speaker_profile,
                progress_callback=update_progress,
            )
            runtime_status = service.describe_runtime()
            progress(1.0, desc="Listo.")
            status_message = (
                f"Audio generado correctamente. Runtime: {runtime_status}. "
                f"Bloques: {result.segment_count}. Job: {result.job_dir}. "
                f"Perfil referencia: {result.reference_profile}. Fidelidad: {result.fidelity_mode}."
            )
            operation_note = result.operation_note
            if operation_note:
                status_message = f"{status_message} {operation_note}"

            if result.resynthesized_count > 0:
                status_message = f"{status_message} Re-sintetizados: {result.resynthesized_count}."

            if result.quality_issues:
                issues_summary = "; ".join(result.quality_issues[:5])
                if len(result.quality_issues) > 5:
                    issues_summary += f" ... y {len(result.quality_issues) - 5} más"
                status_message = f"{status_message} Avisos de calidad: {issues_summary}"

            if result.speaker_similarity_avg is not None:
                status_message = (
                    f"{status_message} Similitud speaker media/mínima: "
                    f"{result.speaker_similarity_avg:.3f}/{(result.speaker_similarity_min or 0.0):.3f}."
                )

            if result.similarity_warnings:
                similarity_summary = "; ".join(result.similarity_warnings[:3])
                if len(result.similarity_warnings) > 3:
                    similarity_summary += f" ... y {len(result.similarity_warnings) - 3} más"
                status_message = f"{status_message} Avisos de similitud: {similarity_summary}"

            if result.speaker_profile_used:
                status_message = f"{status_message} Speaker profile usado: {result.speaker_profile_used}."

            if result.speaker_profile_saved:
                status_message = f"{status_message} Speaker profile guardado/actualizado: {result.speaker_profile_saved}."
                if result.speaker_profile_recommendation:
                    status_message = f"{status_message} {result.speaker_profile_recommendation}"

            return result.final_audio_path, result.final_audio_path, status_message
        except ValidationError as exc:
            return None, None, str(exc)
        except Exception as exc:  # pragma: no cover - depende del entorno/modelo
            return None, None, f"{latest_status} Error de sintesis: {exc}"

    with gr.Blocks(title="XTTS Spanish App", analytics_enabled=False) as app:
        gr.Markdown("# XTTS Spanish App")
        gr.Markdown(
            "Clonado de voz en espanol con XTTS-v2. "
            "Sube un WAV o MP3 limpio y escribe el texto que quieres sintetizar. "
            "Si el audio es largo, la app prepara automaticamente varias referencias utiles para clonar la voz."
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

        with gr.Row():
            reference_profile = gr.Radio(
                label="Perfil de referencia",
                choices=[
                    ("Auto", "auto"),
                    ("Corto (1-3 min)", "short"),
                    ("Largo (5-30 min)", "long"),
                ],
                value=REFERENCE_PROFILE_AUTO,
            )
            fidelity_mode = gr.Radio(
                label="Modo de fidelidad",
                choices=[
                    ("Máxima", "maxima"),
                    ("Normal", "normal"),
                ],
                value=FIDELITY_MODE_MAXIMUM,
            )

        with gr.Row():
            speaker_profile_name = gr.Textbox(
                label="Speaker profile (opcional)",
                lines=1,
                max_lines=1,
                placeholder="ejemplo: voz_lara (si no subes audio, intentará usar este perfil)",
            )
            save_speaker_profile = gr.Checkbox(
                label="Guardar/actualizar speaker profile con este audio",
                value=False,
            )

        gr.Markdown(
            "La referencia se sube como archivo para evitar dependencias externas de previsualizacion en Windows. "
            "Si activas speaker profile, la app deja preparado dataset de segmentacion+transcripcion para una fase de entrenamiento."
        )

        generate_button = gr.Button("Generar audio", variant="primary")
        generated_audio = gr.Audio(label="Audio generado", type="filepath", interactive=False)
        generated_file = gr.File(label="Descargar WAV", interactive=False)

        generate_button.click(
            fn=generate_audio,
            inputs=[reference_audio, text_es, reference_profile, fidelity_mode, speaker_profile_name, save_speaker_profile],
            outputs=[generated_audio, generated_file, status_box],
            queue=True,
        )

    return app
