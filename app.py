from __future__ import annotations

import argparse

from xtts_spanish_app.backend import XTTSVoiceBackend
from xtts_spanish_app.service import SpanishVoiceService
from xtts_spanish_app.settings import SERVER_HOST, SERVER_PORT
from xtts_spanish_app.ui import build_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="App local de clonado de voz en espanol con XTTS-v2.")
    parser.add_argument("--host", default=SERVER_HOST, help="Host de Gradio.")
    parser.add_argument("--port", type=int, default=SERVER_PORT, help="Puerto de Gradio.")
    parser.add_argument(
        "--skip-preload",
        action="store_true",
        help="No precarga el modelo XTTS al arrancar. Util para debug o entornos sin dependencias listas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = XTTSVoiceBackend()
    service = SpanishVoiceService(backend=backend)
    startup_error = None

    if not args.skip_preload:
        try:
            backend.load_model()
        except Exception as exc:  # pragma: no cover - depende del entorno
            startup_error = str(exc)

    app = build_app(service=service, startup_error=startup_error)
    app.launch(server_name=args.host, server_port=args.port, show_api=False, share=False, inbrowser=False)


if __name__ == "__main__":
    main()

