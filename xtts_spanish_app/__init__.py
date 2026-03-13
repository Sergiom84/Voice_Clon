"""XTTS Spanish App."""

from .backend import VoiceBackend, XTTSVoiceBackend
from .quality_check import SegmentQualityReport, analyze_segment_quality
from .service import SpanishVoiceService

__all__ = [
    "SpanishVoiceService",
    "VoiceBackend",
    "XTTSVoiceBackend",
    "SegmentQualityReport",
    "analyze_segment_quality",
]

