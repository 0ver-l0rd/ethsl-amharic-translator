"""EthSL Amharic Translator — source package."""
import importlib

# Submodules are imported lazily: `import src.clip_naming` must not pay for
# MediaPipe/TensorFlow, which LandmarkExtractor pulls in (~40 s on CPU).
_EXPORTS = {
    "Vocabulary":          ".vocabulary",
    "LandmarkExtractor":   ".landmark_extractor",
    "SequenceBuilder":     ".sequence_builder",
    "AugmentationPipeline": ".augmentation",
    "AmharicTTS":          ".amharic_tts",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
