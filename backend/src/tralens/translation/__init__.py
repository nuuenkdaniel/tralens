"""Translation API; constructors perform no provider calls."""
from .translator import Translate
from .contracts import (Reading, Utterance, Style, TranslationRequest, TranslationResult, UnitOutcome, automatic_units)
from .contracts import FragmentInput, FragmentRequest, FragmentResult, fragment_plan, FRAGMENT_POLICY_REVISION
from .runtime import (PreviewTranslator, OllamaRuntime, CapabilityProfile, GENERALIST, TRANSLATEGEMMA,
                      Generation, GenerationOptions, compact_unit, compile_source)

__all__ = ["Translate", "Reading", "Utterance", "Style", "TranslationRequest", "TranslationResult", "UnitOutcome",
           "automatic_units", "PreviewTranslator", "OllamaRuntime", "CapabilityProfile", "GENERALIST",
           "TRANSLATEGEMMA", "Generation", "GenerationOptions", "compact_unit", "compile_source",
           "FragmentInput", "FragmentRequest", "FragmentResult", "fragment_plan", "FRAGMENT_POLICY_REVISION"]
