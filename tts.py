"""Módulo de Text-to-Speech. Isola o restante do sistema do motor TTS específico usado."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TTSError(Exception):
    """Erro ao sintetizar ou reproduzir áudio."""


class TTSEngine(ABC):
    """Interface abstrata que todo motor de TTS deve implementar."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Sintetiza e reproduz o texto informado."""


class Pyttsx3Engine(TTSEngine):
    """Motor de TTS offline baseado em pyttsx3.

    A disponibilidade de uma voz em pt-BR depende das vozes instaladas no sistema
    operacional. Em caso de ausência, o motor cai para a voz padrão do sistema.
    """

    def __init__(self, rate: int = 175, volume: float = 1.0, language: str = "pt-BR") -> None:
        import pyttsx3

        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)
        self._engine.setProperty("volume", volume)
        self._select_voice(language)

    def _select_voice(self, language: str) -> None:
        lang_key = language.lower().replace("-", "_")
        for voice in self._engine.getProperty("voices"):
            voice_langs = " ".join(str(v).lower() for v in getattr(voice, "languages", []))
            if lang_key in voice.id.lower() or "pt" in voice_langs or "brazil" in voice.name.lower():
                self._engine.setProperty("voice", voice.id)
                logger.info("Voz selecionada: %s", voice.name)
                return
        logger.warning(
            "Nenhuma voz em %s foi encontrada instalada no sistema; usando voz padrão.", language
        )

    def speak(self, text: str) -> None:
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as exc:
            raise TTSError(f"Falha ao reproduzir áudio via pyttsx3: {exc}") from exc


def create_tts_engine(engine_name: str, rate: int, volume: float, language: str) -> TTSEngine:
    """Fábrica que retorna o motor TTS configurado, desacoplando o restante do código do backend."""
    if engine_name == "pyttsx3":
        return Pyttsx3Engine(rate=rate, volume=volume, language=language)
    raise TTSError(f"Motor de TTS '{engine_name}' não é suportado.")
