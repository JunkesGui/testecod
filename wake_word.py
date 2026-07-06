"""Detecção de palavra-chave (wake word) usando reconhecimento de fala offline (Vosk)."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from vosk import KaldiRecognizer, Model, SetLogLevel

logger = logging.getLogger(__name__)

SetLogLevel(-1)  # silencia o log interno e verboso do Vosk


class WakeWordError(Exception):
    """Erro na inicialização do modelo ou no processamento de áudio."""


class WakeWordDetector:
    """Detecta, em um fluxo contínuo de áudio, se alguma palavra/frase de ativação foi dita."""

    def __init__(self, model_path: str, sample_rate: int, wake_words: Iterable[str]) -> None:
        path = Path(model_path)
        if not path.exists():
            raise WakeWordError(
                f"Modelo Vosk não encontrado em '{model_path}'. Baixe um modelo pt-BR em "
                "https://alphacephei.com/vosk/models e ajuste 'vosk_model_path' no config.json."
            )
        self._model = Model(model_path)
        self._recognizer = KaldiRecognizer(self._model, sample_rate)
        self._wake_words = [w.lower().strip() for w in wake_words]

    def process_chunk(self, chunk: bytes) -> bool:
        """Alimenta um bloco de áudio. Retorna True se uma palavra-chave foi detectada."""
        try:
            if self._recognizer.AcceptWaveform(chunk):
                result = json.loads(self._recognizer.Result())
                text = result.get("text", "").lower()
            else:
                partial = json.loads(self._recognizer.PartialResult())
                text = partial.get("partial", "").lower()
            return self._matches(text)
        except Exception as exc:
            raise WakeWordError(
                f"Falha ao processar áudio para detecção de palavra-chave: {exc}"
            ) from exc

    def _matches(self, text: str) -> bool:
        if not text:
            return False
        return any(wake_word in text for wake_word in self._wake_words)

    def reset(self) -> None:
        """Reinicia o estado do reconhecedor após uma ativação, evitando re-disparos indevidos."""
        self._recognizer.Reset()
