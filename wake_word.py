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

    # Recebe um dicionário agora
    def __init__(self, model_path: str, sample_rate: int, wake_words: dict[str, list[str]]) -> None:
        path = Path(model_path)
        if not path.exists():
            raise WakeWordError(f"Modelo Vosk não encontrado em '{model_path}'.")
        self._model = Model(model_path)
        self._recognizer = KaldiRecognizer(self._model, sample_rate)
        
        # Converte todas as palavras do dicionário para minúsculas
        self._wake_words = {
            mode: [w.lower().strip() for w in words] 
            for mode, words in wake_words.items()
        }

    # Retorna uma string (o modo) ou None
    def process_chunk(self, chunk: bytes) -> str | None:
        """Alimenta um bloco de áudio. Retorna o modo detectado ou None."""
        try:
            if self._recognizer.AcceptWaveform(chunk):
                result = json.loads(self._recognizer.Result())
                text = result.get("text", "").lower()
            else:
                partial = json.loads(self._recognizer.PartialResult())
                text = partial.get("partial", "").lower()
            return self._get_detected_mode(text)
        except Exception as exc:
            raise WakeWordError(f"Falha ao processar áudio: {exc}") from exc

    def _get_detected_mode(self, text: str) -> str | None:
        if not text:
            return None
        # Verifica qual modo teve uma de suas palavras ditas
        for mode, words in self._wake_words.items():
            if any(wake_word in text for wake_word in words):
                return mode
        return None

    def reset(self) -> None:
        """Reinicia o estado do reconhecedor após uma ativação, evitando re-disparos indevidos."""
        self._recognizer.Reset()
