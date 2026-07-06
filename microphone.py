"""Módulo responsável pela captura contínua de áudio do microfone."""
from __future__ import annotations

import logging
import queue
from typing import Optional

import sounddevice as sd

logger = logging.getLogger(__name__)


class MicrophoneError(Exception):
    """Erro relacionado à abertura ou leitura do microfone."""


class Microphone:
    """Captura áudio continuamente em background, sem bloquear o restante do sistema."""

    def __init__(
        self,
        sample_rate: int = 16000,
        device: Optional[int] = None,
        block_size: int = 8000,
    ) -> None:
        self._sample_rate = sample_rate
        self._device = device
        self._block_size = block_size
        self._queue: "queue.Queue[bytes]" = queue.Queue()
        self._stream: Optional[sd.RawInputStream] = None

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.debug("Status do stream de áudio: %s", status)
        self._queue.put(bytes(indata))

    def start(self) -> None:
        """Inicia a captura contínua em uma stream de baixa latência."""
        try:
            self._stream = sd.RawInputStream(
                samplerate=self._sample_rate,
                blocksize=self._block_size,
                device=self._device,
                dtype="int16",
                channels=1,
                callback=self._callback,
            )
            self._stream.start()
            logger.info("Microfone iniciado (taxa de amostragem=%s).", self._sample_rate)
        except Exception as exc:
            raise MicrophoneError(f"Não foi possível iniciar o microfone: {exc}") from exc

    def read_chunk(self, timeout: float = 1.0) -> Optional[bytes]:
        """Retorna o próximo bloco de áudio, ou None se nada chegar dentro do timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        
    def clear_queue(self) -> None:
        """Limpa todo o áudio acumulado na fila."""
        with self._queue.mutex:
            self._queue.queue.clear()
        logger.debug("Fila do microfone limpa.")

    def stop(self) -> None:
        """Para e libera o stream de áudio."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Microfone parado.")
