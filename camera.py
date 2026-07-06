"""Módulo responsável exclusivamente pela captura de imagens da câmera."""
from __future__ import annotations

import base64
import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraError(Exception):
    """Erro relacionado à abertura, leitura ou liberação da câmera."""


class Camera:
    """Encapsula o acesso ao dispositivo de câmera via OpenCV."""

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._capture: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Abre a câmera padrão. Lança CameraError se indisponível."""
        self._capture = cv2.VideoCapture(self._device_index)
        if not self._capture.isOpened():
            self._capture = None
            raise CameraError(f"Não foi possível abrir a câmera de índice {self._device_index}.")
        logger.info("Câmera %s aberta com sucesso.", self._device_index)

    def is_available(self) -> bool:
        """Indica se a câmera está aberta e pronta para uso."""
        return self._capture is not None and self._capture.isOpened()

    def capture_frame(self) -> np.ndarray:
        """Captura um único frame sob demanda. Lança CameraError em caso de falha."""
        if not self.is_available():
            raise CameraError("Câmera não está disponível para captura.")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraError("Falha ao capturar frame da câmera.")
        return frame

    def close(self) -> None:
        """Libera os recursos da câmera."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Câmera liberada.")


def encode_frame_to_jpeg_base64(frame: np.ndarray) -> str:
    """Codifica um frame BGR (numpy array) em uma string base64 JPEG para envio à API."""
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise CameraError("Falha ao codificar imagem para JPEG.")
    return base64.b64encode(buffer).decode("utf-8")
