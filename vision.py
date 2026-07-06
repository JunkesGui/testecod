"""Módulo responsável apenas pela comunicação com o modelo multimodal de visão."""
from __future__ import annotations

import logging

import numpy as np
from openai import OpenAI

from camera import encode_frame_to_jpeg_base64

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Você descreve imagens para uma pessoa com deficiência visual. Seja preciso e objetivo, "
    "descrevendo apenas o que realmente pode ser observado na imagem. Nunca invente objetos, "
    "pessoas ou detalhes que não estejam claramente visíveis. Priorize informações úteis para "
    "orientação e segurança: obstáculos próximos, posição relativa dos objetos e pessoas, "
    "expressões faciais quando visíveis, e qualquer placa ou texto legível. Use frases curtas "
    "e linguagem direta, em Português Brasileiro."
)


class VisionError(Exception):
    """Erro ao obter a descrição do modelo multimodal."""


class VisionDescriber:
    """Envia uma imagem a um modelo multimodal e retorna uma descrição em português."""

    def __init__(self, model: str, temperature: float = 0.2, max_description_length: int = 400) -> None:
        self._model = model
        self._temperature = temperature
        self._max_length = max_description_length
        self._client = OpenAI()

    def describe(self, frame: np.ndarray) -> str:
        """Envia o frame capturado e retorna a descrição textual em português."""
        try:
            image_b64 = encode_frame_to_jpeg_base64(frame)
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_length,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Descreva esta imagem."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                            },
                        ],
                    },
                ],
            )
            description = response.choices[0].message.content
            if not description:
                raise VisionError("O modelo retornou uma resposta vazia.")
            return description.strip()
        except VisionError:
            raise
        except Exception as exc:
            raise VisionError(f"Falha ao obter descrição do modelo multimodal: {exc}") from exc
