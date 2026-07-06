"""Configuração centralizada do assistente de acessibilidade por voz."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.json"
DEFAULT_WAKE_WORDS = ["descrever", "o que estou vendo", "olhar", "visão"]


@dataclass
class AppConfig:
    """Todos os parâmetros ajustáveis do sistema, sem valores fixos no código."""

    wake_words: dict[str, list[str]] = field(
        default_factory=lambda: {
            "descrever": ["descrever", "o que estou vendo", "olhar", "visão"],
            "ler": ["ler texto", "o que está escrito", "leia isso", "leitura"]
        }
    )
    language: str = "pt-BR"
    camera_index: int = 0
    audio_device: Optional[int] = None
    sample_rate: int = 16000
    model_temperature: float = 0.2
    max_description_length: int = 400
    voice_rate: int = 175
    voice_volume: float = 1.0
    vision_provider: str = "openai"
    vision_model: str = "gpt-4o-mini"
    tts_engine: str = "pyttsx3"
    vosk_model_path: str = "models/vosk-model-small-pt-br"

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        """Carrega a configuração de um JSON, ou usa padrões se o arquivo não existir."""
        path = path or DEFAULT_CONFIG_PATH
        if not path.exists():
            logger.warning(
                "Arquivo de configuração não encontrado em %s. Usando valores padrão.", path
            )
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: Optional[Path] = None) -> None:
        """Persiste a configuração atual em disco."""
        path = path or DEFAULT_CONFIG_PATH
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
