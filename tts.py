"""Módulo de Text-to-Speech. Isola o restante do sistema do motor TTS específico usado."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

class TTSError(Exception):
    """Erro ao sintetizar ou reproduzir áudio."""

class TTSEngine(ABC):
    """Interface abstrata que todo motor de TTS deve implementar."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Sintetiza e reproduz o texto informado."""

class PiperEngine(TTSEngine):
    """Motor de TTS neural 100% offline baseado no Piper."""

    def __init__(self) -> None:
        from piper.voice import PiperVoice
        
        # Define o caminho onde os arquivos do Piper foram salvos
        self._model_path = Path("models/pt_BR-faber-medium.onnx")
        
        if not self._model_path.exists():
            raise TTSError(
                f"Modelo Piper não encontrado em '{self._model_path}'. "
                "Certifique-se de baixar o arquivo .onnx e o .onnx.json correspondente."
            )
        
        logger.info("Carregando modelo de voz neural Piper (Offline)...")
        try:
            self._voice = PiperVoice.load(str(self._model_path))
            self._sample_rate = self._voice.config.sample_rate
            logger.info("Voz Piper carregada com sucesso.")
        except Exception as exc:
            raise TTSError(f"Falha ao carregar o modelo Piper: {exc}") from exc

    def speak(self, text: str) -> None:
        import sounddevice as sd
        import numpy as np
        
        try:
            # Na nova versão do piper-tts, iteramos sobre os chunks estruturados
            audio_data = bytearray()
            for chunk in self._voice.synthesize(text):
                # Extraímos os bytes puros de 16-bit de cada pedaço gerado
                audio_data.extend(chunk.audio_int16_bytes)
            
            # Convertemos os bytes acumulados em um array numérico do NumPy
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Toca o áudio utilizando o sounddevice (a taxa de amostragem padrão costuma ser 22050)
            sd.play(audio_np, samplerate=self._sample_rate)
            
            # Bloqueia a execução até a fala terminar, evitando cruzamento com a escuta
            sd.wait()
            
        except Exception as exc:
            raise TTSError(f"Falha ao reproduzir áudio via Piper: {exc}") from exc

class Pyttsx3Engine(TTSEngine):
    """Motor de TTS offline baseado em pyttsx3."""

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
    if engine_name == "piper":
        return PiperEngine()
    elif engine_name == "pyttsx3":
        return Pyttsx3Engine(rate=rate, volume=volume, language=language)
    
    raise TTSError(f"Motor de TTS '{engine_name}' não é suportado.")