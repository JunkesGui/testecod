"""Ponto de entrada do assistente de acessibilidade por voz com descrição de imagens."""
from __future__ import annotations

import logging

from camera import Camera, CameraError
from config import AppConfig
from microphone import Microphone, MicrophoneError
from tts import TTSError, create_tts_engine
from vision import VisionDescriber, VisionError
from wake_word import WakeWordDetector, WakeWordError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def initialize_components(config: AppConfig):
    """Inicializa câmera, microfone, detector de palavra-chave, visão e TTS."""
    camera = Camera(device_index=config.camera_index)
    camera.open()

    microphone = Microphone(sample_rate=config.sample_rate, device=config.audio_device)
    microphone.start()

    wake_word_detector = WakeWordDetector(
        model_path=config.vosk_model_path,
        sample_rate=config.sample_rate,
        wake_words=config.wake_words,
    )

    vision = VisionDescriber(
        model=config.vision_model,
        temperature=config.model_temperature,
        max_description_length=config.max_description_length,
    )

    tts_engine = create_tts_engine(
        engine_name=config.tts_engine,
        rate=config.voice_rate,
        volume=config.voice_volume,
        language=config.language,
    )

    return camera, microphone, wake_word_detector, vision, tts_engine


def handle_activation(camera: Camera, vision: VisionDescriber, tts_engine) -> None:
    """Captura um frame, gera a descrição e reproduz em áudio, sem interromper o loop em erro."""
    try:
        frame = camera.capture_frame()
    except CameraError as exc:
        logger.error("Erro de câmera: %s", exc)
        return

    try:
        description = vision.describe(frame)
        logger.info("Descrição gerada: %s", description)
    except VisionError as exc:
        logger.error("Erro no modelo multimodal: %s", exc)
        return

    try:
        tts_engine.speak(description)
    except TTSError as exc:
        logger.error("Erro no TTS: %s", exc)



def run() -> None:
    """Loop principal: escuta continuamente e ativa o fluxo de descrição ao detectar a palavra-chave."""
    config = AppConfig.load()

    try:
        camera, microphone, wake_word_detector, vision, tts_engine = initialize_components(config)
    except (CameraError, MicrophoneError, WakeWordError) as exc:
        logger.critical("Falha na inicialização: %s", exc)
        return

    logger.info("Assistente pronto. Diga uma das palavras-chave: %s", config.wake_words)

    try:
        while True:
            chunk = microphone.read_chunk(timeout=1.0)
            if chunk is None:
                continue

            try:
                detected = wake_word_detector.process_chunk(chunk)
            except WakeWordError as exc:
                logger.error("Erro na detecção de palavra-chave: %s", exc)
                continue

            if detected:
                logger.info("Palavra-chave detectada. Capturando imagem...")
                wake_word_detector.reset()
                handle_activation(camera, vision, tts_engine)
                microphone.clear_queue()
                logger.info("Retornando ao modo de escuta.")
    except KeyboardInterrupt:
        logger.info("Encerrando por solicitação do usuário.")
    except Exception as exc:
        logger.critical("Erro inesperado no loop principal: %s", exc, exc_info=True)
    finally:
        camera.close()
        microphone.stop()


if __name__ == "__main__":
    run()
