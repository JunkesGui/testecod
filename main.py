"""Ponto de entrada do assistente de acessibilidade por voz com descrição de imagens."""
from __future__ import annotations

import logging
import time

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


def handle_activation(camera: Camera, vision: VisionDescriber, tts_engine, modo: str) -> None:
    """Fluxo dinâmico baseado no modo de ativação."""
    try:
        frame = camera.capture_frame()
    except CameraError as exc:
        logger.error("Erro de câmera: %s", exc)
        return

    try:
        # Passa o modo para o modelo de visão
        resultado = vision.describe(frame, modo=modo)
        logger.info("Resultado gerado (%s): %s", modo, resultado)
    except VisionError as exc:
        logger.error("Erro no modelo multimodal: %s", exc)
        return

    try:
        # Lógica de silêncio apenas se o modo for leitura
        if modo == "ler" and "NENHUM_TEXTO" in resultado.upper():
            tts_engine.speak("Não encontrei nenhum texto na imagem.")
        else:
            tts_engine.speak(resultado)
    except TTSError as exc:
        logger.error("Erro no TTS: %s", exc)


def run() -> None:
    """Loop principal: escuta continuamente e ativa o fluxo dinâmico."""
    config = AppConfig.load()

    try:
        camera, microphone, wake_word_detector, vision, tts_engine = initialize_components(config)
    except (CameraError, MicrophoneError, WakeWordError) as exc:
        logger.critical("Falha na inicialização: %s", exc)
        return

    logger.info("Assistente pronto. Modos disponíveis: %s", config.wake_words)

    # =====================================================================
    # ESTAS DUAS LINHAS PRECISAM FICAR AQUI, DE FORA DO LOOP 'while True'
    # =====================================================================
    last_activation_time = 0.0
    COOLDOWN_SECONDS = 2.0

    try:
        while True:
            chunk = microphone.read_chunk(timeout=1.0)
            if chunk is None:
                continue

            try:
                detected_mode = wake_word_detector.process_chunk(chunk)
            except WakeWordError as exc:
                logger.error("Erro na detecção de palavra-chave: %s", exc)
                continue

            if detected_mode is not None:
                current_time = time.time()
                
                # Como last_activation_time começou com 0.0 lá em cima, 
                # a conta matemática aqui agora vai funcionar perfeitamente!
                if current_time - last_activation_time < COOLDOWN_SECONDS:
                    logger.warning("Gatilho ignorado: aguardando tempo de recarga (cooldown).")
                    wake_word_detector.reset()
                    microphone.clear_queue()
                    continue

                logger.info("Comando detectado: Modo '%s'. Capturando imagem...", detected_mode)
                wake_word_detector.reset()
                
                handle_activation(camera, vision, tts_engine, modo=detected_mode)
                
                microphone.clear_queue()
                
                # Atualiza a variável para o momento atual após falar
                last_activation_time = time.time()
                
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
