"""
main.py

Sistema de acessibilidade multimodal

Fluxo:

Microfone (voz)
      ↓
Detecta palavras-chave
      ↓
Captura frame atual da câmera
      ↓
BLIP Image Captioning
      ↓
Descrição em texto
      ↓
Text-to-Speech

Dependências:

pip install opencv-python SpeechRecognition pyaudio transformers torch pillow pyttsx3
"""

import threading
import traceback

import cv2
import pyttsx3
import speech_recognition as sr

from PIL import Image
from transformers import pipeline

# ==========================================================
# CONFIGURAÇÕES
# ==========================================================

KEYWORDS = [
    "o que você vê",
    "o que é isso",
]

CAMERA_INDEX = 0

# ==========================================================
# VARIÁVEIS GLOBAIS COMPARTILHADAS
# ==========================================================

last_frame = None
frame_lock = threading.Lock()

processing = False
processing_lock = threading.Lock()

running = True

# ==========================================================
# INICIALIZAÇÃO TTS
# ==========================================================

print("Inicializando TTS...")

tts = pyttsx3.init()

tts.setProperty("rate", 170)

# Tenta selecionar uma voz em português
try:
    for voice in tts.getProperty("voices"):
        text = (
            voice.name.lower()
            + " "
            + str(voice.languages).lower()
        )

        if "portuguese" in text or "brazil" in text or "pt" in text:
            tts.setProperty("voice", voice.id)
            break

except Exception:
    pass

tts_lock = threading.Lock()

# ==========================================================
# MODELO DE IMAGE CAPTIONING
# ==========================================================

print("Carregando modelo BLIP...")

caption_pipeline = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base"
)

print("Modelo carregado.")

# ==========================================================
# RECONHECIMENTO DE VOZ
# ==========================================================

recognizer = sr.Recognizer()

# ==========================================================
# FUNÇÕES AUXILIARES
# ==========================================================

def speak(text):
    """
    Fala um texto utilizando pyttsx3.
    Lock evita que duas falas ocorram ao mesmo tempo.
    """

    with tts_lock:
        tts.say(text)
        tts.runAndWait()


def describe_current_frame():
    """
    Captura o frame mais recente,
    gera uma descrição utilizando BLIP
    e reproduz via TTS.
    """

    global processing

    with processing_lock:
        if processing:
            return
        processing = True

    try:

        with frame_lock:

            if last_frame is None:
                speak("Ainda não há imagem disponível.")
                return

            frame = last_frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(rgb)

        result = caption_pipeline(image)

        if len(result) == 0:
            description = "Não consegui descrever a imagem."

        else:
            description = result[0]["generated_text"]

        print("\nDescrição:", description)

        speak(description)

    except Exception:

        traceback.print_exc()

        speak("Ocorreu um erro durante a análise da imagem.")

    finally:

        with processing_lock:
            processing = False


# ==========================================================
# CALLBACK DO MICROFONE
# ==========================================================

def callback(recognizer, audio):
    """
    Executado automaticamente sempre que o SpeechRecognition
    captura um trecho de áudio.
    """

    global running

    if not running:
        return

    try:

        text = recognizer.recognize_google(
            audio,
            language="pt-BR"
        )

        text = text.lower()

        print("Você disse:", text)

        for keyword in KEYWORDS:

            if keyword in text:

                print("Palavra-chave detectada!")

                threading.Thread(
                    target=describe_current_frame,
                    daemon=True
                ).start()

                break

    except sr.UnknownValueError:
        # Ruído ou fala não compreendida
        pass

    except sr.RequestError as e:

        print("Erro SpeechRecognition:", e)

    except Exception:

        traceback.print_exc()


# ==========================================================
# INICIALIZAÇÃO MICROFONE
# ==========================================================

print("Inicializando microfone...")

try:

    microphone = sr.Microphone()

    with microphone as source:

        print("Calibrando ruído ambiente...")

        recognizer.adjust_for_ambient_noise(
            source,
            duration=2
        )

    stop_listening = recognizer.listen_in_background(
        microphone,
        callback
    )

    print("Microfone pronto.")

except Exception:

    traceback.print_exc()

    raise RuntimeError(
        "Não foi possível inicializar o microfone."
    )

# ==========================================================
# INICIALIZAÇÃO DA CÂMERA
# ==========================================================

print("Abrindo câmera...")

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():

    raise RuntimeError(
        "Não foi possível abrir a câmera."
    )

print("Câmera iniciada.")

# ==========================================================
# LOOP PRINCIPAL
# ==========================================================

print("\nSistema iniciado.")
print("Fale:")
print("- descrever")
print("- o que é isso")
print("- analisar")
print("- ler")
print("\nPressione Q para sair.\n")

try:

    while True:

        success, frame = camera.read()

        if not success:
            continue

        with frame_lock:
            last_frame = frame.copy()

        cv2.imshow(
            "Acessibilidade Multimodal",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

except KeyboardInterrupt:
    pass

finally:

    print("Encerrando...")

    running = False

    try:
        stop_listening(wait_for_stop=False)
    except Exception:
        pass

    camera.release()

    cv2.destroyAllWindows()

    print("Programa encerrado.")