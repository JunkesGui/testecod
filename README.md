# Assistente de Acessibilidade por Voz com Descrição de Imagens

Assistente que escuta o microfone continuamente e, ao ouvir uma palavra-chave,
captura uma imagem da câmera, envia para um modelo multimodal e reproduz a
descrição em voz (Português Brasileiro), voltando em seguida ao modo de escuta.

## Instalação

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Modelo de reconhecimento de voz (Vosk, offline)

1. Baixe um modelo pt-BR em https://alphacephei.com/vosk/models
   (ex.: `vosk-model-small-pt-0.3`).
2. Extraia em `models/vosk-model-small-pt-br/` (ou ajuste `vosk_model_path`
   em `config.json`).

### Modelo multimodal (visão)

Por padrão o projeto usa a API da OpenAI (`gpt-4o-mini`). Defina a variável
de ambiente `OPENAI_API_KEY` antes de rodar. Para usar outro provedor
(ex.: Ollama local), adapte apenas `vision.py` — o restante do sistema não
depende diretamente dele.

### Voz (TTS)

Por padrão usa `pyttsx3` (offline). A qualidade e disponibilidade de uma voz
em pt-BR dependem das vozes já instaladas no seu sistema operacional. Para
trocar de motor (ex.: Piper, Coqui TTS), implemente uma nova classe em
`tts.py` que herde de `TTSEngine` e registre-a em `create_tts_engine`.

## Configuração

Todos os parâmetros ficam em `config.json` — nada é fixo no código:

```json
{
  "wake_words": ["descrever", "o que estou vendo", "olhar", "visão"],
  "language": "pt-BR",
  "camera_index": 0,
  "audio_device": null,
  "sample_rate": 16000,
  "model_temperature": 0.2,
  "max_description_length": 400,
  "voice_rate": 175,
  "voice_volume": 1.0,
  "vision_provider": "openai",
  "vision_model": "gpt-4o-mini",
  "tts_engine": "pyttsx3",
  "vosk_model_path": "models/vosk-model-small-pt-br"
}
```

## Executar

```bash
python main.py
```

Diga uma das palavras-chave configuradas (ex.: "descrever") para ativar a
captura e a descrição da cena.

## Estrutura do projeto

```text
main.py          # orquestra o loop principal
camera.py        # captura da câmera
microphone.py     # captura contínua de áudio
wake_word.py      # detecção offline de palavra-chave (Vosk)
vision.py         # comunicação com o modelo multimodal
tts.py            # Text-to-Speech (motor plugável)
config.py         # carregamento/gravação de config.json
config.json       # parâmetros ajustáveis
requirements.txt  # dependências
```

## Substituindo componentes

A arquitetura é modular por design:

- **Reconhecimento de voz**: troque `wake_word.py` por outra engine
  (ex.: Porcupine, SpeechRecognition) mantendo a mesma interface
  (`process_chunk` retornando `bool`).
- **Visão computacional**: troque o cliente dentro de `vision.py` (ex.: Ollama
  local) mantendo `describe(frame) -> str`.
- **TTS**: implemente uma nova subclasse de `TTSEngine` em `tts.py`.

## Limitações conhecidas

- A qualidade da voz em pt-BR do `pyttsx3` depende do sistema operacional;
  em alguns ambientes Linux pode ser necessário instalar `espeak-ng` com
  suporte a português, ou trocar para um motor como Piper TTS para melhor
  qualidade.
- A API de visão usada por padrão (OpenAI) requer conexão com a internet;
  para uso 100% offline, adapte `vision.py` para um modelo local (ex.: via
  Ollama com um modelo multimodal compatível).
