"""Módulo responsável pela comunicação com o modelo local multimodal Qwen-VL."""
from __future__ import annotations

import logging
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

PROMPTS = {
    "descrever": {
        "system": "Você descreve imagens para uma pessoa com deficiência visual. Seja preciso e objetivo, descrevendo apenas o que realmente pode ser observado na imagem. Nunca invente objetos. Priorize segurança e orientação. Use frases curtas em Português Brasileiro.",
        "user": "Descreva o que está nesta imagem."
    },
    "ler": {
        "system": "Você é um assistente de acessibilidade estritamente focado em leitura de textos (OCR). Sua ÚNICA tarefa é transcrever exatamente o texto que aparece na imagem. NUNCA descreva o ambiente. Se não contiver texto, responda EXATAMENTE com a palavra: NENHUM_TEXTO.",
        "user": "Leia todo o texto visível nesta imagem."
    }
}

class VisionError(Exception):
    """Erro ao obter a descrição do modelo multimodal."""

class VisionDescriber:
    """Processa uma imagem localmente usando Qwen-VL e retorna a descrição em português."""

    def __init__(self, model: str, temperature: float = 0.2, max_description_length: int = 400) -> None:
        self._model_name = model
        self._temperature = temperature
        self._max_length = max_description_length
        
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Carregando o modelo %s no dispositivo: %s", self._model_name, self._device)
        
        try:
            # Seleciona precisão bfloat16 se houver GPU para economizar metade da memória VRAM
            torch_dtype = torch.bfloat16 if self._device == "cuda" else torch.float32
            
            # Inicializa o modelo aplicando o mapeamento automático se houver aceleração por hardware
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self._model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self._device == "cuda" else None
            )
            
            if self._device == "cpu":
                self._model = self._model.to(self._device)
                
            self._processor = AutoProcessor.from_pretrained(self._model_name)
            logger.info("Modelo Qwen-VL carregado com sucesso!")
            
        except Exception as exc:
            logger.critical("Falha ao carregar o modelo Qwen-VL: %s", exc)
            raise VisionError(f"Erro na inicialização do modelo de visão: {exc}") from exc

    def describe(self, frame: np.ndarray, modo: str = "descrever") -> str:
        """Processa o frame capturado com base no modo selecionado."""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Pega as instruções corretas baseadas no modo ("descrever" ou "ler")
            instrucoes = PROMPTS.get(modo, PROMPTS["descrever"])
            
            messages = [
                {"role": "system", "content": instrucoes["system"]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": instrucoes["user"]},
                    ],
                }
            ]
            
            # Prepara os textos e dados de visão usando os utilitários nativos
            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
            
            # Realiza a inferência sem calcular gradientes (modo estritamente de predição)
            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs, 
                    max_new_tokens=self._max_length,
                    do_sample=True if self._temperature > 0 else False,
                    temperature=self._temperature if self._temperature > 0 else None
                )
            
            # Recorta os tokens de prompt da resposta para obter apenas o texto gerado de fato
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self._processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            if not output_text or not output_text[0]:
                raise VisionError("O modelo local retornou uma resposta vazia.")
                
            return output_text[0].strip()
            
        except VisionError:
            raise
        except Exception as exc:
            raise VisionError(f"Falha na inferência local com Qwen-VL: {exc}") from exc