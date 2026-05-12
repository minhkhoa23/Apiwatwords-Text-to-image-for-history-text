import os
import torch

from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig

from src.config import LORA_PATH


def load_image_pipeline(lora_path=LORA_PATH):
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }

    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors",
                **vram_config,
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="text_encoder/model*.safetensors",
                **vram_config,
            ),
            ModelConfig(
                model_id="Qwen/Qwen-Image",
                origin_file_pattern="vae/diffusion_pytorch_model.safetensors",
                **vram_config,
            ),
        ],
        tokenizer_config=ModelConfig(
            model_id="Qwen/Qwen-Image",
            origin_file_pattern="tokenizer/",
        ),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
    )

    if os.path.exists(lora_path):
        pipe.load_lora(pipe.dit, lora_path)

    return pipe


def generate_images(
    pipe,
    prompt,
    num_images=2,
    height=576,
    width=1024,
    num_inference_steps=18,
):
    images = []

    for _ in range(num_images):
        image = pipe(
            prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
        )
        images.append(image)

    return images