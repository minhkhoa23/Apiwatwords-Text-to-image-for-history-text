import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import TEXT_MODEL_NAME


def load_text_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        TEXT_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    return tokenizer, model


def build_context_string(retrieved_hits):
    context_str = ""

    for hit in retrieved_hits:
        entity = hit["entity"]

        context_str += f"--- Source Image: {entity['image_path']} ---\n"
        context_str += f"Caption: {entity['caption']}\n"
        context_str += f"Text Content: {entity['ocr_text']}\n\n"

    return context_str


def generate_diffusion_prompt(
    user_query,
    retrieved_hits,
    tokenizer,
    model,
):
    context_str = build_context_string(retrieved_hits)

    system_instruction = """
You are an AI Image Prompt Expert.
You convert Vietnamese historical context into concise English prompts for image generation.
Output only the final prompt.
"""

    user_content = f"""
RETRIEVED CONTEXT:
{context_str}

USER TOPIC:
{user_query}

TASK:
Write a text-to-image prompt in English.

Rules:
- Use Vietnamese historical context.
- Focus on visual details: clothing, weapon, action, background.
- No style words.
- No proper names.
- Ancient Vietnamese context only.
- Under 100 tokens.
- Use comma-separated keywords.
"""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        [text],
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=100,
        do_sample=False,
        repetition_penalty=1.2,
    )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(
            model_inputs.input_ids,
            generated_ids
        )
    ]

    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return response.replace("\n", ", ").strip()