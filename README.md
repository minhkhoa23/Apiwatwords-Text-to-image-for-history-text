# Apiwatwords: Text-to-Image Generation for Vietnamese Historical Texts

## Introduction

Apiwatwords ("A picture is worth a thousand words") is a specialized Text-to-Image framework designed for generating historically grounded Vietnamese historical images from natural language prompts.

The project addresses two major limitations of modern Text-to-Image models:

- Lack of Vietnamese historical and cultural knowledge
- Prompt ambiguity in complex historical scenarios

To solve these issues, our framework combines:

- Retrieval-Augmented Generation (RAG)
- Prompt enhancement using Qwen2.5-3B-Instruct
- Qwen-Image diffusion model
- Image-to-LoRA (I2LoRA) fine-tuning
- Best-of-N image selection using CLIP + DPG evaluation

---

# Overall Pipeline

1. User enters a Vietnamese historical prompt
2. RAG retrieves related historical context from textbooks
3. Qwen2.5-3B-Instruct enhances the prompt
4. Fine-tuned Qwen-Image + LoRA generates multiple candidate images
5. CLIP Score + DPG Score rerank generated images
6. Best image is selected and returned

---

# Key Features

## Context-Aware Prompt Enhancement

We use a RAG-based retrieval pipeline built from Vietnamese history textbooks.

The system:
- extracts OCR text
- generates dense captions
- stores embeddings in Milvus vector database
- retrieves relevant historical context dynamically

---

## Visual Style Fine-tuning

We apply Image-to-LoRA (I2LoRA) fine-tuning using DiffSynth-Studio.

Advantages:
- lightweight training
- few-shot adaptation
- low VRAM requirement

---

## Best-of-N Image Selection

The system generates multiple candidate images and reranks them using:
- CLIP Score
- DPG Score
- Human preference alignment

---

# Project Structure

```bash
Apiwatwords-Text-to-image-for-history-text/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py
│   ├── rag.py
│   ├── prompt_generator.py
│   ├── image_generator.py
│   ├── reranker.py
│   └── utils.py
│
├── data/
│   └── milvus_history.db
│
├── models/
│   └── model_style.safetensors
│
├── assets/
│   └── pipeline.png
│
└── notebooks/
    └── Apiwatwords_Framework.ipynb
```

---

# Models Used

| Component | Model |
|---|---|
| Diffusion Backbone | Qwen-Image |
| Prompt Enhancement | Qwen2.5-3B-Instruct |
| Embedding Model | vietnamese-bi-encoder |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Captioning | Vintern-1B-v3_5 |
| Style Adaptation | Qwen-Image-i2LoRA |

---

# Experimental Results

| Model | DPG Score | CLIP Score | Human Score |
|---|---|---|---|
| Qwen | 80.6707 | 25.0879 | 7.7916 |
| FFusionXL | 68.6256 | 26.4230 | 5.6645 |
| SD-XL | 65.6452 | 25.7080 | 4.7259 |
| SD-v1.5 | 58.6875 | 25.0230 | 2.1418 |

---

# Download Model Files

Google Drive:

https://drive.google.com/file/d/1TZWcgIJSUro8z-7u2PQ_iPfs4BkFtIZM/view

Place model file here:

```bash
models/model_style.safetensors
```

---

# Installation

## 1. Clone repository

```bash
git clone https://github.com/minhkhoa23/Apiwatwords-Text-to-image-for-history-text.git

cd Apiwatwords-Text-to-image-for-history-text
```

---

## 2. Create conda environment

```bash
conda create -n apiwatwords python=3.10 -y

conda activate apiwatwords
```

---

## 3. Install PyTorch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Install DiffSynth-Studio

```bash
pip install git+https://github.com/modelscope/DiffSynth-Studio.git
```

---

# Running Streamlit Application

Run:

```bash
streamlit run app.py
```

Default local URL:

```bash
http://localhost:8501
```

---

# Hardware Requirements

| Component | Requirement |
|---|---|
| GPU | NVIDIA GPU |
| VRAM | >= 16GB |
| RAM | >= 16GB |
| CUDA | 12.x |
| Python | 3.10 |

Recommended platforms:
- Kaggle Tesla T4
- Google Colab
- RTX 3090 / 4090

---

# Limitations

Current limitations:
- High inference latency
- Large VRAM consumption
- Heavy diffusion pipeline
- Limited Vietnamese historical dataset

---

# Future Work

Future improvements:
- Model quantization
- Faster inference
- Larger Vietnamese historical dataset
- Multi-hop historical retrieval
- Broader dynasty coverage

---

# Authors

- Linh Nguyen Thi Khanh
- Khoa Thai Minh
- Khuong Dinh Xuan
- Dinh Tran Phung

Faculty of Information Technology  
VNUHCM - University of Science
