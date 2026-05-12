import os

import streamlit as st

from src.config import DB_PATH, LORA_PATH
from src.rag import load_rag_components, retrieve_context
from src.prompt_generator import load_text_model, generate_diffusion_prompt
from src.image_generator import load_image_pipeline, generate_images
from src.reranker import rerank_images_with_gpt
from src.utils import image_to_bytes


st.set_page_config(
    page_title="Vietnamese History Image Generator",
    page_icon="🎨",
    layout="wide",
)


st.title("🎨 Vietnamese History Text-to-Image Generator")
st.caption("RAG + Qwen Prompt Generator + Qwen-Image + GPT Vision Reranking")


with st.sidebar:
    st.header("⚙️ Settings")

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password"
    )

    db_path = st.text_input("Milvus DB path", DB_PATH)
    lora_path = st.text_input("LoRA path", LORA_PATH)

    top_k = st.slider("Top-K retrieved documents", 1, 10, 3)
    score_threshold = st.slider("Rerank threshold", 0.0, 1.0, 0.2, 0.05)

    num_images = st.slider("Number of images", 1, 4, 2)
    num_inference_steps = st.slider("Inference steps", 5, 50, 18)

    height = st.selectbox("Height", [512, 576, 768], index=1)
    width = st.selectbox("Width", [768, 1024, 1280], index=1)

    use_gpt_rerank = st.checkbox("Use GPT vision reranking", value=False)


@st.cache_resource
def cached_load_rag(db_path):
    return load_rag_components(db_path)


@st.cache_resource
def cached_load_text_model():
    return load_text_model()


@st.cache_resource
def cached_load_image_pipeline(lora_path):
    return load_image_pipeline(lora_path)


user_query = st.text_area(
    "Nhập chủ đề lịch sử Việt Nam",
    placeholder="Ví dụ: Đinh Bộ Lĩnh dẹp loạn 12 sứ quân",
    height=100,
)

run_button = st.button("🚀 Generate Image", type="primary")


if run_button:
    if not user_query.strip():
        st.warning("Bạn cần nhập chủ đề trước.")
        st.stop()

    if not os.path.exists(db_path):
        st.error(f"Không tìm thấy Milvus DB: {db_path}")
        st.stop()

    with st.status("Đang load model...", expanded=True):
        st.write("Loading RAG components...")
        embed_model, rag_reranker, milvus_client = cached_load_rag(db_path)

        st.write("Loading Qwen text model...")
        tokenizer, text_model = cached_load_text_model()

        st.write("Loading Qwen image pipeline...")
        image_pipe = cached_load_image_pipeline(lora_path)

    with st.status("Đang truy xuất ngữ cảnh lịch sử...", expanded=True):
        retrieved_hits = retrieve_context(
            user_query=user_query,
            embed_model=embed_model,
            reranker=rag_reranker,
            milvus_client=milvus_client,
            top_k=top_k,
            score_threshold=score_threshold,
        )

        st.write(f"Retrieved sources: {len(retrieved_hits)}")

    with st.status("Đang tạo prompt tiếng Anh...", expanded=True):
        enhanced_prompt = generate_diffusion_prompt(
            user_query=user_query,
            retrieved_hits=retrieved_hits,
            tokenizer=tokenizer,
            model=text_model,
        )

        st.subheader("Generated Prompt")
        st.code(enhanced_prompt)

    with st.expander("📚 Retrieved Sources"):
        if len(retrieved_hits) == 0:
            st.warning("Không có source nào vượt threshold.")
        else:
            for i, hit in enumerate(retrieved_hits, start=1):
                entity = hit["entity"]

                st.markdown(f"### Source {i}")
                st.write(f"**Score:** {hit['rerank_score']:.4f}")
                st.write(f"**Image path:** {entity['image_path']}")
                st.write(f"**Caption:** {entity['caption']}")
                st.write(f"**OCR text:** {entity['ocr_text']}")

    with st.status("Đang sinh ảnh...", expanded=True):
        images = generate_images(
            pipe=image_pipe,
            prompt=enhanced_prompt,
            num_images=num_images,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
        )

    best_idx = 0
    scores = None

    if use_gpt_rerank:
        if not openai_api_key:
            st.warning("Bạn chưa nhập OpenAI API Key nên bỏ qua reranking.")
        else:
            with st.status("Đang đánh giá ảnh bằng GPT vision...", expanded=True):
                best_idx, scores = rerank_images_with_gpt(
                    images=images,
                    prompt=enhanced_prompt,
                    api_key=openai_api_key,
                )

    st.divider()
    st.header("✅ Result")

    st.subheader("Best Image")
    st.image(
        images[best_idx],
        caption=f"Best image #{best_idx + 1}",
        use_container_width=True,
    )

    st.download_button(
        label="Download Best Image",
        data=image_to_bytes(images[best_idx]),
        file_name="best_generated_image.png",
        mime="image/png",
    )

    st.subheader("All Generated Images")

    cols = st.columns(len(images))

    for i, image in enumerate(images):
        with cols[i]:
            caption = f"Image {i + 1}"

            if scores is not None:
                caption += f" | Score: {scores[i]:.2f}"

            st.image(
                image,
                caption=caption,
                use_container_width=True,
            )

            st.download_button(
                label=f"Download Image {i + 1}",
                data=image_to_bytes(image),
                file_name=f"generated_image_{i + 1}.png",
                mime="image/png",
            )