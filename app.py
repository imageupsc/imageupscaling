import streamlit as st
from PIL import Image
import numpy as np
import torch
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer
from style_transfer import load_style_model, apply_style
from image_generation import ImageGenerator
import io

st.set_page_config(page_title="Генерация и обработка изображений", layout="wide")
st.markdown(
    """
    <style>
      html, body { overflow-y: scroll; }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Генерация и обработка изображений с помощью нейросетей")

# -------- Session state --------
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "upscaled_image" not in st.session_state:
    st.session_state.upscaled_image = None
if "styled_image" not in st.session_state:
    st.session_state.styled_image = None


# -------- Models loading (cached) --------
@st.cache_resource
def load_upsampler():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SRVGGNetCompact(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path="realesr-general-x4v3.pth",
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device,
    )
    return upsampler


@st.cache_resource
def load_generator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ImageGenerator(device=device)


upsampler = load_upsampler()
generator = load_generator()

# -------- UI: Text-to-Image --------
st.subheader("Генерация по тексту")

prompt = st.text_area(
    "Текстовое описание (prompt)",
    value="A fir tree in a winter forest, digital illustration, soft colors, detailed, cinematic lighting",
    height=110,
)

negative = st.text_input(
    "Нежелательные элементы (negative prompt)",
    value="blurry, low quality, artifacts, text, watermark",
)

steps = st.slider("Шаги диффузии", 10, 50, 25)
guidance = st.slider("Guidance scale", 1.0, 12.0, 7.5)

col_btn1, col_btn2 = st.columns([1, 2])
with col_btn1:
    if st.button("Сгенерировать изображение"):
        with st.spinner("Генерация изображения..."):
            gen_img = generator.generate(
                prompt=prompt,
                negative_prompt=negative,
                steps=steps,
                guidance=guidance,
            )
            st.session_state.original_image = gen_img
            st.session_state.upscaled_image = None
            st.session_state.styled_image = None

with col_btn2:
    if st.button("Сбросить результат"):
        st.session_state.original_image = None
        st.session_state.upscaled_image = None
        st.session_state.styled_image = None

# -------- Preview generated image --------
if st.session_state.original_image is not None:
    st.divider()
    st.subheader("Сгенерированное изображение")
    st.image(st.session_state.original_image, width="stretch")

    gen_buffer = io.BytesIO()
    st.session_state.original_image.save(gen_buffer, format="PNG")
    st.download_button(
        label="Скачать сгенерированное изображение",
        data=gen_buffer.getvalue(),
        file_name="generated.png",
        mime="image/png",
    )

    # -------- Upscale --------
    st.divider()
    st.subheader("Увеличение разрешения (Real-ESRGAN x4)")

    if st.button("Увеличить разрешение"):
        with st.spinner("Увеличение разрешения..."):
            img_np = np.array(st.session_state.original_image.convert("RGB"))
            output, _ = upsampler.enhance(img_np, outscale=4)
            st.session_state.upscaled_image = Image.fromarray(output)
            st.session_state.styled_image = None

    if st.session_state.upscaled_image is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("До")
            st.image(st.session_state.original_image, width="stretch")
        with col2:
            st.subheader("После (x4)")
            st.image(st.session_state.upscaled_image, width="stretch")

        up_buffer = io.BytesIO()
        st.session_state.upscaled_image.save(up_buffer, format="PNG")
        st.download_button(
            label="Скачать увеличенное изображение",
            data=up_buffer.getvalue(),
            file_name="upscaled.png",
            mime="image/png",
        )

        # -------- Style transfer --------
        st.divider()
        st.subheader("Художественные стили")

        STYLE_LABELS = {
            "candy": "Конфетный",
            "mosaic": "Мозаика",
            "rain_princess": "Принцесса дождя",
            "udnie": "Удни (абстракция)",
        }

        style_display = st.selectbox("Выберите стиль:", list(STYLE_LABELS.values()))
        style = [k for k, v in STYLE_LABELS.items() if v == style_display][0]

        if st.button("Применить стиль"):
            with st.spinner("Применение стиля..."):
                model = load_style_model(style)
                styled = apply_style(model, st.session_state.upscaled_image)
                st.session_state.styled_image = styled

        if st.session_state.styled_image is not None:
            st.subheader("Стилизованное изображение")
            st.image(st.session_state.styled_image, width="stretch")

            styled_buffer = io.BytesIO()
            st.session_state.styled_image.save(styled_buffer, format="PNG")
            st.download_button(
                label="Скачать стилизованное изображение",
                data=styled_buffer.getvalue(),
                file_name=f"styled_{style}.png",
                mime="image/png",
            )
else:
    st.info("Введите текстовое описание и нажмите «Сгенерировать изображение».")
