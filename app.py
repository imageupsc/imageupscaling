import streamlit as st
from PIL import Image
import numpy as np
import torch
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer
from style_transfer import load_style_model, apply_style
import io
from image_generation import ImageGenerator



st.set_page_config(page_title="Увеличение разрешения изображения", layout="wide")
st.title("Генерация и обработка изображений с помощью нейросетей")

if "upscaled_image" not in st.session_state:
    st.session_state.upscaled_image = None
if "original_image" not in st.session_state:
    st.session_state.original_image = None
if "styled_image" not in st.session_state:
    st.session_state.styled_image = None


@st.cache_resource
def load_model():
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

generator = load_generator()


upsampler = load_model()

tab_upload, tab_generate = st.tabs(["Загрузка изображения", "Генерация по тексту"])

with tab_upload:
    uploaded = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.session_state.original_image = img
        st.session_state.upscaled_image = None
        st.session_state.styled_image = None

with tab_generate:
    prompt = st.text_area(
        "Текстовое описание (prompt)",
        value="A low quality photo of a cat",
        height=100
    )
    negative = st.text_input("Нежелательные элементы (negative prompt)", value="blurry, low quality, artifacts")
    steps = st.slider("Шаги диффузии", 10, 50, 25)
    guidance = st.slider("Guidance scale", 1.0, 12.0, 7.5)

    if st.button("Сгенерировать изображение"):
        with st.spinner("Генерация изображения..."):
            gen_img = generator.generate(prompt, negative_prompt=negative, steps=steps, guidance=guidance)
            st.session_state.original_image = gen_img
            st.session_state.upscaled_image = None
            st.session_state.styled_image = None
            st.rerun()
#    if st.session_state.original_image is not None:
#        st.subheader("Изображение создано")
#        st.image(st.session_state.original_image, use_container_width=True)
#        st.rerun()

if st.session_state.original_image is not None:
    img = st.session_state.original_image

    if st.button("Увеличить разрешение"):
        with st.spinner("Обработка изображения..."):
            img_np = np.array(img)
            output, _ = upsampler.enhance(img_np, outscale=4)
            result = Image.fromarray(output)
            st.session_state.upscaled_image = result
            st.session_state.styled_image = None
            st.rerun()

    if st.session_state.upscaled_image is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Оригинал")
            st.image(st.session_state.original_image, width="stretch")
        with col2:
            st.subheader("Увеличенное")
            st.image(st.session_state.upscaled_image, width="stretch")

        img_buffer = io.BytesIO()
        st.session_state.upscaled_image.save(img_buffer, format="PNG")
        st.download_button(
            label="Скачать изображение",
            data=img_buffer.getvalue(),
            file_name="upscaled.png",
            mime="image/png",
        )

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
                st.rerun()

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
