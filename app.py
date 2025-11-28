import streamlit as st
from PIL import Image
import numpy as np
import torch
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer
from style_transfer import load_style_model, apply_style


st.set_page_config(page_title="Увеличение разрешения изображения", layout="wide")
st.title("Увеличение изображения + художественные стили")


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


upsampler = load_model()


uploaded = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])

if "upscaled_image" not in st.session_state:
    st.session_state.upscaled_image = None

if "styled_image" not in st.session_state:
    st.session_state.styled_image = None


if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Оригинал", use_column_width=True)

    if st.button("Увеличить изображение"):
        st.write("Обработка...")
        img_np = np.array(img)
        output, _ = upsampler.enhance(img_np, outscale=4)
        result = Image.fromarray(output)

        st.session_state.upscaled_image = result

        st.success("Изображение успешно увеличено!")

if st.session_state.upscaled_image is not None:

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Оригинал")
        st.image(uploaded, use_column_width=True)

    with col2:
        st.subheader("Увеличенное ×4")
        st.image(st.session_state.upscaled_image, use_column_width=True)

    st.download_button(
        "Скачать увеличенное изображение",
        data=st.session_state.upscaled_image.save("upscaled.png"),
        file_name="upscaled.png",
    )

    st.subheader("Художественные стили")
    style = st.selectbox(
        "Выберите стиль:", ["candy", "mosaic", "rain_princess", "udnie"]
    )

    if st.button("Применить стиль"):
        model = load_style_model(style)
        styled = apply_style(model, st.session_state.upscaled_image)
        st.session_state.styled_image = styled
        st.success("Стиль применён!")

if st.session_state.styled_image is not None:
    st.subheader("Стилизованное изображение")
    st.image(st.session_state.styled_image, use_column_width=True)

    st.download_button(
        "Скачать стилизованное изображение",
        data=st.session_state.styled_image.save("styled.png"),
        file_name="styled.png",
    )
