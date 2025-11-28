import streamlit as st
from PIL import Image
import numpy as np
import torch
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer
from style_transfer import load_style_model, apply_style


st.set_page_config(page_title="Увеличение разрешения изображения", layout="wide")
st.title("Увеличение разрешения с помощью Real-ESRGAN")


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
if uploaded:
    img = Image.open(uploaded).convert("RGB")

    if st.button("Увеличить изображение"):
        st.write("Обработка изображения...")
        img_np = np.array(img)
        output, _ = upsampler.enhance(img_np, outscale=4)
        result = Image.fromarray(output)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Оригинал")
            st.image(img, use_column_width=True)
        with col2:
            st.subheader("Увеличенное ×4")
            st.image(result, use_column_width=True)

        result.save("upscaled_result.png")
        st.download_button(
            label="Скачать увеличенное изображение",
            data=open("upscaled_result.png", "rb"),
            file_name="upscaled.png",
        )


st.subheader("Художественные стили")
style = st.selectbox("Выберите стиль:", ["candy", "mosaic", "rain_princess", "udnie"])

if st.button("Применить стиль"):
    model = load_style_model(style)
    styled = apply_style(model, img)
    st.image(styled)
