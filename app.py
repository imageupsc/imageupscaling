import streamlit as st
from PIL import Image
import numpy as np
import torch
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer

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
