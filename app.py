import streamlit as st
from PIL import Image
import numpy as np
import torch
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from realesrgan import RealESRGANer

st.set_page_config(page_title="Увеличение разрешения изображения", layout="wide")
st.title("Увеличение разрешения с помощью Real-ESRGAN")
