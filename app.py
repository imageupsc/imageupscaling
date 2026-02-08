import io
import numpy as np
import streamlit as st
import torch
from PIL import Image
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from image_generation import ImageGenerator
from style_transfer import load_style_model, apply_style


# ---------------- Page config ----------------
st.set_page_config(page_title="Генерация и обработка изображений", layout="wide")
st.title("Генерация и обработка изображений с помощью нейросетей")

# фикс "дёрганья" интерфейса из-за появления/исчезновения скроллбара
st.markdown(
    """
    <style>
      html { scrollbar-gutter: stable; }
      [data-testid="stAppViewContainer"] { overflow-y: scroll; overflow-x: hidden; }
      [data-testid="stMain"] { overflow-y: scroll; overflow-x: hidden; }
      section.main { overflow-y: scroll; overflow-x: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Session state ----------------
defaults = {
    "original_image": None,
    "upscaled_image": None,
    "styled_image": None,
    "is_generating": False,
    "is_upscaling": False,
    "is_styling": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------- Cached model loaders ----------------
@st.cache_resource
def load_generator():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ImageGenerator(device=device)


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
        half=False,  # если будет GPU, можно поставить True (но не всегда стабильно)
        device=device,
    )
    return upsampler


generator = load_generator()
upsampler = load_upsampler()


# ---------------- UI: Source (Upload or Generate) ----------------
st.subheader("Источник изображения")

src_col1, src_col2 = st.columns([1.2, 1])

with src_col1:
    st.file_uploader(
        "Загрузите изображение (drag&drop или выбор файла)",
        type=["png", "jpg", "jpeg", "webp"],
        key="uploaded_file",
        help="Можно перетащить файл сюда или выбрать вручную.",
    )

# ВАЖНО: читаем из session_state
uploaded = st.session_state.get("uploaded_file", None)

with src_col2:
    use_uploaded_clicked = st.button(
        "Использовать загруженное изображение",
        key="btn_use_uploaded",
        disabled=(uploaded is None)
        or st.session_state.is_generating
        or st.session_state.is_upscaling
        or st.session_state.is_styling,
        use_container_width=True,
    )

if use_uploaded_clicked and uploaded is not None:
    try:
        img = Image.open(uploaded).convert("RGB")
        st.session_state.original_image = img
        st.session_state.upscaled_image = None
        st.session_state.styled_image = None
        st.success("Загруженное изображение установлено как исходное.")
    except Exception as e:
        st.session_state.original_image = None
        st.error(f"Не удалось прочитать изображение: {e}")

st.divider()


# ---------------- UI: Controls (Text-to-Image) ----------------
st.subheader("Генерация по тексту")

prompt = st.text_area(
    "Текстовое описание (prompt)",
    value="small dog and cat",
    height=110,
    key="prompt",
)

negative = st.text_input(
    "Нежелательные элементы (negative prompt)",
    value="blurry, low quality, artifacts, text, watermark",
    key="negative",
)

c1, c2 = st.columns(2)
with c1:
    steps = st.slider("Шаги диффузии", 10, 50, 25, key="steps")
with c2:
    guidance = st.slider(
        "Коэффициент следования текстовому описанию", 1.0, 12.0, 7.5, key="guidance"
    )

btn_col1, btn_col2 = st.columns([1, 1])
with btn_col1:
    gen_clicked = st.button(
        "Сгенерировать изображение",
        key="btn_generate",
        disabled=st.session_state.is_generating
        or st.session_state.is_upscaling
        or st.session_state.is_styling,
        use_container_width=True,
    )
with btn_col2:
    reset_clicked = st.button(
        "Сбросить результат",
        key="btn_reset",
        disabled=st.session_state.is_generating
        or st.session_state.is_upscaling
        or st.session_state.is_styling,
        use_container_width=True,
    )

# placeholders (чтобы UI не прыгал)
status_ph = st.empty()
preview_ph = st.empty()
download_gen_ph = st.empty()

upscale_section_ph = st.empty()
style_section_ph = st.empty()


# ---------------- Actions: Reset ----------------
if reset_clicked:
    st.session_state.original_image = None
    st.session_state.upscaled_image = None
    st.session_state.styled_image = None

    # сброс загрузчика
    st.session_state.uploaded_file = None

    status_ph.empty()
    preview_ph.empty()
    download_gen_ph.empty()
    upscale_section_ph.empty()
    style_section_ph.empty()


# ---------------- Actions: Generate ----------------
progress_ph = st.empty()
percent_ph = st.empty()

if gen_clicked:
    st.session_state.is_generating = True
    st.session_state.upscaled_image = None
    st.session_state.styled_image = None

    bar = progress_ph.progress(0)
    percent_ph.write("Прогресс: 0%")

    def on_progress(step, total):
        p = int((step / total) * 100)
        p = max(0, min(p, 100))
        bar.progress(p)
        percent_ph.write(f"Прогресс: {p}%")

    try:
        with st.spinner("Генерация изображения..."):
            img = generator.generate(
                prompt=prompt,
                negative_prompt=negative,
                steps=int(steps),
                guidance=float(guidance),
                on_progress=on_progress,
            )
        st.session_state.original_image = img
        status_ph.success("Генерация завершена.")
    except Exception as e:
        st.session_state.original_image = None
        status_ph.error(f"Ошибка генерации: {e}")
    finally:
        st.session_state.is_generating = False
        bar.progress(100)
        percent_ph.write("Прогресс: 100%")


# ---------------- Preview (original) ----------------
if st.session_state.original_image is None:
    preview_ph.info("Загрузите изображение сверху или сгенерируйте по тексту ниже.")
else:
    preview_ph.subheader("Исходное изображение (загруженное или сгенерированное)")
    preview_ph.image(st.session_state.original_image, width="stretch")

    original_buffer = io.BytesIO()
    st.session_state.original_image.save(original_buffer, format="PNG")
    download_gen_ph.download_button(
        label="Скачать исходное изображение",
        data=original_buffer.getvalue(),
        file_name="original.png",
        mime="image/png",
        key="dl_original",
    )


# ---------------- Upscale block ----------------
if st.session_state.original_image is not None:
    with upscale_section_ph.container():
        st.divider()
        st.subheader("Увеличение разрешения (Real-ESRGAN x4)")

        up_clicked = st.button(
            "Увеличить разрешение",
            key="btn_upscale",
            disabled=st.session_state.is_upscaling or st.session_state.is_generating,
            use_container_width=True,
        )

        if up_clicked:
            st.session_state.is_upscaling = True
            status_ph.info("Увеличение разрешения...")

            try:
                with st.spinner("Увеличение разрешения..."):
                    img_np = np.array(st.session_state.original_image.convert("RGB"))
                    output, _ = upsampler.enhance(img_np, outscale=4)
                    st.session_state.upscaled_image = Image.fromarray(output)
                    st.session_state.styled_image = None
                status_ph.success("Увеличение разрешения завершено.")
            except Exception as e:
                st.session_state.upscaled_image = None
                status_ph.error(f"Ошибка апскейла: {e}")
            finally:
                st.session_state.is_upscaling = False

        if st.session_state.upscaled_image is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("До")
                st.image(st.session_state.original_image, width="stretch")
            with col_b:
                st.caption("После (x4)")
                st.image(st.session_state.upscaled_image, width="stretch")

            up_buffer = io.BytesIO()
            st.session_state.upscaled_image.save(up_buffer, format="PNG")
            st.download_button(
                label="Скачать увеличенное изображение",
                data=up_buffer.getvalue(),
                file_name="upscaled.png",
                mime="image/png",
                key="dl_upscaled",
            )


# ---------------- Style transfer block ----------------
if st.session_state.upscaled_image is not None:
    with style_section_ph.container():
        st.divider()
        st.subheader("Художественные стили")

        STYLE_LABELS = {
            "candy": "Конфетный",
            "mosaic": "Мозаика",
            "rain_princess": "Принцесса дождя",
            "udnie": "Удни (абстракция)",
        }

        style_display = st.selectbox(
            "Выберите стиль:",
            list(STYLE_LABELS.values()),
            key="style_select",
        )
        style_key = [k for k, v in STYLE_LABELS.items() if v == style_display][0]

        style_clicked = st.button(
            "Применить стиль",
            key="btn_style",
            disabled=st.session_state.is_styling
            or st.session_state.is_generating
            or st.session_state.is_upscaling,
            use_container_width=True,
        )

        if style_clicked:
            st.session_state.is_styling = True
            status_ph.info("Применение стиля...")

            try:
                with st.spinner("Применение стиля..."):
                    model = load_style_model(style_key)
                    styled = apply_style(model, st.session_state.upscaled_image)
                    st.session_state.styled_image = styled
                status_ph.success("Стилизация завершена.")
            except Exception as e:
                st.session_state.styled_image = None
                status_ph.error(f"Ошибка стилизации: {e}")
            finally:
                st.session_state.is_styling = False

        if st.session_state.styled_image is not None:
            st.subheader("Стилизованное изображение")
            st.image(st.session_state.styled_image, width="stretch")

            styled_buffer = io.BytesIO()
            st.session_state.styled_image.save(styled_buffer, format="PNG")
            st.download_button(
                label="Скачать стилизованное изображение",
                data=styled_buffer.getvalue(),
                file_name=f"styled_{style_key}.png",
                mime="image/png",
                key="dl_styled",
            )
