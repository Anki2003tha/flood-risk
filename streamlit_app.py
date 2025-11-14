from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image
import io

MODEL_PATH = Path('model_demo.h5')


def inject_theme_css():
    css = r"""
    <style>
    :root{--bg1:#071022;--bg2:#041027;--card:#071528;--accent1:#06b6d4;--accent2:#3b82f6;--muted:#94a3b8}
    .stApp { background: linear-gradient(180deg,var(--bg1),var(--bg2)); color: #e6eef8; }
    .stApp .main .block-container{ background: rgba(255,255,255,0.02); border-radius:10px; padding:1rem; box-shadow:0 8px 30px rgba(0,0,0,0.6); }
    .stButton>button{ background: linear-gradient(90deg,var(--accent1),var(--accent2)); color:white; border-radius:8px; }
    .stSidebar { background: rgba(2,6,23,0.6); color: #cbd5e1 }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        import tensorflow as tf
        if MODEL_PATH.exists():
            model = tf.keras.models.load_model(str(MODEL_PATH))
            return ('tf', model)
        else:
            return ('tf-no-file', None)
    except Exception:
        return ('no-tf', None)


def get_sample_image(size=(128, 128)):
    # generate a synthetic RGB image
    H, W = size
    img = np.random.rand(H, W, 3).astype('float32')
    return img


def image_to_array(uploaded_file, target_size=(128, 128)):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype('float32') / 255.0
    return arr


def patch_predictions(model_tuple, img: np.ndarray, patch_size: int = 32, batch_size: int = 8):
    """Run patch-wise predictions across the image and return a heatmap.

    The image is split into non-overlapping patches of size patch_size.
    Returns heatmap of shape (grid_h, grid_w) with probability of class 'high risk'.
    """
    H, W, C = img.shape
    gh = H // patch_size
    gw = W // patch_size
    if gh == 0 or gw == 0:
        # patch size too large for image; return small heatmap
        return np.zeros((1, 1), dtype='float32')

    patches = []
    coords = []
    for i in range(gh):
        for j in range(gw):
            y0 = i * patch_size
            x0 = j * patch_size
            patch = img[y0:y0+patch_size, x0:x0+patch_size]
            patches.append(patch)
            coords.append((i, j))
    patches = np.stack(patches, axis=0)

    mode, model = model_tuple
    if mode == 'tf' and model is not None:
        preds = model.predict(patches, batch_size=batch_size)
        # assume binary softmax [low, high]
        probs = preds[:, 1]
    else:
        probs = np.random.rand(patches.shape[0])

    heat = np.zeros((gh, gw), dtype='float32')
    for (i, j), p in zip(coords, probs):
        heat[i, j] = float(p)
    return heat


def overlay_heatmap(img: np.ndarray, heat: np.ndarray, cmap_name='viridis', alpha=0.6):
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    H, W, _ = img.shape
    gh, gw = heat.shape
    # upsample heat to image size
    tile_h = H // gh
    tile_w = W // gw
    heat_resized = np.kron(heat, np.ones((tile_h, tile_w)))
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = cm.get_cmap(cmap_name)
    heat_rgba = cmap(norm(heat_resized))  # HxWx4

    over = (1 - alpha) * img + alpha * heat_rgba[..., :3]
    over = np.clip(over, 0, 1)
    return over, heat_resized


def main():
    st.set_page_config(layout='wide', page_title='Flood Risk Visualizer')
    inject_theme_css()
    st.title('Flood Risk Visualizer — Interactive Heatmap')
    st.write('Upload an image or use a synthetic sample. The app predicts patch-wise flood risk and overlays a heatmap.')

    model_tuple = load_model()
    status = model_tuple[0]
    if status == 'tf':
        st.success('Loaded TensorFlow model from model_demo.h5')
    elif status == 'tf-no-file':
        st.warning('TensorFlow available but model_demo.h5 not found — using random demo predictor')
    else:
        st.info('TensorFlow not available — using random demo predictor')

    sidebar = st.sidebar
    sidebar.header('Controls')
    patch_size = int(sidebar.selectbox('Patch size', [16, 32, 64], index=1))
    cmap_name = sidebar.selectbox('Colormap', ['viridis', 'magma', 'plasma', 'inferno', 'coolwarm'])
    alpha = float(sidebar.slider('Overlay alpha', 0.0, 1.0, 0.6))
    run_btn = sidebar.button('Run prediction')

    col1, col2 = st.columns([1, 1])
    with col1:
        choice = st.radio('Input', ['Synthetic sample', 'Upload image'], index=0)
        if choice == 'Upload image':
            uploaded = st.file_uploader('Upload an image (PNG/JPG)')
            if uploaded is not None:
                img_arr = image_to_array(uploaded, target_size=(128, 128))
            else:
                img_arr = get_sample_image((128, 128))
        else:
            img_arr = get_sample_image((128, 128))

        st.image(img_arr, caption='Input image (128x128)', use_container_width=True)

    with col2:
        st.header('Heatmap overlay')
        if run_btn:
            with st.spinner('Running patch-wise prediction...'):
                heat = patch_predictions(model_tuple, img_arr, patch_size=patch_size)
                over, heat_resized = overlay_heatmap(img_arr, heat, cmap_name=cmap_name, alpha=alpha)
                st.image(over, caption='Overlay', use_container_width=True)
                st.subheader('Heatmap (upsampled)')
                st.image(heat_resized, clamp=True, use_container_width=True)

                # allow download of overlay
                buf = io.BytesIO()
                Image.fromarray((over * 255).astype('uint8')).save(buf, format='PNG')
                buf.seek(0)
                st.download_button('Download overlay PNG', buf, file_name='overlay.png')
        else:
            st.info('Adjust controls in the sidebar and click "Run prediction" to generate a heatmap.')

    st.markdown('---')
    st.caption('Tip: run `python src\\train.py --demo` to (re)train and generate `model_demo.h5`.')


if __name__ == '__main__':
    main()
