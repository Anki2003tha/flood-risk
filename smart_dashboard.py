"""Smart Flood Alert Map — Streamlit dashboard with Folium map and live visuals.

Features:
- Interactive Folium map with sample flood-risk zones (Low/Medium/High).
- Click on the map to fetch a sample image and a live model prediction (demo/simulated).
- Time-series charts for rainfall, elevation and risk trends (synthetic data for demo).

Run:
  pip install streamlit folium streamlit-folium matplotlib pandas
  streamlit run smart_dashboard.py
"""
from datetime import datetime, timedelta
import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import io

from pathlib import Path
from src.model import build_small_model


def inject_theme_css():
        """Inject a polished, modern theme using CSS for Streamlit components."""
        css = r"""
        <style>
        :root{--bg1:#0f1724;--bg2:#071022;--card:#0b1726;--accent1:#06b6d4;--accent2:#3b82f6;--muted:#94a3b8}
        /* Page background gradient */
        .stApp {
            background: linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
            color: #e6eef8;
            font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
        }
        /* Card container */
        .stApp .main .block-container{
            background: rgba(255,255,255,0.02);
            padding: 1.2rem 1.6rem;
            border-radius: 12px;
            box-shadow: 0 8px 30px rgba(2,6,23,0.6);
            border: 1px solid rgba(255,255,255,0.03);
        }
        /* Header banner */
        .top-banner{
            background: linear-gradient(90deg, rgba(6,182,212,0.12), rgba(59,130,246,0.08));
            padding: 0.6rem 1rem;
            border-radius: 10px;
            margin-bottom: 0.8rem;
            display:flex;align-items:center;gap:12px
        }
        .top-banner h1{margin:0;font-size:20px;color:white}
        .top-banner p{margin:0;color:var(--muted);font-size:12px}
        /* Buttons: full-width, solid gradient and stronger color */
        .stButton>button, button.stButton>button, .stForm button{
            width: 100% !important;
            display: block !important;
            background: linear-gradient(90deg,var(--accent2),var(--accent1)) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 6px 20px rgba(59,130,246,0.22) !important;
            padding: 10px 14px !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
        }
        /* Sidebar tweaks */
        .css-1d391kg .stSidebar, .stSidebar {
            background: rgba(2,6,23,0.6);
            color: #cbd5e1;
        }
        /* Metric styling */
        .stMetric>div>div>div:nth-child(1){color:#e6eef8}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)


def make_sample_zones():
    """Return a small GeoJSON-like feature collection of polygon zones with risk levels."""
    # simple square zones in lat/lon around a point
    base_lat = 24.0
    base_lon = 90.0
    features = []
    # sizes, risk levels and readable English names for each demo zone
    sizes = [0.05, 0.06, 0.07]
    risks = ['Low', 'Medium', 'High']
    names = ['Riverbank', 'Downtown', 'Upland']
    for i, (s, r, name) in enumerate(zip(sizes, risks, names)):
        lat0 = base_lat + i * 0.08
        lon0 = base_lon
        poly = [
            [lat0, lon0],
            [lat0, lon0 + s],
            [lat0 + s, lon0 + s],
            [lat0 + s, lon0],
            [lat0, lon0],
        ]
        features.append({
            'type': 'Feature',
            'properties': {'id': f'zone-{i+1}', 'risk': r, 'name': name},
            'geometry': {'type': 'Polygon', 'coordinates': [[ [p[1], p[0]] for p in poly ]]}
        })
    return {'type': 'FeatureCollection', 'features': features}


def risk_color(risk: str):
    return {'Low': '#2ECC71', 'Medium': '#F1C40F', 'High': '#E74C3C'}.get(risk, '#95A5A6')


def sample_rainfall_timeseries(days=30):
    # use timezone-aware current date to avoid deprecation warnings
    try:
        from datetime import timezone
        today = datetime.now(timezone.utc).date()
    except Exception:
        today = datetime.now().date()
    dates = [today - timedelta(days=i) for i in range(days)][::-1]
    rains = np.clip(np.random.randn(days).cumsum() * 2 + 50, 0, 200)
    return pd.DataFrame({'date': dates, 'rain_mm': rains})


def sample_elevation(lat, lon):
    # synthetic elevation based on coords
    return 50 + (lat % 1) * 100 - (lon % 1) * 10


def predict_for_point(lat, lon):
    # produce synthetic image and a synthetic prediction (do not initialize TF inside Streamlit callbacks)
    img = (np.random.rand(128, 128, 3) * 255).astype('uint8')
    probs = np.random.rand(2)
    probs = probs / probs.sum()
    return img, probs


# --- Chatbot helpers ---
MODEL = None


def load_demo_model():
    """Attempt to load a demo Keras model from workspace (model_demo.h5). Returns model or None."""
    global MODEL
    if MODEL is not None:
        return MODEL
    try:
        from tensorflow import keras  # type: ignore
        model_path = Path('model_demo.h5')
        if model_path.exists():
            MODEL = keras.models.load_model(str(model_path))
    except Exception:
        MODEL = None
    return MODEL


def analyze_uploaded_image(pil_image):
    """Analyze a PIL image and return (risk_label, probability, suggested_action).

    Uses a loaded Keras model if available, otherwise falls back to a simple heuristic
    based on blue-channel dominance as a proxy for water presence.
    """
    model = load_demo_model()
    # prepare a small array for model or heuristic
    arr_small = np.array(pil_image.resize((32, 32))).astype('float32') / 255.0
    x = np.expand_dims(arr_small, 0)
    prob = 0.0
    heatmap_img = None
    try:
        if model is not None:
            preds = model.predict(x)
            # handle single-output or two-class outputs
            if preds.ndim == 2 and preds.shape[1] >= 2:
                prob = float(preds[0, 1])
                pred_class = int(preds[0].argmax())
            else:
                prob = float(preds[0, 0])
                pred_class = 0
            # attempt Grad-CAM explainability (use local compute_gradcam)
            try:
                heat = compute_gradcam(pil_image, model)
                if heat is not None:
                    heatmap_img = heat
            except Exception:
                heatmap_img = None
        else:
            # heuristic: blue dominance indicates water; higher -> higher flood probability
            arr = np.array(pil_image).astype('float32') / 255.0
            blue = arr[:, :, 2]
            green = arr[:, :, 1]
            bd = (blue - green).mean()
            prob = float(np.clip(0.5 + bd, 0.0, 1.0))
    except Exception:
        # fallback random but deterministic-ish using image mean
        meanv = arr_small.mean()
        prob = float(np.clip(meanv, 0.0, 1.0))

    # Map probability to risk label
    if prob >= 0.66:
        label = 'High'
    elif prob >= 0.33:
        label = 'Medium'
    else:
        label = 'Low'

    actions = {
        'High': 'Build embankment / improve drainage / issue evacuation alerts',
        'Medium': 'Improve drainage, strengthen early warning and monitor river gauges',
        'Low': 'No immediate action; monitor weather and inspect drainage',
    }
    return label, prob, actions[label], heatmap_img


def generate_chat_response(message: str, context: dict) -> str:
    """Generate a simple, context-aware response based on the analysis context.

    This is a lightweight rule-based responder that uses the analysis results
    (risk label, probability, suggested action, heatmap presence) to answer
    common questions about the prediction and consequences.
    """
    m = message.strip().lower()
    label = context.get('label')
    prob = context.get('prob')
    action = context.get('action')
    heat = context.get('heat')

    # direct question intents
    # detect repeated identical user questions (avoid identical canned replies)
    try:
        last_user = st.session_state.get('_last_user_msg')
    except Exception:
        last_user = None

    if any(q in m for q in ('why', 'reason', 'cause')):
        return f"The model estimated **{label}** risk with probability {prob*100:.1f}%. The prediction is influenced by water-like patterns in the image (shown in the Grad-CAM overlay if available)." \
               + "\n\nIf you want more detail, ask about 'which areas' or 'how confident'."

    if any(q in m for q in ('confidence', 'confident', 'how sure')):
        return f"The model's flood probability is approximately **{prob*100:.1f}%**. Values closer to 100% indicate higher confidence; consider ground truth checks for production use."

    # direct harm/danger intent — provide richer detail and vary reply if user repeats
    if any(q in m for q in ('harm', 'harmful', 'danger', 'dangerous', 'is it harmful')):
        base = f"This image indicates a **{label}** flood risk (probability ~{prob*100:.1f}%)."
        extra = (
            "This may be harmful to low-lying infrastructure, vehicles, and people near watercourses. "
            "Avoid driving through floodwater, move valuables to higher ground, and follow local evacuation guidance if risk is High."
        )
        suggestions = (
            "Immediate actions: check river gauge readings, issue local alerts if levels are rising, and prepare temporary sandbags or barriers."
        )
        if last_user == m:
            # repeated question — give a more detailed follow-up
            return base + " I already summarized this. " + extra + " " + suggestions
        else:
            return base + " " + extra

    if any(q in m for q in ('mitigat', 'action', 'what to do', 'suggest')):
        return f"Suggested mitigation: {action}. Prioritize life safety first — evacuation and alerts — then infrastructure measures like embankments and drainage upgrades."

    if any(q in m for q in ('areas', 'where', 'which')):
        if heat is not None:
            return "The Grad-CAM overlay highlights the image regions that influenced the model most (red = strongly influential). Inspect the overlay for flooded channels or riverbanks."
        else:
            return "No Grad-CAM explanation was generated for this prediction. Provide a model file (`model_demo.h5`) for explainability or re-run with a smaller image."

    if any(q in m for q in ('next', 'steps', 'follow')):
        return "Next steps: (1) Verify with local sensors or high-resolution imagery, (2) Notify authorities if risk is high, (3) Start temporary flood defenses and alert communities."

    # fallback: echo with context summary
    if len(m.split()) <= 3:
        # short messages often are quick questions — avoid repeating identical replies
        if last_user == m:
            return f"As I mentioned: risk={label}, probability={prob*100:.1f}%. For more, ask 'why', 'which areas', or 'what to do'."
        return f"Summary: risk={label}, probability={prob*100:.1f}%, action='{action}'. Ask me 'why', 'which areas', or 'what to do'."

    return "I can help explain the prediction or suggest actions. Ask about 'why', 'confidence', 'which areas', or 'what to do next'."


def compute_gradcam(pil_image, model, last_conv_layer_name=None, eps=1e-8):
    """Compute Grad-CAM overlay for a PIL image and a compiled Keras model.

    Returns a PIL.Image with the heatmap overlay (RGBA) or None on failure.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import backend as K
        # Prepare image for model
        input_shape = model.input_shape
        # model.input_shape can be (None, h, w, c) or similar
        _, h, w, c = input_shape if len(input_shape) == 4 else (None, 32, 32, 3)
        img_resized = pil_image.resize((w, h))
        img_arr = np.array(img_resized).astype('float32') / 255.0
        x = np.expand_dims(img_arr, axis=0)

        # Find last conv layer if not provided
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name or 'Conv' in layer.__class__.__name__:
                    last_conv_layer_name = layer.name
                    break
        if last_conv_layer_name is None:
            return None

        last_conv_layer = model.get_layer(last_conv_layer_name)

        # GradCAM implementation
        grad_model = tf.keras.models.Model([
            model.inputs], [last_conv_layer.output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]
        grads = tape.gradient(loss, conv_outputs)
        # guided gradients (simple global-average pooling)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap) if np.max(heatmap) != 0 else eps
        heatmap /= max_val
        heatmap = np.clip(heatmap, 0, 1)

        # Resize heatmap to original image
        heatmap = np.uint8(255 * heatmap)
        import cv2
        heatmap = cv2.resize(heatmap, (pil_image.width, pil_image.height))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        img = np.array(pil_image).astype('uint8')
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        from PIL import Image as PILImage
        return PILImage.fromarray(overlay)
    except Exception:
        return None


def plot_timeseries(df: pd.DataFrame, ycol: str, title: str):
    fig, ax = plt.subplots(figsize=(6, 2.5)) # pyright: ignore[reportUnknownMemberType]
    ax.plot(df['date'], df[ycol], '-o', color='#2b75b6') # pyright: ignore[reportUnknownMemberType]
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel(ycol) # pyright: ignore[reportUnknownMemberType]
    fig.autofmt_xdate()
    return fig


def main():
    st.set_page_config(layout='wide', page_title='Smart Flood Alert Map')
    inject_theme_css()
    # header banner
    st.markdown(
        '''
        <div class="top-banner">
          <div style="width:48px;height:48px;border-radius:8px;background:linear-gradient(135deg,#06b6d4,#3b82f6);display:flex;align-items:center;justify-content:center;font-weight:700;color:#021124">FL</div>
          <div>
            <h1>Smart Flood Alert Map</h1>
            <p style="margin:0;color:var(--muted);font-size:13px">Interactive dashboard — upload images, get predictions and explanations</p>
          </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    # Simple page navigation using session state: 'map' or 'chatbot'
    if 'page' not in st.session_state:
        st.session_state['page'] = 'map'

    cols = st.columns([3, 1])
    with cols[1]:
        if st.session_state['page'] != 'chatbot':
            if st.button('Open Chatbot (Upload & predict)'):
                st.session_state['page'] = 'chatbot'
                return
        else:
            if st.button('Back to Map'):
                st.session_state['page'] = 'map'
                return

    if st.session_state['page'] == 'map':
        zones = make_sample_zones()

        left, right = st.columns([2, 1])
        with left:
            m = folium.Map(location=[24.1, 90.05], zoom_start=10)
            for f in zones['features']:
                geom = f['geometry']
                coords = geom['coordinates'][0]
                props = f['properties']
                risk = props.get('risk', 'Unknown')
                name = props.get('name', props.get('id', 'Zone')) # pyright: ignore[reportUnknownMemberType]
                popup_html = f"<b>{name}</b><br>Risk: {risk}"
                folium.Polygon(locations=[(lat, lon) for lon, lat in coords], # pyright: ignore[reportUnknownArgumentType]
                               color=risk_color(risk), fill=True, fill_opacity=0.4,
                               popup=folium.Popup(popup_html, max_width=300),
                               tooltip=name).add_to(m)

            folium.TileLayer('OpenStreetMap').add_to(m)
            st.write('Click on the map to sample a point and get a model prediction')
            map_data = st_folium(m, width=700, height=500)

            last_clicked = map_data.get('last_clicked') if map_data else None
            if last_clicked:
                lat = last_clicked['lat']
                lon = last_clicked['lng']
                st.success(f'Clicked at {lat:.5f}, {lon:.5f}')
                img, probs = predict_for_point(lat, lon)
                st.image(img, caption='Sample image at clicked location', width=300)
                st.metric('High-risk probability', f'{probs[1]:.2f}')

        with right:
            st.subheader('Trends')
            df = sample_rainfall_timeseries(30)
            st.pyplot(plot_timeseries(df, 'rain_mm', 'Rainfall (mm)'))

            st.write('Elevation (approx)')
            if last_clicked:
                elev = sample_elevation(lat, lon) # pyright: ignore[reportPossiblyUnboundVariable]
                st.write(f'{elev:.1f} m')
            else:
                st.write('Click on map to sample elevation')

            st.subheader('Risk breakdown')
            if last_clicked:
                st.write(f'Low: {1-probs[1]:.2f}  |  High: {probs[1]:.2f}')
            else:
                st.write('Click on map to run a prediction')

        st.markdown('---')
        st.caption('Demo data is synthetic. Integrate real Sentinel tiles and sensors for production.')

    else:
        # Chatbot page: upload image -> run classifier -> show outputs
        st.header('Upload Satellite Image — Chatbot')
        st.write('Upload a satellite image (JPEG/PNG/TIFF). The demo uses a small CNN if available, otherwise a heuristic fallback.')

        uploaded = st.file_uploader('Choose an image', type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])
        if uploaded is not None:
            try:
                image = Image.open(uploaded).convert('RGB')
                st.image(image, caption='Uploaded image', use_container_width=True)
                if st.button('Analyze'):
                    with st.spinner('Analyzing...'):
                        risk_label, prob, action, heatmap_img = analyze_uploaded_image(image) # pyright: ignore[reportUnknownVariableType]
                        st.subheader('Results')
                        st.write(f'Risk level: **{risk_label}**')
                        st.write(f'Flood probability: **{prob*100:.1f}%**')
                        st.write('Suggested mitigation:')
                        st.info(action)
                        if heatmap_img is not None:
                            st.write('Model explanation (Grad-CAM overlay):')
                            st.image(heatmap_img, use_container_width=True)
                        # store last analysis context for chat
                        st.session_state['last_analysis'] = {
                            'label': risk_label,
                            'prob': prob,
                            'action': action,
                            'heat': heatmap_img is not None,
                        }

                        # init chat history if not present
                        if 'chat_history' not in st.session_state:
                            st.session_state['chat_history'] = []

                # Chat UI (below analyze button) — allow follow-up questions about the result
                st.markdown('---')
                st.subheader('Ask about this result')
                if 'last_analysis' not in st.session_state:
                    st.info('Run analysis first to enable contextual chat.')
                else:
                    # display chat history
                    chat_col1, chat_col2 = st.columns([4, 1])
                    with chat_col1:
                        for i, (who, text) in enumerate(st.session_state.get('chat_history', [])): # pyright: ignore[reportUnusedVariable]
                            if who == 'user':
                                st.markdown(f"**You:** {text}")
                            else:
                                st.markdown(f"**AI:** {text}")

                    # input area using a form (clears on submit)
                    with st.form(key='chat_form', clear_on_submit=True):
                        user_msg = st.text_input('Type a question about the result and press Enter', key='chat_input_form')
                        submitted = st.form_submit_button('Send')
                    if submitted and user_msg:
                        # append user
                        st.session_state['chat_history'].append(('user', user_msg))
                        # remember last user message (normalized) to avoid repeated identical replies
                        try:
                            st.session_state['_last_user_msg'] = user_msg.strip().lower()
                        except Exception:
                            pass
                        ctx = st.session_state.get('last_analysis', {})
                        resp = generate_chat_response(user_msg, ctx)
                        st.session_state['chat_history'].append(('ai', resp))
            except Exception as e:
                st.error(f'Failed to process image: {e}')


if __name__ == '__main__':
    main()
