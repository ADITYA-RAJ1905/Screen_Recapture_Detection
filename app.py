import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern
import time
import threading

# streamlit-webrtc gives us access to the VISITOR'S browser camera (works on
# desktop and mobile) instead of the camera attached to the server.
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="Screen Recapture Detector",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------------
# CUSTOM CSS
# -------------------------------

st.markdown("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 10% 10%, #1c1f3f 0%, #0f1024 45%, #060714 100%);
        color: #f1f1f6;
    }

    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* header {visibility: hidden;} */

    /* Hero header */
    .hero-title {
        text-align: center;
        font-size: 2.7rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7f5af0, #2cb67d, #ff8906);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 6s linear infinite;
        margin-bottom: 0;
        padding-top: 0.5rem;
    }

    @keyframes shine {
        to { background-position: 200% center; }
    }

    .hero-subtitle {
        text-align: center;
        color: #b8b8d1;
        font-size: 1.05rem;
        font-weight: 300;
        margin-top: 0.3rem;
        margin-bottom: 2rem;
    }

    /* Glassmorphism card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 1.8rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.5rem;
    }

    /* Uploader styling */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.04);
        border: 2px dashed rgba(127, 90, 240, 0.5);
        border-radius: 16px;
        padding: 1.2rem;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #7f5af0;
        background: rgba(127, 90, 240, 0.08);
    }

    /* Image display */
    [data-testid="stImage"] img {
        border-radius: 16px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.45);
    }

    /* Result banners */
    .result-real {
        background: linear-gradient(135deg, rgba(44,182,125,0.18), rgba(44,182,125,0.05));
        border: 1px solid #2cb67d;
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        text-align: center;
        animation: fadeIn 0.6s ease;
    }

    .result-screen {
        background: linear-gradient(135deg, rgba(255,84,84,0.18), rgba(255,84,84,0.05));
        border: 1px solid #ff5454;
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        text-align: center;
        animation: fadeIn 0.6s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-emoji {
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }

    .result-label {
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 0.2rem;
    }

    .result-confidence {
        font-size: 1rem;
        color: #d3d3e3;
        font-weight: 300;
    }

    /* Metric pills */
    .metric-pill {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 0.9rem;
        text-align: center;
    }
    .metric-pill .value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #7f5af0;
    }
    .metric-pill .label {
        font-size: 0.78rem;
        color: #b8b8d1;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    /* Progress bar recolor */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2cb67d, #ff8906, #ff5454);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #14152e 0%, #0a0b1a 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Live badge */
    .live-badge {
        display: inline-block;
        background: #ff5454;
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        margin-left: 0.5rem;
        animation: pulse 1.4s infinite;
        vertical-align: middle;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }

    .footer-note {
        text-align: center;
        color: #6c6c8a;
        font-size: 0.8rem;
        margin-top: 2.5rem;
        padding-bottom: 1rem;
    }

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HERO HEADER
# -------------------------------

st.markdown('<div class="hero-title">📱 Screen Recapture Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">AI-powered forensic analysis — instantly tell whether an image is a '
    '<b>genuine photo</b> or a <b>photo of a screen</b>.</div>',
    unsafe_allow_html=True
)

# -------------------------------
# SIDEBAR
# -------------------------------

with st.sidebar:
    st.markdown("### ⚙️ How it works")
    st.markdown(
        """
        This detector combines two models:

        - **🧠 EfficientNet-B0** — deep visual pattern recognition  
        - **🌲 XGBoost** — handcrafted forensic features (sharpness, glare, edge density, frequency & texture analysis)

        Scores are blended (60% / 40%) into one final confidence score.
        """
    )
    st.markdown("---")
    st.markdown("### 🎯 Score Guide")
    st.markdown(
        """
        - **0.00 – 0.49** → ✅ Real Photo  
        - **0.50 – 1.00** → ⚠️ Screen Recapture
        """
    )
    st.markdown("---")
    st.markdown("### 🎥 Live Camera Settings")
    facing_mode = st.radio(
        "Camera",
        options=["environment", "user"],
        format_func=lambda x: "📷 Back camera" if x == "environment" else "🤳 Front camera",
        horizontal=True,
        help="On phones, 'Back camera' usually gives sharper results for document/photo checks."
    )
    analyze_every_n = st.slider(
        "Analyze every N frames", min_value=1, max_value=15, value=5,
        help="Higher = smoother video but less frequent score updates."
    )
    st.caption(
        "Live Camera streams video straight from **your device's browser** "
        "(works on phones/tablets too) using WebRTC — nothing is accessed on the server."
    )
    st.markdown("---")
    st.caption("Built with PyTorch, XGBoost & Streamlit")

# -------------------------------
# DEVICE
# -------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# LOAD EFFICIENTNET
# -------------------------------

@st.cache_resource
def load_effnet():

    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 2)
    )

    model.load_state_dict(
        torch.load("enet.pth", map_location=device)
    )

    model.to(device)
    model.eval()

    return model

# -------------------------------
# LOAD XGBOOST
# -------------------------------

@st.cache_resource
def load_xgb():
    return joblib.load("xgboost.pkl")

eff_model = load_effnet()
xgb_model = load_xgb()

# -------------------------------
# TRANSFORM
# -------------------------------

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# FEATURE EXTRACTION (works on an in-memory BGR array)
# -------------------------------

def extract_features(img_bgr):

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    features = []

    # Sharpness
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    features.append(lap_var)

    # Edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    features.append(edge_density)

    # Glare ratio
    glare_ratio = np.sum(gray > 240) / gray.size
    features.append(glare_ratio)

    # FFT high frequency energy
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude = np.abs(fshift)

    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2

    radius = min(h, w) // 8

    mask = np.ones_like(magnitude)
    mask[
        center_h-radius:center_h+radius,
        center_w-radius:center_w+radius
    ] = 0

    high_freq_energy = np.mean(magnitude * mask)

    features.append(high_freq_energy)

    # LBP histogram
    lbp = local_binary_pattern(
        gray,
        P=8,
        R=1,
        method="uniform"
    )

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, 11),
        density=True
    )

    features.extend(hist)

    return np.array(features)

# -------------------------------
# PREDICTION (works on an in-memory BGR array)
# -------------------------------

def predict_upload(img_bgr):

    image = Image.fromarray(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    )

    x = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = eff_model(x)
        eff_score = torch.softmax(output, dim=1)[0][1].item()

    feats = extract_features(img_bgr).reshape(1, -1)
    xgb_score = xgb_model.predict_proba(feats)[0][1]

    final_score = 0.6 * eff_score + 0.4 * xgb_score

    return final_score, eff_score, xgb_score
def predict_live(img_bgr):

    image = Image.fromarray(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    )

    x = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = eff_model(x)
        eff_score = torch.softmax(output, dim=1)[0][1].item()

    return eff_score, eff_score, 0.0


def render_result_html(score):
    if score >= 0.5:
        return f"""
        <div class="result-screen">
            <div class="result-emoji">⚠️</div>
            <div class="result-label" style="color:#ff5454;">SCREEN PHOTO DETECTED</div>
            <div class="result-confidence">Confidence: {score:.2%}</div>
        </div>
        """
    else:
        return f"""
        <div class="result-real">
            <div class="result-emoji">✅</div>
            <div class="result-label" style="color:#2cb67d;">AUTHENTIC REAL PHOTO</div>
            <div class="result-confidence">Confidence: {1 - score:.2%}</div>
        </div>
        """


def render_metrics_html(score, eff_score, xgb_score):
    return f"""
    <div style="display:flex; gap:0.8rem; margin-top:0.8rem;">
        <div class="metric-pill" style="flex:1;">
            <div class="value">{score:.2f}</div>
            <div class="label">Final Score</div>
        </div>
        <div class="metric-pill" style="flex:1;">
            <div class="value">{eff_score:.2f}</div>
            <div class="label">EfficientNet</div>
        </div>
        <div class="metric-pill" style="flex:1;">
            <div class="value">{xgb_score:.2f}</div>
            <div class="label">XGBoost</div>
        </div>
    </div>
    """

# -------------------------------
# WEBRTC VIDEO PROCESSOR (runs in its own thread per browser connection)
# -------------------------------

if WEBRTC_AVAILABLE:

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class ScreenRecaptureProcessor(VideoProcessorBase):
        def __init__(self):
            self.frame_count = 0
            self.analyze_every_n = 5
            self.lock = threading.Lock()
            self.score = 0.0
            self.eff_score = 0.0
            self.xgb_score = 0.0

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            self.frame_count += 1

            # Only run the (heavier) models every N frames to keep video smooth
            if self.frame_count % max(self.analyze_every_n, 1) == 0:
                try:
                    score, eff_score, xgb_score = predict_live(img)
                    with self.lock:
                        self.score = score
                        self.eff_score = eff_score
                        self.xgb_score = xgb_score
                except Exception:
                    pass

            with self.lock:
                score = self.score

            # Overlay the latest score directly on the video
            label = "SCREEN" if score >= 0.5 else "REAL"
            color = (84, 84, 255) if score >= 0.5 else (125, 182, 44)  # BGR
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 50), (20, 20, 20), -1)
            img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
            cv2.putText(
                img,
                f"{label}  |  score: {score:.2f}",
                (15, 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
                cv2.LINE_AA
            )

            return av.VideoFrame.from_ndarray(img, format="bgr24")

# -------------------------------
# UI — TABS
# -------------------------------

upload_tab, camera_tab = st.tabs(["📤 Upload Image", "🎥 Live Camera"])

# =========================================================
# TAB 1 — UPLOAD IMAGE
# =========================================================
with upload_tab:

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file is None:
        st.markdown(
            "<div style='text-align:center; color:#b8b8d1;'>"
            "📤 Drag & drop or click to upload a JPG / PNG image"
            "</div>",
            unsafe_allow_html=True
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with st.spinner("🔍 Analyzing pixels, frequencies & textures..."):
            score, eff_score, xgb_score = predict_upload(img_bgr)
            time.sleep(0.3)

        st.markdown("### Result")
        st.markdown(render_result_html(score), unsafe_allow_html=True)

        st.write("")
        st.progress(float(score))
        st.write(f"Screen Probability: **{score:.4f}**")

        st.write("")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""<div class="metric-pill">
                        <div class="value">{score:.2f}</div>
                        <div class="label">Final Score</div>
                    </div>""",
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f"""<div class="metric-pill">
                        <div class="value">{eff_score:.2f}</div>
                        <div class="label">EfficientNet</div>
                    </div>""",
                unsafe_allow_html=True
            )
        with c3:
            st.markdown(
                f"""<div class="metric-pill">
                        <div class="value">{xgb_score:.2f}</div>
                        <div class="label">XGBoost</div>
                    </div>""",
                unsafe_allow_html=True
            )

# =========================================================
# TAB 2 — LIVE CAMERA (browser / mobile camera via WebRTC)
# =========================================================
with camera_tab:

    st.markdown(
        '<div class="glass-card">'
        '<b>Live Detection</b><span class="live-badge">● LIVE</span>'
        '<div style="color:#b8b8d1; margin-top:0.5rem; font-size:0.92rem;">'
        'Streams your camera straight from the browser — works on phones and tablets. '
        'Tap "Start", then allow camera access when prompted.'
        '</div></div>',
        unsafe_allow_html=True
    )

    if not WEBRTC_AVAILABLE:
        st.error(
            "Live camera requires the `streamlit-webrtc` package, which isn't installed.\n\n"
            "Install it with:\n\n"
            "```\npip install streamlit-webrtc av\n```\n\n"
            "then restart the app."
        )
    else:
        st.caption(
            "📶 Note: browsers only allow camera access over **HTTPS** (or `localhost`). "
            "If you're testing over a plain `http://` LAN address on your phone, the camera "
            "prompt won't appear — deploy behind HTTPS or use `localhost`/ngrok/Streamlit "
            "Cloud instead."
        )

        ctx = webrtc_streamer(
            key="screen-recapture-live",
            video_processor_factory=ScreenRecaptureProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": {"facingMode": {"ideal": facing_mode}},
                "audio": False,
            },
            async_processing=True,
        )

        result_placeholder = st.empty()
        metrics_placeholder = st.empty()

        if ctx.video_processor:
            ctx.video_processor.analyze_every_n = analyze_every_n

        if ctx.state.playing:
            # Poll the processor's latest score and refresh the UI below the video
            while ctx.state.playing:
                if ctx.video_processor:
                    with ctx.video_processor.lock:
                        score = ctx.video_processor.score
                        eff_score = ctx.video_processor.eff_score
                        xgb_score = ctx.video_processor.xgb_score

                    result_placeholder.markdown(render_result_html(score), unsafe_allow_html=True)
                    metrics_placeholder.markdown(
                        render_metrics_html(score, eff_score, xgb_score), unsafe_allow_html=True
                    )
                time.sleep(0.4)
        else:
            st.info("Tap **Start** above to begin live analysis.")

st.markdown(
    '<div class="footer-note">Made with 💜 using PyTorch · XGBoost · Streamlit</div>',
    unsafe_allow_html=True
)
