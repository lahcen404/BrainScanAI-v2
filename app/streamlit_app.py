
import os
import base64
from io import BytesIO

import requests
import streamlit as st
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BrainScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Internal Docker address: used for API requests.
API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

# Browser-accessible address: displayed to the user.
API_DISPLAY_URL = os.getenv(
    "API_DISPLAY_URL",
    "http://localhost:8000",
)

CLASSIFICATION_ENDPOINT = f"{API_URL}/predict/classification"
DETECTION_ENDPOINT = f"{API_URL}/predict/detection"

REQUEST_TIMEOUT = 180

CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary",
]


# ============================================================
# CUSTOM CSS - STATIC, COMPACT DESIGN
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #080d19;
        --panel: #111c2e;
        --border: rgba(148,163,184,.16);
        --cyan: #22d3ee;
        --blue: #3b82f6;
        --text: #edf5ff;
        --muted: #94a3b8;
    }

    .stApp {
        background: #080d19;
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #101a2b;
        border-right: 1px solid var(--border);
    }

    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 1rem;
        max-width: 1500px;
    }

    .brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.65rem;
        font-weight: 800;
        color: white;
    }

    .brand span {
        color: var(--cyan);
    }

    .brand-caption {
        color: var(--muted);
        font-size: .72rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .hero {
        padding: 20px 24px;
        border-radius: 16px;
        border: 1px solid rgba(34,211,238,.2);
        background: linear-gradient(110deg, #142b4c, #111c2e 65%, #211b3b);
        margin-bottom: 18px;
    }

    .hero-tag {
        color: var(--cyan);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        color: white;
        margin: 8px 0 4px;
    }

    .hero p {
        color: #a9bad0;
        margin: 0;
        font-size: .9rem;
    }

    .section-heading {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        color: white;
        margin: 10px 0;
    }

    .glass-card {
        background: #111c2e;
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 15px;
    }

    .metric-label {
        color: var(--muted);
        font-size: .75rem;
        margin-bottom: 5px;
    }

    .metric-value {
        color: white;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
    }

    .metric-accent {
        color: var(--cyan);
    }

    .stButton > button {
        background: linear-gradient(100deg, #2563eb, #6d28d9);
        border: 1px solid rgba(255,255,255,.1);
        border-radius: 10px;
        color: white;
        font-weight: 700;
        padding: .65rem 1rem;
    }

    .stButton > button:hover {
        border-color: var(--cyan);
        color: white;
    }

    [data-testid="stFileUploader"] {
        background: #111c2e;
        border: 1px dashed rgba(34,211,238,.4);
        border-radius: 13px;
        padding: 8px;
    }

    [data-testid="stMetric"] {
        background: #111c2e;
        border: 1px solid var(--border);
        padding: 12px;
        border-radius: 12px;
    }

    [data-testid="stMetricValue"] {
        color: white;
    }

    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #2563eb, #22d3ee);
    }

    .disclaimer {
        background: #2a2112;
        border: 1px solid rgba(245,158,11,.25);
        border-radius: 10px;
        padding: 11px;
        color: #fcd34d;
        font-size: .78rem;
        margin-top: 20px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: .75rem;
        padding: 12px 0 0;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data(ttl=10)
def check_api_health():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def validate_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()

        image = Image.open(BytesIO(image_bytes))
        image.load()

        return image
    except Exception:
        return None


def call_prediction_api(endpoint, uploaded_file):
    try:
        uploaded_file.seek(0)
        image_bytes = uploaded_file.read()

        response = requests.post(
            endpoint,
            files={
                "file": (
                    uploaded_file.name,
                    image_bytes,
                    uploaded_file.type,
                )
            },
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()
        return response.json()

    except requests.Timeout:
        st.error("Analysis timed out. Please try again.")

    except requests.ConnectionError:
        st.error("Cannot connect to the FastAPI backend.")

    except requests.HTTPError as error:
        try:
            detail = error.response.json().get(
                "detail",
                error.response.text,
            )
        except Exception:
            detail = str(error)

        st.error(f"API error: {detail}")

    except requests.RequestException as error:
        st.error(f"Request failed: {error}")

    except ValueError:
        st.error("The API returned invalid JSON.")

    return None


def display_metric(label, value):
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value metric-accent">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CLASSIFICATION RESULTS
# ============================================================

def display_classification_results(result):

    st.markdown(
        '<div class="section-heading">🧬 Classification Results</div>',
        unsafe_allow_html=True,
    )

    predicted_class = result.get("predicted_class", "Unknown")
    confidence = float(result.get("confidence", 0))

    col1, col2 = st.columns(2)

    with col1:
        display_metric("PREDICTED CLASS", predicted_class)

    with col2:
        display_metric("CONFIDENCE", f"{confidence:.2f}%")

    st.markdown("**Class Probabilities**")

    probabilities = result.get("probabilities", {})

    for class_name in CLASS_NAMES:
        probability = float(probabilities.get(class_name, 0))

        st.caption(f"{class_name} — {probability:.2f}%")
        st.progress(max(0.0, min(probability / 100, 1.0)))


# ============================================================
# DETECTION RESULTS
# ============================================================

def display_detection_results(result):

    st.markdown(
        '<div class="section-heading">🎯 Detection Results</div>',
        unsafe_allow_html=True,
    )

    detections = result.get("detections", [])

    count = result.get(
        "number_of_detections",
        len(detections),
    )

    st.metric("Detected Regions", count)

    annotated_image = result.get("annotated_image")

    if annotated_image:

        try:
            image_bytes = base64.b64decode(annotated_image)

            annotated_pil = Image.open(
                BytesIO(image_bytes)
            )

            st.markdown("**Annotated MRI Scan**")

            # Display at original image resolution where possible.
            st.image(
                annotated_pil,
                caption="YOLOv8 detection result",
                use_container_width=False,
            )

            st.caption(
                f"Resolution: {annotated_pil.width} × "
                f"{annotated_pil.height} pixels"
            )

            st.download_button(
                "⬇ Download Annotated Image",
                data=image_bytes,
                file_name="brainscan_detection.jpg",
                mime="image/jpeg",
            )

        except Exception:
            st.warning("Unable to display the annotated image.")

    if detections:

        st.markdown("**Detection Details**")

        for index, detection in enumerate(detections, start=1):

            class_name = detection.get("class_name", "Unknown")
            confidence = float(detection.get("confidence", 0))
            bbox = detection.get("bbox", [])

            with st.expander(
                f"Detection {index} — {class_name}"
            ):

                col1, col2 = st.columns(2)

                col1.metric("Class", class_name)
                col2.metric("Confidence", f"{confidence:.2%}")

                if len(bbox) == 4:
                    st.write("Bounding box")
                    st.code(
                        f"x1: {bbox[0]:.2f} | y1: {bbox[1]:.2f}\n"
                        f"x2: {bbox[2]:.2f} | y2: {bbox[3]:.2f}"
                    )

    else:
        st.info("No tumor regions were detected.")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">BrainScan<span>AI</span></div>
        <div class="brand-caption">Intelligent Medical Imaging</div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### Analysis Mode")

    analysis_mode = st.radio(
        "Choose an analysis",
        ["Classification", "Detection", "Both"],
    )

    st.divider()

    st.markdown("### System Status")

    if check_api_health():
        st.success("API Connected")
    else:
        st.error("API Offline")

    # This is the browser-accessible address, not the Docker hostname.
    st.caption(f"API address: {API_DISPLAY_URL}")

    st.divider()

    st.markdown("### AI Models")
    st.write("🧠 InceptionV3")
    st.caption("Tumor classification")

    st.write("🎯 YOLOv8")
    st.caption("Tumor localization")

    st.divider()

    st.caption("BrainScanAI-V2")


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">AI-powered medical imaging</div>
        <h1>Brain MRI Analysis</h1>
        <p>
            Analyze MRI scans with deep learning classification
            and tumor localization.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# IMAGE UPLOAD AND PREVIEW
# ============================================================

st.markdown(
    '<div class="section-heading">01 — Upload MRI Scan</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:

    image_bytes = uploaded_file.getvalue()
    image = validate_image(image_bytes)

    if image is None:

        st.error("Invalid image. Upload a valid JPG or PNG.")

    else:

        preview_col, info_col = st.columns([1.3, 1])

        with preview_col:
            st.image(
                image,
                caption="Original MRI scan",
                use_container_width=True,
            )

        with info_col:

            st.markdown(
                '<div class="section-heading">Image Information</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="metric-label">FILENAME</div>
                    <div>{uploaded_file.name}</div>
                    <br>
                    <div class="metric-label">FORMAT</div>
                    <div>{image.format}</div>
                    <br>
                    <div class="metric-label">RESOLUTION</div>
                    <div>{image.width} × {image.height}</div>
                    <br>
                    <div class="metric-label">FILE SIZE</div>
                    <div>{len(image_bytes) / 1024:.1f} KB</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        st.markdown(
            '<div class="section-heading">02 — Run AI Analysis</div>',
            unsafe_allow_html=True,
        )

        st.caption(f"Selected mode: {analysis_mode}")

        if st.button(
            "✦ Start AI Analysis",
            use_container_width=True,
        ):

            if not check_api_health():

                st.error("FastAPI is unavailable.")

            else:

                classification_result = None
                detection_result = None

                if analysis_mode in ["Classification", "Both"]:

                    with st.spinner(
                        "InceptionV3 is analyzing the MRI..."
                    ):

                        classification_result = call_prediction_api(
                            CLASSIFICATION_ENDPOINT,
                            uploaded_file,
                        )

                if analysis_mode in ["Detection", "Both"]:

                    with st.spinner(
                        "YOLOv8 is detecting tumor regions..."
                    ):

                        detection_result = call_prediction_api(
                            DETECTION_ENDPOINT,
                            uploaded_file,
                        )

                if classification_result or detection_result:

                    st.divider()

                    # Display results side by side when both are requested.
                    if analysis_mode == "Both":

                        classification_col, detection_col = st.columns(2)

                        with classification_col:
                            if classification_result:
                                display_classification_results(
                                    classification_result
                                )

                        with detection_col:
                            if detection_result:
                                display_detection_results(
                                    detection_result
                                )

                        if (
                            classification_result is None
                            or detection_result is None
                        ):
                            st.warning(
                                "Only part of the analysis completed."
                            )

                    elif classification_result:

                        display_classification_results(
                            classification_result
                        )

                    elif detection_result:

                        display_detection_results(
                            detection_result
                        )

                else:

                    st.error("Analysis failed. Check the API logs.")


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">
        <strong>Medical Disclaimer:</strong>
        BrainScan AI is an educational and research project.
        Its predictions are not a medical diagnosis and must
        not replace professional medical advice.
    </div>

    <div class="footer">
        BrainScanAI-V2 | InceptionV3 · YOLOv8 · FastAPI
    </div>
    """,
    unsafe_allow_html=True,
)
