import json
import os
import base64
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

# 1. API Setup
API_KEY = "AQ.Ab8RN6KTF8F8CTlTIYkWvDyEOgZmxDKdytNfDZV-VWCzVi4IUA"
client = genai.Client(api_key=API_KEY)

def preprocess_image(pil_image, max_size=(1024, 1024)):
    """Downscale large images and normalize color mode to optimize network payload."""
    img = pil_image.copy()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    return img

def segment_and_quantify_leaf(pil_image):
    """
    OpenCV Computer Vision pipeline to quantify leaf disease severity:
    1. Converts image to HSV color space for illumination-invariant segmentation.
    2. Segments total leaf boundary (A_leaf).
    3. Segments diseased lesions (chlorosis, necrosis, fungal spots) (A_disease).
    4. Computes quantitative severity % = (A_disease / A_leaf) * 100.
    5. Returns segmented mask, diagnostic overlay with highlights, and metrics.
    """
    # Convert PIL Image to RGB NumPy array
    img_rgb = np.array(preprocess_image(pil_image).convert("RGB"))
    
    # Convert to HSV color space (OpenCV scale: H in [0, 179], S in [0, 255], V in [0, 255])
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # -------------------------------------------------------------------------
    # HSV COLOR RANGE EXPLANATIONS:
    # 1. Healthy Foliage (Green):
    #    H: [25, 88] spans yellow-green to deep emerald green.
    #    S: [35, 255] excludes low-saturation grays/whites.
    #    V: [35, 255] excludes dark shadow clippings.
    lower_green = np.array([25, 35, 35])
    upper_green = np.array([88, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 2. Chlorosis / Yellow Discoloration (Early stage disease / nutrient deficiency):
    #    H: [12, 28] captures yellow to golden tones.
    #    S: [40, 255] requires distinct coloration.
    #    V: [50, 255] captures medium to bright yellow spots.
    lower_yellow = np.array([12, 40, 50])
    upper_yellow = np.array([28, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 3. Necrotic Lesions / Rust / Blight (Brown / Reddish-brown dead tissue):
    #    H: [5, 22] captures rust, reddish-brown, and dark brown lesions.
    #    S: [35, 255] avoids grayscale background noise.
    #    V: [25, 200] excludes total black clipping.
    lower_brown = np.array([5, 35, 25])
    upper_brown = np.array([22, 255, 200])
    mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

    # 4. Severe Necrosis / Black Rot / Canker:
    #    Low brightness spots within leaf tissue.
    lower_dark = np.array([0, 0, 10])
    upper_dark = np.array([179, 120, 70])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)

    # 5. Powdery Mildew / White fungal spores:
    #    Very low saturation with high brightness.
    lower_white = np.array([0, 0, 185])
    upper_white = np.array([179, 45, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # -------------------------------------------------------------------------
    # STEP 1: LEAF SEGMENTATION (A_leaf)
    # The whole leaf encompasses both healthy green tissue and diseased portions.
    raw_leaf_mask = cv2.bitwise_or(mask_green, mask_yellow)
    raw_leaf_mask = cv2.bitwise_or(raw_leaf_mask, mask_brown)

    # Morphological operations to clean background noise and fill interior leaf gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_leaf_mask = cv2.morphologyEx(raw_leaf_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    clean_leaf_mask = cv2.morphologyEx(clean_leaf_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Retain the main leaf contour(s) to eliminate stray background artifacts
    contours, _ = cv2.findContours(clean_leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_leaf_mask = np.zeros_like(clean_leaf_mask)
    if contours:
        min_area = 0.015 * (img_rgb.shape[0] * img_rgb.shape[1])
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        if valid_contours:
            cv2.drawContours(final_leaf_mask, valid_contours, -1, 255, thickness=cv2.FILLED)
        else:
            final_leaf_mask = clean_leaf_mask
    else:
        final_leaf_mask = clean_leaf_mask

    # Total leaf pixel count
    a_leaf = np.count_nonzero(final_leaf_mask) 

    # -------------------------------------------------------------------------
    # STEP 2: DISEASED AREA SEGMENTATION (A_disease)
    # Combine chlorotic, necrotic, and fungal masks
    disease_candidates = cv2.bitwise_or(mask_yellow, mask_brown)
    disease_candidates = cv2.bitwise_or(disease_candidates, mask_white)
    
    # Dark rot spots are included only if they lie inside the leaf boundary
    dark_in_leaf = cv2.bitwise_and(mask_dark, mask_dark, mask=final_leaf_mask)
    disease_candidates = cv2.bitwise_or(disease_candidates, dark_in_leaf)

    # CRITICAL: Confine diseased mask strictly to the detected leaf area
    disease_mask = cv2.bitwise_and(disease_candidates, disease_candidates, mask=final_leaf_mask)
    
    # Filter tiny pixel noise from the disease mask
    kernel_disease = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel_disease)

    # Total diseased pixel count
    a_disease = np.count_nonzero(disease_mask)

    # -------------------------------------------------------------------------
    # STEP 3: SEVERITY QUANTIFICATION
    if a_leaf > 0:
        severity_percentage = round((a_disease / a_leaf) * 100.0, 2)
    else:
        severity_percentage = 0.0

    if severity_percentage < 15.0:
        severity_class = "Mild"
    elif severity_percentage <= 40.0:
        severity_class = "Moderate"
    else:
        severity_class = "Severe"

    # -------------------------------------------------------------------------
    # STEP 4: VISUALIZATION GENERATION
    # Diagnostic Image (Original + Neon Red Overlay + Contour Borders)
    red_tint = np.zeros_like(img_rgb)
    red_tint[:, :] = [255, 30, 30] # High-visibility red
    overlay = cv2.addWeighted(img_rgb, 0.45, red_tint, 0.55, 0)
    
    # Apply red overlay only where disease_mask is non-zero
    diagnostic_img = np.where(disease_mask[:, :, None] == 255, overlay, img_rgb)
    
    # Draw crisp yellow boundaries around individual lesions for precision
    disease_contours, _ = cv2.findContours(disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(diagnostic_img, disease_contours, -1, (255, 230, 0), 1)

    return {
        "diagnostic_img": diagnostic_img,
        "a_leaf": a_leaf,
        "a_disease": a_disease,
        "severity_percentage": severity_percentage,
        "severity_class": severity_class
    }

def analyze_crop(pil_image):
    optimized_image = preprocess_image(pil_image)
    prompt = """
    Analyze this crop leaf image for diseases and severity.
    Respond ONLY in valid JSON with this exact structure:
    {
      "has_disease": boolean,
      "crop_type": "string",
      "disease_name": "string",
      "confidence": float,
      "severity": {
        "percentage": int,
        "level": "Low" | "Moderate" | "High",
        "symptoms": ["string"]
      },
      "remedies": {
        "organic": ["string"],
        "chemical": ["string"]
      }
    }
    """
    response = client.models.generate_content(
        model='gemini-3.5-flash',
        contents=[optimized_image, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2
        )
    )
    if not response.text:
        raise ValueError("Model response is empty or blocked by safety filters.")
    return json.loads(response.text)

# 2. User Interface Setup
st.set_page_config(
    page_title="Croppie",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CROPPPIE DESIGN SYSTEM
# Emerald Green / Warm Beige / White
# 20% Emerald | 70% Beige | 10% White
# =============================================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');

/* ==========================================================================
   DESIGN TOKENS
   ========================================================================== */

:root {
    --emerald-950: #092e1f;
    --emerald-900: #0d3b28;
    --emerald-800: #145c3a;
    --emerald-700: #1b6b45;
    --emerald-600: #287a50;
    --emerald-500: #3b8f63;

    --beige-100: #f8f5ed;
    --beige-200: #f1ece0;
    --beige-300: #e7dfcf;

    --white: #ffffff;

    --text-primary: #173b2a;
    --text-secondary: #587063;
    --text-muted: #829188;

    --border: #e3ddcf;

    --shadow-soft: 0 8px 30px rgba(22, 59, 42, 0.06);
    --shadow-medium: 0 14px 40px rgba(22, 59, 42, 0.09);
}


/* ==========================================================================
   GLOBAL APP
   ========================================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 8% 8%,
            rgba(40, 122, 80, 0.055) 0,
            transparent 28%
        ),
        radial-gradient(
            circle at 92% 90%,
            rgba(40, 122, 80, 0.04) 0,
            transparent 25%
        ),
        var(--beige-100) !important;

    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary) !important;
}

.block-container {
    max-width: 920px !important;
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
}


/* ==========================================================================
   REMOVE STREAMLIT CHROME
   ========================================================================== */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}


/* ==========================================================================
   HERO / BRAND
   ========================================================================== */

.croppie-hero {
    text-align: center;
    padding: 1.2rem 0 1.5rem 0;
}

.croppie-logo-img {
    width: 150px;
    max-width: 150px;
    height: auto;
    display: block;
    margin: 0 auto 0.45rem auto;
    filter: drop-shadow(0 8px 18px rgba(20, 92, 58, 0.12));
    transition: transform 0.25s ease;
}

.croppie-logo-img:hover {
    transform: translateY(-2px);
}

.croppie-title {
    font-family: 'Outfit', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -2.5px;
    line-height: 0.95;
    color: var(--emerald-900);
    margin: 0;
}

.croppie-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--emerald-700);
    letter-spacing: 0.15px;
    margin: 0.2rem 0 0 0;
}


/* ==========================================================================
   HERO INTRO
   ========================================================================== */

.hero-copy {
    text-align: center;
    max-width: 650px;
    margin: 0 auto 1.8rem auto;
}

.hero-copy h2 {
    font-family: 'Outfit', sans-serif;
    color: var(--emerald-900);
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.4px;
    margin: 0 0 0.45rem 0;
}

.hero-copy p {
    color: var(--text-secondary);
    font-size: 0.98rem;
    line-height: 1.65;
    margin: 0;
}


/* ==========================================================================
   SECTION LABELS
   ========================================================================== */

.section-label {
    display: flex;
    align-items: center;
    gap: 9px;
    margin: 1.8rem 0 0.75rem 0;
}

.section-number {
    width: 26px;
    height: 26px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: var(--emerald-900);
    color: white;
    font-family: 'Outfit', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
}

.section-label-text {
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--emerald-900);
    letter-spacing: -0.1px;
}


/* ==========================================================================
   SAMPLE GALLERY
   ========================================================================== */

.sample-wrapper {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.15rem 1.25rem 1.25rem 1.25rem;
    box-shadow: var(--shadow-soft);
    margin-bottom: 1.2rem;
}

.sample-heading {
    font-family: 'Outfit', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--emerald-900);
    margin-bottom: 0.9rem;
}

.sample-heading span {
    color: var(--text-muted);
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    margin-left: 6px;
}


/* ==========================================================================
   IMAGE UPLOADER
   ========================================================================== */

.upload-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.3rem;
    box-shadow: var(--shadow-soft);
    margin-bottom: 1.2rem;
}

.upload-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--emerald-900);
    margin-bottom: 0.2rem;
}

.upload-description {
    color: var(--text-secondary);
    font-size: 0.82rem;
    margin-bottom: 0.9rem;
}


/* Streamlit uploader itself */
[data-testid="stFileUploader"] {
    background: var(--beige-100) !important;
    border: 2px dashed #b9c8bb !important;
    border-radius: 15px !important;
    padding: 0.8rem !important;
    transition: border-color 0.2s ease, background-color 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--emerald-600) !important;
    background: #f5f2e9 !important;
}

[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
}


/* ==========================================================================
   BUTTONS
   ========================================================================== */

div.stButton > button {
    width: 100%;
    min-height: 44px;
    background: var(--emerald-900) !important;
    color: var(--white) !important;

    border: none !important;
    border-radius: 11px !important;

    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;

    box-shadow: 0 5px 15px rgba(13, 59, 40, 0.15) !important;

    transition:
        transform 0.18s ease,
        box-shadow 0.18s ease,
        background-color 0.18s ease !important;
}

div.stButton > button:hover {
    background: var(--emerald-800) !important;
    color: var(--white) !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(13, 59, 40, 0.22) !important;
}

div.stButton > button:active {
    transform: translateY(0);
}


/* ==========================================================================
   ACTIVE IMAGE
   ========================================================================== */

.active-image-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1rem;
    box-shadow: var(--shadow-soft);
    margin: 1rem 0 1.2rem 0;
}

.active-image-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--emerald-900);
    margin-bottom: 0.75rem;
}


/* ==========================================================================
   DIAGNOSIS BUTTON
   ========================================================================== */

.diagnose-area {
    margin: 1.1rem 0 1.8rem 0;
}

.diagnose-area div.stButton > button {
    min-height: 52px !important;
    border-radius: 14px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.02rem !important;
    letter-spacing: 0.1px;
}


/* ==========================================================================
   RESULTS HEADER
   ========================================================================== */

.results-header {
    margin-top: 2rem;
    margin-bottom: 1rem;
    padding: 1.3rem 1.4rem;
    background: var(--emerald-900);
    border-radius: 18px;
    color: white;
    box-shadow: var(--shadow-medium);
}

.results-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    opacity: 0.72;
    margin-bottom: 0.25rem;
}

.results-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.4px;
    margin: 0;
}

.results-subtitle {
    font-size: 0.82rem;
    opacity: 0.78;
    margin-top: 0.3rem;
}


/* ==========================================================================
   DIAGNOSTIC IMAGE
   ========================================================================== */

.diagnostic-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 0.9rem;
    box-shadow: var(--shadow-soft);
    margin-bottom: 1rem;
}


/* ==========================================================================
   METRICS
   ========================================================================== */

[data-testid="stMetric"] {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.1rem 1.2rem !important;
    box-shadow: var(--shadow-soft) !important;
}

[data-testid="stMetricLabel"] p {
    color: var(--text-secondary) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] div {
    color: var(--emerald-900) !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
}


/* ==========================================================================
   DISEASE / HEALTH STATUS
   ========================================================================== */

.status-card {
    border-radius: 16px;
    padding: 1rem 1.15rem;
    margin: 0.7rem 0 1rem 0;
}

.status-card.disease {
    background: #fff7f3;
    border: 1px solid #ead5c9;
}

.status-card.healthy {
    background: #f2f8f3;
    border: 1px solid #cfe0d2;
}

.status-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.status-card.disease .status-title {
    color: #88432e;
}

.status-card.healthy .status-title {
    color: var(--emerald-800);
}

.status-text {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.5;
}


/* ==========================================================================
   SYMPTOMS
   ========================================================================== */

.symptom-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    box-shadow: var(--shadow-soft);
    margin: 1rem 0;
}

.symptom-title {
    font-family: 'Outfit', sans-serif;
    color: var(--emerald-900);
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.65rem;
}


/* ==========================================================================
   REMEDIES
   ========================================================================== */

.remedy-heading {
    font-family: 'Outfit', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--emerald-900);
    margin: 1.5rem 0 0.7rem 0;
}

button[data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: var(--text-secondary) !important;
    padding: 0.55rem 1.2rem !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--emerald-900) !important;
}

div[data-baseweb="tab-highlight"] {
    background-color: var(--emerald-700) !important;
}


/* ==========================================================================
   STREAMLIT ALERTS
   ========================================================================== */

div[data-testid="stAlert"] {
    border-radius: 14px !important;
    border-width: 1px !important;
    font-size: 0.85rem !important;
}


/* ==========================================================================
   SPINNER
   ========================================================================== */

.stSpinner > div {
    border-top-color: var(--emerald-700) !important;
}


/* ==========================================================================
   DIVIDER
   ========================================================================== */

hr {
    border: none !important;
    height: 1px !important;
    background: var(--border) !important;
    margin: 1.7rem 0 !important;
}


/* ==========================================================================
   FOOTER
   ========================================================================== */

.croppie-footer {
    text-align: center;
    margin-top: 2.8rem;
    padding-top: 1.2rem;
    border-top: 1px solid var(--border);
    color: var(--text-muted);
    font-size: 0.72rem;
}

.croppie-footer strong {
    color: var(--emerald-700);
}


/* ==========================================================================
   MOBILE
   ========================================================================== */

@media (max-width: 640px) {

    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1rem !important;
    }

    .croppie-logo-img {
        width: 125px;
    }

    .croppie-title {
        font-size: 3rem;
    }

    .hero-copy h2 {
        font-size: 1.35rem;
    }

    .hero-copy p {
        font-size: 0.88rem;
    }

    .sample-wrapper,
    .upload-card {
        padding: 0.9rem;
    }
}

</style>
""", unsafe_allow_html=True)


# Helper to load and embed local logo as base64
# Helper to load and embed local logo as base64
logo_path = os.path.join(os.path.dirname(__file__), "Croppie_Logo.png")

if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <div class="croppie-hero">
        <img src="data:image/png;base64,{logo_b64}" class="croppie-logo-img" alt="Croppie" />
        <p class="croppie-subtitle">Your Crop's Personal AI Assistant</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="croppie-hero">
        <h1 class="croppie-title">Croppie</h1>
        <p class="croppie-subtitle">Your Crop's Personal AI Assistant</p>
    </div>
    """, unsafe_allow_html=True)
# ---------------------------------------------------------------------------
# SAMGAL: SAMPLE IMAGE GALLERY
# Create the SamGal directory if it doesn't already exist.
# Place sample crop images inside: SamGal/healthy_leaf.jpg, etc.
# ---------------------------------------------------------------------------
SAMGAL_DIR = Path(__file__).parent / "SamGal"
SAMGAL_DIR.mkdir(exist_ok=True)  # No-op if directory already exists

# Dynamically discover all image files in the SamGal folder
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}
samgal_images = sorted(
    [p for p in SAMGAL_DIR.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]
)

# Render sample gallery only when images are present in SamGal
if samgal_images:
    st.markdown('<div class="sampgal-section"><p class="sampgal-title">Try a Sample Image</p></div>', unsafe_allow_html=True)

    # Build thumbnail row using Streamlit columns (up to 5 per row)
    cols_per_row = 5
    rows = [samgal_images[i:i + cols_per_row] for i in range(0, len(samgal_images), cols_per_row)]
    selected_sample_path = None

    for row in rows:
        cols = st.columns(len(row))
        for col, img_path in zip(cols, row):
            with col:
                thumb = Image.open(img_path)
                st.image(thumb, use_container_width=True)
                # Clean display name: strip extension, replace underscores with spaces, title-case
                display_name = img_path.stem.replace("_", " ").title()
                if st.button(display_name, key=f"samgal_{img_path.stem}"):
                    selected_sample_path = img_path
else:
    selected_sample_path = None

# ---------------------------------------------------------------------------
# FILE UPLOADER SECTION
# ---------------------------------------------------------------------------
st.markdown('<p class="farmer-upload-label">Upload a photo of your crop leaf to check for diseases and get easy remedies:</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a photo of your crop leaf to check for diseases and get easy remedies:",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ---------------------------------------------------------------------------
# DETERMINE ACTIVE IMAGE: prefer uploaded file, then gallery selection,
# persisting gallery choice in session_state so it survives reruns.
# ---------------------------------------------------------------------------
if "samgal_selected" not in st.session_state:
    st.session_state["samgal_selected"] = None

if selected_sample_path is not None:
    # User just clicked a gallery thumbnail - store it
    st.session_state["samgal_selected"] = str(selected_sample_path)

if uploaded_file is not None:
    # Fresh upload takes priority; clear any previous gallery selection
    st.session_state["samgal_selected"] = None
    active_image = Image.open(uploaded_file)
    image_caption = "Uploaded Leaf Photo"
elif st.session_state["samgal_selected"]:
    active_image = Image.open(st.session_state["samgal_selected"])
    image_caption = Path(st.session_state["samgal_selected"]).stem.replace("_", " ").title()
else:
    active_image = None
    image_caption = ""

# ---------------------------------------------------------------------------
# SHARED DIAGNOSIS PIPELINE (runs for both uploaded and gallery images)
# ---------------------------------------------------------------------------
if active_image is not None:
    st.image(active_image, caption=image_caption, width=350)

    if st.button("Diagnose Crop"):
        with st.spinner("Diagnosing..."):
            try:
                # 1. Run OpenCV quantitative segmentation
                cv_results = segment_and_quantify_leaf(active_image)

                # 2. Run Gemini multimodal model for disease classification & treatment
                ai_data = analyze_crop(active_image)

                st.markdown("---")

                # 1. Highlighted image
                st.image(cv_results["diagnostic_img"], caption="Diagnostic Overlay (Diseased Spots)", use_container_width=True)

                # 2. Disease Detected
                if ai_data.get("has_disease"):
                    st.error(f"**Disease Detected:** {ai_data['disease_name']} ({ai_data['crop_type']})")
                else:
                    st.success("The crop appears healthy! No diseases detected by the AI model.")

                # 3. Area of the crop affected
                st.metric("Area of the crop affected", f"{cv_results['severity_percentage']}%")

                if ai_data.get("has_disease"):
                    # Symptoms
                    st.write("**Visual Symptoms Observed:**")
                    for symptom in ai_data.get('severity', {}).get('symptoms', []):
                        st.write(f"- {symptom}")

                # 4. The severity
                st.metric("Estimated Severity", cv_results["severity_class"])
                if ai_data.get("has_disease"):
                    sev = ai_data.get('severity', {})
                    st.caption(f"AI Qualitative Assessment: {sev.get('level', cv_results['severity_class'])} ({sev.get('percentage', 0)}% estimated) | Confidence: {ai_data.get('confidence', 0.0):.2f}")

                # 5. Remedies (Last section)
                if ai_data.get("has_disease"):
                    st.write("")
                    tab1, tab2 = st.tabs(["Organic", "Chemical"])
                    with tab1:
                        st.markdown("### Organic Remedies")
                        for remedy in ai_data.get('remedies', {}).get('organic', []):
                            st.success(remedy)
                    with tab2:
                        st.markdown("### Chemical Remedies")
                        for remedy in ai_data.get('remedies', {}).get('chemical', []):
                            st.warning(remedy)

            except Exception as e:
                err_msg = str(e)
                if "getaddrinfo" in err_msg.lower() or "11001" in err_msg:
                    st.error("**Network Connection Error:** Could not reach the AI service. Please check your internet connection or DNS and try again.")
                else:
                    st.error(f"Error during analysis: {err_msg}")