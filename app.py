from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TB-XRay AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

MODELS_DIR = Path(__file__).parent / "models"

MODEL_PATH = MODELS_DIR / "best_tb_cnn.keras"

CLASS_NAMES = ["Normal", "Tuberculosis"]


# ============================================================
# FULL THEME
# ============================================================

st.markdown(
    """
    <style>

    /* MAIN BACKGROUND */
    .stApp {
        background-color: #FFF7EF;
    }

    .main {
        background-color: #FFF7EF;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* HEADINGS */
    h1 {
        color: #292522 !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #292522 !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #292522 !important;
        font-weight: 700 !important;
    }

    p {
        color: #625B55;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #292522;
        border-right: 3px solid #E76F51;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #FFF7EF !important;
    }

    /* SIDEBAR RADIO */
    [data-testid="stSidebar"] [role="radiogroup"] {
        gap: 6px;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background-color: #35302C;
        border-radius: 10px;
        padding: 7px 10px;
        border: 1px solid #46403A;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background-color: #443A34;
        border-color: #E76F51;
    }

    /* CONTAINERS / CARDS */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFDF9;
        border-radius: 18px;
        border: 1px solid #E5D9CE;
        box-shadow: 0 5px 18px rgba(70, 55, 45, 0.06);
    }

    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background-color: #FFFDF9;
        border: 2px dashed #D9B9A7;
        border-radius: 18px;
        padding: 12px;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #E76F51;
        background-color: #FFF8F3;
    }

    /* BUTTON */
    .stButton > button {
        background-color: #E76F51;
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background-color: #C9563D;
        color: white;
    }

    /* METRICS */
    [data-testid="stMetric"] {
        background-color: #FFFDF9;
        border: 1px solid #E4D9D0;
        border-radius: 15px;
        padding: 15px;
    }

    /* PROGRESS */
    [data-testid="stProgressBar"] > div > div {
        background-color: #E76F51;
    }

    /* EXPANDER */
    [data-testid="stExpander"] {
        background-color: #FFFDF9;
        border: 1px solid #E3D8CE;
        border-radius: 14px;
    }

    /* ALERTS */
    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* DIVIDER */
    hr {
        border-color: #E5D9CE;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():

    return tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess(image):

    image = image.convert("L")

    image = image.resize(IMG_SIZE)

    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    array = array[None, ..., None]

    return array


# ============================================================
# PREDICTION
# ============================================================

def predict(model, image):

    array = preprocess(image)

    prediction = model.predict(
        array,
        verbose=0
    )

    tb_probability = float(
        prediction[0][0]
    )

    normal_probability = 1.0 - tb_probability

    if tb_probability >= 0.5:

        label = "Tuberculosis"

        confidence = tb_probability

    else:

        label = "Normal"

        confidence = normal_probability

    return (
        label,
        normal_probability,
        tb_probability,
        confidence
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🫁 TB-XRay AI")

    st.caption(
        "Intelligent Chest X-Ray Analysis"
    )

    st.divider()

    st.subheader("🧭 Navigation")

    page = st.radio(
        "Select Page",
        [
            "🏠 Home",
            "🔬 AI Analysis",
            "🧠 About Model",
            "ℹ️ About Project",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    st.subheader("🖥️ System Status")

    if MODEL_PATH.exists():

        st.success(
            "🟢 AI Model Ready"
        )

        st.caption(
            "Model file detected and ready for prediction."
        )

    else:

        st.error(
            "🔴 Model Not Found"
        )

        st.caption(
            "Check the models folder."
        )


    # --------------------------------------------------------
    # SYSTEM INFORMATION
    # --------------------------------------------------------

    st.subheader("⚡ System Information")

    with st.container(border=True):

        st.write("🧠 **Model**")

        st.caption("Custom CNN")

        st.write("🖼️ **Input**")

        st.caption("224 × 224 grayscale")

        st.write("🎯 **Classes**")

        st.caption("Normal / Tuberculosis")

        st.write("🔧 **Framework**")

        st.caption("TensorFlow / Keras")


    st.divider()

    st.warning(
        "⚠️ Academic research prototype. "
        "Not for clinical diagnosis."
    )


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title("🫁 TB-XRay AI")

    st.subheader(
        "Pulmonary Chest X-Ray Classification"
    )

    st.write(
        "An AI-powered research application that uses "
        "a Convolutional Neural Network to analyze "
        "chest X-ray images."
    )

    st.success(
        "🟢 AI system is ready for image analysis."
    )

    st.divider()

    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.header("⚡ How Does It Work?")

    st.write(
        "The system follows three simple steps."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.subheader("📤 Upload")

            st.write(
                "Upload a chest X-ray image in "
                "PNG, JPG or JPEG format."
            )

            st.caption(
                "STEP 01 • INPUT"
            )


    with col2:

        with st.container(border=True):

            st.subheader("🧠 Analyze")

            st.write(
                "The image is converted to grayscale, "
                "resized to 224 × 224 pixels and "
                "normalized."
            )

            st.caption(
                "STEP 02 • PROCESSING"
            )


    with col3:

        with st.container(border=True):

            st.subheader("📊 Predict")

            st.write(
                "The CNN analyzes the image and "
                "predicts Normal or Tuberculosis."
            )

            st.caption(
                "STEP 03 • RESULT"
            )


    st.divider()

    # ========================================================
    # FEATURES
    # ========================================================

    st.header("✨ What This System Provides")

    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.subheader("🩻 X-Ray Analysis")

            st.write(
                "Upload and analyze pulmonary "
                "chest X-ray images."
            )


    with col2:

        with st.container(border=True):

            st.subheader("🎯 Confidence Score")

            st.write(
                "View the probability associated "
                "with the model prediction."
            )


    with col3:

        with st.container(border=True):

            st.subheader("⚡ Fast Prediction")

            st.write(
                "Analyze a new image quickly after "
                "the model has been loaded."
            )


# ============================================================
# AI ANALYSIS
# ============================================================

elif page == "🔬 AI Analysis":

    st.title("🔬 AI Chest X-Ray Analysis")

    st.write(
        "Upload a chest X-ray image to get a prediction "
        "from the trained CNN."
    )

    uploaded_file = st.file_uploader(
        "📤 Upload Chest X-Ray",
        type=["png", "jpg", "jpeg"],
        help="Supported formats: PNG, JPG and JPEG.",
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        st.divider()

        image_col, result_col = st.columns(
            [1, 1],
            gap="large"
        )


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        with image_col:

            st.subheader("🩻 Uploaded X-Ray")

            with st.container(border=True):

                st.image(
                    image,
                    use_container_width=True
                )

            st.caption(
                f"Original image size: "
                f"{image.width} × {image.height} pixels"
            )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        with result_col:

            st.subheader("🤖 AI Prediction")

            try:

                with st.spinner(
                    "Analyzing X-ray..."
                ):

                    model = load_model()

                    (
                        label,
                        normal_probability,
                        tb_probability,
                        confidence
                    ) = predict(
                        model,
                        image
                    )


                if label == "Tuberculosis":

                    st.error(
                        "⚠️ Prediction: Tuberculosis"
                    )

                    st.warning(
                        "The model detected patterns that "
                        "are more consistent with Tuberculosis."
                    )

                else:

                    st.success(
                        "✅ Prediction: Normal"
                    )

                    st.info(
                        "The model detected patterns that "
                        "are more consistent with Normal."
                    )


                st.write("")

                metric1, metric2 = st.columns(2)

                with metric1:

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.1%}"
                    )

                with metric2:

                    st.metric(
                        "TB Probability",
                        f"{tb_probability:.1%}"
                    )


                st.write("")

                st.subheader(
                    "📊 Prediction Probabilities"
                )

                st.write(
                    f"🟢 Normal: "
                    f"**{normal_probability:.1%}**"
                )

                st.progress(
                    normal_probability
                )

                st.write(
                    f"🔴 Tuberculosis: "
                    f"**{tb_probability:.1%}**"
                )

                st.progress(
                    tb_probability
                )


            except Exception as e:

                st.error(
                    "Unable to load or run the model."
                )

                st.exception(e)


    else:

        st.info(
            "👆 Upload a chest X-ray image to begin."
        )


# ============================================================
# ABOUT MODEL
# ============================================================

elif page == "🧠 About Model":

    st.title("🧠 About the AI Model")

    st.write(
        "TB-XRay AI uses a Convolutional Neural Network "
        "for binary classification of pulmonary chest "
        "X-ray images."
    )

    st.divider()

    # ========================================================
    # MODEL
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.subheader("🧠 Convolutional Neural Network")

            st.write(
                "The CNN learns visual features from "
                "chest X-ray images and uses these "
                "features to classify the image."
            )


    with col2:

        with st.container(border=True):

            st.subheader("🎯 Binary Classification")

            st.write(
                "The model has two output classes:"
            )

            st.success(
                "🟢 Normal"
            )

            st.error(
                "🔴 Tuberculosis"
            )


    st.write("")

    # ========================================================
    # PREPROCESSING
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.subheader("🖼️ Image Preprocessing")

            st.write(
                "Every uploaded image follows the "
                "same preprocessing process."
            )

            st.write(
                "1. Convert image to grayscale"
            )

            st.write(
                "2. Resize to 224 × 224"
            )

            st.write(
                "3. Normalize pixel values"
            )

            st.write(
                "4. Add required CNN dimensions"
            )


    with col2:

        with st.container(border=True):

            st.subheader("📚 Dataset")

            st.write(
                "The model was trained using pulmonary "
                "chest X-ray data from the Montgomery "
                "and Shenzhen collections."
            )

            st.write(
                "The data contains chest X-ray images "
                "used for research involving pulmonary "
                "abnormalities and Tuberculosis."
            )


    st.divider()

    # ========================================================
    # MODEL SPECIFICATIONS
    # ========================================================

    st.header("📋 Model Specifications")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Input",
            "224 × 224"
        )

    with col2:

        st.metric(
            "Channels",
            "1"
        )

    with col3:

        st.metric(
            "Classes",
            "2"
        )

    with col4:

        st.metric(
            "Framework",
            "TensorFlow"
        )


    st.divider()

    with st.expander(
        "🔬 View Preprocessing Code"
    ):

        st.code(
            """
image = image.convert("L")

image = image.resize((224, 224))

array = np.asarray(
    image,
    dtype=np.float32
) / 255.0

array = array[None, ..., None]
            """,
            language="python"
        )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About the Project")

    st.write(
        "TB-XRay AI is an academic deep learning project "
        "demonstrating the use of CNNs for medical image "
        "classification."
    )

    st.divider()

    # ========================================================
    # OBJECTIVE
    # ========================================================

    st.header("🎯 Project Objective")

    with st.container(border=True):

        st.write(
            "The main objective is to develop an AI-based "
            "system capable of classifying pulmonary "
            "chest X-ray images into Normal and "
            "Tuberculosis categories."
        )


    st.divider()

    # ========================================================
    # TECHNOLOGIES
    # ========================================================

    st.header("🛠️ Technologies Used")

    st.write(
        "Simple explanation of why each technology "
        "was used in the project."
    )


    # --------------------------------------------------------
    # PYTHON
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.subheader("🐍 Python")

            st.caption("Why use it?")

            st.write(
                "Python is simple and powerful for "
                "AI, machine learning and image processing."
            )


    # --------------------------------------------------------
    # TENSORFLOW
    # --------------------------------------------------------

    with col2:

        with st.container(border=True):

            st.subheader("🧠 TensorFlow / Keras")

            st.caption("Why use it?")

            st.write(
                "Used to build, train, save and run "
                "the CNN model."
            )


    # --------------------------------------------------------
    # PILLOW
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.subheader("🖼️ Pillow")

            st.caption("Why use it?")

            st.write(
                "Used to open, resize and prepare "
                "uploaded X-ray images."
            )


    # --------------------------------------------------------
    # NUMPY
    # --------------------------------------------------------

    with col2:

        with st.container(border=True):

            st.subheader("🔢 NumPy")

            st.caption("Why use it?")

            st.write(
                "Used to convert images into numerical "
                "arrays that the CNN can process."
            )


    # --------------------------------------------------------
    # STREAMLIT
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.subheader("🎈 Streamlit")

            st.caption("Why use it?")

            st.write(
                "Used to create the interactive web "
                "application directly with Python."
            )


    # --------------------------------------------------------
    # MATPLOTLIB
    # --------------------------------------------------------

    with col2:

        with st.container(border=True):

            st.subheader("📊 Matplotlib")

            st.caption("Why use it?")

            st.write(
                "Used during development to visualize "
                "images and model training results."
            )


    st.divider()

    # ========================================================
    # WORKFLOW
    # ========================================================

    st.header("🔄 Project Workflow")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        with st.container(border=True):

            st.subheader("1️⃣ Data")

            st.write(
                "Collect chest X-ray images."
            )


    with col2:

        with st.container(border=True):

            st.subheader("2️⃣ Training")

            st.write(
                "Train the CNN to learn image patterns."
            )


    with col3:

        with st.container(border=True):

            st.subheader("3️⃣ Testing")

            st.write(
                "Evaluate the trained model on images."
            )


    with col4:

        with st.container(border=True):

            st.subheader("4️⃣ Deployment")

            st.write(
                "Deploy the model through Streamlit."
            )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    ⚠️ **Important Medical Disclaimer**

    This application is an academic and research prototype.
    It has not been clinically validated and is not a
    certified medical device.

    Its predictions should not be used for diagnosis,
    treatment, or other medical decisions.

    Always consult a qualified physician or radiologist
    for professional medical evaluation.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🫁 TB-XRay AI • CNN-Based Pulmonary Chest X-Ray Classification"
)

st.caption(
    "Academic Machine Learning Project"
)

