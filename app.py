import json
import platform
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

MODEL_PATH = "best_model.keras"
METRICS_PATH = "metrics.json"
CONFUSION_MATRIX_PATH = "confusion_matrix.png"
IMAGE_SIZE = 28

st.set_page_config(page_title="Digit AI Dashboard", page_icon = "🔢", layout = "wide")

st.markdown("""
<style>
.block-container {padding-top:1rem;}
.metric-card{border:1px solid #333;padding:.5rem;border-radius:10px;}
</style>
""", unsafe_allow_html = True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_metrics():
    p = Path(METRICS_PATH)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)

def preprocess_digit(image_array):
    if image_array.ndim == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(image_array, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,11,2
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No digit detected")

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    digit = binary[y:y+h, x:x+w]

    scale = 20 / max(h, w)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    digit = cv2.resize(digit, (nw, nh))

    canvas = np.zeros((28, 28), dtype=np.uint8)
    xo, yo = (28 - nw)//2, (28 - nh)//2
    canvas[yo:yo+nh, xo:xo+nw] = digit

    processed = canvas.astype("float32") / 255.0
    model_input = processed.reshape(1, 28, 28, 1)
    return processed, model_input

def predict(model, model_input):
    start = time.perf_counter()
    probs = model.predict(model_input, verbose=0)[0]
    latency = (time.perf_counter() - start) * 1000
    return int(np.argmax(probs)), float(np.max(probs)), probs, latency

model = load_model()
metrics = load_metrics()

if "history" not in st.session_state:
    st.session_state.history = []

if "last_probs" not in st.session_state:
    st.session_state.last_probs = None

if "last_processed" not in st.session_state:
    st.session_state.last_processed = None

if "last_input" not in st.session_state:
    st.session_state.last_input = None

def clear_prediction_state():
    st.session_state.history = []
    st.session_state.last_probs = None
    st.session_state.last_processed = None
    st.session_state.last_input = None


with st.sidebar:
    st.title("✍️ Digit AI")

    st.success("Model Online")

    st.metric(
        "Parameters",
        f"{model.count_params():,}"
    )

    st.metric(
        "Accuracy",
        f"{metrics.get('accuracy', 0):.2%}"
    )

    st.metric(
        "F1 Score",
        f"{metrics.get('f1_score', 0):.2%}"
    )

    if st.button(
        "🗑 Clear History",
        use_container_width=True,
    ):
        clear_prediction_state()
        st.success("History cleared")
        st.rerun()

st.title("Handwritten Digit Recognition Dashboard")

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Accuracy", f"{metrics.get('accuracy',0):.4f}")
c2.metric("Precision", f"{metrics.get('precision',0):.4f}")
c3.metric("Recall", f"{metrics.get('recall',0):.4f}")
c4.metric("F1", f"{metrics.get('f1_score',0):.4f}")
c5.metric("ROC-AUC", f"{metrics.get('roc_auc',0):.4f}")

prediction_tab, analytics_tab, metrics_tab, model_tab = st.tabs(
    ["Prediction","Analytics","Metrics","Model Info"]
)

with prediction_tab:
    mode = st.radio("Input Source", ["Upload Image", "Draw Digit"], horizontal=True)

    image_array = None
    preview = None

    if mode == "Upload Image":
        file = st.file_uploader("Upload", type=["png","jpg","jpeg"])
        if file:
            preview = Image.open(file).convert("RGB")
            image_array = np.array(preview)
    else:
        canvas = st_canvas(
            stroke_width=15,
            stroke_color="white",
            background_color="black",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="draw"
        )
        if (canvas.image_data is not None and np.any(canvas.image_data[:, :, :3] > 0)):
            image_array = canvas.image_data.astype(np.uint8)
            preview = Image.fromarray(image_array)

    predict_clicked = st.button(
    "🔍 Predict Digit",
    type="primary",
    use_container_width=True,
    )

    if image_array is not None and predict_clicked:
        try:
            processed, model_input = preprocess_digit(image_array)
            pred, conf, probs, latency = predict(model, model_input)

            st.session_state.last_probs = probs
            st.session_state.last_processed = processed
            st.session_state.last_input = model_input

            record = {
                    "timestamp": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                    ),
                    
                    "prediction": pred,
                    "confidence": round(conf, 4),
                    "latency_ms": round(latency, 2)
                }

            if (
                not st.session_state.history
                or st.session_state.history[-1] != record
                ):
                st.session_state.history.append(record)

            a,b = st.columns(2)
            a.image(preview, caption="Original")
            b.image(processed, caption="Processed")

            m1,m2,m3 = st.columns(3)
            m1.metric("Prediction", pred)
            m2.metric("Confidence", f"{conf:.2%}")
            m3.metric("Latency", f"{latency:.2f} ms")

            top = sorted(enumerate(probs), key=lambda x:x[1], reverse=True)[:5]
            st.subheader("Top Predictions")
            cols = st.columns(5)
            for col,(digit,score) in zip(cols,top):
                col.metric(str(digit), f"{score:.2%}")

        except Exception as e:
            st.error(str(e))

with analytics_tab:
    if(st.session_state.last_probs is None or len(st.session_state.history) == 0):
        st.info("Run a prediction first.")
    else:
        probs = st.session_state.last_probs

        df = pd.DataFrame({
            "Digit":[str(i) for i in range(10)],
            "Probability":probs
        })

        left,right = st.columns([2,1])

        with left:
            fig = px.bar(df,x="Digit",y="Probability",title="Probability Distribution")
            st.plotly_chart(fig,use_container_width=True)

        with right:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(np.max(probs))*100,
                title={"text":"Confidence"}
            ))
            st.plotly_chart(gauge,use_container_width=True)

        st.subheader("Prediction History")

        history_df = pd.DataFrame(st.session_state.history)

        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True)

            trend = px.line(
                history_df.reset_index(),
                x="index",
                y="confidence",
                markers=True,
                title="Confidence Trend"
            )
            st.plotly_chart(trend,use_container_width=True)

            st.download_button(
                "Download CSV",
                history_df.to_csv(index=False),
                "prediction_history.csv"
            )

            st.download_button(
                "Download JSON",
                history_df.to_json(orient="records", indent=2),
                "prediction_history.json"
            )

        st.subheader("Processed Image")
        st.image(st.session_state.last_processed, width=250)

        st.subheader("Batch Prediction")

        files = st.file_uploader(
            "Upload Multiple Images",
            type=["png","jpg","jpeg"],
            accept_multiple_files=True,
            key="batch"
        )

        if files:
            results = []

            for f in files:
                try:
                    img = np.array(Image.open(f).convert("RGB"))
                    _, inp = preprocess_digit(img)
                    p, c, _, _ = predict(model, inp)
                    results.append(
                        {"file":f.name,"prediction":p,"confidence":round(c,4)}
                    )
                except Exception:
                    results.append(
                        {"file":f.name,"prediction":"Error","confidence":0}
                    )

            batch_df = pd.DataFrame(results)
            st.dataframe(batch_df, use_container_width=True)

with metrics_tab:
    st.subheader("Evaluation Metrics")
    st.json(metrics)

    radar = go.Figure()
    radar.add_trace(go.Scatterpolar(
        r=[
            metrics.get("accuracy",0),
            metrics.get("precision",0),
            metrics.get("recall",0),
            metrics.get("f1_score",0),
        ],
        theta=["Accuracy","Precision","Recall","F1"],
        fill="toself"
    ))
    st.plotly_chart(radar, use_container_width=True)

    score = (
        metrics.get("accuracy",0)+
        metrics.get("precision",0)+
        metrics.get("recall",0)+
        metrics.get("f1_score",0)
    ) / 4

    st.metric("Health Score", f"{score:.2%}")

    cm = Path(CONFUSION_MATRIX_PATH)
    if cm.exists():
        st.image(str(cm), caption="Confusion Matrix")

with model_tab:
    st.metric("Parameters", f"{model.count_params():,}")

    summary = []
    model.summary(print_fn=summary.append)
    st.code("\n".join(summary))

    architecture = pd.DataFrame([
        ["Input","28x28x1"],
        ["Conv Block 1","32 Filters"],
        ["Conv Block 2","64 Filters"],
        ["Dense","128 Units"],
        ["Output","10 Classes"]
    ], columns=["Layer","Details"])

    st.dataframe(architecture, use_container_width=True)

    runtime = pd.DataFrame({
        "Property":["Python","TensorFlow","Platform"],
        "Value":[
            platform.python_version(),
            tf.__version__,
            platform.system()
        ]
    })

    st.dataframe(runtime, use_container_width=True)
