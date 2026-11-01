from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("models/best_model.pkl")
CONFIG_PATH = Path("models/feature_config.json")

st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Wellness Tourism Package Prediction")
st.write(
    "Predict whether a customer is likely to purchase the Wellness Tourism Package "
    "before the sales team contacts them."
)

if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
    st.error(
        "Deployment files are missing. Ensure models/best_model.pkl and "
        "models/feature_config.json are committed by the training pipeline."
    )
    st.stop()

model = joblib.load(MODEL_PATH)
config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

st.subheader("Customer Inputs")

values = {}
with st.form("prediction_form"):
    cols = st.columns(3)

    for i, (feature, meta) in enumerate(config.items()):
        container = cols[i % 3]
        with container:
            if meta["type"] == "categorical":
                options = meta["values"]
                values[feature] = st.selectbox(feature, options)
            else:
                low = meta["min"]
                high = meta["max"]
                default = meta["median"]

                if meta.get("is_integer", False):
                    low_i = int(low)
                    high_i = int(high)
                    default_i = int(round(default))
                    values[feature] = st.number_input(
                        feature,
                        min_value=low_i,
                        max_value=high_i,
                        value=max(low_i, min(high_i, default_i)),
                        step=1,
                    )
                else:
                    values[feature] = st.number_input(
                        feature,
                        min_value=float(low),
                        max_value=float(high),
                        value=float(default),
                    )

    submitted = st.form_submit_button(
        "Predict Purchase Propensity",
        use_container_width=True,
    )

if submitted:
    input_df = pd.DataFrame([values])

    st.subheader("Input DataFrame")
    st.dataframe(input_df, use_container_width=True)

    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])

    st.subheader("Prediction Result")
    st.metric("Probability of Purchase", f"{probability:.1%}")

    if prediction == 1:
        st.success(
            "Prediction: Customer is likely to purchase the package. "
            "Prioritize this lead for outreach."
        )
    else:
        st.info(
            "Prediction: Customer is less likely to purchase the package. "
            "Consider lower-cost nurturing before direct sales follow-up."
        )

    st.caption(
        "The prediction is a decision-support tool and should not be the sole "
        "basis for customer treatment."
    )
