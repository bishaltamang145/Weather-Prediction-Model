import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# -----------------------------------------------------------------------------
# 1. Custom Transformer Definition (Fixes Unpickling Error)
# -----------------------------------------------------------------------------
class IQRCapper(BaseEstimator, TransformerMixin):
    """Custom transformer used during model training to cap outliers using IQR."""
    def __init__(self, factor=1.5, lower_bound_=None, upper_bound_=None):
        self.factor = factor
        self.lower_bound_ = lower_bound_
        self.upper_bound_ = upper_bound_

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        q1 = X_df.quantile(0.25)
        q3 = X_df.quantile(0.75)
        iqr = q3 - q1
        self.lower_bound_ = q1 - self.factor * iqr
        self.upper_bound_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X)
        if not hasattr(self, 'lower_bound_') or self.lower_bound_ is None:
            return X_df
        return X_df.clip(lower=self.lower_bound_, upper=self.upper_bound_, axis=1)


# -----------------------------------------------------------------------------
# 2. Page Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Seattle Weather Predictor",
    page_icon="🌦️",
    layout="centered"
)

WEATHER_EMOJIS = {
    "sun": "☀️",
    "rain": "🌧️",
    "fog": "🌫️",
    "drizzle": "🌦️",
    "snow": "❄️"
}


# -----------------------------------------------------------------------------
# 3. Model Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model_artifact():
    try:
        artifact = joblib.load("seattle_weather_model.joblib")
        return artifact
    except Exception as e:
        st.error(f"❌ Error loading model file: {e}")
        st.stop()

artifact = load_model_artifact()
model = artifact["model"]
features = artifact["features"]
classes = artifact.get("classes", ["drizzle", "fog", "rain", "snow", "sun"])


# -----------------------------------------------------------------------------
# 4. User Interface
# -----------------------------------------------------------------------------
st.title("🌦️ Seattle Weather Predictor")
st.markdown(
    "Enter daily meteorological readings and calendar dates to classify the weather outcome."
)
st.divider()

# Sidebar / Top info
st.sidebar.header("📌 App Info")
st.sidebar.write("This application uses a Machine Learning classification pipeline trained on historical Seattle weather data.")

# Input Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌡️ Weather Measurements")
    precipitation = st.number_input("🌧️ Precipitation (mm)", min_value=0.0, value=0.0, step=0.1)
    temp_max = st.number_input("🔥 Max Temperature (°C)", value=15.0, step=0.1)
    temp_min = st.number_input("❄️ Min Temperature (°C)", value=7.0, step=0.1)
    wind = st.number_input("💨 Wind Speed", min_value=0.0, value=3.0, step=0.1)

with col2:
    st.subheader("📅 Calendar Details")
    year = st.number_input("📆 Year", min_value=2012, max_value=2100, value=2026, step=1)
    month = st.slider("🗓️ Month", 1, 12, 6)
    day_of_year = st.slider("📆 Day of Year", 1, 366, 150)

# Feature engineering (cyclic month features)
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

# Construct input dataframe
input_dict = {
    "precipitation": precipitation,
    "temp_max": temp_max,
    "temp_min": temp_min,
    "wind": wind,
    "year": year,
    "month": month,
    "day_of_year": day_of_year,
    "month_sin": month_sin,
    "month_cos": month_cos
}

input_df = pd.DataFrame([input_dict])

if features is not None:
    input_df = input_df[features]

st.divider()

# -----------------------------------------------------------------------------
# 5. Prediction Logic
# -----------------------------------------------------------------------------
if st.button("🔮 Predict Weather Condition", type="primary", use_container_width=True):
    prediction = str(model.predict(input_df)[0]).lower()
    emoji = WEATHER_EMOJIS.get(prediction, "🌤️")

    st.success(f"### Predicted Weather: {emoji} **{prediction.upper()}**")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        
        # Determine active class labels
        model_classes = getattr(model, "classes_", classes)
        
        prob_df = pd.DataFrame({
            "Weather": [f"{WEATHER_EMOJIS.get(str(cls).lower(), '')} {str(cls).capitalize()}" for cls in model_classes],
            "Probability": probabilities
        }).sort_values("Probability", ascending=False)

        st.subheader("📊 Class Probabilities")
        st.dataframe(
            prob_df,
            column_config={
                "Probability": st.column_config.ProgressColumn(
                    "Probability",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                )
            },
            hide_index=True,
            use_container_width=True
        )