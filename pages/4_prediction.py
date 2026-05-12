import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

from assets.lexicon import second_page, fourth_page

# =========================
# CONFIG
# =========================

MODEL_PATHS = {
    "Bagging": "assets/bagging.pkl",
    "CatBoost": "assets/catboost.pkl",
    "KNN": "assets/knn_model.pkl",
    "FCNN": "assets/FCNN.h5",
    "Gradient Boosting": "assets/gradientboosting.pkl",
    "Stacking": "assets/stacking.pkl"
}

# =========================
# LOAD MODELS
# =========================

@st.cache_resource
def load_pickle_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_keras_model(path):
    return tf.keras.models.load_model(path)

models = {}

for name, path in MODEL_PATHS.items():
    try:
        if name == "FCNN":
            models[name] = load_keras_model(path)
        else:
            models[name] = load_pickle_model(path)
    except Exception as e:
        st.warning(f"Не удалось загрузить {name}: {e}")

# =========================
# FEATURES
# =========================

features_info = {
    k: v for k, v in second_page["features"].items()
    if k != "color"
}

# =========================
# UI
# =========================

st.title(fourth_page["title"])

mode = st.radio("Режим ввода данных", ["CSV", "Manual"])

# =========================
# VALIDATION
# =========================

def validate_input_data(df, feature_names):
    errors = []

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        return [f"Отсутствуют признаки: {missing}"]

    df_check = df.copy()

    for col in feature_names:

        df_check[col] = pd.to_numeric(df_check[col], errors="coerce")

        if df_check[col].isna().any():
            errors.append(f"'{col}' содержит некорректные значения (NaN)")

        if (df_check[col] < 0).any():
            errors.append(f"'{col}' содержит отрицательные значения")

    return errors

# =========================
# INPUT PIPELINE
# =========================

input_data = None

# ---------- CSV MODE ----------
if mode == "CSV":

    uploaded_file = st.file_uploader(
        "Загрузите CSV файл",
        type=["csv"]
    )

    if uploaded_file:

        try:
            df = pd.read_csv(uploaded_file, sep=";")

            errors = validate_input_data(df, features_info.keys())

            if errors:
                st.error("❌ Ошибки в данных:")
                for e in errors:
                    st.error(e)
                st.stop()

            input_data = df[list(features_info.keys())]
            st.success("CSV успешно загружен")

        except Exception as e:
            st.error(f"Ошибка чтения файла: {e}")
            st.stop()

# ---------- MANUAL MODE ----------
else:

    st.subheader("Ручной ввод")

    manual_inputs = {}

    for feat, descr in features_info.items():

        if feat == "quality":
            val = st.slider(feat, 0, 10, 5)
        else:
            val = st.number_input(
                feat,
                min_value=0.0,
                step=0.01,
                format="%.4f"
            )

        manual_inputs[feat] = val

    input_data = pd.DataFrame([manual_inputs])

    errors = validate_input_data(input_data, features_info.keys())

    if errors:
        st.error("❌ Ошибки во вводе:")
        for e in errors:
            st.error(e)
        st.stop()

# =========================
# MODEL SELECTION
# =========================

selected_models = st.multiselect(
    "Выберите модели",
    list(models.keys())
)

# =========================
# KERAS PREDICT
# =========================

def predict_with_keras(model, df):
    x = df.values.astype(np.float32)
    output = model.predict(x, verbose=0)

    if output.shape[1] > 1:
        return np.argmax(output, axis=1)

    return (output.flatten() > 0.5).astype(int)

# =========================
# PREDICT
# =========================

if st.button("Получить предсказание"):

    if not selected_models:
        st.warning("Выберите хотя бы одну модель")
        st.stop()

    if input_data is None:
        st.error("Нет входных данных")
        st.stop()

    for model_name in selected_models:

        model = models.get(model_name)

        if model is None:
            st.error(f"Модель {model_name} не загружена")
            continue

        try:

            if model_name == "FCNN":
                preds = predict_with_keras(model, input_data)
            else:
                preds = model.predict(input_data)

            preds = np.array(preds).flatten()

            st.markdown(f"### {model_name}")

            for i, p in enumerate(preds):

                if p in [0, 1]:
                    color = "white" if p == 0 else "red"
                else:
                    color = str(p)

                st.success(f"{fourth_page['target']} #{i+1}: {color}")

        except Exception as e:
            st.error(f"Ошибка в {model_name}: {e}")