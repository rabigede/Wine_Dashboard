import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from pathlib import Path
from assets.lexicon import second_page, fourth_page

# =========================
# Пути к моделям (FIXED)
# =========================

BASE_DIR = Path("assets")

MODEL_PATHS = {
    "Bagging": BASE_DIR / "bagging.pkl",
    "CatBoost": BASE_DIR / "catboost.pkl",
    "KNN": BASE_DIR / "knn_model.pkl",
    "FCNN": BASE_DIR / "FCNN.h5",
    "Gradient Boosting": BASE_DIR / "gradientboosting.pkl",
    "Stacking": BASE_DIR / "stacking.pkl"
}

# =========================
# Загрузка моделей
# =========================

@st.cache_resource
def load_pickle_model(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_keras_model(path: Path):
    return tf.keras.models.load_model(path)

models = {}

for name, path in MODEL_PATHS.items():
    try:
        if name == "FCNN":
            models[name] = load_keras_model(path)
        else:
            models[name] = load_pickle_model(path)

    except Exception as e:
        st.warning(f"Не удалось загрузить модель {name}: {e}")

# =========================
# Features
# =========================

features_info = {
    k: v for k, v in second_page["features"].items()
    if k != "color"
}

# =========================
# Title
# =========================

st.title(fourth_page["title"])

st.markdown("""
Введите параметры вина вручную  
или загрузите CSV-файл.
""")

# =========================
# Validation
# =========================

def validate_input_data(df, feature_names):
    errors = []

    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        return [f"Отсутствуют признаки: {missing}"]

    for col in feature_names:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            errors.append(f"Признак '{col}' содержит некорректные значения")

    if df[feature_names].isnull().values.any():
        errors.append("Обнаружены NaN значения")

    negative_cols = [c for c in feature_names if (df[c] < 0).any()]
    if negative_cols:
        errors.append(f"Отрицательные значения недопустимы: {negative_cols}")

    return errors

# =========================
# Upload CSV (FIXED PATH SAFE)
# =========================

uploaded_file = st.file_uploader(
    "Загрузите CSV-файл с признаками",
    type=["csv"]
)

input_data = None

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file, sep=";")

        validation_errors = validate_input_data(
            df,
            list(features_info.keys())
        )

        if validation_errors:
            for err in validation_errors:
                st.error(err)
        else:
            input_data = df[list(features_info.keys())]
            st.success("Файл успешно загружен")

    except Exception as e:
        st.error(f"Ошибка загрузки файла: {e}")

# =========================
# Manual input
# =========================

if input_data is None:

    st.subheader("Ручной ввод параметров")

    manual_inputs = {}

    for feat, descr in features_info.items():

        if feat == "quality":
            value = st.slider(
                f"{feat} ({descr})",
                0, 10, 5
            )
        else:
            value = st.number_input(
                f"{feat} ({descr})",
                min_value=0.0,
                value=0.0,
                step=0.1,
                format="%.4f"
            )

        manual_inputs[feat] = value

    input_data = pd.DataFrame([manual_inputs])

# =========================
# Model selection
# =========================

selected_models = st.multiselect(
    "Выберите модели для предсказания",
    list(models.keys())
)

# =========================
# Keras prediction
# =========================

def predict_with_keras(model, df):
    x = df.values.astype(np.float32)
    output = model.predict(x, verbose=0)

    if output.shape[1] > 1:
        return np.argmax(output, axis=1)
    return (output.flatten() > 0.5).astype(int)

# =========================
# Predict button
# =========================

if st.button("Получить предсказание"):

    if not selected_models:
        st.warning("Выберите хотя бы одну модель")
        st.stop()

    errors = validate_input_data(input_data, list(features_info.keys()))

    if errors:
        for e in errors:
            st.error(e)
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

            # normalize output
            preds = np.array(preds).flatten()

            color_map = {0: "white", 1: "red"}

            if set(np.unique(preds)).issubset({0, 1}):
                results = [color_map[int(p)] for p in preds]
            else:
                results = preds.astype(str)

            st.markdown(f"### Результаты модели **{model_name}**")

            for i, r in enumerate(results):
                st.success(f"Объект {i+1}: {fourth_page['target']} — **{r}**")

        except Exception as e:
            st.error(f"Ошибка в модели {model_name}: {e}")