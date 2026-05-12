import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

from assets.lexicon import second_page, fourth_page

# =========================
# МОДЕЛИ
# =========================

MODEL_PATHS = {
    "Bagging": "assets/bagging.pkl",
    "CatBoost": "assets/catboost.pkl",
    "KNN": "assets/knn_model.pkl",
    "FCNN": "assets/FCNN.h5",
    "Gradient Boosting": "assets/gradientboosting.pkl",
    "Stacking": "assets/stacking.pkl"
}

@st.cache_data
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
        st.warning(f"Не удалось загрузить модель {name}: {e}")

# =========================
# ПРИЗНАКИ
# =========================

features_info = {
    k: v for k, v in second_page["features"].items()
    if k != "color"
}

# =========================
# РЕАЛЬНЫЕ ДИАПАЗОНЫ (ВАЖНО)
# =========================

FEATURE_RANGES = {
    "fixed acidity": (4, 16),
    "volatile acidity": (0, 1.5),
    "citric acid": (0, 1),
    "residual sugar": (0, 65),
    "chlorides": (0, 0.6),
    "free sulfur dioxide": (0, 150),
    "total sulfur dioxide": (0, 300),
    "density": (0.98, 1.01),
    "pH": (2.8, 3.8),
    "sulphates": (0, 2),
    "alcohol": (8, 15),
    "quality": (0, 10)
}

# =========================
# UI
# =========================

st.title(fourth_page["title"])

st.markdown("Введите параметры вина или загрузите CSV-файл")

# =========================
# ВАЛИДАЦИЯ
# =========================

def validate_input_data(df, feature_names):
    errors = []

    # missing columns
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        errors.append(f"❌ Отсутствуют признаки: {missing}")
        return errors

    # numeric conversion
    for col in feature_names:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # NaN check
    if df[feature_names].isnull().values.any():
        errors.append("❌ Обнаружены NaN или некорректные значения")

    # negative values
    negative_cols = [col for col in feature_names if (df[col] < 0).any()]
    if negative_cols:
        errors.append(f"❌ Отрицательные значения: {negative_cols}")

    # range check
    out_of_range = []
    for col in feature_names:
        if col in FEATURE_RANGES:
            min_v, max_v = FEATURE_RANGES[col]
            if ((df[col] < min_v) | (df[col] > max_v)).any():
                out_of_range.append(col)

    if out_of_range:
        errors.append(f"❌ Значения вне диапазона: {out_of_range}")

    return errors

# =========================
# CSV
# =========================

uploaded_file = st.file_uploader("CSV файл", type=["csv"])

input_data = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=";")

        errors = validate_input_data(df, list(features_info.keys()))

        if errors:
            for e in errors:
                st.error(e)
        else:
            input_data = df[list(features_info.keys())]
            st.success("CSV загружен успешно")

    except Exception as e:
        st.error(f"Ошибка загрузки файла: {e}")

# =========================
# РУЧНОЙ ВВОД
# =========================

if input_data is None:

    st.subheader("Ручной ввод")

    manual_inputs = {}

    for feat, descr in features_info.items():

        min_v, max_v = FEATURE_RANGES.get(feat, (0.0, 100.0))

        if feat == "quality":
            value = st.slider(
                f"{feat} ({descr})",
                min_value=int(min_v),
                max_value=int(max_v),
                value=5
            )
        else:
            value = st.number_input(
                f"{feat} ({descr})",
                min_value=float(min_v),
                max_value=float(max_v),
                value=float(min_v),
                step=0.01,
                format="%.4f"
            )

        manual_inputs[feat] = value

    input_data = pd.DataFrame([manual_inputs])

# =========================
# МОДЕЛИ
# =========================

selected_models = st.multiselect(
    "Выберите модели",
    list(models.keys())
)

# =========================
# FCNN
# =========================

def predict_with_keras(model, df):
    x = df.values.astype(np.float32)
    output = model.predict(x, verbose=0)

    if output.shape[1] > 1:
        return np.argmax(output, axis=1)
    return (output.flatten() > 0.5).astype(int)

# =========================
# КНОПКА
# =========================

if st.button("Получить предсказание"):

    if not selected_models:
        st.warning("Выберите хотя бы одну модель")
        st.stop()

    # финальная валидация
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

            # обработка
            if isinstance(preds, np.ndarray) and preds.dtype.kind in "ifu":

                if set(np.unique(preds)).issubset({0, 1}):
                    mapping = {0: "white", 1: "red"}
                    results = [mapping.get(int(p), "unknown") for p in preds]
                else:
                    results = preds.astype(str)

            else:
                results = preds

            st.markdown(f"### {model_name}")

            for i, r in enumerate(results):
                st.success(f"Объект {i+1}: {fourth_page['target']} — {r}")

        except Exception as e:
            st.error(f"{model_name}: {e}")