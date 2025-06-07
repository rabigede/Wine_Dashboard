import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from assets.lexicon import second_page, fourth_page

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


features_info = {
    k: v for k, v in second_page["features"].items() if k != "color"
}

st.title(fourth_page['title'])

uploaded_file = st.file_uploader("Загрузите CSV-файл с признаками", type=["csv"])
input_data = None

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep=';')
        missing = [feat for feat in features_info if feat not in df.columns]
        if missing:
            st.error(f"В файле отсутствуют признаки: {missing}")
        else:
            input_data = df[list(features_info.keys())]
    except Exception as e:
        st.error(f"Ошибка загрузки файла: {e}")

if input_data is None:
    st.markdown("Или введите данные вручную:")
    manual_inputs = {}
    for feat, descr in features_info.items():
        if feat == "quality":
            val = st.slider(f"{feat} ({descr})", 0, 10, 5)
        else:
            val = st.number_input(f"{feat} ({descr})", format="%.4f")
        manual_inputs[feat] = val
    input_data = pd.DataFrame([manual_inputs])


model_name = st.selectbox("Выберите модель для предсказания", list(models.keys()))


def predict_with_keras(model, df):
    x = df.values.astype(np.float32)
    output = model.predict(x)
    if output.shape[1] > 1:
        preds = np.argmax(output, axis=1)
    else:
        preds = (output.flatten() > 0.5).astype(int)
    return preds


if st.button("Получить предсказание"):
    if model_name not in models:
        st.error("Модель не загружена")
    else:
        model = models[model_name]
        try:
            if model_name == "FCNN":
                preds = predict_with_keras(model, input_data)
            else:
                preds = model.predict(input_data)

            if isinstance(preds, np.ndarray) and preds.dtype.kind in 'ifu':
                if set(np.unique(preds)).issubset({0, 1}):
                    color_map = {0: "white", 1: "red"}
                    pred_colors = [color_map.get(p, "Неизвестно") for p in preds]
                else:
                    pred_colors = preds.astype(str)
            else:
                pred_colors = preds

            for i, c in enumerate(pred_colors):
                st.success(f"Объект {i + 1}: {fourth_page['target']} — **{c}**")
        except Exception as e:
            st.error(f"Ошибка при предсказании: {e}")
