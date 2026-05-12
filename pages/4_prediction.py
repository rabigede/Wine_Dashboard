import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from assets.lexicon import second_page, fourth_page

# =========================
# Пути к моделям
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
# Загрузка моделей
# =========================

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
# Информация о признаках
# =========================

features_info = {
    k: v for k, v in second_page["features"].items()
    if k != "color"
}

# =========================
# Заголовок
# =========================

st.title(fourth_page["title"])

st.markdown("""
Введите параметры вина вручную
или загрузите CSV-файл.
""")

# =========================
# Валидация данных
# =========================

def validate_input_data(df, feature_names):
    errors = []

    # Проверка отсутствующих колонок
    missing = [feat for feat in feature_names if feat not in df.columns]

    if missing:
        errors.append(
            f"Отсутствуют признаки: {missing}"
        )
        return errors

    # Проверка и преобразование типов
    for col in feature_names:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            errors.append(
                f"Признак '{col}' содержит некорректные значения"
            )

    # Проверка NaN
    if df[feature_names].isnull().values.any():
        errors.append(
            "Обнаружены пропущенные значения (NaN)"
        )

    # Проверка отрицательных значений
    negative_columns = []

    for col in feature_names:
        if (df[col] < 0).any():
            negative_columns.append(col)

    if negative_columns:
        errors.append(
            f"Отрицательные значения недопустимы: {negative_columns}"
        )

    return errors

# =========================
# Загрузка CSV
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
# Ручной ввод
# =========================

if input_data is None:

    st.subheader("Ручной ввод параметров")

    manual_inputs = {}

    for feat, descr in features_info.items():

        if feat == "quality":

            value = st.slider(
                f"{feat} ({descr})",
                min_value=0,
                max_value=10,
                value=5
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
# Выбор моделей
# =========================

selected_models = st.multiselect(
    "Выберите модели для предсказания",
    list(models.keys())
)

# =========================
# Предсказание Keras
# =========================

def predict_with_keras(model, df):

    x = df.values.astype(np.float32)

    output = model.predict(x, verbose=0)

    if output.shape[1] > 1:
        preds = np.argmax(output, axis=1)
    else:
        preds = (output.flatten() > 0.5).astype(int)

    return preds

# =========================
# Кнопка предсказания
# =========================

if st.button("Получить предсказание"):

    if not selected_models:

        st.warning(
            "Пожалуйста, выберите хотя бы одну модель."
        )

    else:

        validation_errors = validate_input_data(
            input_data,
            list(features_info.keys())
        )

        if validation_errors:

            for err in validation_errors:
                st.error(err)

        else:

            for model_name in selected_models:

                if model_name not in models:
                    st.error(
                        f"Модель {model_name} не загружена"
                    )
                    continue

                model = models[model_name]

                try:

                    # Предсказание
                    if model_name == "FCNN":
                        preds = predict_with_keras(
                            model,
                            input_data
                        )
                    else:
                        preds = model.predict(input_data)

                    # Обработка результата
                    if (
                        isinstance(preds, np.ndarray)
                        and preds.dtype.kind in "ifu"
                    ):

                        if set(np.unique(preds)).issubset({0, 1}):

                            color_map = {
                                0: "white",
                                1: "red"
                            }

                            pred_colors = [
                                color_map.get(p, "unknown")
                                for p in preds
                            ]

                        else:
                            pred_colors = preds.astype(str)

                    else:
                        pred_colors = preds

                    # Вывод результата
                    st.markdown(
                        f"### Результаты модели "
                        f"**{model_name}**"
                    )

                    for i, color in enumerate(pred_colors):

                        st.success(
                            f"Объект {i + 1}: "
                            f"{fourth_page['target']} — "
                            f"**{color}**"
                        )

                except Exception as e:

                    st.error(
                        f"Ошибка в модели "
                        f"{model_name}: {e}"
                    )
