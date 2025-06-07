import pandas as pd
import streamlit as st
from assets.lexicon import second_page

st.set_page_config(page_title="Информация о датасете", layout="wide")

st.title("Информация о наборе данных")

st.markdown(second_page["domain_description"])

DATA_PATH = "assets\\union-wines.csv"
df = pd.read_csv(DATA_PATH, sep=';')

features_dict = second_page["features"]
columns_info = {
    "Признак": [],
    "Тип данных": [],
    "Описание": []
}

for col in df.columns:
    columns_info["Признак"].append(col)
    columns_info["Тип данных"].append(df[col].dtype)
    columns_info["Описание"].append(features_dict.get(col, "Описание отсутствует"))

features_df = pd.DataFrame(columns_info)

st.markdown("### Описание признаков:")
st.dataframe(features_df, use_container_width=True)

st.markdown("### Предобработка данных:")
st.markdown(second_page["preprocessing_notes"])