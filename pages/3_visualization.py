import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from assets.lexicon import third_page

st.set_page_config(page_title="Визуализации зависимостей", layout="wide")

st.title("Визуализации зависимостей в наборе данных")
if 'description' in third_page:
    st.markdown(third_page['description'])

@st.cache_data
def load_data():
    DATA_PATH = "assets/union-wines.csv"
    df = pd.read_csv(DATA_PATH, sep=';')
    return df

df = load_data()

def plot_boxplot(df):
    sns.set(style='whitegrid', palette='Set2')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='color', y='volatile acidity', ax=ax)
    ax.set_title('Летучая кислотность vs Цвет вина')
    ax.set_xlabel('Цвет вина')
    ax.set_ylabel('Летучая кислотность (г/дм³)')
    plt.tight_layout()
    return fig

def plot_histogram(df):
    sns.set(style='whitegrid', palette='Set2')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x='alcohol', hue='color', kde=True, element='step',
                 stat='density', common_norm=False, ax=ax)
    ax.set_title('Распределение алкоголя по цвету вина')
    ax.set_xlabel('Содержание алкоголя (% об.)')
    ax.set_ylabel('Плотность распределения')
    plt.tight_layout()
    return fig

def plot_scatter(df):
    sns.set(style='whitegrid', palette='Set2')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x='residual sugar', y='density', hue='color',
                    alpha=0.6, ax=ax)
    ax.set_title('Остаточный сахар и Плотность по цвету вина')
    ax.set_xlabel('Остаточный сахар (г/дм³)')
    ax.set_ylabel('Плотность (г/см³)')
    plt.tight_layout()
    return fig

def plot_violin(df):
    sns.set(style='whitegrid', palette='Set2')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=df, x='color', y='citric acid', inner='quartile', ax=ax)
    ax.set_title('Лимонная кислота по цвету вина')
    ax.set_xlabel('Цвет вина')
    ax.set_ylabel('Лимонная кислота (г/дм³)')
    plt.tight_layout()
    return fig

col1, col2 = st.columns(2)

with col1:
    st.pyplot(plot_boxplot(df))
    st.pyplot(plot_scatter(df))

with col2:
    st.pyplot(plot_histogram(df))
    st.pyplot(plot_violin(df))
