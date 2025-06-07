import streamlit as st
from PIL import Image
from assets.lexicon import first_page

st.set_page_config(page_title=first_page['page_title'], page_icon=first_page['page_icon'])

st.title(first_page['title'])

image = Image.open("assets/photo.png")

col1, col2 = st.columns([1, 2])
with col1:
    st.image(image, caption="Разработчик", width=200)
with col2:
    st.subheader(first_page['FIO'])
    st.text(f"Группа: {first_page['GROUP']}")
    st.markdown(f"**Тема РГР:** _{first_page['TOPIC']}_")

st.markdown("---")
st.markdown(first_page['markdown'])

#streamlit run pages/1_author.py
