import glob
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Readme",
    page_icon="📖",
)


st.sidebar.success("Please select a page from above.")

    

with open("README.md", "r") as file:
    st.markdown(file.read())



