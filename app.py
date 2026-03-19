"""Spotify Diffusion Predictor — Entry point."""

import streamlit as st

st.set_page_config(
    page_title="Spotify Diffusion Predictor",
    page_icon="🎵",
    layout="wide",
)

# Sidebar
st.sidebar.title("Spotify Diffusion Predictor")
st.sidebar.markdown(
    "Predict which countries a song will spread to on Spotify's Top-200 charts, "
    "and how quickly it will get there."
)
st.sidebar.divider()

mode = st.sidebar.radio(
    "Mode",
    ["Demo", "Production"],
    captions=[
        "Browse pre-computed predictions from the 2021 test set",
        "Input a custom song and get top-5 country predictions",
    ],
)

# Route to the selected page
if mode == "Demo":
    from views.demo import render
    render()
else:
    from views.production import render
    render()
