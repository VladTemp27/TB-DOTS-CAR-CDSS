import streamlit as st

st.title("Model")
st.write("This is where the model will be displayed.")

tab1, tab2 = st.tabs(["Temporal"," Non Temporal"])

with tab1:
    st.write("This is where the temporal model will be displayed.")
with tab2:
    st.write("This is where the non temporal model will be displayed.")
