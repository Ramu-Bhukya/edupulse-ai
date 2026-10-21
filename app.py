import streamlit as st

st.set_page_config(page_title="EduPulse AI", page_icon="📚")

st.title("EduPulse AI")
st.write("Your personal AI study companion.")

subject = st.text_input("What are you studying today?")

if st.button("Start"):
    if subject.strip():
        st.success(f"Great, let's get started with {subject.strip()}!")
    else:
        st.warning("Type a subject first.")
