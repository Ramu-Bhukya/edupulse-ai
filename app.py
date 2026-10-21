import streamlit as st

from edupulse.presets import find_by_name, load_presets
from edupulse.prompts import TASKS, build_system_prompt, build_user_prompt

st.set_page_config(page_title="EduPulse AI", page_icon="📚", layout="wide")

personas, tones = load_presets()

if "generated" not in st.session_state:
    st.session_state.generated = None

with st.sidebar:
    st.header("Settings")
    persona_name = st.selectbox("AI persona", [p.name for p in personas])
    tone_name = st.selectbox("Tone", [t.name for t in tones])
    persona = find_by_name(personas, persona_name)
    tone = find_by_name(tones, tone_name)
    st.caption(persona.description)

st.title("EduPulse AI")
st.subheader("Prompt Architect")
st.write(
    "Paste some study material, pick a persona and a tone, and EduPulse builds "
    "a ready-to-use prompt for it."
)

left, right = st.columns(2)

with left:
    material = st.text_area(
        "Study material",
        height=300,
        placeholder="Paste a paragraph from your notes or textbook...",
    )
    task = st.selectbox("What should the AI do?", list(TASKS))
    if st.button("Generate", type="primary", key="generate"):
        if not material.strip():
            st.warning("Paste some study material first.")
        else:
            st.session_state.generated = {
                "task": task,
                "persona": persona.name,
                "tone": tone.name,
                "system": build_system_prompt(persona, tone),
                "user": build_user_prompt(task, material),
            }

with right:
    result = st.session_state.generated
    if result:
        st.markdown(f"**{result['task']}** · {result['persona']} · {result['tone']}")
        st.caption("System prompt")
        st.code(result["system"], language=None, wrap_lines=True)
        st.caption("User prompt")
        st.code(result["user"], language=None, wrap_lines=True)
    else:
        st.info("Your generated prompt will appear here.")
