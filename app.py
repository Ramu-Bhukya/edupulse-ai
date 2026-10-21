import json
from datetime import datetime

import streamlit as st

from edupulse.presets import find_by_name, load_presets
from edupulse.prompts import TASKS, build_system_prompt, build_user_prompt

MAX_MATERIAL_CHARS = 8000
MAX_HISTORY = 10

st.set_page_config(page_title="EduPulse AI", page_icon="📚", layout="wide")

personas, tones = load_presets()

if "generated" not in st.session_state:
    st.session_state.generated = None
if "history" not in st.session_state:
    st.session_state.history = []


def prompt_as_text(result: dict) -> str:
    return f"# System\n{result['system']}\n\n# User\n{result['user']}\n"


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
        max_chars=MAX_MATERIAL_CHARS,
        placeholder="Paste a paragraph from your notes or textbook...",
    )
    task = st.selectbox("What should the AI do?", list(TASKS))
    if st.button("Generate", type="primary", key="generate"):
        if len(material.strip()) < 20:
            st.warning("Paste a bit more study material first (at least a sentence or two).")
        else:
            result = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "task": task,
                "persona": persona.name,
                "tone": tone.name,
                "system": build_system_prompt(persona, tone),
                "user": build_user_prompt(task, material),
            }
            st.session_state.generated = result
            st.session_state.history.insert(0, result)
            del st.session_state.history[MAX_HISTORY:]

with right:
    result = st.session_state.generated
    if result:
        st.markdown(f"**{result['task']}** · {result['persona']} · {result['tone']}")
        st.caption("System prompt")
        st.code(result["system"], language=None, wrap_lines=True)
        st.caption("User prompt")
        st.code(result["user"], language=None, wrap_lines=True)

        dl_txt, dl_json = st.columns(2)
        dl_txt.download_button(
            "Download .txt",
            data=prompt_as_text(result),
            file_name="edupulse_prompt.txt",
            mime="text/plain",
        )
        dl_json.download_button(
            "Download .json",
            data=json.dumps(result, indent=2),
            file_name="edupulse_prompt.json",
            mime="application/json",
        )
    else:
        st.info("Your generated prompt will appear here.")

if st.session_state.history:
    with st.expander(f"Prompt history ({len(st.session_state.history)})"):
        for i, item in enumerate(st.session_state.history):
            col_text, col_btn = st.columns([4, 1])
            col_text.write(
                f"`{item['created_at']}` · **{item['task']}** · {item['persona']} · {item['tone']}"
            )
            if col_btn.button("Show", key=f"show_{i}"):
                st.session_state.generated = item
                st.rerun()
