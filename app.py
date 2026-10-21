import json
from datetime import datetime

import streamlit as st

from edupulse.config import MissingAPIKeyError
from edupulse.llm import complete, get_client
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


def result_as_text(result: dict) -> str:
    return (
        f"# System\n{result['system']}\n\n"
        f"# User\n{result['user']}\n\n"
        f"# Response\n{result['response']}\n"
    )


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
    "Paste some study material, pick a persona and a tone, and let the AI "
    "explain, summarize or quiz you on it."
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
            system_prompt = build_system_prompt(persona, tone)
            user_prompt = build_user_prompt(task, material)
            try:
                client = get_client()
                with st.spinner(f"{persona.name} is thinking..."):
                    response = complete(
                        client,
                        [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
            except MissingAPIKeyError as exc:
                st.error(str(exc))
            else:
                result = {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "task": task,
                    "persona": persona.name,
                    "tone": tone.name,
                    "system": system_prompt,
                    "user": user_prompt,
                    "response": response,
                }
                st.session_state.generated = result
                st.session_state.history.insert(0, result)
                del st.session_state.history[MAX_HISTORY:]

with right:
    result = st.session_state.generated
    if result:
        st.markdown(f"**{result['task']}** · {result['persona']} · {result['tone']}")
        with st.container(border=True):
            st.markdown(result["response"])
        with st.expander("See the prompt that was sent"):
            st.caption("System prompt")
            st.code(result["system"], language=None, wrap_lines=True)
            st.caption("User prompt")
            st.code(result["user"], language=None, wrap_lines=True)

        dl_txt, dl_json = st.columns(2)
        dl_txt.download_button(
            "Download .txt",
            data=result_as_text(result),
            file_name="edupulse_response.txt",
            mime="text/plain",
        )
        dl_json.download_button(
            "Download .json",
            data=json.dumps(result, indent=2),
            file_name="edupulse_response.json",
            mime="application/json",
        )
    else:
        st.info("The AI's answer will appear here.")

if st.session_state.history:
    with st.expander(f"History ({len(st.session_state.history)})"):
        for i, item in enumerate(st.session_state.history):
            col_text, col_btn = st.columns([4, 1])
            col_text.write(
                f"`{item['created_at']}` · **{item['task']}** · {item['persona']} · {item['tone']}"
            )
            if col_btn.button("Show", key=f"show_{i}"):
                st.session_state.generated = item
                st.rerun()
