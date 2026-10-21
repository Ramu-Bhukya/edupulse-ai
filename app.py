import html
import json
import random
from datetime import datetime

import streamlit as st

from edupulse.config import MissingAPIKeyError, get_api_key, get_settings, mask_key
from edupulse.embeddings import OpenAIEmbedder
from edupulse.history import export_json, export_markdown
from edupulse.ingest import index_pdf
from edupulse.llm import LLM_ERRORS, complete, friendly_error, get_client, stream_chat
from edupulse.pdf_parser import PDFParseError
from edupulse.presets import find_by_name, load_presets
from edupulse.prompts import (
    TASKS,
    build_chat_messages,
    build_system_prompt,
    build_user_prompt,
)
from edupulse.rag_engine import answer_stream, is_idk
from edupulse.study_tools import (
    deck_to_csv,
    generate_flashcards,
    generate_quiz,
    grade_quiz,
    sample_context,
    summarize_takeaways,
)
from edupulse.text_utils import escape_markdown, highlight_terms
from edupulse.vector_store import VectorStore, document_id

MAX_MATERIAL_CHARS = 8000
MAX_HISTORY = 10
MAX_CONTEXT_MESSAGES = 10
AVATARS = {"user": "🧑‍🎓", "assistant": "🎓"}

CARD_CSS = """
<style>
.flashcard {
    min-height: 220px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2rem;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
    margin-bottom: 1rem;
}
.flashcard .side { font-size: 0.8rem; letter-spacing: 0.1em; text-transform: uppercase; }
.flashcard .content { font-size: 1.5rem; margin-top: 0.75rem; }
.flashcard.front { background: #eef2ff; color: #1e1b4b; }
.flashcard.back { background: #ecfdf5; color: #064e3b; }
.flashcard.back .content { font-size: 1.15rem; }
</style>
"""

st.set_page_config(page_title="EduPulse AI", page_icon="📚", layout="wide")

personas, tones = load_presets()

if "generated" not in st.session_state:
    st.session_state.generated = None
if "history" not in st.session_state:
    st.session_state.history = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "takeaways" not in st.session_state:
    st.session_state.takeaways = None
if "document" not in st.session_state:
    st.session_state.document = None
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "search_query" not in st.session_state:
    st.session_state.search_query = ""
if "quiz" not in st.session_state:
    st.session_state.quiz = None
    st.session_state.quiz_answers = []
    st.session_state.quiz_submitted = False
    st.session_state.quiz_round = 0  # bumped for every new quiz so radio keys reset
if "deck" not in st.session_state:
    st.session_state.deck = None
    st.session_state.card_order = []
    st.session_state.card_pos = 0
    st.session_state.card_flipped = False


@st.cache_resource
def get_store(path: str) -> VectorStore:
    return VectorStore.persistent(path)


store = get_store(get_settings().chroma_dir)


def set_document(info) -> None:
    """Switch the active document and forget results that belonged to the old one."""
    current = st.session_state.document
    if current is not None and info is not None and current.doc_id == info.doc_id:
        return
    st.session_state.document = info
    st.session_state.search_results = None
    st.session_state.quiz = None
    st.session_state.deck = None


def result_as_text(result: dict) -> str:
    return (
        f"# System\n{result['system']}\n\n"
        f"# User\n{result['user']}\n\n"
        f"# Response\n{result['response']}\n"
    )


with st.sidebar:
    st.header("Your tutor")
    persona_name = st.selectbox("Teaching style", [p.name for p in personas])
    tone_name = st.selectbox("Tone", [t.name for t in tones])
    persona = find_by_name(personas, persona_name)
    tone = find_by_name(tones, tone_name)
    st.caption(persona.description)

    try:
        env_key = get_api_key()
    except MissingAPIKeyError:
        env_key = None
    with st.expander("OpenAI API key", expanded=env_key is None):
        user_key = st.text_input(
            "Use your own key",
            type="password",
            help="Only kept for this browser session. It is never saved.",
        )
        if user_key.strip():
            st.caption(f"Using your key {mask_key(user_key.strip())}")
        elif env_key:
            st.caption(f"Using the app's key {mask_key(env_key)}")
        else:
            st.caption("No key configured yet. Paste one above to start chatting.")

    st.divider()
    st.subheader("Your document")
    uploaded = st.file_uploader("Upload a PDF textbook or syllabus", type=["pdf"])
    if uploaded is not None:
        data = uploaded.getvalue()
        current = st.session_state.document
        if current is None or current.doc_id != document_id(data):
            try:
                with st.spinner("Processing document..."):
                    embedder = OpenAIEmbedder(get_client(user_key))
                    info, cached = index_pdf(data, uploaded.name, store, embedder)
                set_document(info)
                if cached:
                    st.toast(f"{info.filename} was indexed before, loaded it instantly.")
            except PDFParseError as exc:
                st.error(str(exc))
            except LLM_ERRORS as exc:
                st.error(friendly_error(exc))
    else:
        # Documents indexed in earlier sessions are still in Chroma, so they
        # can be reopened without uploading (or paying for embeddings) again.
        library = {d.doc_id: d for d in store.list_documents()}
        choice = None
        if library:
            choice = st.selectbox(
                "...or open one you indexed before",
                [None, *library],
                format_func=lambda i: "Choose a document" if i is None else library[i].filename,
                key="library_choice",
            )
        set_document(library.get(choice))
        if st.session_state.document and st.button("Remove from library", key="remove_doc"):
            store.delete_document(st.session_state.document.doc_id)
            set_document(None)
            st.rerun()

    if st.session_state.document:
        doc = st.session_state.document
        st.success(f"**{doc.filename}**  \n{doc.page_count} pages · {doc.chunk_count} chunks")

    st.divider()
    st.subheader("This session")
    st.metric("Messages", len(st.session_state.messages))
    if st.session_state.messages:
        st.download_button(
            "Export chat (.json)",
            data=export_json(st.session_state.messages, persona=persona.name, tone=tone.name),
            file_name="edupulse_chat.json",
            mime="application/json",
            width="stretch",
        )
        st.download_button(
            "Export chat (.md)",
            data=export_markdown(st.session_state.messages, persona=persona.name, tone=tone.name),
            file_name="edupulse_chat.md",
            mime="text/markdown",
            width="stretch",
        )

st.title("EduPulse AI")

chat_tab, quiz_tab, cards_tab, search_tab, architect_tab = st.tabs(
    ["💬 Chat with Notes", "📝 Quiz Generator", "🃏 Flashcards", "🔎 Search", "🛠️ Prompt Architect"]
)

# --- Chat with Notes ----------------------------------------------------------
with chat_tab:
    active_doc = st.session_state.document
    caption_col, takeaways_col, clear_col = st.columns([3, 1, 1])
    if active_doc:
        caption_col.caption(
            f"Answering from **{active_doc.filename}** as the **{persona.name}** "
            f"({tone.name.lower()} tone)."
        )
    else:
        caption_col.caption(
            f"Learning with the **{persona.name}** in a **{tone.name.lower()}** tone. "
            "Upload a PDF in the sidebar to chat with your own notes."
        )
    if takeaways_col.button("Key takeaways", key="get_takeaways"):
        if not st.session_state.messages:
            st.info("Ask a few questions first, then I can summarize them.")
        else:
            try:
                with st.spinner("Summarizing the conversation..."):
                    st.session_state.takeaways = summarize_takeaways(
                        get_client(user_key), st.session_state.messages
                    )
            except LLM_ERRORS as exc:
                st.error(friendly_error(exc))
    if clear_col.button("Clear chat", key="clear_chat"):
        st.session_state.messages = []
        st.session_state.takeaways = None
        st.rerun()

    if st.session_state.takeaways:
        with st.expander(f"Key takeaways: {st.session_state.takeaways.title}", expanded=True):
            for point in st.session_state.takeaways.points:
                st.markdown(f"- {point}")

    welcome = st.empty()
    if not st.session_state.messages:
        welcome.info(
            f"Hi! I'm your {persona.name.lower()}. Ask me about anything you're studying."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=AVATARS[message["role"]]):
            if message["role"] == "assistant":
                label = message.get("persona", "")
                if message.get("document"):
                    label += f" · from {message['document']}"
                st.caption(label)
            st.markdown(message["content"])
            if message.get("document") and is_idk(message["content"]):
                st.caption(
                    "That doesn't seem to be covered in your document. Try rephrasing, "
                    "or search for related terms in the Knowledge Base tab."
                )
            elif message.get("sources"):
                with st.expander(f"Sources ({len(message['sources'])})"):
                    for i, source in enumerate(message["sources"], start=1):
                        st.markdown(
                            f"**[{i}]** {source['page_label']} · "
                            f"{source['similarity']:.0%} match"
                        )
                        snippet = source["text"].replace("\n", " ")
                        if len(snippet) > 300:
                            snippet = snippet[:300] + "..."
                        st.caption(escape_markdown(snippet))

    question = st.chat_input(f"Ask the {persona.name.lower()} anything...")
    if question:
        welcome.empty()
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=AVATARS["user"]):
            st.markdown(question)
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            st.caption(persona.name)
            try:
                client = get_client(user_key)
                system_prompt = build_system_prompt(persona, tone)
                sources = []
                if active_doc:
                    sources, stream = answer_stream(
                        client,
                        store,
                        OpenAIEmbedder(client),
                        active_doc.doc_id,
                        system_prompt,
                        st.session_state.messages,
                        max_messages=MAX_CONTEXT_MESSAGES,
                    )
                else:
                    stream = stream_chat(
                        client,
                        build_chat_messages(
                            system_prompt,
                            st.session_state.messages,
                            max_messages=MAX_CONTEXT_MESSAGES,
                        ),
                    )
                reply = st.write_stream(stream)
            except LLM_ERRORS as exc:
                st.session_state.messages.pop()  # don't keep a question with no answer
                st.error(friendly_error(exc))
            else:
                message = {"role": "assistant", "content": reply, "persona": persona.name}
                if active_doc:
                    message["document"] = active_doc.filename
                    message["sources"] = [
                        {
                            "page_label": r.chunk.page_label,
                            "text": r.chunk.text,
                            "similarity": round(r.similarity, 3),
                        }
                        for r in sources
                    ]
                st.session_state.messages.append(message)
                st.rerun()  # refresh the sidebar counter and export buttons

# --- Quiz Generator -----------------------------------------------------------
with quiz_tab:
    if st.session_state.document is None:
        st.info("Upload a PDF in the sidebar and EduPulse will quiz you on it.")
    else:
        doc = st.session_state.document
        st.write(f"Multiple-choice practice questions from **{doc.filename}**.")
        with st.form("quiz_settings"):
            topic_col, count_col = st.columns([3, 1])
            topic = topic_col.text_input(
                "Focus on a topic (optional)",
                key="quiz_topic",
                placeholder="e.g. cell membrane",
            )
            n_questions = count_col.number_input(
                "Questions", min_value=3, max_value=10, value=5, key="quiz_n"
            )
            make_quiz = st.form_submit_button("Generate quiz", type="primary")
        if make_quiz:
            try:
                with st.spinner("Writing your quiz..."):
                    client = get_client(user_key)
                    chunks = sample_context(store, OpenAIEmbedder(client), doc.doc_id, topic)
                    st.session_state.quiz = generate_quiz(client, chunks, n=int(n_questions))
                st.session_state.quiz_answers = []
                st.session_state.quiz_submitted = False
                st.session_state.quiz_round += 1
            except LLM_ERRORS as exc:
                st.error(friendly_error(exc))

        quiz = st.session_state.quiz
        if quiz:
            submitted = st.session_state.quiz_submitted
            with st.form("quiz_answers_form"):
                answers = []
                for i, q in enumerate(quiz.questions):
                    answers.append(
                        st.radio(
                            f"**{i + 1}. {q.question}**",
                            options=range(len(q.options)),
                            format_func=lambda j, q=q: q.options[j],
                            index=None,
                            key=f"quiz_{st.session_state.quiz_round}_{i}",
                            disabled=submitted,
                        )
                    )
                    if submitted:
                        chosen = st.session_state.quiz_answers[i]
                        if chosen == q.correct_index:
                            st.success(f"Correct! {q.explanation}")
                        else:
                            st.error(
                                f"Not quite. The answer is **{q.correct_option}**. {q.explanation}"
                            )
                check = st.form_submit_button("Check answers", disabled=submitted)

            if check:
                if None in answers:
                    st.warning("Answer every question before checking.")
                else:
                    st.session_state.quiz_answers = answers
                    st.session_state.quiz_submitted = True
                    st.rerun()

            if submitted:
                result = grade_quiz(quiz, st.session_state.quiz_answers)
                score_col, retry_col = st.columns([3, 1])
                score_col.metric("Your score", f"{result.score} / {result.total}")
                if result.score == result.total:
                    st.balloons()
                if retry_col.button("Try again", key="quiz_retry"):
                    st.session_state.quiz_answers = []
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_round += 1
                    st.rerun()

# --- Flashcards ---------------------------------------------------------------
with cards_tab:
    if st.session_state.document is None:
        st.info("Upload a PDF in the sidebar to turn it into flashcards.")
    else:
        doc = st.session_state.document
        st.write(f"Key terms and definitions from **{doc.filename}**.")
        with st.form("cards_settings"):
            topic_col, count_col = st.columns([3, 1])
            card_topic = topic_col.text_input(
                "Focus on a topic (optional)",
                key="cards_topic",
                placeholder="e.g. organelles",
            )
            n_cards = count_col.number_input(
                "Cards", min_value=5, max_value=20, value=10, key="cards_n"
            )
            make_cards = st.form_submit_button("Generate flashcards", type="primary")
        if make_cards:
            try:
                with st.spinner("Making your flashcards..."):
                    client = get_client(user_key)
                    chunks = sample_context(
                        store, OpenAIEmbedder(client), doc.doc_id, card_topic
                    )
                    deck = generate_flashcards(client, chunks, n=int(n_cards))
                st.session_state.deck = deck
                st.session_state.card_order = list(range(len(deck.cards)))
                st.session_state.card_pos = 0
                st.session_state.card_flipped = False
            except LLM_ERRORS as exc:
                st.error(friendly_error(exc))

        deck = st.session_state.deck
        if deck:
            st.markdown(CARD_CSS, unsafe_allow_html=True)
            order = st.session_state.card_order
            pos = st.session_state.card_pos
            card = deck.cards[order[pos]]
            flipped = st.session_state.card_flipped
            side, text = ("Definition", card.definition) if flipped else ("Term", card.term)
            st.markdown(
                f'<div class="flashcard {"back" if flipped else "front"}">'
                f'<div class="side">{side}</div>'
                f'<div class="content">{html.escape(text)}</div></div>',
                unsafe_allow_html=True,
            )
            st.caption(f"Card {pos + 1} of {len(order)}")

            prev_col, flip_col, next_col, shuffle_col = st.columns(4)
            if prev_col.button("◀ Prev", key="card_prev", disabled=pos == 0, width="stretch"):
                st.session_state.card_pos -= 1
                st.session_state.card_flipped = False
                st.rerun()
            if flip_col.button("Flip", key="card_flip", type="primary", width="stretch"):
                st.session_state.card_flipped = not flipped
                st.rerun()
            if next_col.button(
                "Next ▶", key="card_next", disabled=pos == len(order) - 1, width="stretch"
            ):
                st.session_state.card_pos += 1
                st.session_state.card_flipped = False
                st.rerun()
            if shuffle_col.button("Shuffle", key="card_shuffle", width="stretch"):
                random.shuffle(st.session_state.card_order)
                st.session_state.card_pos = 0
                st.session_state.card_flipped = False
                st.rerun()

            with st.expander(f"All {len(deck.cards)} cards"):
                for c in deck.cards:
                    st.markdown(f"**{escape_markdown(c.term)}**: {escape_markdown(c.definition)}")
                st.download_button(
                    "Download for Anki / Quizlet (.csv)",
                    data=deck_to_csv(deck),
                    file_name="edupulse_flashcards.csv",
                    mime="text/csv",
                )

# --- Search -------------------------------------------------------------------
with search_tab:
    doc = st.session_state.document
    if doc is None:
        st.info("Upload a PDF in the sidebar to search it by meaning, not just keywords.")
    else:
        st.write(f"Search **{doc.filename}** by meaning. Try a question or a few key words.")
        with st.form("kb_search"):
            query = st.text_input(
                "Search your document",
                key="kb_query",
                placeholder="e.g. How do cells make energy?",
            )
            k = st.slider("Number of results", min_value=1, max_value=5, value=3)
            submitted = st.form_submit_button("Search", type="primary")
        if submitted:
            if not query.strip():
                st.warning("Type something to search for.")
            else:
                try:
                    with st.spinner("Searching..."):
                        embedder = OpenAIEmbedder(get_client(user_key))
                        st.session_state.search_results = store.search(
                            doc.doc_id, embedder.embed_query(query), k=k
                        )
                        st.session_state.search_query = query
                except LLM_ERRORS as exc:
                    st.error(friendly_error(exc))

        results = st.session_state.search_results
        if results is not None and not results:
            st.info("No matches found.")
        for rank, result in enumerate(results or [], start=1):
            with st.container(border=True):
                title_col, score_col = st.columns([3, 2])
                title_col.markdown(
                    f"**#{rank}** · {result.chunk.page_label} · chunk {result.chunk.index + 1}"
                )
                score = max(0.0, min(1.0, result.similarity))
                score_col.progress(score, text=f"{score:.0%} match")
                st.markdown(highlight_terms(result.chunk.text, st.session_state.search_query))

# --- Prompt Architect ---------------------------------------------------------
with architect_tab:
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
                    client = get_client(user_key)
                    with st.spinner(f"{persona.name} is thinking..."):
                        response = complete(
                            client,
                            [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                        )
                except LLM_ERRORS as exc:
                    st.error(friendly_error(exc))
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
                    f"`{item['created_at']}` · **{item['task']}** · "
                    f"{item['persona']} · {item['tone']}"
                )
                if col_btn.button("Show", key=f"show_{i}"):
                    st.session_state.generated = item
                    st.rerun()
