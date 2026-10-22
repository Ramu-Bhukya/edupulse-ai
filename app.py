"""EduPulse AI: Streamlit front end.

Run with:  streamlit run app.py
"""

import json
import random
from datetime import datetime

import streamlit as st
from openai import OpenAI

from edupulse.config import MissingAPIKeyError, get_api_key, get_settings, mask_key
from edupulse.embeddings import OpenAIEmbedder
from edupulse.history import export_json, export_markdown, prompt_result_as_text
from edupulse.ingest import index_pdf
from edupulse.llm import LLM_ERRORS, complete, friendly_error, get_client, stream_chat
from edupulse.pdf_parser import PDFParseError
from edupulse.presets import Persona, Tone, find_by_name, load_presets
from edupulse.prompts import TASKS, build_chat_messages, build_system_prompt, build_user_prompt
from edupulse.rag_engine import answer_stream, source_to_dict
from edupulse.study_tools import (
    deck_to_csv,
    generate_flashcards,
    generate_quiz,
    grade_quiz,
    sample_context,
    summarize_takeaways,
)
from edupulse.text_utils import escape_markdown
from edupulse.ui_components import (
    AVATARS,
    init_session_state,
    render_chat_message,
    render_document_badge,
    render_flashcard,
    render_graded_question,
    render_search_result,
)
from edupulse.vector_store import DocumentInfo, VectorStore, document_id

MAX_MATERIAL_CHARS = 8000
MAX_HISTORY = 10
MAX_CONTEXT_MESSAGES = 10
TAB_LABELS = [
    "💬 Chat with Notes",
    "📝 Quiz Generator",
    "🃏 Flashcards",
    "🔎 Search",
    "🛠️ Prompt Architect",
]


@st.cache_resource
def get_store(path: str) -> VectorStore:
    return VectorStore.persistent(path)


@st.cache_resource(show_spinner=False)
def get_ai(api_key: str) -> tuple[OpenAI, OpenAIEmbedder]:
    """One OpenAI client + embedder per API key, reused across reruns.

    Keeps the HTTP connection pool (and the embedder's query cache) alive
    instead of rebuilding them on every click.
    """
    client = get_client(api_key)
    return client, OpenAIEmbedder(client)


def ai() -> tuple[OpenAI, OpenAIEmbedder]:
    return get_ai(get_api_key(st.session_state.get("user_api_key")))


def set_document(info: DocumentInfo | None) -> None:
    """Switch the active document and reset everything tied to the old one.

    Called on every rerun, so it must be a no-op when nothing changed
    (including "no document" -> "no document").
    """
    current = st.session_state.document
    old_id = current.doc_id if current else None
    new_id = info.doc_id if info else None
    if old_id == new_id:
        return
    st.session_state.document = info
    st.session_state.search_results = None
    st.session_state.quiz = None
    st.session_state.deck = None
    if st.session_state.messages:
        # answers about the old document would be confusing next to the new one
        st.session_state.messages = []
        st.session_state.takeaways = None
        st.toast("Started a fresh chat for the new document.")


# --- sidebar ------------------------------------------------------------------------


def sidebar_tutor(personas: list[Persona], tones: list[Tone]) -> tuple[Persona, Tone]:
    st.header("Your tutor")
    persona = find_by_name(personas, st.selectbox("Teaching style", [p.name for p in personas]))
    tone = find_by_name(tones, st.selectbox("Tone", [t.name for t in tones]))
    st.caption(persona.description)
    return persona, tone


def sidebar_api_key() -> None:
    try:
        env_key = get_api_key()
    except MissingAPIKeyError:
        env_key = None
    with st.expander("OpenAI API key", expanded=env_key is None):
        user_key = st.text_input(
            "Use your own key",
            type="password",
            key="user_api_key",
            help="Only kept for this browser session. It is never saved.",
        )
        if user_key.strip():
            st.caption(f"Using your key {mask_key(user_key.strip())}")
        elif env_key:
            st.caption(f"Using the app's key {mask_key(env_key)}")
        else:
            st.caption("No key configured yet. Paste one above to start chatting.")


def index_upload(uploaded, store: VectorStore) -> None:
    data = uploaded.getvalue()
    current = st.session_state.document
    if current is not None and current.doc_id == document_id(data):
        return

    with st.status(f"Processing {uploaded.name}...", expanded=True) as status:
        bar = st.empty()

        def show_progress(done: int, total: int) -> None:
            bar.progress(done / total, text=f"Embedded {done} of {total} chunks")

        try:
            _, embedder = ai()
            info, cached = index_pdf(
                data, uploaded.name, store, embedder, on_step=st.write, on_progress=show_progress
            )
        except PDFParseError as exc:
            status.update(label="Couldn't read this PDF", state="error")
            st.error(str(exc))
        except LLM_ERRORS as exc:
            status.update(label="Indexing failed", state="error")
            st.error(friendly_error(exc))
        else:
            set_document(info)
            status.update(label=f"{info.filename} is ready", state="complete", expanded=False)
            if cached:
                st.toast(f"{info.filename} was indexed before, loaded it instantly.")


def sidebar_library(store: VectorStore) -> None:
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


def sidebar_document(store: VectorStore) -> None:
    st.subheader("Your document")
    uploaded = st.file_uploader("Upload a PDF textbook or syllabus", type=["pdf"])
    if uploaded is None:
        st.session_state.upload_attempt = None
        sidebar_library(store)
    else:
        # Hashing a large PDF on every rerun is wasteful, so remember which upload
        # (and key, in case indexing failed for lack of one) was already handled.
        attempt = (uploaded.file_id, st.session_state.get("user_api_key"))
        if attempt != st.session_state.upload_attempt:
            st.session_state.upload_attempt = attempt
            index_upload(uploaded, store)

    if st.session_state.document:
        render_document_badge(st.session_state.document)


def sidebar_session(persona: Persona, tone: Tone) -> None:
    st.subheader("This session")
    messages = st.session_state.messages
    st.metric("Messages", len(messages))
    if messages:
        st.download_button(
            "Export chat (.json)",
            data=export_json(messages, persona=persona.name, tone=tone.name),
            file_name="edupulse_chat.json",
            mime="application/json",
            width="stretch",
        )
        st.download_button(
            "Export chat (.md)",
            data=export_markdown(messages, persona=persona.name, tone=tone.name),
            file_name="edupulse_chat.md",
            mime="text/markdown",
            width="stretch",
        )


# --- tabs ------------------------------------------------------------------------------


def chat_tab(store: VectorStore, persona: Persona, tone: Tone) -> None:
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
                        ai()[0], st.session_state.messages
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
        welcome.info(f"Hi! I'm your {persona.name.lower()}. Ask me about anything you're studying.")

    for message in st.session_state.messages:
        render_chat_message(message)

    question = st.chat_input(f"Ask the {persona.name.lower()} anything...")
    if not question:
        return

    welcome.empty()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(question)
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        st.caption(persona.name)
        try:
            client, embedder = ai()
            system_prompt = build_system_prompt(persona, tone)
            sources = []
            if active_doc:
                sources, stream = answer_stream(
                    client,
                    store,
                    embedder,
                    active_doc.doc_id,
                    system_prompt,
                    st.session_state.messages,
                    max_messages=MAX_CONTEXT_MESSAGES,
                )
            else:
                messages = build_chat_messages(
                    system_prompt, st.session_state.messages, max_messages=MAX_CONTEXT_MESSAGES
                )
                stream = stream_chat(client, messages)
            reply = st.write_stream(stream)
        except LLM_ERRORS as exc:
            st.session_state.messages.pop()  # don't keep a question with no answer
            st.error(friendly_error(exc))
            return

    message = {"role": "assistant", "content": reply, "persona": persona.name}
    if active_doc:
        message["document"] = active_doc.filename
        message["sources"] = [source_to_dict(r) for r in sources]
    st.session_state.messages.append(message)
    st.rerun()  # refresh the sidebar counter and export buttons


def quiz_tab(store: VectorStore) -> None:
    doc = st.session_state.document
    if doc is None:
        st.info("Upload a PDF in the sidebar and EduPulse will quiz you on it.")
        return

    st.write(f"Multiple-choice practice questions from **{doc.filename}**.")
    with st.form("quiz_settings"):
        topic_col, count_col = st.columns([3, 1])
        topic = topic_col.text_input(
            "Focus on a topic (optional)", key="quiz_topic", placeholder="e.g. cell membrane"
        )
        n_questions = count_col.number_input(
            "Questions", min_value=3, max_value=10, value=5, key="quiz_n"
        )
        make_quiz = st.form_submit_button("Generate quiz", type="primary")
    if make_quiz:
        try:
            with st.spinner("Writing your quiz..."):
                client, embedder = ai()
                chunks = sample_context(store, embedder, doc.doc_id, topic)
                st.session_state.quiz = generate_quiz(client, chunks, n=int(n_questions))
            st.session_state.quiz_answers = []
            st.session_state.quiz_submitted = False
            st.session_state.quiz_round += 1
        except LLM_ERRORS as exc:
            st.error(friendly_error(exc))

    quiz = st.session_state.quiz
    if not quiz:
        return

    if st.session_state.quiz_submitted:
        answers = st.session_state.quiz_answers
        for i, q in enumerate(quiz.questions):
            render_graded_question(i + 1, q, answers[i])
        result = grade_quiz(quiz, answers)
        score_col, retry_col = st.columns([3, 1])
        score_col.metric("Your score", f"{result.score} / {result.total}")
        if result.score == result.total:
            st.balloons()
        if retry_col.button("Try again", key="quiz_retry"):
            st.session_state.quiz_answers = []
            st.session_state.quiz_submitted = False
            st.session_state.quiz_round += 1
            st.rerun()
        return

    with st.form("quiz_answers_form"):
        answers = [
            st.radio(
                f"**{i + 1}. {q.question}**",
                options=range(len(q.options)),
                format_func=lambda j, q=q: q.options[j],
                index=None,
                key=f"quiz_{st.session_state.quiz_round}_{i}",
            )
            for i, q in enumerate(quiz.questions)
        ]
        check = st.form_submit_button("Check answers")

    if check:
        if None in answers:
            st.warning("Answer every question before checking.")
        else:
            st.session_state.quiz_answers = answers
            st.session_state.quiz_submitted = True
            st.rerun()


def show_card(pos: int, flipped: bool = False) -> None:
    st.session_state.card_pos = pos
    st.session_state.card_flipped = flipped
    st.rerun()


def flashcards_tab(store: VectorStore) -> None:
    doc = st.session_state.document
    if doc is None:
        st.info("Upload a PDF in the sidebar to turn it into flashcards.")
        return

    st.write(f"Key terms and definitions from **{doc.filename}**.")
    with st.form("cards_settings"):
        topic_col, count_col = st.columns([3, 1])
        topic = topic_col.text_input(
            "Focus on a topic (optional)", key="cards_topic", placeholder="e.g. organelles"
        )
        n_cards = count_col.number_input(
            "Cards", min_value=5, max_value=20, value=10, key="cards_n"
        )
        make_cards = st.form_submit_button("Generate flashcards", type="primary")
    if make_cards:
        try:
            with st.spinner("Making your flashcards..."):
                client, embedder = ai()
                chunks = sample_context(store, embedder, doc.doc_id, topic)
                deck = generate_flashcards(client, chunks, n=int(n_cards))
            st.session_state.deck = deck
            st.session_state.card_order = list(range(len(deck.cards)))
            st.session_state.card_pos = 0
            st.session_state.card_flipped = False
        except LLM_ERRORS as exc:
            st.error(friendly_error(exc))

    deck = st.session_state.deck
    if not deck:
        return

    order, pos = st.session_state.card_order, st.session_state.card_pos
    flipped = st.session_state.card_flipped
    render_flashcard(deck.cards[order[pos]], flipped)
    st.caption(f"Card {pos + 1} of {len(order)}")

    prev_col, flip_col, next_col, shuffle_col = st.columns(4)
    if prev_col.button("◀ Prev", key="card_prev", disabled=pos == 0, width="stretch"):
        show_card(pos - 1)
    if flip_col.button("Flip", key="card_flip", type="primary", width="stretch"):
        show_card(pos, flipped=not flipped)
    if next_col.button("Next ▶", key="card_next", disabled=pos == len(order) - 1, width="stretch"):
        show_card(pos + 1)
    if shuffle_col.button("Shuffle", key="card_shuffle", width="stretch"):
        random.shuffle(st.session_state.card_order)
        show_card(0)

    with st.expander(f"All {len(deck.cards)} cards"):
        for c in deck.cards:
            st.markdown(f"**{escape_markdown(c.term)}**: {escape_markdown(c.definition)}")
        st.download_button(
            "Download for Anki / Quizlet (.csv)",
            data=deck_to_csv(deck),
            file_name="edupulse_flashcards.csv",
            mime="text/csv",
        )


def search_tab(store: VectorStore) -> None:
    doc = st.session_state.document
    if doc is None:
        st.info("Upload a PDF in the sidebar to search it by meaning, not just keywords.")
        return

    st.write(f"Search **{doc.filename}** by meaning. Try a question or a few key words.")
    with st.form("kb_search"):
        query = st.text_input(
            "Search your document", key="kb_query", placeholder="e.g. How do cells make energy?"
        )
        k = st.slider("Number of results", min_value=1, max_value=5, value=3)
        submitted = st.form_submit_button("Search", type="primary")
    if submitted:
        if not query.strip():
            st.warning("Type something to search for.")
        else:
            try:
                with st.spinner("Searching..."):
                    _, embedder = ai()
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
        render_search_result(rank, result, st.session_state.search_query)


def prompt_architect_tab(persona: Persona, tone: Tone) -> None:
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
                    client, _ = ai()
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
        if not result:
            st.info("The AI's answer will appear here.")
        else:
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
                data=prompt_result_as_text(result),
                file_name="edupulse_response.txt",
                mime="text/plain",
            )
            dl_json.download_button(
                "Download .json",
                data=json.dumps(result, indent=2),
                file_name="edupulse_response.json",
                mime="application/json",
            )

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


# --- page --------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="EduPulse AI", page_icon="📚", layout="wide")
    init_session_state()
    personas, tones = load_presets()
    store = get_store(get_settings().chroma_dir)

    with st.sidebar:
        persona, tone = sidebar_tutor(personas, tones)
        sidebar_api_key()
        st.divider()
        sidebar_document(store)
        st.divider()
        sidebar_session(persona, tone)

    st.title("EduPulse AI")
    chat, quiz, cards, search, architect = st.tabs(TAB_LABELS)
    with chat:
        chat_tab(store, persona, tone)
    with quiz:
        quiz_tab(store)
    with cards:
        flashcards_tab(store)
    with search:
        search_tab(store)
    with architect:
        prompt_architect_tab(persona, tone)


main()
