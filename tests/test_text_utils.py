from edupulse.text_utils import escape_markdown, highlight_terms, query_terms


def test_escape_markdown():
    assert escape_markdown("cost is $5 * 2") == r"cost is \$5 \* 2"
    assert escape_markdown("snake_case #1") == r"snake\_case \#1"


def test_escape_markdown_list_and_heading_markers_at_line_start():
    text = "6. Cellular Respiration\n- a note\n2) second\n---\nwell-known in 1855. Fine"
    assert escape_markdown(text) == (
        "6\\. Cellular Respiration\n\\- a note\n2\\) second\n\\---\nwell-known in 1855. Fine"
    )


def test_query_terms_skip_short_words_and_stopwords():
    assert query_terms("What does the Golgi apparatus do?") == ["apparatus", "golgi"]
    assert query_terms("is it?") == []


def test_highlight_matches_word_starts_case_insensitively():
    text = "Cells use cellular respiration. The cell wall is rigid."
    result = highlight_terms(text, "cell")
    assert result.count("**") == 6
    assert "**Cells**" in result
    assert "**cellular**" in result


def test_highlight_escapes_before_bolding():
    result = highlight_terms("glucose * oxygen", "glucose")
    assert result == r"**glucose** \* oxygen"


def test_highlight_without_terms_just_escapes():
    assert highlight_terms("a_b", "it") == r"a\_b"
