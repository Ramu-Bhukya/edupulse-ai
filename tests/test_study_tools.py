from edupulse.study_tools import KeyTakeaways, summarize_takeaways


def test_summarize_takeaways(fake_client):
    fake_client.chat.completions.parsed = {
        "title": "Heart basics",
        "points": ["The heart has four chambers.", "  ", " Atria receive blood. "],
    }
    history = [
        {"role": "user", "content": "How many chambers?"},
        {"role": "assistant", "content": "Four."},
    ]
    result = summarize_takeaways(fake_client, history)
    assert isinstance(result, KeyTakeaways)
    assert result.points == ["The heart has four chambers.", "Atria receive blood."]

    sent = fake_client.chat.completions.calls[0]["messages"]
    assert "USER: How many chambers?" in sent[1]["content"]
    assert "ASSISTANT: Four." in sent[1]["content"]
