import pytest

from edupulse.embeddings import OpenAIEmbedder


def test_embeds_in_batches_and_keeps_order(fake_client):
    embedder = OpenAIEmbedder(fake_client, model="test-embed", batch_size=2)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]
    progress = []
    vectors = embedder.embed(texts, on_progress=lambda done, total: progress.append((done, total)))

    calls = fake_client.embeddings.calls
    assert [len(c["input"]) for c in calls] == [2, 2, 1]
    assert all(c["model"] == "test-embed" for c in calls)
    # fake vectors are [len(text), index_in_batch]
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert progress == [(2, 5), (4, 5), (5, 5)]


def test_empty_strings_are_not_sent(fake_client):
    OpenAIEmbedder(fake_client).embed(["", "text"])
    assert fake_client.embeddings.calls[0]["input"] == [" ", "text"]


def test_embed_query(fake_client):
    assert OpenAIEmbedder(fake_client).embed_query("hello") == [5.0, 0.0]


def test_default_model_comes_from_settings(fake_client, monkeypatch):
    monkeypatch.setenv("EMBED_MODEL", "my-embedding-model")
    assert OpenAIEmbedder(fake_client).model == "my-embedding-model"


def test_no_texts_no_calls(fake_client):
    assert OpenAIEmbedder(fake_client).embed([]) == []
    assert fake_client.embeddings.calls == []


def test_bad_batch_size(fake_client):
    with pytest.raises(ValueError):
        OpenAIEmbedder(fake_client, batch_size=0)
