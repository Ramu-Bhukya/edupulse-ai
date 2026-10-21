import json

import pytest

from edupulse.presets import Persona, Tone, find_by_name, load_presets


def test_default_presets_load():
    personas, tones = load_presets()
    assert personas and tones
    assert all(isinstance(p, Persona) for p in personas)
    assert all(isinstance(t, Tone) for t in tones)


def test_preset_names_are_unique():
    personas, tones = load_presets()
    assert len({p.name for p in personas}) == len(personas)
    assert len({t.name for t in tones}) == len(tones)


def test_find_by_name_is_case_insensitive():
    personas, _ = load_presets()
    assert find_by_name(personas, "medical tutor").name == "Medical Tutor"
    assert find_by_name(personas, "  MATH COACH ").name == "Math Coach"
    assert find_by_name(personas, "Astronaut") is None


def test_invalid_file_raises_value_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"personas": [{"name": "x"}], "tones": []}))
    with pytest.raises(ValueError):
        load_presets(bad)


def test_empty_lists_rejected(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"personas": [], "tones": []}))
    with pytest.raises(ValueError):
        load_presets(empty)
