"""
Tests for coalition_formation.py (AI textual version).

Paper: "AI-Generated Compromises for Coalition Formation",
       by Eyal Briman, Ehud Shapiro, and Nimrod Talmon (2024),
       https://arxiv.org/abs/2410.21440

Programmers: Hillel Ohayon.
Date: 2025-04-04.
"""

import math
import os
import random

import numpy as np
import pytest
from dotenv import load_dotenv
from unittest.mock import patch

from pref_voting.coalition_formation import (
    embed_text,
    cosine_dissimilarity,
    agent_votes,
    generate_compromise_sentences,
    choose_best_sentence,
    coalition_formation,
)

load_dotenv()
API_KEY = os.environ.get("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# cosine_dissimilarity
# ---------------------------------------------------------------------------

class TestCosineDissimilarity:
    def test_identical_vectors_is_zero(self):
        # Distance between a vector and itself must be exactly 0.
        assert cosine_dissimilarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_orthogonal_vectors(self):
        # Perpendicular vectors should have distance sqrt(2).
        assert cosine_dissimilarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(math.sqrt(2), rel=1e-3)

    def test_opposite_vectors(self):
        # Opposite vectors should have maximum distance 2.
        assert cosine_dissimilarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0, rel=1e-3)

    def test_symmetry(self):
        # d(A, B) == d(B, A).
        a, b = np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
        assert cosine_dissimilarity(a, b) == pytest.approx(cosine_dissimilarity(b, a))

    def test_result_in_valid_range(self):
        # Distance must lie in [0, 2] for any two vectors (paper footnote 6).
        rng = np.random.default_rng(0)
        a, b = rng.standard_normal(512), rng.standard_normal(512)
        assert 0.0 <= cosine_dissimilarity(a, b) <= 2.0 + 1e-9


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------

class TestEmbedText:
    def test_returns_512_dimensions(self):
        # Universal Sentence Encoder must return exactly 512 dimensions.
        assert len(embed_text("Reduce carbon emissions globally.")) == 512

    def test_returns_floats(self):
        # All 512 elements must be floating-point numbers.
        v = embed_text("Climate change is urgent.")
        assert all(isinstance(x, float) for x in v)

    def test_similar_sentences_are_closer(self):
        # Semantically similar sentences should be closer than unrelated ones.
        v1 = embed_text("Plant trees to fight climate change.")
        v2 = embed_text("Grow forests to combat global warming.")
        v3 = embed_text("Increase military spending now.")
        assert cosine_dissimilarity(v1, v2) < cosine_dissimilarity(v1, v3)

    def test_same_sentence_distance_zero(self):
        # Embedding the same text twice yields distance 0.
        v = embed_text("We must act on climate change.")
        assert cosine_dissimilarity(v, v) == pytest.approx(0.0, abs=1e-5)

    def test_result_is_numpy_array(self):
        # embed_text must return a numpy ndarray, not a list.
        assert isinstance(embed_text("Test sentence."), np.ndarray)


# ---------------------------------------------------------------------------
# agent_votes
# ---------------------------------------------------------------------------

class TestAgentVotes:
    def test_deterministic_accepts_identical_to_ideal(self):
        # Agent always accepts its own ideal sentence.
        s = "Cut carbon emissions now."
        assert agent_votes(s, s, "Do nothing.", sigma=0.0) is True

    def test_deterministic_rejects_farther_proposal(self):
        # Deterministic agent rejects a proposal farther than the status quo.
        assert agent_votes(
            "Plant trees everywhere.",
            "Increase military spending.",
            "Plant trees everywhere.",
            sigma=0.0,
        ) is False

    def test_deterministic_accepts_closer_proposal(self):
        # Deterministic agent accepts a proposal closer than the status quo.
        assert agent_votes(
            "Use solar energy.",
            "Switch to renewable energy.",
            "Keep burning coal forever.",
            sigma=0.0,
        ) is True

    def test_probabilistic_sometimes_accepts_worse(self):
        # High sigma allows occasional acceptance of a worse proposal.
        random.seed(42)
        results = [
            agent_votes("Cut emissions.", "Do nothing.", "Cut emissions.", sigma=10.0)
            for _ in range(100)
        ]
        assert any(results), "High sigma should allow occasional acceptance of worse proposal"

    def test_returns_bool(self):
        # agent_votes must return a standard Python bool.
        assert isinstance(agent_votes("a", "b", "c", sigma=0.0), bool)


# ---------------------------------------------------------------------------
# generate_compromise_sentences
# ---------------------------------------------------------------------------

class TestGenerateCompromiseSentences:
    def test_returns_correct_count(self):
        # Must return exactly n candidate sentences.
        sentences = generate_compromise_sentences(
            "Plant trees to fight climate change.",
            "Switch to renewable energy.",
            n=5,
            api_key=API_KEY,
        )
        assert len(sentences) == 5

    def test_returns_non_empty_strings(self):
        # All returned sentences must be non-empty strings.
        sentences = generate_compromise_sentences(
            "Plant trees to fight climate change.",
            "Switch to renewable energy.",
            n=3,
            api_key=API_KEY,
        )
        assert all(isinstance(s, str) and len(s) > 0 for s in sentences)

    def test_default_count_is_10(self):
        # Default n must be 10.
        sentences = generate_compromise_sentences(
            "Reduce emissions.", "Plant trees.", api_key=API_KEY
        )
        assert len(sentences) == 10

    def test_sentences_max_15_words(self):
        # Paper constraint: at most 15 words per sentence.
        sentences = generate_compromise_sentences(
            "Protect rainforests by reducing deforestation rates globally.",
            "Invest in solar and wind energy technology.",
            n=5,
            api_key=API_KEY,
        )
        for s in sentences:
            assert len(s.split()) <= 15, f"Too long: '{s}'"

    def test_gpt_path_called_with_api_key(self):
        # When api_key is provided, GPT path must be invoked (not fallback).
        import json
        mock_content = json.dumps({"compromises": ["Save the planet now.", "Go green today.", "Act on climate."]})
        mock_response = type("R", (), {
            "choices": [type("C", (), {
                "message": type("M", (), {"content": mock_content})()
            })()]
        })()
        with patch("openai.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.return_value = mock_response
            sentences = generate_compromise_sentences(
                "Plant trees.", "Use solar power.", n=3, api_key="sk-test-fake-key"
            )
        mock_client.chat.completions.create.assert_called_once()
        assert len(sentences) == 3


# ---------------------------------------------------------------------------
# choose_best_sentence
# ---------------------------------------------------------------------------

class TestChooseBestSentence:
    def test_returns_string(self):
        # Must return a single string.
        target = np.array([1.0] + [0.0] * 511)
        assert isinstance(choose_best_sentence(["hello", "world"], target), str)

    def test_returns_one_of_the_candidates(self):
        # Returned sentence must come from the candidates list.
        candidates = ["Cut emissions now.", "Plant more trees.", "Use green energy."]
        target = embed_text("Reduce carbon through renewable energy and reforestation.")
        assert choose_best_sentence(candidates, target) in candidates

    def test_single_candidate_always_returned(self):
        # Single candidate must always be returned.
        assert choose_best_sentence(["only option"], np.zeros(512)) == "only option"


# ---------------------------------------------------------------------------
# coalition_formation (main algorithm)
# ---------------------------------------------------------------------------

class TestCoalitionFormation:
    def test_single_agent_returns_itself(self):
        # Single agent halts immediately (100% of votes).
        sentence, agents = coalition_formation(
            {0: "Protect the forests."}, "Do nothing.", api_key=API_KEY
        )
        assert agents == [0]
        assert isinstance(sentence, str)

    def test_returns_majority(self):
        # Algorithm must halt with >= 50% of agents in the winning coalition.
        ideal = {
            0: "Plant trees to absorb CO2 emissions.",
            1: "Switch to solar and wind energy.",
            2: "Reduce meat consumption to lower emissions.",
            3: "Invest in carbon capture technologies.",
            4: "Improve public transport to cut car use.",
        }
        sentence, agents = coalition_formation(
            ideal, "Do nothing about climate change.", seed=0, api_key=API_KEY
        )
        assert len(agents) >= math.ceil(len(ideal) / 2)

    def test_returned_agents_are_valid_indices(self):
        # Returned agent IDs must be a subset of the input dict keys.
        ideal = {i: f"Policy proposal number {i}." for i in range(6)}
        _, agents = coalition_formation(ideal, "Status quo.", seed=1, api_key=API_KEY)
        assert all(a in ideal for a in agents)

    def test_compromise_is_string(self):
        # Final output must be a non-empty string.
        ideal = {0: "Tax carbon heavily.", 1: "Subsidise green energy."}
        result_sentence, _ = coalition_formation(
            ideal, "Do nothing.", majority_quota=0.51, seed=0, api_key=API_KEY
        )
        assert isinstance(result_sentence, str) and len(result_sentence) > 0

    def test_large_random_input_converges(self):
        # 20 agents on related topics must coalesce into a majority.
        topics = [
            "Plant trees globally.", "Ban fossil fuels immediately.",
            "Invest in nuclear energy.", "Tax carbon emissions heavily.",
            "Improve public transport networks.", "Subsidise electric vehicles now.",
            "Reduce meat consumption worldwide.", "Install rooftop solar panels.",
            "Protect existing rainforests legally.", "Develop carbon capture technologies.",
        ] * 2
        ideal = {i: topics[i] for i in range(20)}
        _, agents = coalition_formation(
            ideal, "Do nothing about climate change.", sigma=1.0, seed=77, api_key=API_KEY
        )
        assert len(agents) >= 10

    def test_deterministic_reproducible(self):
        # Same seed must produce identical results.
        ideal = {0: "Cut emissions.", 1: "Plant trees.", 2: "Use renewables."}
        r1 = coalition_formation(ideal, "Do nothing.", seed=42, api_key=API_KEY)
        r2 = coalition_formation(ideal, "Do nothing.", seed=42, api_key=API_KEY)
        assert r1[1] == r2[1]

    def test_empty_input(self):
        # Zero agents must return status quo and empty list.
        sentence, agents = coalition_formation({}, "Status quo remains.", api_key=API_KEY)
        assert sentence == "Status quo remains." and agents == []

    def test_invalid_majority_quota_raises_error(self):
        # Quota > 1.0 is mathematically impossible and must raise ValueError.
        with pytest.raises(ValueError):
            coalition_formation({0: "A", 1: "B"}, "Status quo", majority_quota=1.5, api_key=API_KEY)

    def test_invalid_alpha_raises_error(self):
        # Alpha outside [-1, 1] must raise ValueError.
        with pytest.raises(ValueError):
            coalition_formation({0: "A", 1: "B"}, "Status quo", alpha=5, api_key=API_KEY)

    def test_very_large_random_input_subset_property(self):
        # 100 agents: result must be a valid subset meeting the quota.
        rng = np.random.default_rng(100)
        topics = ["Solar", "Wind", "Nuclear", "Geothermal", "Hydro", "Biomass"]
        ideal = {i: rng.choice(topics) for i in range(100)}
        sentence, agents = coalition_formation(
            ideal, "Status quo", majority_quota=0.6, seed=100, api_key=API_KEY
        )
        assert set(agents).issubset(set(ideal.keys()))
        assert len(agents) >= 60
        assert isinstance(sentence, str)
