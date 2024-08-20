from pref_voting.pairwise_profiles import PairwiseComparisons
from io import StringIO
from contextlib import redirect_stdout
import pytest


@pytest.fixture
def sample_comparisons():
    comparisons = [
        (("A", "B"), ("A",)),
        (("A", "C"), ("C",)),
        (("B", "C"), ("B",))
    ]
    return PairwiseComparisons(comparisons, cmap={"A": "Candidate A", "B": "Candidate B", "C": "Candidate C"})

def test_initialization(sample_comparisons):
    assert sample_comparisons.is_coherent()

def test_weak_preference(sample_comparisons):
    assert sample_comparisons.weak_pref("A", "B")
    assert sample_comparisons.weak_pref("C", "A")
    assert not sample_comparisons.weak_pref("A", "C")

def test_strict_preference(sample_comparisons):
    assert sample_comparisons.strict_pref("A", "B")
    assert not sample_comparisons.strict_pref("B", "A")

def test_indifference(sample_comparisons):
    assert not sample_comparisons.indiff("A", "B")
    sample_comparisons.add_comparison({"A", "D"}, {"A", "D"})
    assert sample_comparisons.indiff("A", "D")

def test_has_comparison(sample_comparisons):
    assert sample_comparisons.has_comparison("A", "B")
    assert not sample_comparisons.has_comparison("A", "D")

def test_get_comparison(sample_comparisons):
    assert sample_comparisons.get_comparison("A", "B") == ({"A", "B"}, {"A"})
    assert sample_comparisons.get_comparison("A", "D") is None

def test_add_comparison(sample_comparisons):
    sample_comparisons.add_comparison({"A", "D"}, {"D"})
    assert sample_comparisons.has_comparison("A", "D")
    assert sample_comparisons.strict_pref("D", "A")

def test_add_strict_preference(sample_comparisons):
    sample_comparisons.add_strict_preference("D", "E")
    assert sample_comparisons.strict_pref("D", "E")

def test_display(sample_comparisons):
    expected_output = "{Candidate A, Candidate B} -> {Candidate A}\n{Candidate A, Candidate C} -> {Candidate C}\n{Candidate B, Candidate C} -> {Candidate B}\n"
    with StringIO() as buf, redirect_stdout(buf):
        sample_comparisons.display()
        output = buf.getvalue()
    assert output == expected_output

def test_str(sample_comparisons):
    expected_output = "{Candidate A, Candidate B} -> {Candidate A}, {Candidate A, Candidate C} -> {Candidate C}, {Candidate B, Candidate C} -> {Candidate B}"
    assert str(sample_comparisons) == expected_output
