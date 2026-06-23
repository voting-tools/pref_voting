import pytest

from pref_voting.axiom import Axiom
from pref_voting.scoring_methods import plurality, borda


@pytest.fixture
def axiom():
    # the two callables are stubs; this test exercises the Axiom container itself
    return Axiom(
        "Test Axiom",
        has_violation=lambda edata, vm, verbose=False: False,
        find_all_violations=lambda edata, vm, verbose=False: [],
    )

def test_init(axiom):
    assert axiom.name == "Test Axiom"
    assert callable(axiom.has_violation)
    assert callable(axiom.find_all_violations)
    assert axiom.satisfying_vms == []
    assert axiom.violating_vms == []

def test_has_violation_and_find_all_violations_are_the_injected_functions():
    ax = Axiom(
        "A",
        has_violation=lambda edata, vm, verbose=False: True,
        find_all_violations=lambda edata, vm, verbose=False: ["x"],
    )
    assert ax.has_violation(None, None) is True
    assert ax.find_all_violations(None, None) == ["x"]

def test_add_satisfying_vms(axiom):
    axiom.add_satisfying_vms(["Borda"])
    assert axiom.satisfying_vms == ["Borda"]
    axiom.add_satisfying_vms(["Plurality"])
    assert axiom.satisfying_vms == ["Borda", "Plurality"]

def test_add_violating_vms(axiom):
    axiom.add_violating_vms(["Plurality"])
    assert axiom.violating_vms == ["Plurality"]

def test_satisfies(axiom):
    # satisfies/violates look up by the voting method's name
    axiom.add_satisfying_vms(["Borda"])
    assert axiom.satisfies(borda) is True
    assert axiom.satisfies(plurality) is False

def test_violates(axiom):
    axiom.add_violating_vms(["Plurality"])
    assert axiom.violates(plurality) is True
    assert axiom.violates(borda) is False

def test_satisfying_and_violating_are_independent(axiom):
    axiom.add_satisfying_vms(["Borda"])
    axiom.add_violating_vms(["Plurality"])
    assert axiom.satisfies(borda) and not axiom.violates(borda)
    assert axiom.violates(plurality) and not axiom.satisfies(plurality)
