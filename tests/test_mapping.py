import pytest
import numpy as np
from pref_voting.mappings import _Mapping
# Assume needed imports and setups for _Mapping are done here.

@pytest.fixture
def simple_mapping():
    return _Mapping({1: 100, 2: 200, 3: 300})

def test_initialization():
    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2}, codomain={10, 20})
    assert mapping.domain == {1, 2}
    assert mapping.codomain == {10, 20}
    assert mapping.val(1) == 10

def test_val(simple_mapping):
    assert simple_mapping.val(1) == 100

    with pytest.raises(AssertionError) as excinfo:
        simple_mapping.val(4)
    assert "4 not in the domain [1, 2, 3]" in str(excinfo.value) 

def test_has_value(simple_mapping):
    assert simple_mapping.has_value(1)
    assert not simple_mapping.has_value(4)

def test_defined_domain(simple_mapping):
    assert sorted(simple_mapping.defined_domain) == [1, 2, 3]
    mapping2 = _Mapping({1: 10, 2: 20}, domain=[0, 1, 2, 3])
    assert sorted(mapping2.defined_domain) == [1, 2]

def test_inverse_image(simple_mapping):
    assert simple_mapping.inverse_image(100) == [1]
    assert simple_mapping.inverse_image(400) == []

def test_image(simple_mapping):
    assert simple_mapping.image() == [100, 200, 300]
    assert simple_mapping.image([1, 3]) == [100, 300]

def test_range(simple_mapping):
    assert simple_mapping.range == [100, 200, 300]

def test_average(simple_mapping):
    assert simple_mapping.average() == np.mean([100, 200, 300])

def test_median(simple_mapping):
    assert simple_mapping.median() == np.median([100, 200, 300])

def test_compare(simple_mapping):
    assert simple_mapping.compare(1, 2) == -1
    assert simple_mapping.compare(2, 2) == 0
    assert simple_mapping.compare(2, 1) == 1
    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2, 3, 4})
    assert mapping.compare(1, 2) == -1
    assert mapping.compare(2, 2) == 0
    assert mapping.compare(2, 1) == 1
    assert mapping.compare(3, 1) is None
    assert mapping.compare(1, 3) is None
    assert mapping.compare(3, 4) is None

def test_extended_compare():
    
    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2, 3, 4})
    assert mapping.extended_compare(1, 2) == -1
    assert mapping.extended_compare(2, 2) == 0
    assert mapping.extended_compare(2, 1) == 1
    assert mapping.extended_compare(3, 1) == -1
    assert mapping.extended_compare(1, 3) == 1
    assert mapping.extended_compare(3, 4) == 0

def test_strict_pref(simple_mapping):
    assert simple_mapping.strict_pref(3, 1)
    assert not simple_mapping.strict_pref(1, 3)
    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2, 3, 4})
    assert not mapping.strict_pref(1, 2) 
    assert not mapping.strict_pref(2, 2) 
    assert mapping.strict_pref(2, 1)
    assert not mapping.strict_pref(3, 1) 
    assert not mapping.strict_pref(1, 3) 
    assert not mapping.strict_pref(3, 4) 

def test_extended_strict_pref():
    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2, 3, 4})
    assert not mapping.extended_strict_pref(1, 2) 
    assert not mapping.extended_strict_pref(2, 2) 
    assert mapping.extended_strict_pref(2, 1)
    assert not mapping.extended_strict_pref(3, 1) 
    assert mapping.extended_strict_pref(1, 3) 
    assert not mapping.extended_strict_pref(3, 4) 

def test_indiff(simple_mapping):
    assert not simple_mapping.indiff(1, 2)
    assert simple_mapping.indiff(2, 2)

    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2, 3, 4})
    assert not mapping.indiff(1, 2) 
    assert mapping.indiff(2, 2) 
    assert not mapping.indiff(3, 1) 
    assert not mapping.indiff(1, 3) 
    assert not mapping.indiff(3, 4) 

def test_extended_indiff():
    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2, 3, 4})
    assert not mapping.extended_indiff(1, 2) 
    assert mapping.extended_indiff(2, 2) 
    assert not mapping.extended_indiff(3, 1) 
    assert not mapping.extended_indiff(1, 3) 
    assert mapping.extended_indiff(3, 4) 

def test_weak_pref(simple_mapping):
    assert simple_mapping.weak_pref(3, 1)
    assert not simple_mapping.weak_pref(1, 3)
    assert simple_mapping.weak_pref(3, 3)
    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2, 3, 4})
    assert not mapping.weak_pref(1, 2) 
    assert  mapping.weak_pref(2, 2) 
    assert mapping.weak_pref(2, 1)
    assert not mapping.weak_pref(3, 1) 
    assert not mapping.weak_pref(1, 3) 
    assert not mapping.weak_pref(3, 4) 

def test_extended_weak_pref():
    mapping = _Mapping({1: 10, 2: 20}, domain={1, 2, 3, 4})
    assert not mapping.extended_weak_pref(1, 2) 
    assert  mapping.extended_weak_pref(2, 2) 
    assert mapping.extended_weak_pref(2, 1)
    assert not mapping.extended_weak_pref(3, 1) 
    assert  mapping.extended_weak_pref(1, 3) 
    assert  mapping.extended_weak_pref(3, 4) 


def test_indifference_class(simple_mapping):
    assert simple_mapping._indifference_classes(simple_mapping.domain) == [[1], [2], [3]]
    assert simple_mapping._indifference_classes([2, 3]) == [[2], [3]]
    mapping = _Mapping({1: 10, 2: 10}, domain={1, 2, 3, 4})
    assert mapping._indifference_classes([1, 2]) == [[1, 2]]
    assert mapping._indifference_classes([1, 3]) == [[1]]
    assert mapping._indifference_classes([3, 4]) == []

    assert mapping._indifference_classes([1, 2, 3, 4], use_extended=True) == [[1, 2], [3, 4]]

    assert mapping._indifference_classes([1, 2], use_extended=True) == [[1, 2]]
    assert mapping._indifference_classes([1, 3], use_extended=True) == [[1], [3]]
    assert mapping._indifference_classes([3, 4], use_extended=True) == [[3, 4]]

def test_sorted_domain(simple_mapping):
    assert simple_mapping.sorted_domain(simple_mapping.domain) == [[3], [2], [1]]
    mapping = _Mapping({1: 10, 2: 10}, domain={1, 2, 3, 4})
    assert mapping.sorted_domain() == [[1, 2]]
    assert mapping.sorted_domain(extended=True) == [[1, 2], [3, 4]]

def test_as_dict(simple_mapping):
    assert simple_mapping.as_dict() == {1: 100, 2: 200, 3: 300}

def test_display_str(simple_mapping):
    
    assert simple_mapping.display_str("F") == "F(1) = 100, F(2) = 200, F(3) = 300"

def test_call(simple_mapping):
    assert simple_mapping(1) == 100

def test_repr(simple_mapping):
    assert repr(simple_mapping) == "{1: 100, 2: 200, 3: 300}"

def test_str(simple_mapping):

    assert str(simple_mapping) == "1:100, 2:200, 3:300"
