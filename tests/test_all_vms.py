from pref_voting.voting_methods_registry import *
from pref_voting.voting_method_properties import *
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies

def test_all_profile_vms():
    prof = Profile([[0, 1], [1, 0]], rcounts=[1, 2])
    for vm in voting_methods: 
        if ElectionTypes.PROFILE in vm.input_types:
            if vm.name != "Pareto":
                assert vm(prof) == [1]
            else:
                assert vm(prof) == [0, 1]

def test_all_profile_with_ties_vms():
    prof = ProfileWithTies([{0:1, 1:2}, {1:1, 0:2}], rcounts=[1, 2])
    for vm in voting_methods: 
        if ElectionTypes.PROFILE_WITH_TIES in vm.input_types:
            if vm.name != "Pareto":
                assert vm(prof) == [1]
            else:
                assert vm(prof) == [0, 1]

def test_all_margin_graph_vms():
    mg = Profile([[0, 1], [1, 0]], rcounts=[1, 2]).margin_graph()
    for vm in voting_methods: 
        if ElectionTypes.MARGIN_GRAPH in vm.input_types:
            assert vm(mg) == [1]

def test_all_majority_graph_vms():
    mg = Profile([[0, 1], [1, 0]], rcounts=[1, 2]).majority_graph()
    for vm in voting_methods: 
        if ElectionTypes.MAJORITY_GRAPH in vm.input_types:
            assert vm(mg) == [1]

