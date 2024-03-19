import pytest
from pref_voting.profiles_with_ties import ProfileWithTies
from  pref_voting.io.writers import  *
from  pref_voting.io.readers import  *
import json

def test_write_abif(condorcet_cycle, tmp_path):

    write_abif(condorcet_cycle, str(tmp_path / "condorcet_cycle.abif"))
    assert (tmp_path / "condorcet_cycle.abif").exists()
    with open(tmp_path / "condorcet_cycle.abif", 'r') as f:
        contents = f.read()
    assert contents == "# 3 candidates\n=0 : [0]\n=1 : [1]\n=2 : [2]\n1:0>1>2\n1:1>2>0\n1:2>0>1\n"

    prof = ProfileWithTies([
        {0:1, 1:2},
        {0:1, 1:1, 2:2}
    ],
    rcounts=[2, 1], 
    candidates=[0, 1, 2, 3])
    write_abif(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.abif").exists()
    with open(tmp_path / "prof.abif", 'r') as f:
        contents = f.read()
    assert contents == "# 4 candidates\n=0 : [0]\n=1 : [1]\n=2 : [2]\n=3 : [3]\n2:0>1\n1:0=1>2\n"


def test_abif_to_profile(condorcet_cycle, tmp_path):

    write_abif(condorcet_cycle, str(tmp_path / "condorcet_cycle.abif"))
    assert (tmp_path / "condorcet_cycle.abif").exists()

    prof = abif_to_profile(str(tmp_path / "condorcet_cycle.abif"))
    assert prof == condorcet_cycle

    prof = ProfileWithTies([
        {0:1, 1:2},
        {0:1, 1:1, 2:2}
    ],
    rcounts=[2, 1], 
    candidates=[0, 1, 2, 3])
    write_abif(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.abif").exists()

    # should raise an error since it is not a profile of linear orders
    with pytest.raises(AssertionError) as excinfo:
        prof = abif_to_profile(str(tmp_path / "prof.abif"))

    assert "The election must contain linear orders on the candidates to create a Profile." in str(excinfo.value)


    prof = ProfileWithTies([
        {0:1, 1:2, 2:3},
        {0:1, 1:3, 2:2}
    ],
    rcounts=[2, 1], 
    candidates=[0, 1, 2, 3])
    write_abif(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.abif").exists()

    # should raise an error since it is not a profile of linear orders
    with pytest.raises(AssertionError) as excinfo:
        prof = abif_to_profile(str(tmp_path / "prof.abif"))

    assert "The election must contain linear orders on the candidates to create a Profile." in str(excinfo.value)

    prof = ProfileWithTies([
        {0:1, 1:2, 2:3},
        {0:1, 1:3, 2:2}
    ],
    rcounts=[2, 1], 
    candidates=[0, 1, 2])
    write_abif(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.abif").exists()

    prof2 = abif_to_profile(str(tmp_path / "prof.abif"))
    assert type(prof2) == Profile
    assert prof2 == prof.to_linear_profile()

def test_abif_to_profile_with_ties(tmp_path):

    condorcet_cycle = ProfileWithTies([ 
        {0:1, 1:2, 2:3},
        {0:2, 1:3, 2:1},
        {0:3, 1:1, 2:2}
    ])
    write_abif(condorcet_cycle, str(tmp_path / "condorcet_cycle.abif"))
    assert (tmp_path / "condorcet_cycle.abif").exists()

    prof = abif_to_profile_with_ties(str(tmp_path / "condorcet_cycle.abif"), cand_type=int)

    assert type(prof) == ProfileWithTies
    assert prof == condorcet_cycle

    prof = ProfileWithTies([
        {0:1, 1:2},
        {0:1, 1:1, 2:2}
    ],
    rcounts=[2, 1], 
    candidates=[0, 1, 2, 3])
    write_abif(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.abif").exists()

    prof2 = abif_to_profile_with_ties(str(tmp_path / "prof.abif"), cand_type=int)
    
    assert type(prof2) == ProfileWithTies
    assert prof == prof2

def test_to_preflib_instance():
    prof = ProfileWithTies(
        [
            {0:1, 1:2},
            {1:1, 2:2, 0:3},
            {2:1, 0:1}
        ], 
        [2, 3, 1])
    inst = to_preflib_instance(prof)
    assert isinstance(inst, OrdinalInstance)
    assert inst.num_voters == 6
    assert inst.num_alternatives == 3
    assert inst.full_profile() == [((0,), (1,)), ((0,), (1,)), ((1,), (2,), (0,)), ((1,), (2,), (0,)), ((1,), (2,), (0,)), ((2, 0),)]

    prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    inst = to_preflib_instance(prof)
    assert isinstance(inst, OrdinalInstance)
    assert inst.num_voters == 3
    assert inst.num_alternatives == 3
    assert inst.full_profile() == [((0,), (1,), (2,)), ((1,), (2,), (0,)), ((2,), (0,), (1,))]

def test_write_preflib(tmp_path):
    prof = ProfileWithTies(
        [
            {0:1, 1:2},
            {1:1, 2:2, 0:3},
            {2:1, 0:1}
        ], 
        [2, 3, 1])
    write_preflib(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.toi").exists()
    with open(tmp_path / "prof.toi", 'r') as f:
        contents = f.read()
    print(contents)
    assert "1: {2, 0}" in contents

    prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    write_preflib(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.soc").exists()
    with open(tmp_path / "prof.soc", 'r') as f:
        contents = f.read()
    assert '1: 0, 1, 2' in contents
    assert '1: 1, 2, 0' in contents
    assert '1: 2, 0, 1' in contents

def test_preflib_to_profiles(tmp_path):
    prof = ProfileWithTies(
        [
            {0:1, 1:2},
            {1:1, 2:2, 0:3},
            {2:1, 0:1}
        ], 
        [2, 3, 1])
    write_preflib(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.toi").exists()
    prof2 = preflib_to_profile(str(tmp_path / "prof.toi"))
    assert type(prof2) == ProfileWithTies   
    assert prof == prof2

    prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    filename = write_preflib(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.soc").exists()
    prof2 = preflib_to_profile(filename, as_linear_profile=True)
    assert type(prof2) == Profile
    assert prof == prof2

def test_write_csv_candidate_columns(tmp_path):
    prof = ProfileWithTies(
        [
            {0:1, 1:2},
            {1:1, 2:2, 0:3},
            {2:1, 0:1}
        ], 
        [2, 3, 1])
    write_csv(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.csv").exists()
    with open(tmp_path / "prof.csv", 'r') as f:
        contents = f.readlines()
    assert '0,1,2,#\n' in contents
    assert '1,2,,2\n' in contents
    assert '3,1,2,3\n' in contents
    assert '1,,1,1\n' in contents

    prof = Profile(
        [[0, 1, 2], [0, 1, 2], [2, 1, 0]], 
        rcounts=[2, 3, 1])
    write_csv(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.csv").exists()
    with open(tmp_path / "prof.csv", 'r') as f:
        contents = f.readlines()
    assert '0,1,2,#\n' in contents
    assert '1,2,3,5\n' in contents
    assert '3,2,1,1\n' in contents


def test_write_csv_rank_columns(tmp_path):

    prof = Profile(
        [[0, 1, 2], [0, 1, 2], [2, 1, 0]], 
        rcounts=[2, 3, 1])
    write_csv(prof, str(tmp_path / "prof"), csv_format="rank_columns")
    assert (tmp_path / "prof.csv").exists()
    with open(tmp_path / "prof.csv", 'r') as f:
        contents = f.readlines()
    assert 'Rank1,Rank2,Rank3\n' in contents
    assert '0,1,2\n' in contents
    assert '2,1,0\n' in contents

    prof = ProfileWithTies(
        [{0:1, 1:2}, {0:2, 1:1, 2:3}, {0:3, 1:1, 2:2}], 
        rcounts=[2, 3, 1])
    write_csv(prof, str(tmp_path / "prof"), csv_format="rank_columns")
    assert (tmp_path / "prof.csv").exists()
    with open(tmp_path / "prof.csv", 'r') as f:
        contents = f.readlines()
    assert 'Rank1,Rank2,Rank3\n' in contents
    assert '0,1,skipped\n' in contents
    assert '1,0,2\n' in contents
    assert '1,2,0\n' in contents


def test_write_json(tmp_path):

    prof = Profile(
        [[0, 1, 2], [0, 1, 2], [2, 1, 0]], 
        rcounts=[2, 3, 1])
    write_json(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.json").exists()
    # read the json file
    with open(tmp_path / "prof.json", 'r') as f:
        prof_as_dict = json.load(f)

    assert prof_as_dict["candidates"] == prof.candidates
    assert {'ranking': {'0': 0, '1': 1, '2': 2}, 'count': 2} in prof_as_dict["rankings"]
    assert {'ranking': {'0': 0, '1': 1, '2': 2}, 'count': 3} in prof_as_dict["rankings"]
    assert {'ranking': {'0': 2, '1': 1, '2': 0}, 'count': 1} in prof_as_dict["rankings"]
    assert prof_as_dict["cmap"] == {str(cidx): str(c) for cidx, c in prof.cmap.items()}


    prof = ProfileWithTies(
        [{0:1, 1:2}, {0:2, 1:1, 2:3}, {0:3, 1:1, 2:2}], 
        rcounts=[2, 3, 1])
    write_json(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.json").exists()
    # read the json file
    with open(tmp_path / "prof.json", 'r') as f:
        prof_as_dict = json.load(f)

    assert prof_as_dict["candidates"] == prof.candidates
    assert {'ranking': {'0': 1, '1': 2}, 'count': 2} in prof_as_dict["rankings"]
    assert {'ranking': {'0': 2, '1': 1, '2': 3}, 'count': 3} in prof_as_dict["rankings"]
    assert {'ranking': {'0': 3, '1': 1, '2': 2}, 'count': 1} in prof_as_dict["rankings"]
    assert prof_as_dict["cmap"] == {str(cidx): c for cidx, c in prof.cmap.items()}


def test_csv_to_profile_candidate_columns(tmp_path):
    prof = ProfileWithTies(
        [
            {0:1, 1:2},
            {1:1, 2:2, 0:3},
            {2:1, 0:1}
        ], 
        [2, 3, 1])
    write_csv(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.csv").exists()
    prof2 = csv_to_profile(str(tmp_path / "prof.csv"), cand_type=int)
    assert prof == prof2
    prof = Profile(
        [[0, 1, 2], [0, 1, 2], [2, 1, 0]], 
        rcounts=[2, 3, 1])
    write_csv(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.csv").exists()
    prof2 = csv_to_profile(str(tmp_path / "prof.csv"), cand_type=int, as_linear_profile=True)
    assert prof == prof2

def test_csv_to_profile_rank_columns(tmp_path):
    prof = ProfileWithTies(
        [
            {0:1, 1:2},
            {1:1, 2:2, 0:3},
            {2:1, 0:2}
        ], 
        [2, 3, 1])
    write_csv(prof, str(tmp_path / "prof"), csv_format="rank_columns")
    assert (tmp_path / "prof.csv").exists()
    prof2 = csv_to_profile(str(tmp_path / "prof.csv"), csv_format="rank_columns", cand_type=int)
    assert prof == prof2
    prof = Profile(
        [[0, 1, 2], [0, 1, 2], [2, 1, 0]], 
        rcounts=[2, 3, 1])
    write_csv(prof, str(tmp_path / "prof"), csv_format="rank_columns")
    assert (tmp_path / "prof.csv").exists()
    prof2 = csv_to_profile(str(tmp_path / "prof.csv"), csv_format="rank_columns", cand_type=int, as_linear_profile=True)
    assert prof == prof2


def test_json_to_profile(tmp_path):

    prof = Profile(
        [[0, 1, 2], [0, 1, 2], [2, 1, 0]], 
        rcounts=[2, 3, 1])
    write_json(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.json").exists()

    prof2 = json_to_profile(str(tmp_path / "prof.json"),  as_linear_profile=True)

    assert type(prof2) == Profile
    assert prof2 == prof

    prof = ProfileWithTies(
        [{0:1, 1:2}, {0:2, 1:1, 2:3}, {0:3, 1:1, 2:2}], 
        rcounts=[2, 3, 1])
    write_json(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.json").exists()

    prof2 = json_to_profile(str(tmp_path / "prof.json"))

    assert type(prof2) == ProfileWithTies
    assert prof2 == prof


def test_write_read_profile(tmp_path):
    prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]])

    fname = write(prof, str(tmp_path / "prof"), file_format="abif")
    prof2 = read(fname, file_format="abif", as_linear_profile=True)
    assert prof == prof2

    fname = write(prof, str(tmp_path / "prof"), file_format="csv")
    prof2 = read(fname, file_format="csv", as_linear_profile=True)
    assert prof == prof2

    fname = write(prof, str(tmp_path / "prof"), file_format="csv", csv_format="rank_columns")
    prof2 = read(fname, file_format="csv", csv_format="rank_columns",  as_linear_profile=True)
    assert prof == prof2

    fname = write(prof, str(tmp_path / "prof"), file_format="json")
    prof2 = read(fname, file_format="json", as_linear_profile=True)
    assert prof == prof2


def test_write_read_profile_with_ties(tmp_path):
    prof = ProfileWithTies([
        {0:1, 1:2},
        {0:1, 1:1, 2:2},
        {0:2, 1:1, 2:2}
    ],
    rcounts=[2, 1, 1],
    candidates=[0, 1, 2, 3])

    fname = write(prof, str(tmp_path / "prof"), file_format="abif")
    prof2 = read(fname, file_format="abif", cand_type=int)
    assert prof == prof2

    fname = write(prof, str(tmp_path / "prof"), file_format="csv")
    prof2 = read(fname, file_format="csv", cand_type=int)
    assert prof == prof2

    with pytest.raises(AssertionError) as excinfo:
        fname = write(prof, str(tmp_path / "prof"), file_format="csv", csv_format="rank_columns")

    assert "The profile must be truncated linear to use the rank_columns csv format." in str(excinfo.value)

    fname = write(prof, str(tmp_path / "prof"), file_format="json")
    prof2 = read(fname, file_format="json")
    assert prof == prof2

def test_spatial_profile_to_json(tmp_path): 
    cand_pos = {1: np.array([0.1, 0.2]), 2: np.array([0.3, 0.4])}
    voter_pos = {1: np.array([0.5, 0.6]), 2: np.array([0.7, 0.8])}
    sp = SpatialProfile(cand_pos, voter_pos)
    write_spatial_profile_to_json(sp, str(tmp_path / "test"))
    assert (tmp_path / "test.json").exists()
    with open(tmp_path / "test.json", 'r') as f:
        sp_as_dict = json.load(f)
        for voter, pos in voter_pos.items():
            np.testing.assert_array_equal(sp_as_dict["voters"][str(voter)], pos, err_msg=f"Incorrect position for voter {voter}")
        for candidate, pos in cand_pos.items():
            np.testing.assert_array_equal(sp_as_dict["candidates"][str(candidate)], pos, err_msg=f"Incorrect position for voter {voter}")

def test_spatial_profile_to_json_json_to_spatial_profile(tmp_path): 
    cand_pos = {1: np.array([0.1, 0.2]), 2: np.array([0.3, 0.4])}
    voter_pos = {1: np.array([0.5, 0.6]), 2: np.array([0.7, 0.8])}
    sp = SpatialProfile(cand_pos, voter_pos)
    fname = write_spatial_profile_to_json(sp, str(tmp_path / "test"))
    sp2 = json_to_spatial_profile(fname)
    
    for voter, pos in voter_pos.items():
        np.testing.assert_array_equal(sp.voter_position(voter), pos, err_msg=f"Incorrect position for voter {voter}")
        np.testing.assert_array_equal(sp2.voter_position(voter), pos, err_msg=f"Incorrect position for voter {voter}")
    for candidate, pos in cand_pos.items():
        np.testing.assert_array_equal(sp.candidate_position(candidate), pos, err_msg=f"Incorrect position for candidate {candidate}")
        np.testing.assert_array_equal(sp2.candidate_position(candidate), pos, err_msg=f"Incorrect position for candidate {candidate}")
