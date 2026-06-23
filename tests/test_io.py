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

def test_write_preflib_profile_with_ties_alpha(tmp_path):
    prof = ProfileWithTies(
        [
            {'a':1, 'b':2},
            {'b':1, 'c':2, 'a':3},
            {'c':1, 'a':1}
        ], 
        [2, 3, 1])

    write_preflib(prof, str(tmp_path / "prof"))
    assert (tmp_path / "prof.toi").exists()
    prof2 = preflib_to_profile(str(tmp_path / "prof.toi"), 
                               use_cand_names=True,
                               include_cmap=True)
    assert type(prof2) == ProfileWithTies   
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


# ===========================================================================
#  Coverage additions
# ===========================================================================

# --- abif ----------------------------------------------------------------

def test_abif_to_profile_custom_candidate_names_roundtrip(tmp_path):
    # Bug 7.1: abif_to_profile keyed its candidate lookup by the display NAME, but
    # vote lines reference the id TOKEN. So any profile whose names != str(id) failed
    # to read back. The original tests only used default names (name == str(id)), so
    # they missed it. This round-trips a profile with genuine custom names.
    prof = Profile([[0, 1, 2], [1, 2, 0], [2, 0, 1]],
                   cmap={0: "Alice", 1: "Bob", 2: "Carol"})
    fname = write_abif(prof, str(tmp_path / "named"))
    back = abif_to_profile(fname)
    assert back.cmap == {0: "Alice", 1: "Bob", 2: "Carol"}
    for a in prof.candidates:
        for b in prof.candidates:
            assert back.margin(a, b) == prof.margin(a, b)

def test_abif_to_profile_no_candidate_declarations(tmp_path):
    # an abif file with no `=` declaration lines: the candidate map is built from the
    # rankings, and a leading `{...}` metadata line is skipped
    p = tmp_path / "nodecl.abif"
    p.write_text("# a comment\n{meta: data}\n2:0>1>2\n1:2>1>0\n")
    prof = abif_to_profile(str(p))
    assert type(prof) == Profile
    assert sorted(prof.candidates) == [0, 1, 2]
    assert prof.num_voters == 3

def test_abif_to_profile_with_ties_string_candidates(tmp_path):
    # no cand_type -> candidates stay strings; also covers the no-cmap return branch
    p = tmp_path / "ties.abif"
    p.write_text("2:0>1>2\n1:0=1>2\n")
    prof = abif_to_profile_with_ties(str(p))
    assert type(prof) == ProfileWithTies
    assert prof.rankings[0].rmap == {"0": 1, "1": 2, "2": 3}

def test_abif_to_profile_with_ties_skips_metadata(tmp_path):
    p = tmp_path / "ties_meta.abif"
    p.write_text("# comment\n{meta}\n=0 : [A]\n=1 : [B]\n2:0>1\n")
    prof = abif_to_profile_with_ties(str(p), cand_type=int)
    assert type(prof) == ProfileWithTies
    assert prof.cmap == {0: "A", 1: "B"}

# --- preflib -------------------------------------------------------------

def test_preflib_to_profile_from_ordinal_instance():
    # passing an OrdinalInstance directly (not a filename) hits the else branch
    inst = to_preflib_instance(Profile([[0, 1, 2], [2, 1, 0]]))
    prof = preflib_to_profile(inst)
    assert type(prof) == ProfileWithTies
    assert prof.num_voters == 2

def test_preflib_to_profile_include_cmap(tmp_path):
    prof = ProfileWithTies([{"a": 1, "b": 2}, {"b": 1, "a": 2}], [1, 1])
    fname = write_preflib(prof, str(tmp_path / "named"))
    back = preflib_to_profile(fname, include_cmap=True, use_cand_names=False)
    assert back.cmap == {0: "a", 1: "b"}

def test_preflib_to_profile_linear_include_cmap(tmp_path):
    fname = write_preflib(Profile([[0, 1, 2], [2, 1, 0]]), str(tmp_path / "lin"))
    back = preflib_to_profile(fname, include_cmap=True, as_linear_profile=True)
    assert type(back) == Profile
    assert back.candidates == [0, 1, 2]

def test_preflib_to_profile_rejects_bad_extension(tmp_path):
    bad = tmp_path / "prof.txt"
    bad.write_text("nonsense")
    with pytest.raises(AssertionError):
        preflib_to_profile(str(bad))

# --- csv -----------------------------------------------------------------

def test_csv_rank_columns_writein_sorted_last(tmp_path):
    import csv as _csv
    f = tmp_path / "wi.csv"
    with open(f, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["Rank1", "Rank2"])
        w.writerow(["A", "writein"])
        w.writerow(["writein", "A"])
    prof = csv_to_profile(str(f), csv_format="rank_columns")
    # 'writein' is always sorted to the last candidate index
    assert prof.cmap == {0: "A", 1: "writein"}

def test_csv_rank_columns_no_candidates(tmp_path, capsys):
    import csv as _csv
    f = tmp_path / "empty.csv"
    with open(f, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["Rank1"])
        w.writerow(["skipped"])
    csv_to_profile(str(f), csv_format="rank_columns")
    assert "No candidates found" in capsys.readouterr().out

# --- json ----------------------------------------------------------------

def test_json_to_profile_with_cand_type(tmp_path):
    fname = write_json(Profile([[0, 1, 2], [2, 1, 0]]), str(tmp_path / "j"))
    prof = json_to_profile(str(fname), cand_type=int)
    assert prof.candidates == [0, 1, 2]
    assert all(isinstance(c, int) for c in prof.candidates)

def test_convert_key_type_fallthrough():
    from pref_voting.io.readers import _convert_key_type
    # convertible: "1" matches int candidate 1
    assert _convert_key_type("1", [0, 1, 2]) == 1
    # not convertible to any candidate type -> returned unchanged
    assert _convert_key_type("xyz", [0, 1, 2]) == "xyz"

# --- read / write dispatchers -------------------------------------------

def test_read_preflib_dispatch(tmp_path):
    fname = write_preflib(Profile([[0, 1, 2], [2, 1, 0]]), str(tmp_path / "p"))
    prof = read(fname, file_format="preflib")
    assert type(prof) == ProfileWithTies

def test_read_unrecognized_format_raises(tmp_path):
    fname = write_preflib(Profile([[0, 1]]), str(tmp_path / "p"))
    with pytest.raises(ValueError):
        read(fname, file_format="bogus")

def test_write_preflib_dispatch(tmp_path):
    fname = write(Profile([[0, 1, 2]]), str(tmp_path / "wp"), file_format="preflib")
    assert fname.endswith(".soc")

def test_write_unrecognized_format_raises(tmp_path):
    with pytest.raises(ValueError):
        write(Profile([[0, 1]]), str(tmp_path / "x"), file_format="bogus")

# --- to_preflib_instance aggregation ------------------------------------

def test_to_preflib_instance_aggregates_identical_rankings():
    # two voters with the same ranking must be aggregated into one vote-map entry
    inst = to_preflib_instance(Profile([[0, 1], [0, 1], [1, 0]], rcounts=[1, 1, 1]))
    assert inst.num_voters == 3
    assert inst.num_alternatives == 2

# --- writers: filename already carries the extension (skip the append) ---

def test_write_csv_filename_already_has_extension(tmp_path):
    fname = write_csv(Profile([[0, 1, 2]]), str(tmp_path / "p.csv"))
    assert fname.endswith(".csv") and not fname.endswith(".csv.csv")

def test_write_json_filename_already_has_extension(tmp_path):
    fname = write_json(Profile([[0, 1, 2]]), str(tmp_path / "p.json"))
    assert fname.endswith(".json") and not fname.endswith(".json.json")

def test_write_abif_filename_already_has_extension(tmp_path):
    fname = write_abif(Profile([[0, 1, 2]]), str(tmp_path / "p.abif"))
    assert fname.endswith(".abif") and not fname.endswith(".abif.abif")

def test_write_spatial_profile_filename_already_has_extension(tmp_path):
    sp = SpatialProfile({0: np.array([0.1])}, {0: np.array([0.2])})
    fname = write_spatial_profile_to_json(sp, str(tmp_path / "p.json"))
    assert fname.endswith(".json") and not fname.endswith(".json.json")

def test_write_grade_profile_to_abif_is_a_noop():
    # currently an unimplemented stub; calling it must not raise
    assert write_grade_profile_to_abif(Profile([[0, 1]])) is None

def test_write_preflib_filename_already_has_inferred_extension(tmp_path):
    # a complete linear profile infers type ".soc"; passing that extension already
    # present skips the append branch
    fname = write_preflib(Profile([[0, 1, 2], [2, 1, 0]]), str(tmp_path / "p.soc"))
    assert fname.endswith(".soc") and not fname.endswith(".soc.soc")
