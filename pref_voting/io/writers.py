"""
    File: writers.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: March 17, 2024

    Functions to write election data to a file.
"""

from pref_voting.rankings import Ranking
from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from preflibtools.instances import OrdinalInstance
import csv
import json
import numpy as np

def to_preflib_instance(profile):
    """
    Returns an instance of the ``OrdinalInstance`` class from the ``preflibtools`` package (see https://preflib.github.io/preflibtools/usage.html#ordinal-preferences).  

    Args:
        profile: A Profile or ProfileWithTies object.

    Returns:
        An instance of the ``OrdinalInstance`` class from preflibtools.
        
    """
    assert type(profile) in [Profile, ProfileWithTies], "Must be a Profile or ProfileWithTies object to convert to a preflib OrdinalInstance."

    instance = OrdinalInstance()
    vote_map = dict()
    cand_to_cidx = {c: i for i, c in enumerate(profile.candidates)}
    cmap = {i: profile.cmap[c] for c, i in cand_to_cidx.items()}
    for r,c in zip(*profile.rankings_counts):
        ranking = tuple([tuple([cand_to_cidx[_c] for _c in indiff]) for indiff in r.to_indiff_list()]) if type(r) == Ranking else tuple([(c,) for c in r])
        if ranking in vote_map.keys():
            vote_map[ranking] += c
        else:
            vote_map[ranking] = c
    instance.append_vote_map(vote_map)  
    instance.alternatives_name = cmap  
    return instance   

def write_preflib(profile, filename):
    """
    Write a profile to a file in the PrefLib format.

    Args:
        profile: A Profile or ProfileWithTies object.
        filename: The name of the file to write the profile to.

    Returns:
        The name of the file the profile was written to.
    """
    assert type(profile) in [Profile, ProfileWithTies], "Must be a Profile or ProfileWithTies object to write in the preflib format."

    instance = to_preflib_instance(profile)
    preflib_type = instance.infer_type()
    instance.write(filename)

    if not filename.endswith(preflib_type):
        filename += f".{preflib_type}"

    print(f"Election written to {filename}.")
    return f"{filename}"

def write_csv(profile, filename, csv_format="candidate_columns"):
    """
    Write a profile to a file in CSV format.

    Args:
        profile: A Profile or ProfileWithTies object.
        filename: The name of the file to write the profile to.
        csv_format: The format to write the profile in.  Defaults to "candidate_columns".  The other option is "rank_columns".
    """
    assert type(profile) in [Profile, ProfileWithTies], "Must be a Profile or ProfileWithTies object to write in the csv format."

    candidates = profile.candidates

    if not filename.endswith(".csv"):
        filename += ".csv"

    if csv_format == "rank_columns":

        assert profile.is_truncated_linear, "The profile must be truncated linear to use the rank_columns csv format."

        ranks = range(1, len(candidates) + 1)
        with open(filename, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow([f"Rank{_r}" for _r in ranks])
            for indiff_list in profile.rankings_as_indifference_list:
                ranking = [str(profile.cmap[cs[0]]) for cs in indiff_list]
                writer.writerow(ranking if len(ranking) == len(candidates) else ranking + ["skipped"] * (len(candidates) - len(ranking)))
        
        print(f"Election written to {filename}.")

        return filename

    elif csv_format == "candidate_columns":
        
        prof = profile.to_profile_with_ties() if type(profile) == Profile else profile

        rs, cs = prof.rankings_counts
        anon_rankings = []
        for r, count in zip(rs, cs): 
            r.normalize_ranks()
            found_it = False
            for r_c in anon_rankings: 
                if r_c[0] == r: 
                    found_it = True
                    r_c[1] += count
            if not found_it: 
                anon_rankings.append([r, count])
            
        with open(filename, mode='w') as file:
            writer = csv.writer(file)
            writer.writerow([profile.cmap[c] for c in candidates] + ["#"])
            for r,count in anon_rankings:
                writer.writerow([r.rmap[c] if r.is_ranked(c) else "" for c in candidates] + [count])

        print(f"Election written to {filename}.")

        return filename

def write_json(profile, filename):
    """
    Write a profile to a file in JSON format.

    Args:
        profile: A Profile or ProfileWithTies object.
        filename: The name of the file to write the profile to.
    
    Returns:
        The name of the file the profile was written to.
    """
    assert type(profile) in [Profile, ProfileWithTies], "Cannot write to the abif format."

    if not filename.endswith(".json"):
        filename += ".json"

    prof = profile.to_profile_with_ties() if type(profile) == Profile else profile

    prof_as_dict = {
            "candidates": profile.candidates,
            "rankings": [{"ranking": {
                int(cand) if isinstance(cand, np.int64) else cand: int(rank) 
                for cand,rank in r.rmap.items()}, 
                "count": int(c)} 
                for r,c in zip(*prof.rankings_counts)],
            "cmap": profile.cmap
        }
    with open(filename, "w") as f:
        json.dump(prof_as_dict, f)

    print(f"Election written to {filename}.")
    return filename

def write_abif(profile, filename):
    """
    Write a profile to a file in ABIF format.
    
    The ABIF format is explained here: https://electowiki.org/wiki/ABIF.

    Args:
        profile: A Profile or ProfileWithTies object.
        filename: The name of the file to write the profile to.
    
    Returns:
        The name of the file the profile was written to.
    """

    assert type(profile) in [Profile, ProfileWithTies], "Cannot write to the abif format."

    if not filename.endswith(".abif"):
        filename += ".abif"
        
    with open(filename, mode='w') as file:
        file.write(f"# {len(profile.candidates)} candidates\n")
        for c in profile.candidates:
            file.write(f"={c} : [{profile.cmap[c]}]\n")
        for r, c in zip(*profile.rankings_counts):
            indiff_list = r.to_indiff_list() if type(r) == Ranking else [(c,) for c in r]
            file.write(f"{c}:{'>'.join(['='.join([str(c) for c in cs]) for cs in indiff_list])}\n")
                    
        print(f"Election written to {filename}.")

        return filename

def write_grade_profile_to_abif(profile):
    """
    Write a profile to a file in ABIF format.

    Args:
        profile: A Profile object.
    """
    pass

def write_spatial_profile_to_json(spatial_profile, filename):
    """
    Write a spatial profile to a file in JSON format.

    Args:
        spatial_profile: A SpatialProfile object.

    Returns:
        The name of the file the spatial profile was written to.
    """
     
    if not filename.endswith(".json"):
        filename += ".json"

    with open(filename, "w") as f:
        spatial_profile_dict = {
            "cand_names": spatial_profile.candidates,
            "voter_names": spatial_profile.voters,  
            "candidates": {c: list(spatial_profile.candidate_position(c)) for c in spatial_profile.candidates},
            "voters": {v: list(spatial_profile.voter_position(v)) for v in spatial_profile.voters}
        }
        json.dump(spatial_profile_dict, f)

    print(f"Spatial profile written to {filename}.")

    return filename

def write(
        edata, 
        filename, 
        file_format='preflib', 
        csv_format="candidate_columns"):
    """
    Write election data to ``filename`` in the format specified in ``file_format``.

    Args:
        edata: Election data to write.
        filename: The name of the file to write the election data to.
        file_format: The format to write the election data in. Defaults to "preflib".  The other options are "csv", "json", and "abif".
        csv_format: The format to write the election data in if the file format is "csv".  Defaults to 'candidate_columns'.  The other option is ``rank_columns``.

    Returns:
        The name of the file the election data was written to.
       
    Note: 
        There are two formats for the csv file: "rank_columns" and "candidate_columns".  The "rank_columns" format is used when the csv file contains a column for each rank and the rows are the candidates at that rank (or "skipped" if the ranked is skipped).  The "candidate_columns" format is used when the csv file contains a column for each candidate and the rows are the rank of the candidates (or the empty string if the candidate is not ranked).

    """

    if file_format == 'preflib':
        return write_preflib(edata, filename)
    elif file_format == 'csv':
        return write_csv(edata, filename, csv_format=csv_format)
    elif file_format == 'json':
        return write_json(edata, filename)
    elif file_format == 'abif':
        return write_abif(edata, filename)
    else:
        raise ValueError(f"File format {file_format} not recognized.")