"""
    File: readers.py
    Author: Wes Holliday (wesholliday@berkeley.edu) and Eric Pacuit (epacuit@umd.edu)
    Date: March 17, 2024

    Functions to write election data to a file.
"""

from pref_voting.profiles import Profile
from pref_voting.profiles_with_ties import ProfileWithTies
from pref_voting.spatial_profiles import SpatialProfile
from preflibtools.instances import OrdinalInstance
import os
import csv
import pandas as pd
import json

def abif_to_profile(filename):
    """
    Open filename in the abif format and return a Profile object.

    Args:
        filename: The name of the file to read the profile from.
    
    Returns:
        A Profile object.

    """

    with open(filename, mode='r') as file:
        
        lines = list(file.readlines())

        cmap = {}
        cand_to_indices = {}
        cindx = 0
        # create a candidate map
        for line in lines: 
            if line.startswith("="):
                _, cname = line[1:].strip().split(":")
                cmap[cindx] = cname.strip().strip("[]")
                cand_to_indices[cname.strip().strip("[]")] = cindx
                cindx += 1

        rankings = []
        rcounts = []
        for line in lines:
            if line.startswith("#"):
                # comment
                continue
            elif line.startswith("="):
                # candidate line
                continue
            elif line.startswith("{"):
                # metadata
                continue
            else:
                # ranking line
                count, ranking = line.strip().split(":")
                count = int(count)
                ranking = ranking.split(">")

                assert not any(["=" in cs or "," in cs for cs in ranking]), "The election must contain linear orders on the candidates to create a Profile."
                
                if len(cmap) == 0:
                    # no candidate map provided, so need to create one from the rankings
                    cmap = {cidx: str(sorted(ranking)[cidx].strip()) for cidx in range(len(sorted(ranking)))}
                    cand_to_indices = {c: i for i, c in cmap.items()}
                
                r = list()
                assert len(cmap) > 0 and len(ranking) == len(cmap), "The election must contain linear orders on the candidates to create a Profile."
                for c in ranking:
                    assert len(cmap) > 0 and c in cand_to_indices.keys(), "Candidate found that is not in the candidate map."
                    r.append(cand_to_indices[c.strip()])
                rankings.append(r)
                rcounts.append(count)

        return Profile(
            rankings, 
            rcounts=rcounts, 
            cmap=cmap)           


def abif_to_profile_with_ties(filename, cand_type=None):
    """
    Open filename in the abif format and return a ProfileWithTies object.

    Args:
        filename: The name of the file to read the profile from.
    
    Returns:
        A ProfileWithTies object.

    """
    
    import re

    with open(filename, mode='r') as file:
        lines = list(file.readlines())
        rankings = []
        rcounts = []
        cmap = {}
        for line in lines:
            if line.startswith("#"):
                # comment
                continue
            elif line.startswith("="):
                # candidate line
                cidx, cname = line[1:].strip().split(":")
                cmap[cand_type(cidx.strip()) 
                    if cand_type is not None else cidx.strip()] = cname.strip().strip("[]")
            elif line.startswith("{"):
                # metadata
                continue
            else:
                # ranking line
                count, ranking = line.strip().split(":")
                count = int(count)
                ranking = ranking.split(">")
                r = dict()
                for ridx, cs in enumerate(ranking):
                    cands = re.split(r'[=,]', cs)
                    for c in cands: 
                        if cand_type is not None: 
                            r[cand_type(c.strip())] = ridx + 1
                        else:
                            r[c.strip()] = ridx + 1
                rankings.append(r)
                rcounts.append(count)

        if len(cmap) == 0:
            return ProfileWithTies(
                rankings, 
                rcounts=rcounts)           
        else:
            return ProfileWithTies(
                rankings, 
                rcounts=rcounts, 
                candidates = sorted(list(cmap.keys())),
                cmap = cmap)           

def preflib_to_profile(
        instance_or_preflib_file, 
        include_cmap=False,
        use_cand_names=False,
        as_linear_profile=False): 
    
    """
    Read a profile from an OrdinalInstance or a .soc, .soi, .toc, or .toi file used by PrefLib (https://www.preflib.org/format#types).

    This function uses the ``OrdinalInstance`` class from the ``preflibtools`` package to read the profile from the file (see https://preflib.github.io/preflibtools/usage.html#ordinal-preferences).

    Args:
        preflib_file (str): the path to the file
        include_cmap (bool): if True, then include the candidate map.  Defaults to False.
        use_cand_names (bool): if True, then use the candidate map as the candidate names.  Defaults to False.
        as_linear_profile (bool): if True, then return a Profile object.  Defaults to False.  If False, then return a ProfileWithTies object.

    Returns:    
        Profile or ProfileWithTies: the profile read from the file
        
    """

    assert type(instance_or_preflib_file) == OrdinalInstance or type(instance_or_preflib_file) == str, "The argument must be an instance of OrdinalInstance or a string."

    if type(instance_or_preflib_file) == str:
        preflib_file = instance_or_preflib_file

        assert preflib_file.endswith(".soc") or preflib_file.endswith(".soi") or preflib_file.endswith(".toc") or preflib_file.endswith(".toi"), f"The file must be one of the file types from preflib: https://www.preflib.org/format#types, not {preflib_file}."

        assert os.path.exists(preflib_file), f"The file {preflib_file} does not exist."

        instance = OrdinalInstance()
        instance.parse_file(preflib_file)

    else:
        instance = instance_or_preflib_file

    rankings = []
    rcounts = []
    cmap = {c:str(c) for c in range(instance.num_alternatives)}

    if not as_linear_profile:

        for order in instance.orders:
            rank = dict()
            for r,cs in enumerate(order): 
                for c in cs: 
                    if not use_cand_names:
                        rank[c] = r + 1
                    else: 
                        rank[instance.alternatives_name[c]] = r + 1
                    if include_cmap:
                        if  use_cand_names:
                            cmap[instance.alternatives_name[c]] = instance.alternatives_name[c]
                        else:
                            cmap[c] = instance.alternatives_name[c]

            rankings.append(rank)
            rcounts.append(instance.multiplicity[order])

        return ProfileWithTies(rankings, 
                       rcounts=rcounts,
                       cmap=cmap)

    elif as_linear_profile: 
        
        cand_to_cidx = {c:cidx 
                        for cidx,c in enumerate(sorted(list(instance.alternatives_name.keys())))}

        for order in instance.orders:    
            rank = list()
            cmap = {c:str(c) for c in range(instance.num_alternatives)}
            for _,cs in enumerate(order): 
                for c in cs: 
                    rank.append(cand_to_cidx[c])
                    if include_cmap:
                        cmap[cand_to_cidx[c]] = instance.alternatives_name[c]
            rankings.append(rank)
            rcounts.append(instance.multiplicity[order])

        return Profile(rankings, 
                       rcounts=rcounts,
                       cmap=cmap)

def csv_to_profile(
        filename, 
        csv_format="candidate_columns", 
        as_linear_profile=False,
        items_to_skip=None, 
        cand_type=None):
        """
        Read a profile from a csv file. 

        Args:
            filename (str): the path to the file
            csv_format (str): the format of the csv file.  Defaults to "candidate_columns".  The other option is "rank_columns".
            as_linear_profile (bool): if True, then return a Profile object.  Defaults to False.  If False, then return a ProfileWithTies object.
            items_to_skip (list[str]): a list of items to skip.  Defaults to None.  Items in this list are not included in the profile.  Only relevant for "rank_columns" csv format.

        Returns:
            Profile or ProfileWithTies: the profile read from the file

        Note: 
            There are two formats for the csv file: "rank_columns" and "candidate_columns".  The "rank_columns" format is used when the csv file contains a column for each rank and the rows are the candidates at that rank (or "skipped" if the ranked is skipped).  The "candidate_columns" format is used when the csv file contains a column for each candidate and the rows are the rank of the candidates (or the empty string if the candidate is not ranked).
        """
        
        if csv_format == "rank_columns":
            df = pd.read_csv(filename)
            items_to_skip = items_to_skip if items_to_skip is not None else ["skipped"]
            ranks = []
            rank_columns = [col for col in df.columns if col.startswith('rank') or col.startswith('Rank')]

            # Get unique values from these columns, excluding 'skipped'
            cand_names = pd.unique(df[rank_columns].values.ravel('K'))
            cand_names = [str(value) for value in cand_names if value not in items_to_skip]

            if 'writein' in cand_names:
                cands = list(set([c for c in sorted(cand_names) if c != 'writein'])) + ['writein']
            else: 
                cands = sorted(list(set(cand_names)))
            if len(cands) == 0: 
                print("No candidates found in file", filename)
            cmap = {cidx: c for cidx,c in enumerate(cands)}
            cand_to_cidx = {c:cidx for cidx,c in enumerate(cands)}

            rank_str_to_rank = lambda rank_str: int(rank_str[4:].strip())
            for _, row in df.iterrows():
                ballot_dict = {}
                for rank in rank_columns:
                    candidate = str(row[rank])
                    if candidate not in items_to_skip:
                        ballot_dict[cand_to_cidx[candidate]] = rank_str_to_rank(rank)
                        
                ballot_dict = {cand_type(c) if cand_type is not None else c:r 
                               for c,r in ballot_dict.items()}
                ranks.append(ballot_dict)
            cmap = {cand_to_cidx[c]:str(c) for c in cands}
            prof = ProfileWithTies(ranks, cmap=cmap)
            if as_linear_profile:
                prof = prof.to_linear_profile() 
                assert prof is not None, "The profile could not be converted to a Profile."
            return prof
        
        elif csv_format == "candidate_columns":             
            with open(filename, mode='r') as file:
                reader = csv.reader(file)
                header = next(reader)
                candidates = header[:-1]
                rankings = list()
                rcounts = list()
                for row in reader:
                    ranks = [int(r) if r != "" else None for r in row[:-1]]
                    count = int(row[-1])
                    ranking = {cand_type(c) 
                               if cand_type is not None else c:r 
                               for c,r in zip(candidates, ranks) 
                               if r is not None}
                    rankings.append(ranking)
                    rcounts.append(count)

            prof = ProfileWithTies(rankings, 
                                   rcounts=rcounts, 
                                   cmap={cand_type(c) 
                                         if cand_type is not None else str(c):str(c) 
                                         for c in candidates})
            if as_linear_profile:
                prof = prof.to_linear_profile() 
                assert prof is not None, "The profile could not be converted to a Profile."
            return prof


# helper function for json_to_profile
def _convert_key_type(key, lst):
    for c in lst:
        try:
            # Attempt to convert the key to the same type as the candidate
            if type(c)(key) == c:
                return type(c)(key)
        except ValueError:
            continue
    # Return the original key if no conversion is successful
    return key


def json_to_profile(filename, cand_type=None, as_linear_profile=False): 
    """
    Read a profile from a json file. 

    Args:
        filename (str): the path to the file
        cand_type (type): the type of the candidates.  Defaults to None.  If not None, then the candidates are converted to this type.
        as_linear_profile (bool): if True, then return a Profile object.  Defaults to False.  If False, then return a ProfileWithTies object.

    Returns:
        Profile or ProfileWithTies: the profile read from the file
    """
    with open(filename, mode='r') as file:
        data = json.load(file)
        candidates = data["candidates"]
        cmap = {_convert_key_type(c, candidates): c_str for c, c_str in data["cmap"].items()}

        if cand_type is not None: 
            cmap = {cand_type(c):str(c_str) for c,c_str in cmap.items()}
            candidates = [cand_type(c) for c in candidates]

        rankings = []
        rcounts = []
        for r_data in data["rankings"]:
            rank = {cand_type(c) if cand_type is not None else _convert_key_type(c, candidates):int(r) for c,r in r_data["ranking"].items()}
            rankings.append(rank)
            rcounts.append(int(r_data["count"]))

    if as_linear_profile:
        prof = ProfileWithTies(rankings, 
                               rcounts=rcounts, 
                               candidates=candidates,
                               cmap=cmap)
        
        prof = prof.to_linear_profile() 
        assert prof is not None, "The profile could not be converted to a Profile."
    else:
        prof = ProfileWithTies(rankings, 
                               rcounts=rcounts,
                               candidates=candidates, 
                               cmap=cmap)
    return prof

def read(filename,
         file_format,
         as_linear_profile=False, 
         cand_type=None,
         csv_format="candidate_columns",
         items_to_skip=None): 
    """
    Read election data from ``filename`` in the format ``file_format``. 

    Args:
        filename (str): the path to the file
        file_format (str): the format of the file.  The options are "preflib", "json", "csv", and "abif".
        as_linear_profile (bool): if True, then return a Profile object.  Defaults to False.  If False, then return a ProfileWithTies object.
        cand_type (type): the type of the candidates.  Defaults to None.  If not None, then the candidates are converted to this type.
        csv_format (str): the format of the csv file.  Defaults to "candidate_columns".  The other option is "rank_columns".
        items_to_skip (list[str]): a list of items to skip.  Defaults to None.  Items in this list are not included in the profile.  Only relevant for "rank_columns" csv format.

    Returns:
        Profile or ProfileWithTies: the profile read from the file
    """
    if file_format == "abif":
        if as_linear_profile:
            return abif_to_profile(
                filename)
        else: 
            return abif_to_profile_with_ties(
                filename,
                cand_type=cand_type)
    elif file_format == "json":
        return json_to_profile(
            filename, 
            cand_type=cand_type, 
            as_linear_profile=as_linear_profile)
    elif file_format == "csv":
        return csv_to_profile(
            filename,
            as_linear_profile=as_linear_profile,
            cand_type=cand_type, 
            csv_format=csv_format,
            items_to_skip=items_to_skip)
    elif file_format == "preflib":
        return preflib_to_profile(filename, as_linear_profile=as_linear_profile)
    else:
        raise ValueError(f"File format {file_format} not recognized.")
    
def json_to_spatial_profile(filename): 
    """
    Load a spatial profile from a JSON file.

    Args:
        filename (str): the path to the file

    Returns:
        SpatialProfile: the spatial profile read from the file
    """

    with open(filename, "r") as f:
        spatial_profile_dict = json.load(f)
        candidates = spatial_profile_dict["cand_names"]
        voters = spatial_profile_dict["voter_names"]
        return SpatialProfile(
            {_convert_key_type(c, candidates):c_pos for c,c_pos in spatial_profile_dict["candidates"].items()}, 
            {_convert_key_type(v, voters):v_pos for v,v_pos in spatial_profile_dict["voters"].items()}
            )
