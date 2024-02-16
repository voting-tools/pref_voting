import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pref_voting.utility_functions import *
from pref_voting.utility_profiles import UtilityProfile

class SpatialProfile(object): 
    """
    A spatial profile is a set of candidates and voters in a multi-dimensional space.  Each voter and candidate is assigned vector of floats representing their position on each issue.

    Args:
        cand_pos (dict): A dictionary mapping each candidate to their position in the space.
        voter_pos (dict): A dictionary mapping each voter to their position in the space.

    Attributes:
        candidates (list): A list of candidates.
        voters (list): A list of voters.    
        cand_pos (dict): A dictionary mapping each candidate to their position in the space.    
        voter_pos (dict): A dictionary mapping each voter to their position in the space.   
        num_dims (int): The number of dimensions in the space.  

    """
    def __init__(self, cand_pos, voter_pos):

        self.candidates = sorted(list(cand_pos.keys())) 
        self.voters = sorted(list(voter_pos.keys())) 
        self.cand_pos = cand_pos
        self.voter_pos = voter_pos
        self.num_dims = len(list(cand_pos.values())[0])

    def voter_position(self, v): 
        """
        Given a voter v, returns their position in the space.
        """
        return self.voter_pos[v]
    
    def candidate_position(self, c):
        """
        Given a candidate c, returns their position in the space.
        """
        return self.cand_pos[c]
    
    def to_utility_profile(self, utility_function = None):
        """
        Returns a utility profile corresponding to the spatial profile.  
        
        Args:
            utility_function (optional, function): A function that takes two vectors and returns a float. The default utility function is the quadratic utility function.

        Returns:    
            UtilityProfile: A utility profile corresponding to the spatial profile.

        """
        utility_function = quadratic_utility if utility_function is None else utility_function 

        return UtilityProfile([
            {c:utility_function(self.voter_position(v), self.candidate_position(c)) 
             for c in self.candidates} for v in self.voters
            ])
    
    
    def write(self): 
        """
        Returns a string representation of the spatial profile.
        """

        sp_str = ''
        for c in self.candidates: 
            sp_str += f'C-{c}:{",".join([str(x) for x in self.candidate_position(c)])}_'
        for v in self.voters: 
            sp_str += f'V-{v}:{",".join([str(x) for x in self.voter_position(v)])}_'
        return sp_str[:-1]
    

    @classmethod
    def from_string(cls, sp_str): 
        """
        Returns a spatial profile described by ``sp_str``.

        ``sp_str`` must be in the format produced by the :meth:`pref_voting.SpatialProfile.write` function.
        """

        cand_positions = {}
        voter_positions = {}

        sp_data = sp_str.split('_')

        for d in sp_data: 
            if d.startswith("C-"): 
                cand,positions = d.split(':')
                cand_positions[int(cand[2:])] = np.array([float(x) for x in positions.split(',')])
            elif d.startswith("V-"):
                voter,positions = d.split(':')
                voter_positions[int(voter[2:])] = np.array([float(x) for x in positions.split(',')])

        return cls(cand_positions, voter_positions)

    def view(self, show_labels = False): 
        """ 
        Displays the spatial model in a 1D, 2D, or 3D plot.

        Args:
            show_labels (optional, bool): If True, displays the labels of each candidate and voter. The default is False.

        """
        assert self.num_dims <= 3, "Can only view profiles with 1, 2, or 3 dimensions"

        sns.set_theme(style="darkgrid")

        if self.num_dims == 1: 

            sns.scatterplot(x=[self.voter_position(v)[0] for v in self.voters], y=[1] * len(self.voters), color="blue", label="Voters")

            sns.scatterplot(x=[self.candidate_position(c)[0] for c in self.candidates], y=[1] * len(self.candidates), color="red", marker='X', label="Candidates")
            
            if show_labels:
                # Adding labels to each point
                for v in self.voters:
                    plt.annotate(v + 1, (self.voter_position(v)[0], 1))
                # Adding labels to each point
                for c in self.candidates:
                    plt.annotate(c, (self.candidate_position(c)[0], 1))

            plt.yticks([])  # this hides the y-axis
            plt.show()

        elif self.num_dims == 2:

            sns.scatterplot(x=[self.voter_position(v)[0] for v in self.voters], y=[self.voter_position(v)[1] for v in self.voters], color="blue", label="Voters")

            scatter = sns.scatterplot(x=[self.candidate_position(c)[0] for c in self.candidates], y=[self.candidate_position(c)[1] for c in self.candidates], color="red", marker='X', label="Candidates")

            if show_labels:

                # Adding labels to each point
                for v in self.voters:
                    plt.annotate(v + 1, (self.voter_position(v)[0], self.voter_position(v)[1]))
                for c in self.candidates:
                    plt.annotate(c, (self.candidate_position(c)[0], self.candidate_position(c)[1]))


            scatter.set(xlabel='Dimension 1', ylabel='Dimension 2')
            plt.legend()
            plt.show()
        elif self.num_dims == 3:

            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')

            x = [self.voter_position(v)[0] for v in self.voters]
            y = [self.voter_position(v)[1] for v in self.voters]
            z = [self.voter_position(v)[2] for v in self.voters]
            ax.scatter(x, y, z, color="blue", label="Voters")

            x = [self.candidate_position(c)[0] for c in self.candidates]
            y = [self.candidate_position(c)[1] for c in self.candidates]
            z = [self.candidate_position(c)[2] for c in self.candidates]
            ax.scatter(x, y, z, color="red", marker="X", label="Candidates")

            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')

            plt.legend()
            plt.show()

    
    def display(self): 
        """
        Displays the positions of each candidate and voter in the profile.

        """
        print("Candidates: ")
        for c in self.candidates: 
            print("Candidate ", c, " position: ", self.candidate_position(c))

        print("\nVoters: ")
        for v in self.voters:
            print("Voter ", v, " position: ", self.voter_position(v))
