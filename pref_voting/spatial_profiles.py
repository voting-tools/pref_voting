import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
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
        cand_types (dict): A dictionary mapping each candidate to their type (e.g., party affiliation).

    """
    def __init__(self, cand_pos, voter_pos, candidate_types=None):

        cand_dims = [len(v) for v in cand_pos.values()]
        voter_dims = [len(v) for v in voter_pos.values()]

        assert len(cand_dims) > 0, "There must be at least one candidate."
        assert len(set(cand_dims)) == 1, "All candidate positions must have the same number of dimensions."
        assert len(voter_dims) > 0, "There must be at least one voter."
        assert len(set(voter_dims)) == 1, "All voter positions must have the same number of dimensions."
        assert cand_dims[0] == voter_dims[0], "Candidate and voter positions must have the same number of dimensions."

        self.candidates = sorted(list(cand_pos.keys())) 
        self.voters = sorted(list(voter_pos.keys())) 
        self.cand_pos = cand_pos
        self.voter_pos = voter_pos
        self.num_dims = len(list(cand_pos.values())[0]) 
        self.candidate_types = candidate_types or {c:'unknown' for c in self.candidates}

    @property
    def num_cands(self): 
        """
        Returns the number of candidates in the profile.
        """
        return len(self.candidates)

    @property
    def num_voters(self):
        """
        Returns the number of voters in the profile.
        """
        return len(self.voters) 
    
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
    
    def candidate_type(self, c):
        """
        Given a candidate c, returns their type.
        """
        return self.candidate_types[c]

    def set_candidate_types(self, cand_types): 
        """
        Sets the types of each candidate.
        """

        assert set(cand_types.keys()) == set(self.candidates), "The candidate types must be specified for all candidates."

        self.candidate_types = cand_types

    def to_utility_profile(self, 
                           utility_function = None,
                           uncertainty_function=None,
                           batch=False, 
                           return_virtual_cand_positions=False):
        """
        Returns a utility profile corresponding to the spatial profile.  
        
        Args:
            utility_function (callable, optional): A function that takes two vectors and returns a float. The default utility function is the quadratic utility function.
            uncertainty_function (callable, optional): A function that models uncertainty and returns covariance parameters.
            batch (bool, optional): If True, generate positions in batches. Default is False.
            return_virtual_cand_positions (bool, optional): If True, return virtual candidate positions. Default is False.
            
        Returns:    
            UtilityProfile: A utility profile corresponding to the spatial profile.
            (optional) Tuple[UtilityProfile, dict]: The utility profile and virtual candidate positions if `return_virtual_cand_positions` is True.
        """
        import numpy as np
        from pref_voting.generate_spatial_profiles import generate_covariance

        utility_function = utility_function or quadratic_utility

        if uncertainty_function is not None:
            virtual_cand_positions = {}
            for c in self.candidates:
                if batch:
                    covariance = generate_covariance(self.num_dims, *uncertainty_function(self, c, self.voters[0]))
                    positions = np.random.multivariate_normal(self.candidate_position(c), covariance, size=len(self.voters))
                else:
                    positions = [np.random.multivariate_normal(self.candidate_position(c), generate_covariance(self.num_dims, *uncertainty_function(self, c, v))) for v in self.voters]
                virtual_cand_positions[c] = positions
            
            utility_profile = [
                {c: utility_function(self.voter_position(v), virtual_cand_positions[c][vidx]) for c in self.candidates}
                for vidx, v in enumerate(self.voters)
            ]
            
            if return_virtual_cand_positions:
                return UtilityProfile(utility_profile), virtual_cand_positions
            else:
                return UtilityProfile(utility_profile)
        else:
            utility_profile = [
                {c: utility_function(np.array(self.voter_position(v)), np.array(self.candidate_position(c))) for c in self.candidates}
                for v in self.voters
            ]
            return UtilityProfile(utility_profile)
    
    def add_candidate(self, candidate_positions, add_multiple_candidates = False): 
        """
        Add a candidate to the spatial profile. 

        Args: 
            candidate_positions (list): A list of candidate positions
        """

        if add_multiple_candidates: 
            assert all([len(pos) == self.num_dims for pos in candidate_positions]), f"Candidates positions ({candidate_positions}) must be the same dimension as the profile dimension ({self.num_dims})"

            starting_cand_name = self.num_cands

            for c_pos in candidate_positions: 
                self.cand_pos[starting_cand_name] = c_pos
                starting_cand_name += 1

        elif not add_multiple_candidates: 

            assert  len(candidate_positions) == self.num_dims, f"Candidates position ({candidate_positions}) must be the same dimension as the profile dimension ({self.num_dims})"

            starting_cand_name = self.num_cands
            self.cand_pos[starting_cand_name] = candidate_positions

        self.candidates = sorted(list(self.cand_pos.keys())) 

    def move_candidate(self, cand, new_cand_pos): 
        """
        Move cand to a new position
        """

        assert len(new_cand_pos) == self.num_dims, f"The new position {new_cand_pos} must be the same as the profile dimension: {self.num_dims}"

        assert cand in self.candidates, f"Candidate {cand} is not in the profile."

        self.cand_pos[cand] = new_cand_pos


    def to_string(self): 
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

    def view(self, show_cand_labels=False, show_voter_labels=False, bin_width=None, dpi=150):
        """
        Displays the spatial model in a 1D, 2D, or 3D plot.
        
        Args:
            show_cand_labels (optional, bool): If True, displays the labels of each candidate. The default is False.
            show_voter_labels (optional, bool): If True, displays the labels of each voter. The default is False.
                Note: In 1D visualizations, voter labels are disabled regardless of this setting.
            bin_width (optional, float): Width of bins for grouping voters in 1D visualization. If None, a suitable width is calculated.
            dpi (optional, int): Resolution in dots per inch. Default is 150.
        """
        assert self.num_dims <= 3, "Can only view profiles with 1, 2, or 3 dimensions"
        sns.set_theme(style="darkgrid")
        
        # Define the candidate color consistently across all dimensions
        candidate_color = "red"
        
        if self.num_dims == 1:
            # Get all voter positions
            voter_positions = [self.voter_position(v)[0] for v in self.voters]
            
            # Calculate histogram data
            if bin_width is None:
                # Auto-calculate a reasonable bin width based on data range
                position_range = max(voter_positions) - min(voter_positions)
                bin_width = max(position_range / 20, 0.05)
            
            # Create histogram data - exact counts
            bins = {}
            for pos in voter_positions:
                # Round to nearest bin
                binned_pos = round(pos / bin_width) * bin_width
                bins[binned_pos] = bins.get(binned_pos, 0) + 1
            
            # Get the bin positions and counts
            bin_positions = list(bins.keys())
            bin_counts = list(bins.values())
            max_count = max(bin_counts) if bin_counts else 1
            
            # Create figure with sufficient space for labels and high resolution
            fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
            
            # Calculate the space needed for candidates at the top
            candidate_area_height = max_count * 0.3
            
            # Plot the bars for voters
            bars = ax.bar(
                bin_positions,
                bin_counts,
                width=bin_width*0.8, 
                alpha=0.6,
                color="blue",
                label="Voters"
            )
            
            # Calculate position for candidates at the top
            top_line_y = max_count + 0.2
            candidate_y_pos = top_line_y + candidate_area_height * 0.3
            candidate_positions = [self.candidate_position(c)[0] for c in self.candidates]
            
            # Plot candidates above the histogram
            cand_scatter = ax.scatter(candidate_positions, [candidate_y_pos] * len(self.candidates), 
                    color=candidate_color, marker='X', s=100, zorder=5)
            
            # Create a custom legend box
            legend = ax.legend([cand_scatter, bars], ['Candidates', 'Voters'], 
                            loc='upper right', 
                            bbox_to_anchor=(0.99, 0.99))
            
            # Draw the figure to get legend position for label placement
            fig.canvas.draw()
            
            # Get legend position for detecting label overlap
            if legend:
                legend_bbox = legend.get_window_extent().transformed(ax.transData.inverted())
                legend_left = legend_bbox.x0
                legend_right = legend_bbox.x1
                legend_width = legend_right - legend_left
            else:
                legend_left = float('inf')
                legend_right = float('inf')
                legend_width = 0
            
            if show_cand_labels:
                # Add labels to each candidate
                for c in self.candidates:
                    pos = self.candidate_position(c)[0]
                    
                    # Buffer to detect legend overlap
                    legend_buffer = 0.05 * legend_width
                    
                    # Check if candidate is under or near the legend
                    if (pos >= legend_left - legend_buffer) and (pos <= legend_right + legend_buffer):
                        # Place label below the marker to avoid legend overlap
                        ax.annotate(c, (pos, candidate_y_pos), xytext=(0, -25), 
                                textcoords='offset points', ha='center', va='top',
                                fontsize=13, fontweight='bold', color=candidate_color)
                    else:
                        # Otherwise place it above
                        ax.annotate(c, (pos, candidate_y_pos), xytext=(0, 15), 
                                textcoords='offset points', ha='center', va='bottom',
                                fontsize=13, fontweight='bold', color=candidate_color)
            
            # Set axis labels
            ax.set_xlabel('Position')
            ax.set_ylabel('Number of voters')
            
            # Configure y-axis to show integers for the histogram area
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            
            # Set y-limits to include the candidate area
            ax.set_ylim(0, top_line_y + candidate_area_height)
            
            # Hide y-ticks in the candidate area
            yticks = [t for t in ax.get_yticks() if t <= max_count]
            ax.set_yticks(yticks)
            
            # Adjust the figure layout
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            
            plt.show()
        
        elif self.num_dims == 2:
    
            fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
            
            # Get voter positions
            x_voters = [self.voter_position(v)[0] for v in self.voters]
            y_voters = [self.voter_position(v)[1] for v in self.voters]
            
            # Plot voters with semi-transparency to visualize density
            voter_scatter = ax.scatter(x_voters, y_voters, 
                                    color="blue", alpha=0.2,
                                    edgecolor="black", linewidth=0.3,
                                    s=30,
                                    label="Voters")
            
            # Plot candidates
            x_cand = [self.candidate_position(c)[0] for c in self.candidates]
            y_cand = [self.candidate_position(c)[1] for c in self.candidates]
            cand_scatter = ax.scatter(x_cand, y_cand, 
                                    color=candidate_color, marker='X', s=100,
                                    edgecolor="white", linewidth=0.7,
                                    zorder=5,
                                    label="Candidates")
            
            # Create a legend
            ax.legend([cand_scatter, voter_scatter], ['Candidates', 'Voters'], loc='upper right')
            
            if show_cand_labels:
                for c in self.candidates:
                    pos = self.candidate_position(c)
                    text = ax.annotate(c, (pos[0], pos[1]), xytext=(0, 10), 
                            textcoords='offset points', ha='center', va='bottom',
                            fontsize=11, fontweight='bold', color=candidate_color)
                    
                    # Add white outline to text for better visibility
                    text.set_path_effects([
                        path_effects.Stroke(linewidth=1.5, foreground='white'),
                        path_effects.Normal()
                    ])
                    
            if show_voter_labels:
                for v in self.voters:
                    pos = self.voter_position(v)
                    ax.annotate(v + 1, (pos[0], pos[1]), fontsize=8)
                    
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            plt.tight_layout()
            plt.show()
        
        elif self.num_dims == 3:

            fig = plt.figure(figsize=(10, 6), dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')
            
            # Fetch all positions
            x_voters = [self.voter_position(v)[0] for v in self.voters]
            y_voters = [self.voter_position(v)[1] for v in self.voters]
            z_voters = [self.voter_position(v)[2] for v in self.voters]
            
            x_cand = [self.candidate_position(c)[0] for c in self.candidates]
            y_cand = [self.candidate_position(c)[1] for c in self.candidates]
            z_cand = [self.candidate_position(c)[2] for c in self.candidates]
            
            # Plot voters with high transparency for better visibility through clusters
            voter_scatter = ax.scatter(x_voters, y_voters, z_voters, 
                                    color="blue", alpha=0.1,
                                    edgecolor="black", linewidth=0.5,
                                    s=30,
                                    label="Voters")
            
            # Plot candidate markers with white outline for better visibility
            cand_scatter = ax.scatter(x_cand, y_cand, z_cand, 
                                    color=candidate_color, marker="X", s=40,
                                    edgecolor="white", linewidth=0.7,
                                    label="Candidates")
            
            # Add voter labels if requested
            if show_voter_labels:
                for v in self.voters:
                    pos = self.voter_position(v)
                    ax.text(pos[0], pos[1], pos[2], str(v + 1), fontsize=8)
            
            # Add candidate labels
            if show_cand_labels:
                for c in self.candidates:
                    pos = self.candidate_position(c)
                    text = ax.text(pos[0], pos[1], pos[2] + 0.15, c, 
                        fontsize=11, fontweight='bold', color=candidate_color,
                        ha='center', va='bottom')
                    
                    # Add white outline to text for better visibility
                    text.set_path_effects([
                        path_effects.Stroke(linewidth=1.5, foreground='white'),
                        path_effects.Normal()
                    ])
            
            # Create legend
            ax.legend([cand_scatter, voter_scatter], ['Candidates', 'Voters'], loc='upper right')
            
            # Set axis labels
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.set_zlabel('Dimension 3')
            
            plt.tight_layout()
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