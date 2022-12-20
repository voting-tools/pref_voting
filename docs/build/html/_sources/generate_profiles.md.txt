Generate Profiles
=======================================

We have included  the following probability models: 

1. IC:  Impartial Culture Model - generate a profile by sampling  from a uniform distribution over profiles with $n$ candidates and $m$ voters.   This is the default probaiblity that used to generate a profile. 


3. URN: Urn model -  In the Polya-Eggenberger urn model, to generate a profile given a parameter $\alpha\in [0,\infty)$, each voter in turn randomly draws a linear order from an urn. Initially the urn is the set of all linear orders on the $n$ candidates. If a voter randomly chooses $L$ from the urn, we return $L$ to the urn plus $\alpha n!$ copies of $L$. IC is the special case where $\alpha=0$. The Impartial Anonymous Culture (IAC) is the special case where $\alpha=1/n!$. 

    * URN-$\alpha$ is the URN model with parameter $\alpha$.  
    * URN-R is the random URN model, where, following Boehmer et al. (2021), for each generated profile, we chose $\alpha$ according to a Gamma distribution with shape parameter $k=0.8$ and scale parameter $\theta=1$ for the model we call URN. 

4. MALLOWS, MALLOWS-RELPHI, MALLOWS_2REFS: In the Mallow's model, to generate a profile, the main idea is to fix a reference  linear ordering of the candidates and to assign to each voter a ranking that is "close" to this reference ranking.   Closeness to the reference ranking is defined using the Kendall-tau distance between rankings, depending on a  *dispersion* parameter $\phi$.   Setting $\phi= 0$ means that every voter is assigned the reference ranking, and setting $\phi=1$ is equivalent to the IC model. Formally, to generate a profile given a reference ranking $L_0\in\mathcal{L}(X)$ and $\phi\in (0,1]$, the probability that a voter's ballot is $L\in\mathcal{L}(X)$ is $Pr_{L_0,\phi}(L)=\phi^{\tau(L,L_0)}/C$ where $\tau(L,L_0)= {{|X|}\choose{2}} - |L\cap L_0|$, the Kendell-tau distance of $L$ to $L_0$, and $C$ is a normalization constant. In addition to generating profiles using a single reference ranking $L_0$, we considered generating profiles using two references rankings, which are the reverse of each other (called the MALLOWS_2REF model). E.g., $L_0$ ranks candidates from more liberal to more conservative, while $L_0^{-1}$ ranks candidates in the opposite order. The set of voters is divided into two groups, each associated with one of the reference rankings. Each voter is equally likely to be assigned to either of the two groups. Formally, the probability that a voter's ballot is $L$ is $\frac{1}{2} Pr_{L_0,\phi}(L)+\frac{1}{2}Pr_{L_0^{-1},\phi}(L)$. 

    * MALLOWS-$\phi$ is the MALLOWS model with parameter $\phi$
    * MALLOWS-RELPHI-$\phi$ is the MALLOWS model with a parameter that Boehmer et al. (2021) call rel-$\phi$, which together with the number of candidates determines the $\phi$ value. 
    * MALLOWS-R is the MALLOWS model where $\phi$ is chosen uniformly from the interval $(0,1)$ for each profile. 
    * MALLOWS_2REF-$\phi$ is the MALLOWS_2REF model with parameter $\phi$
    * MALLOWS_2REF-RELPHI-$\phi$ is the MALLOWS_2REF model with a parameter that Boehmer et al. (2021) call rel-$\phi$, which together with the number of candidates determines the $\phi$ value. 
    * MALLOWS_2REF-R is the MALLOWS_2REF model where $\phi$ is chosen uniformly from the interval $(0,1)$ for each profile. 


5. SinglePeaked - a profile $\mathbf{P}$ is *single peaked* if there exists a strict linear order $<$ of $X(\mathbf{P})$ such that for every $i\in V(\mathbf{P})$ and $x,y\in X(\mathbf{P})$, $x<y < max(\mathbf{P}_i)$ implies  $y\mathbf{P}_ix$, and $max(\mathbf{P}_i)< x<y$ implies $x\mathbf{P}_iy$. The probability model we call *single peaked* assigns zero probability to any profile that is not single peaked and equal probability to any two single-peaked profiles. 

```{eval-rst}

.. exec_code:: 

    from pref_voting.generate_profiles import generate_profile # function to generate a profile

    # dictionary where keys are the names of the probability models and 
    # values is a dictionary with two keys: "func" giving the function to generate a profile
    # and "param" giving the default parameter for the profile

    from pref_voting.generate_profiles import prob_models 

    num_cands = 3
    num_voters = 5

    for pm in prob_models.keys():
        print(f"Profile generated using the {pm} probability model")
        prof = generate_profile(num_cands, num_voters, probmod=pm)
        prof.display()
        print("\n")

```


## Generate a Profile

```{eval-rst}
.. autofunction:: pref_voting.generate_profiles.generate_profile

```

## Generate a Truncated ProfileWithTies

```{eval-rst}
.. autofunction:: pref_voting.generate_profiles.generate_truncated_profile

```


