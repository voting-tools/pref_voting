Generate Profiles
=======================================

# Sampling Profiles

We use the [prefsampling](https://comsoc-community.github.io/prefsampling/index.html) package to sample profiles from standard probability functions. There is a single function, ``generate_profile``, to interface with the prefsampling functions.  The following are the available probability models for generating profiles: 

1. Impartial Culture Model: generate a profile by sampling from a uniform distribution over profiles with $n$ candidates and $m$ voters, where each voter is equally likely to have any of the $n!$ linear orders on the candidates.

```{eval-rst}

.. exec_code:: 

    from pref_voting.generate_profiles import generate_profile
    
    # the following all generate a profile with 3 candidates and 5 voters
    # using the Impartial Culture model: 

    prof = generate_profile(3, 5) # the default is the IC model
    prof = generate_profile(3, 5, probmodel="IC")
    prof = generate_profile(3, 5, probmodel="impartial")

```

2. Impartial Anonymous Culture: generate a profile by sampling from a uniform distribution over anonymous profiles.

```{eval-rst}

.. exec_code:: 

    from pref_voting.generate_profiles import generate_profile
    
    # the following all generate a profile with 3 candidates and 5 voters
    # using the Impartial Anonymous Culture model: 

    prof = generate_profile(3, 5, probmodel="IAC")
    prof = generate_profile(3, 5, probmodel="impartial_anonymous")

```

3. The Urn model: In the Polya-Eggenberger urn model, to generate a profile given a parameter $\alpha\in [0,\infty)$, each voter in turn randomly draws a linear order from an urn. Initially the urn is the set of all linear orders on the $n$ candidates. If a voter randomly chooses $L$ from the urn, we return $L$ to the urn plus $\alpha n!$ copies of $L$. IC is the special case where $\alpha=0$. The Impartial Anonymous Culture (IAC) is the special case where $\alpha=1/n!$. 

```{eval-rst}

.. exec_code:: 

    from pref_voting.generate_profiles import generate_profile
    
    # the following all generate a profile with 3 candidates and 5 voters
    # using the Urn model: 
    
    # default is URN with alpha=0.0, which is equivalent to the IC model.
    prof = generate_profile(3, 5, probmodel="urn") 
    prof = generate_profile(3, 5, probmodel="URN", alpha=5)
    prof = generate_profile(3, 5, probmodel="urn", alpha=5)

    # in addition, there is a pre-defined urn models

    prof = generate_profile(3, 5, probmodel="URN-10") # URN with alpha=10

```

4. Urn models where the $\alpha$ parameter depends on the number of candidates: "URN-0.3" is the urn model with $\alpha=n!*0.3$ where $n$ is the number of candidates; and  "URN-R" is the random URN model, where, following Boehmer et al. ("Putting a compass on the map of elections," IJCAI-21), for each generated profile, we chose $\alpha$ according to a Gamma distribution with shape parameter $k=0.8$ and scale parameter $\theta=1$. 

```{eval-rst}

.. exec_code:: 

    from pref_voting.generate_profiles import generate_profile
            
    # the following all generate a profile with 3 candidates and 5 voters
    # using the Urn model: 

    prof = generate_profile(3, 5, probmodel="URN-0.3") 
    prof = generate_profile(3, 5, probmodel="URN-R") 

```    

5. The Mallow's model: generate a profile by fixing a reference linear ordering of the candidates (called the **central_vote**) and assign to each voter a ranking that is "close" to this reference ranking.  Closeness to the reference ranking is defined using the Kendall-tau distance between rankings, depending on a *dispersion* parameter $\phi$.  Setting $\phi= 0$ means that every voter is assigned the reference ranking, and setting $\phi=1$ is equivalent to the IC model. Formally, to generate a profile given a reference ranking $L_0\in\mathcal{L}(X)$ and $\phi\in (0,1]$, the probability that a voter's ballot is $L\in\mathcal{L}(X)$ is $Pr_{L_0,\phi}(L)=\phi^{\tau(L,L_0)}/C$ where $\tau(L,L_0)= {{|X|}\choose{2}} - |L\cap L_0|$, the Kendell-tau distance of $L$ to $L_0$, and $C$ is a normalization constant. 

```{eval-rst}

.. exec_code:: 

    from pref_voting.generate_profiles import generate_profile
            
    # the following all generate a profile with 3 candidates and 5 voters
    # using the Mallow's model: 

    # default is phi = 1.0, which is equivalent to the IC model.
    prof = generate_profile(3, 5, probmodel="MALLOWS") 
    prof = generate_profile(3, 5, probmodel="mallows") 

    prof = generate_profile(3, 5, probmodel="mallows", phi=0.5) 

    # in addition, you can fix the central ranking

    prof = generate_profile(3, 5, 
                            probmodel="mallows", 
                            central_vote=[2, 1, 0],
                            phi=0.5) 

```    


7. The Mallow's model with a parameter that [Boehmer et al. (2021)](https://arxiv.org/abs/2105.07815) call rel-$\phi$, which together with the number of candidates determines the $\phi$ value. 

```{eval-rst}

.. exec_code:: 

    from pref_voting.generate_profiles import generate_profile
            
    # the following all generate a profile with 3 candidates and 5 voters
    # using the Mallow's model: 

    # default is relphi is generated randomly from the interval (0,1).
    prof = generate_profile(3, 5, probmodel="MALLOWS-RELPHI") 

    prof = generate_profile(3, 5, probmodel="MALLOWS-RELPHI", relphi=0.5) 

    # the following are pre-defined Mallow's models with relphi=0.375
    prof = generate_profile(3, 5, probmodel="MALLOWS-RELPHI-0.375") 

    # the relphi parameter is chosen uniformly from the interval (0,1)
    prof = generate_profile(3, 5, probmodel="MALLOWS-RELPHI-R") 


```

8. The Mallow's model with the phi parameter normalized as discussed in [Boehmer, Faliszewski and Kraiczy (2023)](https://proceedings.mlr.press/v202/boehmer23b.html). 

```{eval-rst}

.. exec_code:: 

    from pref_voting.generate_profiles import generate_profile
            
    # the following all generate a profile with 3 candidates and 5 voters
    # using the Mallow's model: 

    prof = generate_profile(3, 5, 
                            probmodel="mallows", 
                            phi=0.5, 
                            normalise_phi=True) 

```

9. SinglePeaked - a profile $\mathbf{P}$ is *single peaked* if there exists a strict linear order $<$ of $X(\mathbf{P})$ such that for every $i\in V(\mathbf{P})$ and $x,y\in X(\mathbf{P})$, $x<y < max(\mathbf{P}_i)$ implies  $y\mathbf{P}_ix$, and $max(\mathbf{P}_i)< x<y$ implies $x\mathbf{P}_iy$. The probability model called *single peaked* assigns zero probability to any profile that is not single peaked and equal probability to any two single-peaked profiles.   

```{eval-rst}

.. exec_code:: 

    from pref_voting.generate_profiles import generate_profile
            
    # the following all generate a profile with 3 candidates and 5 voters
    # using the Single Peaked probability model: 

    prof = generate_profile(3, 5, 
                            probmodel="SinglePeaked") 
    prof = generate_profile(3, 5, 
                            probmodel="single_peaked_walsh") 
    prof = generate_profile(3, 5, 
                            probmodel="single_peaked_conitzer") 

```

10. There are a number of variations of the Single Peaked model available.   

```{eval-rst}

.. exec_code:: 

    from pref_voting.generate_profiles import generate_profile
            
    # the following all generate a profile with 3 candidates and 5 voters
    # using the Single Crossing and Single Peaked Circle probability models: 

    prof = generate_profile(3, 5, 
                            probmodel="single_crossing") 
    prof = generate_profile(3, 5, 
                            probmodel="single_peaked_circle") 

```


11.  Euclidean - In the Euclidean model voters and candidates are assigned random positions in a Euclidean space. A voter then ranks the candidates according to their distance: their most preferred candidate is the closest one to them, etc.


```{eval-rst}

.. exec_code:: 

    from pref_voting.generate_profiles import generate_profile
            
    # the following all generate a profile with 3 candidates and 5 voters
    # using the Euclidean probability model.  The default is 2 dimensions 
    # in a uniform space.

    prof = generate_profile(3, 5, 
                            probmodel="euclidean") 

    # you can specify the number of dimensions and the type of space.

    prof = generate_profile(3, 5, 
                            probmodel="euclidean", 
                            dimensions=3, 
                            space="gaussian_ball")


    prof = generate_profile(3, 5, 
                            probmodel="euclidean", 
                            dimensions=3, 
                            space="gaussian_cube")

    prof = generate_profile(3, 5, 
                            probmodel="euclidean", 
                            dimensions=3, 
                            space="unbounded_gaussian")

    prof = generate_profile(3, 5, 
                            probmodel="euclidean", 
                            dimensions=3, 
                            space="uniform_ball")
    
    prof = generate_profile(3, 5, 
                            probmodel="euclidean", 
                            dimensions=3, 
                            space="uniform_cube")

    prof = generate_profile(3, 5, 
                            probmodel="euclidean", 
                            dimensions=3, 
                            space="uniform_sphere")

```


12.  Plackett-Luce - In the Plackett-Luce model, each candidate is assigned a value representing the candidate's "quality".  The voters rank the candidates according to their quality (the higher quality candidate has a higher chance of being ranked first, etc.). 


```{eval-rst}

.. exec_code:: 

    from pref_voting.generate_profiles import generate_profile
            
    # the following all generate a profile with 3 candidates and 5 voters
    # using the Plackett Luce probability model. 
    

    prof = generate_profile(3, 5, 
                            probmodel="plackett_luce",
                            alphas=[2, 1, 1]) 

```

13.  Direct Dirichlet - The Direct Dirichlet model is very similar to the Plackett-Luce model, but the quality of the candidates is drawn from a Dirichlet distribution. The key difference is that the higher the sum of the alphas, the more correlated the votes are (the more concentrated the Dirichlet distribution is). 


```{eval-rst}

.. exec_code:: 

    from pref_voting.generate_profiles import generate_profile
            
    # the following all generate a profile with 3 candidates and 5 voters
    # using the Didi probability model.  

    prof = generate_profile(3, 5, 
                            probmodel="didi",
                            alphas=[20, 10, 10]) 

```


14.  Stratification - In the Stratification model, the candidates are dividing into groups with the first group being ranked above the second group. The size of the top group is defined by the weight parameter (a number between 0 and 1). 


```{eval-rst}

.. exec_code:: 

    from pref_voting.generate_profiles import generate_profile
            
    # the following all generate a profile with 3 candidates and 5 voters
    # using the Stratification probability model.  

    prof = generate_profile(3, 5, 
                            probmodel="stratification",
                            weight=0.3) 

```

15. Groups of Voters - In the Groups of Voters model, the voters are divided into groups and each group ranks the candidates according to a different probability model.  The weights parameter specifies how likely each voter will be in each group.

```{eval-rst}

.. exec_code:: 

    from pref_voting.generate_profiles import generate_profile_with_groups
            
    # the following all generate a profile with 3 candidates and 5 voters
    # and two groups of voters drawn from  Mallow's models with different 
    # reference rankings.  

    prof = generate_profile_with_groups(4, 3,
                                        [{"probmodel":"mallows", 
                                          "central_vote":[0, 1, 2, 3],
                                          "phi":0.5}, 
                                         {"probmodel":"mallows",
                                          "central_vote":[3, 2, 1, 0], 
                                          "phi":0.5}],
                                        weights=[1, 2]) 

```


## Generate a Profile

```{eval-rst}
.. autofunction:: pref_voting.generate_profiles.get_rankings

```

```{eval-rst}
.. autofunction:: pref_voting.generate_profiles.generate_profile

```

## Generate a Profile with groups

```{eval-rst}
.. autofunction:: pref_voting.generate_profiles.generate_profile_with_groups

```

## Generate a Profile with truncated linear orders

```{eval-rst}
.. autofunction:: pref_voting.generate_profiles.generate_truncated_profile

```

## Generate a Profile for a given ordinal margin graph

```{eval-rst}
.. autofunction:: pref_voting.generate_profiles.minimal_profile_from_edge_order

```

# Enumerating profiles

## Enumerate anonymous profiles

```{eval-rst}
.. autofunction:: pref_voting.generate_profiles.enumerate_anon_profile

```

## Enumerate anonymous profiles with ties

```{eval-rst}
.. autofunction:: pref_voting.generate_profiles.enumerate_anon_profile_with_ties

```