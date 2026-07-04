"""
Interfaces to other voting software packages.

The modules in this subpackage convert between pref_voting election objects
and the election objects of other packages:

* ``pref_voting.interfaces.votekit_interface``: VoteKit (https://votekit.readthedocs.io)
* ``pref_voting.interfaces.abcvoting_interface``: abcvoting (https://abcvoting.readthedocs.io)

The external packages are optional dependencies: they are only imported when
one of the conversion functions is called.  Install them with, e.g.,
``uv pip install "pref_voting[votekit]"``, ``uv pip install "pref_voting[abcvoting]"``,
or ``uv pip install "pref_voting[interfaces]"`` for both.
"""
