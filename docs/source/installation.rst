Installation
================

``pref_voting`` is available on pypi and can be installed using pip (or `uv <https://docs.astral.sh/uv/>`_).

.. code-block:: shell

    pip3 install pref_voting

Alternatively, you can install the latest development version from source:

.. code-block:: shell

    git clone https://github.com/voting-tools/pref_voting
    pip3 install .

Requirements:

* The package requires Python 3.10 or higher and has been tested on Python 3.10 - 3.13.
* The following pip packages are required and installed automatically:  filelock, scipy, networkx, tabulate, matplotlib, nashpy, numba, ortools, pathos, preflibtools, prefsampling, and seaborn

.. warning::

    Since the package uses Numba, refer to the Numba documentation for the latest supported Python version: https://numba.readthedocs.io/en/stable/user/installing.html#version-support-information.

Optional dependencies
------------------------

The interfaces to the `VoteKit <https://votekit.readthedocs.io>`_ and `abcvoting <https://abcvoting.readthedocs.io>`_ packages (see :doc:`interfaces`) require optional dependencies that are not installed by default.  Install them with the corresponding extras:

.. code-block:: shell

    pip3 install "pref_voting[votekit]"      # VoteKit (requires Python >= 3.11)
    pip3 install "pref_voting[abcvoting]"    # abcvoting
    pip3 install "pref_voting[interfaces]"   # both
