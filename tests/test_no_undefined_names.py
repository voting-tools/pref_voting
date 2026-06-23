"""Guard against the "undefined name" class of bug (e.g. a symbol used but never
imported, or a method missing `self`).

Python resolves names at runtime, so such code imports fine and only raises a
NameError when the offending line actually executes -- which means a function that
no test calls can ship completely broken. A static pyflakes scan finds these
without running the code or needing a test per function.

This is the automated version of the manual scan that found bugs 5.1-5.4.
"""
import os

import pyflakes.api
import pyflakes.reporter
import pytest

import pref_voting


PACKAGE_DIR = os.path.dirname(pref_voting.__file__)


class _UndefinedNameCollector(pyflakes.reporter.Reporter):
    """A pyflakes reporter that records only undefined-name messages."""

    def __init__(self):
        self.undefined = []

    def unexpectedError(self, filename, msg):
        # syntax-level problems are reported separately by other tooling
        pass

    def syntaxError(self, filename, msg, lineno, offset, text):
        pass

    def flake(self, message):
        from pyflakes.messages import UndefinedName

        if isinstance(message, UndefinedName):
            self.undefined.append(str(message))


def test_no_undefined_names_in_package():
    collector = _UndefinedNameCollector()
    for root, _dirs, files in os.walk(PACKAGE_DIR):
        for fname in files:
            if fname.endswith(".py"):
                pyflakes.api.checkPath(os.path.join(root, fname), collector)

    assert not collector.undefined, "Undefined names found:\n" + "\n".join(collector.undefined)
