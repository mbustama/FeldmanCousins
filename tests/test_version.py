"""
Tests that the version is written down exactly once and reaches everywhere from there.

WHY THIS FILE EXISTS: `docs/source/conf.py` hardcoded `release = '0.1.0'` while the package
was at 0.10.0. Every page of the published documentation advertised a version nine releases
out of date, and it went unnoticed for all nine, because nothing about a docs build fails
when a string is merely wrong. A second copy of a number that must match `pyproject.toml`
is a copy that will eventually disagree with it.

`pyproject.toml` is now the single source: `pyfc.__version__` reads the installed
distribution's metadata, and `conf.py` imports that rather than repeating the lookup. The
tests below pin all three links in that chain, because the failure mode is silent in every
one of them -- a stale version does not crash anything, it just misinforms.

The module docstrings' `Last modified: vX.Y.Z` markers are gone for the same reason. They
were hand-maintained, nothing could derive them, and they had already drifted: after 0.20.0
was prepared, seven of the nine still read v0.10.0, with no way for a reader to tell whether
that meant "genuinely unchanged since" or "somebody forgot to update it". Git records when a
file last changed, accurately and without anyone remembering to.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import pyfc

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _declared_version():
    """
    The version as written in `pyproject.toml`, read as text.

    Deliberately not parsed with `tomllib`: that is 3.11+, and this suite runs on 3.9
    upward. A regex anchored to a line-initial `version = ` is enough here, and is checked
    to match exactly once so it cannot silently pick up some other table's key.
    """
    text = PYPROJECT.read_text()
    matches = re.findall(r'^version = "([^"]+)"', text, flags=re.MULTILINE)
    assert len(matches) == 1, (
        f"expected exactly one line-initial `version = \"...\"` in pyproject.toml, "
        f"found {len(matches)}: {matches}"
    )
    return matches[0]


needs_source_tree = pytest.mark.skipif(
    not PYPROJECT.is_file(),
    reason="pyproject.toml is not present, so this is an installed copy rather than a "
           "source checkout and there is no declared version to compare against",
)


def test_package_exposes_a_version():
    """`pyfc.__version__` exists, is a string, and is exported."""
    assert isinstance(pyfc.__version__, str)
    assert pyfc.__version__, "__version__ is empty"
    assert "__version__" in pyfc.__all__, "__version__ is not exported in __all__"


def test_version_is_not_the_uninstalled_fallback():
    """
    The fallback means the distribution's metadata could not be found -- normally because
    the package is being used from a source tree that was never installed. If that value
    ever reaches a test run it means the lookup is broken, most likely because the
    DISTRIBUTION name (`PyFeldmanCousins`) was confused with the import package (`pyfc`),
    which raises PackageNotFoundError and lands here silently.
    """
    assert pyfc.__version__ != "0.0.0+unknown", (
        "pyfc.__version__ fell back to the unknown sentinel; the distribution metadata "
        "lookup is failing (check the distribution name, and that the package is installed)"
    )


@needs_source_tree
def test_version_matches_pyproject():
    """
    The version reported at runtime must equal the one declared in `pyproject.toml`.

    These come from genuinely different routes -- installed metadata versus the text of
    the source file -- so they can drift if the package is not reinstalled after a bump,
    which is the everyday case this catches.
    """
    assert pyfc.__version__ == _declared_version(), (
        f"pyfc.__version__ is {pyfc.__version__!r} but pyproject.toml declares "
        f"{_declared_version()!r}; reinstall the package, or the two have genuinely drifted"
    )


@needs_source_tree
def test_docs_take_their_version_from_the_package():
    """
    The specific regression this file exists for: `conf.py` must derive its version rather
    than carry a literal.

    Imported in a subprocess because `conf.py` is a module-level script that mutates
    `sys.path`, and importing it into this interpreter would leak that. The check is on the
    value it produces, not on how it is written, so it stays true if the mechanism changes
    -- what must not come back is a hardcoded string that nothing keeps honest.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, 'docs/source');"
         " import conf; print(conf.release); print(conf.version)"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"importing docs/source/conf.py failed:\n{result.stderr}"

    reported = [line for line in result.stdout.strip().splitlines() if line.strip()]
    release, doc_version = reported[-2], reported[-1]

    assert release == pyfc.__version__, (
        f"docs report version {release!r} but the package is {pyfc.__version__!r}; "
        "conf.py has drifted from the package again"
    )
    assert doc_version == pyfc.__version__, (
        f"conf.py's `version` is {doc_version!r}, not {pyfc.__version__!r}"
    )


@needs_source_tree
def test_no_module_hardcodes_a_version():
    """
    No module may reintroduce a hand-written version string in its docstring.

    Each one used to carry a `Last modified: vX.Y.Z` line. They were provenance rather than
    the package version, so nothing could derive them, and being hand-maintained they drifted
    exactly as a duplicated number always does -- seven of the nine still said v0.10.0 after
    0.20.0 was prepared, and no reader could tell whether that meant "unchanged since" or
    "somebody forgot". They were removed; `pyfc.__version__` is the one version any caller
    needs, and git already records when a file last changed, accurately and for free.

    This guards the removal rather than the old convention: the markers are gone, and this
    fails if one comes back.
    """
    pattern = re.compile(r"^(Last modified|Version):\s*v?\d+\.\d+", flags=re.MULTILINE)

    offenders = [
        f"  {module.name}: {pattern.search(module.read_text()).group(0)!r}"
        for module in sorted((REPO_ROOT / "pyfc").glob("*.py"))
        if pattern.search(module.read_text())
    ]

    assert not offenders, (
        "module docstrings carry hand-written version strings, which drift silently "
        "because nothing checks them:\n" + "\n".join(offenders)
        + "\n\nUse pyfc.__version__, which is read from the distribution metadata."
    )
