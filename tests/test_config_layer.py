"""
Tests for the configuration layer (`pyfc/config.py`, `pyfc/generate_config.py`).

Two groups, in this order: structural sweeps that check the layer's parameters line up
across all four places they are written down, and behavioural tests of the two things the
layer actually promises to do -- resolve Defaults -> JSON -> CLI in that order, and
validate interactive input before accepting it.

WHY THIS FILE EXISTS: the first coverage measurement of PyFC reported `config.py` at 10%
and `generate_config.py` at 6% -- effectively nothing was testing the CLI/JSON config
layer at all.  That layer is nonetheless edited on almost every release (the `n_restarts`,
`neighbor_seeding` and `scipy_method` wiring all landed in v0.10.0), and each time its
correctness was established by running the CLI by hand and reading the output.  Two real
bugs were found that way that a test should have caught: `scipy_method` reaching
`parse_arguments` but never being added to the interactive wizard, and the wizard writing
a default that disagreed with the one `compute_fc_intervals` actually uses.

The structural group is therefore deliberately SELF-DISCOVERING.  None of those tests
carries a hardcoded list of parameter names.  Each derives its subjects from the code --
from the base config dict, from the argparse parser's own actions, from the wizard's
output, and from `inspect.signature(compute_fc_intervals)` -- so a parameter added later is
swept automatically, without anyone having to remember to extend a list here.  A test that
had to be updated by hand every time a parameter was added would have been updated by hand
and would therefore never have caught either of the bugs above.

Every test in this file was verified to fail against a deliberately broken version of the
code before being kept: a test that raises coverage without being able to detect a defect
is worse than no test, because it makes the number look better while catching nothing.
"""

import argparse
import builtins
import inspect
import json
import os
import sys
from unittest.mock import patch

import pytest

from pyfc.config import generate_sample_config, parse_arguments
from pyfc.generate_config import main as wizard_main
from pyfc.orchestrator import compute_fc_intervals


# Argparse metadata flags: these control the CLI itself rather than naming an analysis
# parameter, and `parse_arguments` explicitly filters them out of the config it returns.
# This is the one list in this file that is hardcoded, because it is a statement about
# what is NOT a parameter, and it is enforced below rather than assumed: adding a flag
# here without it being genuinely meta will fail `test_meta_flags_are_really_meta`.
META_FLAGS = {"help", "config_file", "generate_config"}


def _build_parser_and_defaults():
    """
    Run `parse_arguments()` with an empty command line and return both the argparse
    parser it built and the config dict it resolved.

    `parse_arguments` constructs its parser as a local, so the only way to inspect it is
    to intercept the `parse_args` call.  Every flag uses `default=argparse.SUPPRESS`, so
    an empty argv leaves the returned config equal to the hardcoded base defaults, which
    is exactly what these tests want to compare against.
    """
    captured = {}
    real_parse_args = argparse.ArgumentParser.parse_args

    def spy(self, *args, **kwargs):
        captured["parser"] = self
        return real_parse_args(self, *args, **kwargs)

    with patch.object(argparse.ArgumentParser, "parse_args", spy):
        with patch.object(sys, "argv", ["pyfc"]):
            config = parse_arguments()

    return captured["parser"], config


def _cli_flag_dests(parser):
    """The `dest` of every flag the parser defines, meta flags removed."""
    return {action.dest for action in parser._actions} - META_FLAGS


def _run_wizard_with_all_defaults(tmp_path):
    """
    Drive `generate_config.main()` end to end, pressing Enter at every prompt, and return
    the JSON it wrote.

    Every prompt falls back to its default on empty input, including the final one asking
    where to save, which defaults to `fc_config.json` in the working directory -- hence
    the chdir into a temporary directory rather than passing a path.  That also means the
    default output path itself is under test: if it ever changed to something absolute or
    outside the cwd, this would stop finding the file.
    """
    prompts = []

    def fake_input(prompt=""):
        prompts.append(prompt)
        return ""

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        with patch.object(builtins, "input", fake_input):
            wizard_main()
        out = tmp_path / "fc_config.json"
        assert out.exists(), (
            "the wizard accepted every default but wrote no fc_config.json in the "
            "working directory"
        )
        with open(out) as f:
            written = json.load(f)
    finally:
        os.chdir(cwd)

    return written, prompts


@pytest.fixture(scope="module")
def parser_and_defaults():
    return _build_parser_and_defaults()


@pytest.fixture(scope="module")
def orchestrator_defaults():
    """The real default of every keyword parameter of `compute_fc_intervals`."""
    signature = inspect.signature(compute_fc_intervals)
    return {
        name: p.default
        for name, p in signature.parameters.items()
        if p.default is not inspect.Parameter.empty
    }


def test_meta_flags_are_really_meta(parser_and_defaults):
    """
    Guard on this file's own hardcoded list: every name in META_FLAGS must actually be a
    flag the parser defines and must genuinely be absent from the returned config.  Without
    this, a future parameter could be silenced from every other test in this file just by
    being added to META_FLAGS.
    """
    parser, config = parser_and_defaults
    all_dests = {action.dest for action in parser._actions}

    for flag in META_FLAGS:
        assert flag in all_dests, f"META_FLAGS lists '{flag}', which is not a CLI flag"
        assert flag not in config, (
            f"'{flag}' is treated as a meta flag here but parse_arguments() does put it "
            "in the config it returns, so it is a real parameter and must be tested "
            "like one"
        )


def test_every_config_key_has_a_cli_flag(parser_and_defaults):
    """
    Every parameter in the base config dict must be settable from the command line.

    This is the direction that catches a parameter being wired into the defaults and the
    orchestrator but left unreachable from the CLI -- the failure mode is silent, since
    the run simply uses the default and produces plausible output.
    """
    parser, config = parser_and_defaults
    missing = sorted(set(config) - _cli_flag_dests(parser))
    assert not missing, (
        f"config keys with no matching --flag in parse_arguments(): {missing}. "
        "Add a parser.add_argument(..., default=argparse.SUPPRESS) for each."
    )


def test_every_cli_flag_has_a_config_key(parser_and_defaults):
    """
    The reverse direction: a flag that does not correspond to a base-config key.

    Because every flag uses `default=argparse.SUPPRESS`, such a flag works when passed
    explicitly but has no documented default, so the config's meaning depends on whether
    the user happened to pass it.
    """
    parser, config = parser_and_defaults
    orphaned = sorted(_cli_flag_dests(parser) - set(config))
    assert not orphaned, (
        f"--flags with no matching key in the base config dict: {orphaned}. "
        "Add each to the hardcoded defaults in parse_arguments()."
    )


def test_every_config_key_is_accepted_by_compute_fc_intervals(
    parser_and_defaults, orchestrator_defaults
):
    """
    Every key the config layer emits must be a real parameter of `compute_fc_intervals`.

    The orchestrator is called by splatting the config in, so a key that is misspelled or
    left behind after a rename raises a TypeError at the very end of a long run -- or,
    worse, is silently dropped if the call site ever filters keys.
    """
    _, config = parser_and_defaults
    unknown = sorted(set(config) - set(orchestrator_defaults))
    assert not unknown, (
        f"config keys that compute_fc_intervals() does not accept: {unknown}"
    )


def test_sample_config_matches_parse_arguments_defaults(tmp_path):
    """
    `generate_sample_config()` and `parse_arguments()` each carry their own copy of the
    same defaults dict, duplicated verbatim.  They are currently identical; nothing but
    this test stops them drifting, and a drifted sample config is actively misleading,
    since it is the file users copy as their starting point.
    """
    out = tmp_path / "sample.json"
    generate_sample_config(str(out))
    with open(out) as f:
        sample = json.load(f)

    _, config = _build_parser_and_defaults()

    assert sample == config, (
        "the sample config written by generate_sample_config() has drifted from the "
        "hardcoded defaults in parse_arguments(); they are duplicated copies of one dict "
        "and must be updated together"
    )


def test_wizard_covers_exactly_the_config_parameters(tmp_path, parser_and_defaults):
    """
    The interactive wizard must ask about every parameter the config layer knows, and must
    not invent keys of its own.

    This is the test that would have caught `scipy_method` reaching `parse_arguments` in
    v0.10.0 but never being added to the wizard: the wizard's output silently lacked the
    key, so a config generated by it used a different optimizer method than one generated
    by `--generate_config`.
    """
    written, _ = _run_wizard_with_all_defaults(tmp_path)
    _, config = parser_and_defaults

    missing = sorted(set(config) - set(written))
    extra = sorted(set(written) - set(config))

    assert not missing, (
        f"parameters the wizard never asks about: {missing}. A config generated by "
        "`python -m pyfc.generate_config` will silently lack them."
    )
    assert not extra, (
        f"keys the wizard writes that the config layer does not know: {extra}"
    )


def test_wizard_defaults_match_orchestrator_defaults(tmp_path, orchestrator_defaults):
    """
    Pressing Enter at every wizard prompt must produce the same behaviour as calling
    `compute_fc_intervals` with no arguments at all.

    A wizard whose "default" differs from the library's default is a trap: the user
    accepts what is presented as the default and gets something else.  That exact
    mismatch was a real bug fixed in v0.10.0.

    Three keys are exempt, and the exemption is narrow and directional: for `cl`,
    `param_names` and `output_file` the orchestrator's default is `None`, meaning "decide
    later" or "do not save", which is not a usable answer for a config file that has to be
    written to disk and re-read.  The wizard therefore substitutes a concrete value.  The
    assertion below still requires the orchestrator's side of each exemption to actually
    be `None`, so if one of them ever gains a real default the exemption stops applying
    and this test starts checking it like any other key.
    """
    written, _ = _run_wizard_with_all_defaults(tmp_path)

    concrete_substitutions = {"cl", "param_names", "output_file"}

    mismatches = []
    for key, value in sorted(written.items()):
        if key not in orchestrator_defaults:
            continue
        expected = orchestrator_defaults[key]
        if key in concrete_substitutions:
            assert expected is None, (
                f"'{key}' is exempted here only because compute_fc_intervals defaults it "
                f"to None, but it now defaults to {expected!r}; remove the exemption and "
                "make the wizard agree with it"
            )
            continue
        if value != expected:
            mismatches.append(f"  {key}: wizard={value!r} orchestrator={expected!r}")

    assert not mismatches, (
        "the wizard's defaults disagree with compute_fc_intervals' real defaults, so "
        "accepting every prompt does not reproduce the library's own behaviour:\n"
        + "\n".join(mismatches)
    )


def test_wizard_prompts_are_uniquely_numbered(tmp_path, parser_and_defaults):
    """
    The wizard's prompts are hand-numbered ("1. Likelihood Type?", "2. ...").  Adding a
    parameter mid-list without renumbering is easy to do and produces a visibly broken
    CLI, and duplicate or skipped numbers are exactly the artifact left behind.
    """
    _, prompts = _run_wizard_with_all_defaults(tmp_path)

    numbers = []
    for prompt in prompts:
        head = prompt.strip().split(".", 1)[0]
        assert head.isdigit(), f"wizard prompt is not numbered: {prompt.strip()[:60]!r}"
        numbers.append(int(head))

    assert numbers == list(range(1, len(numbers) + 1)), (
        f"wizard prompt numbers are not a gapless 1..N sequence: {numbers}"
    )


# ---------------------------------------------------------------------------
# Behavioural tests: the precedence chain, and interactive input validation.
#
# `config.py`'s module docstring states the layer's contract as "Hardcoded Defaults ->
# JSON Configuration File -> Command Line Arguments", each overriding the one before.
# That ordering is the entire purpose of the module and nothing exercised it; the tests
# above sweep the parameter *names*, but a precedence chain wired backwards would pass
# every one of them while silently ignoring the user's config file.
# ---------------------------------------------------------------------------


def _parse_with_argv(argv):
    with patch.object(sys, "argv", ["pyfc"] + argv):
        return parse_arguments()


def test_json_config_file_overrides_hardcoded_defaults(tmp_path):
    """A value present in the JSON file must beat the hardcoded default."""
    cfg = tmp_path / "user.json"
    cfg.write_text(json.dumps({"n_toys": 4242, "strategy": "grid"}))

    config = _parse_with_argv(["--config_file", str(cfg)])

    assert config["n_toys"] == 4242
    assert config["strategy"] == "grid"
    # Keys absent from the JSON must keep their hardcoded default rather than vanishing.
    assert config["toy_batch_size"] == 200


def test_cli_argument_overrides_json_config_file(tmp_path):
    """
    The top of the precedence chain: an explicit flag must beat the same key in the JSON
    file.  This is the direction that breaks if the two update steps are ever reordered,
    and the symptom -- a `--flag` on the command line quietly doing nothing because a
    stale config file wins -- is close to impossible to diagnose from the output.
    """
    cfg = tmp_path / "user.json"
    cfg.write_text(json.dumps({"n_toys": 4242, "verbose": 2}))

    config = _parse_with_argv(["--config_file", str(cfg), "--n_toys", "7"])

    assert config["n_toys"] == 7, "the CLI flag lost to the JSON config file"
    # The JSON's other keys must survive the CLI overlay.
    assert config["verbose"] == 2


def test_missing_config_file_warns_and_falls_back_to_defaults(tmp_path, capsys):
    """A nonexistent `--config_file` is a warning and a fallback, not a crash."""
    config = _parse_with_argv(["--config_file", str(tmp_path / "nope.json")])

    assert "not found" in capsys.readouterr().out
    assert config["n_toys"] == 500


def test_generate_config_flag_writes_a_sample_and_exits(tmp_path):
    """
    `--generate_config` writes the sample file and terminates rather than continuing into
    a run.  Note the sample lands at `../config/example_fc_config.json`, i.e. *outside*
    the working directory, which is why this test runs from a nested scratch directory.
    """
    workdir = tmp_path / "run"
    workdir.mkdir()
    (tmp_path / "config").mkdir()

    cwd = os.getcwd()
    try:
        os.chdir(workdir)
        with pytest.raises(SystemExit) as excinfo:
            _parse_with_argv(["--generate_config"])
    finally:
        os.chdir(cwd)

    assert excinfo.value.code == 0
    written = tmp_path / "config" / "example_fc_config.json"
    assert written.exists(), "--generate_config exited without writing the sample file"
    with open(written) as f:
        assert json.load(f)["likelihood_type"] == "binned"


@pytest.mark.parametrize("raw", ["y", "Y", "yes", "t", "true", "1", " TRUE "])
def test_parse_bool_accepts_documented_true_spellings(raw):
    from pyfc.generate_config import parse_bool

    assert parse_bool(raw) is True


@pytest.mark.parametrize("raw", ["n", "N", "no", "f", "false", "0", " False "])
def test_parse_bool_accepts_documented_false_spellings(raw):
    from pyfc.generate_config import parse_bool

    assert parse_bool(raw) is False


def test_parse_bool_rejects_anything_else():
    from pyfc.generate_config import parse_bool

    with pytest.raises(ValueError):
        parse_bool("maybe")


def test_parse_bool_passes_through_real_booleans():
    """The wizard's boolean defaults are supplied as strings, but callers may pass a
    genuine bool; casting it again must not go through the string table."""
    from pyfc.generate_config import parse_bool

    assert parse_bool(True) is True
    assert parse_bool(False) is False


def test_wizard_maps_its_sentinels_when_given_real_answers(tmp_path):
    """
    Two wizard answers are sentinels that must be translated before they reach the
    orchestrator: `scipy_method="auto"` means "let the optimizer decide" and `num_cores=0`
    means "use every core", and both are written to JSON as `null`.  The all-defaults run
    above takes the translating branch of each; this one takes the other branch, where the
    user supplies a real value that must survive *untranslated*.

    Answers are selected by matching the prompt text rather than by position, so inserting
    or reordering a question does not silently shift them onto the wrong parameters.
    """
    answers = {
        "scipy.optimize.minimize": "SLSQP",
        "CPU cores": "4",
        "Path to save": str(tmp_path / "nested" / "deeper" / "cfg.json"),
    }

    def fake_input(prompt=""):
        matched = [v for k, v in answers.items() if k in prompt]
        assert len(matched) <= 1, f"ambiguous answer key for prompt: {prompt[:60]!r}"
        return matched[0] if matched else ""

    with patch.object(builtins, "input", fake_input):
        wizard_main()

    out = tmp_path / "nested" / "deeper" / "cfg.json"
    assert out.exists(), "the wizard did not create the parent directories it was given"
    with open(out) as f:
        written = json.load(f)

    assert written["scipy_method"] == "SLSQP", "an explicit method was overwritten"
    assert written["num_cores"] == 4, "an explicit core count was overwritten"


def test_wizard_reports_write_failures_instead_of_crashing(tmp_path, capsys):
    """
    A wizard that raised here would lose every answer the user just typed.  It catches the
    failure and says so.  The unwritable destination is a path under an existing *file*,
    which cannot be turned into a directory.
    """
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")

    def fake_input(prompt=""):
        return str(blocker / "cfg.json") if "Path to save" in prompt else ""

    with patch.object(builtins, "input", fake_input):
        wizard_main()

    assert "ERROR" in capsys.readouterr().out


def test_parse_helpers_pass_through_existing_lists():
    """
    Both list parsers accept an already-parsed list unchanged.  That path is what makes
    the wizard's list-valued defaults work, since a default may be supplied either as the
    raw comma-separated string a user would type or as a real list.
    """
    from pyfc.generate_config import parse_float_list, parse_str_list

    assert parse_float_list([0.68, 0.9]) == [0.68, 0.9]
    assert parse_str_list(["a", "b"]) == ["a", "b"]
    assert parse_float_list("0.68, 0.90") == [0.68, 0.90]
    assert parse_str_list(" a , b ") == ["a", "b"]


def test_get_input_rejects_then_accepts(capsys):
    """
    `get_input` must loop rather than propagate on bad input, and must apply all three
    rejection rules: a cast failure, a value outside `choices`, and a failed `validator`.

    This is the wizard's only defence against a statistically meaningless configuration --
    a negative toy count, or a confidence level outside (0, 1) -- and a `continue` that
    became a `return` would accept the bad value silently while still looking interactive.
    """
    from pyfc.generate_config import get_input

    supplied = iter(["not-a-number", "-5", "250"])
    with patch.object(builtins, "input", lambda _prompt="": next(supplied)):
        value = get_input(
            "toys?", default_val=500, cast_func=int,
            validator=lambda x: x > 0, error_msg="must be positive",
        )

    assert value == 250, "get_input returned before the first valid entry"
    out = capsys.readouterr().out
    assert "must be positive" in out, "the validator rejection was not reported"

    supplied = iter(["nonsense", "binned"])
    with patch.object(builtins, "input", lambda _prompt="": next(supplied)):
        value = get_input("type?", default_val="binned", cast_func=str,
                          choices=["binned", "unbinned"])

    assert value == "binned"
    assert "not in the allowed list" in capsys.readouterr().out


def test_get_input_returns_default_on_empty_input():
    """Pressing Enter yields the default, cast through the same function as real input."""
    from pyfc.generate_config import get_input

    with patch.object(builtins, "input", lambda _prompt="": ""):
        assert get_input("toys?", default_val=500, cast_func=int) == 500


class _Bail(BaseException):
    """Escape hatch for the EOF tests below.

    Deliberately a BaseException: `get_input`'s catch-all is `except Exception`, so this
    passes straight through it. That is what lets an unfixed `get_input` fail this test
    quickly instead of spinning forever, which would hang the whole suite rather than
    report a failure.
    """


def _eof_then_bail(calls, limit=3):
    """An `input` that always hits EOF, but gives up after `limit` retries."""
    def _fake(prompt=""):
        calls.append(prompt)
        if len(calls) > limit:
            raise _Bail("get_input retried after EOF instead of stopping")
        raise EOFError
    return _fake


def test_get_input_stops_at_end_of_input_instead_of_retrying_forever():
    """
    `input()` raises EOFError when stdin is closed or exhausted. Nothing can be retried
    at that point -- no further input will ever arrive -- so `get_input` must stop.

    Before this was fixed, the bare `except Exception` swallowed EOFError and `while True`
    retried immediately, at full CPU, forever: `pyfc-config < /dev/null` produced ~7.4
    million prompts in 5 seconds, and one such process was found still running after
    nearly three hours. That makes the wizard unusable in CI, in cron, in a container
    without a TTY, and behind any pipe whose input runs out.

    The retry count is the real assertion here. Exiting is necessary but not sufficient --
    what matters is that EOF is not treated as a retryable error.
    """
    from pyfc.generate_config import get_input

    calls = []
    with patch.object(builtins, "input", _eof_then_bail(calls)):
        with pytest.raises(SystemExit) as excinfo:
            get_input("toys?", default_val=500, cast_func=int)

    assert excinfo.value.code != 0, "reaching end of input is a failure, not a clean exit"
    assert len(calls) == 1, (
        f"get_input called input() {len(calls)} times after EOF; it must not retry, "
        "because stdin cannot produce anything new"
    )


def test_get_input_does_not_silently_accept_the_default_at_end_of_input():
    """
    The tempting alternative fix -- treat EOF as "just press Enter" -- would be worse than
    the hang it replaces. It would write a configuration file the user never saw, let
    alone approved, and a wizard that fabricates answers when nobody is listening is a
    quieter failure than one that spins.
    """
    from pyfc.generate_config import get_input

    calls = []
    with patch.object(builtins, "input", _eof_then_bail(calls)):
        with pytest.raises(SystemExit):
            get_input("toys?", default_val=500, cast_func=int)


def test_wizard_exits_rather_than_hanging_when_stdin_is_exhausted(tmp_path):
    """
    The same thing one level up, through `main()`: a pipe that runs out mid-run ends the
    wizard instead of wedging it. This is the shape of the real-world failure -- a script
    that pipes a few answers and then closes -- and it must not write a partial config.
    """
    calls = []
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        with patch.object(builtins, "input", _eof_then_bail(calls, limit=5)):
            with pytest.raises(SystemExit):
                wizard_main()
    finally:
        os.chdir(cwd)

    assert not (tmp_path / "fc_config.json").exists(), (
        "the wizard wrote a config file despite never receiving an answer"
    )
