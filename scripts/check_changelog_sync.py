#!/usr/bin/env python3
"""
Structural sync check between CHANGELOG.md and docs/source/changelog.rst.

CHANGELOG.md (Markdown) and docs/source/changelog.rst (RST) are maintained
BY HAND, in parallel, bullet-for-bullet. This script does not regenerate
changelog.rst -- it only checks that the two files haven't drifted, and
fails CI with a specific pointer to where they disagree so a human (or an
AI session) can reconcile them manually. There is no auto-fix mode.

Why not full-text diffing, and why not a generic Markdown->RST converter:
changelog.rst carries RST-only enhancements that do not exist in, and
cannot be mechanically derived from, CHANGELOG.md's Markdown:
  - `.. warning::` admonitions wrapping specific callouts within a bullet.
  - `:ref:`...`` / `:doc:`...`` Sphinx cross-references.
  - Double-backtick ``code`` spans (RST's inline-code syntax) where the
    Markdown source uses single-backtick `code`.
A byte-for-byte or full-prose diff would flag all of these as "drift" on
every single sync, even when the files are correctly in sync by the
convention this project actually uses. So this check instead verifies
STRUCTURE, which is the part that must never legitimately differ:

WHAT IS CHECKED (must match exactly, or CI fails):
  1. The sequence of version headings (`## [X.Y.Z]` in the .md vs the
     matching underlined heading in the .rst), in the same order, with
     the same text (brackets aside -- "[0.10.0]" and "0.10.0" are treated
     as equivalent).
  2. Within each version, the sequence of subsection headings
     (`### BREAKING CHANGES` / `### Changed` / `### Added`, etc.), in the
     same order, with the same text.
  3. Within each subsection, the number of top-level bullets, and the
     number of nested (sub-)bullets.
  4. For each top-level bullet, a normalized "fingerprint" of its first
     4 words (RST double-backtick code spans folded to single-backtick,
     `:ref:`/`:doc:` roles unwrapped to their visible text, `.. warning::`/
     `.. _label:` directive markers stripped) must match between the two
     files. Every bullet in this changelog opens with a **bolded key
     phrase** before any admonition/cross-reference/prose divergence can
     occur, so 4 words is enough to confirm "this is the same bullet, in
     the same position" without being so wide a window that it starts
     tripping on legitimate line-wrap artifacts or the deliberate prose
     rewording described below.

WHAT IS DELIBERATELY NOT CHECKED (free to differ, will never fail CI):
  - Everything past a bullet's first 4 normalized words: prose wording,
    `.. warning::` admonitions, `:ref:`/`:doc:` cross-references, and any
    other RST-only elaboration are all invisible to this check. In
    practice this project's changelog.rst sometimes rewords a bullet's
    back half entirely (e.g. citing a `:doc:` cross-reference instead of
    spelling out a file path the way CHANGELOG.md does) -- that is
    accepted, existing style, not drift, which is why the check window
    stops at the bolded lead-in rather than trying to diff full prose.
  - The exact wording/content of nested bullets (only their COUNT per
    subsection is checked, not their text).
  - Inline formatting choices beyond the single/double-backtick
    normalization above (e.g. line-wrap width, trailing whitespace).

WHEN THIS CHECK FAILS: it means a version, a subsection, a whole bullet,
or a bullet's opening wording was added/removed/reworded in one file but
not the other. Fix it by hand in docs/source/changelog.rst: mirror the
new/changed CHANGELOG.md entries using the existing RST bullets as a
style template, and re-add any `.. warning::`/cross-reference embellishing
you judge worthwhile for the new content -- exactly how every prior entry
in that file was written.
"""
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = REPO_ROOT / "CHANGELOG.md"
RST_PATH = REPO_ROOT / "docs" / "source" / "changelog.rst"

FINGERPRINT_WORDS = 4


@dataclass
class Subsection:
    name: str
    top_bullets: list = field(default_factory=list)  # normalized fingerprints
    nested_count: int = 0


@dataclass
class VersionSection:
    key: str
    subsections: list = field(default_factory=list)  # list[Subsection]


def normalize_text(s: str) -> str:
    s = re.sub(r"``(.+?)``", r"`\1`", s)  # RST double-backtick code -> single
    s = re.sub(r":(?:ref|doc):`([^`]+)`", r"\1", s)  # strip RST cross-ref roles
    s = re.sub(r"\.\.\s+warning::", " ", s)  # strip warning directive marker
    s = re.sub(r"\.\.\s+_[^:]+:", " ", s)  # strip RST internal target labels
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fingerprint(text: str) -> str:
    return " ".join(normalize_text(text).split()[:FINGERPRINT_WORDS])


def parse_markdown(path: Path) -> list:
    lines = path.read_text().splitlines()
    versions = []
    cur_version = None
    cur_sub = None
    cur_bullet_lines = []
    cur_bullet_is_top = None

    def flush_bullet():
        nonlocal cur_bullet_lines, cur_bullet_is_top
        if cur_sub is not None and cur_bullet_lines:
            text = " ".join(cur_bullet_lines)
            if cur_bullet_is_top:
                cur_sub.top_bullets.append(fingerprint(text))
            else:
                cur_sub.nested_count += 1
        cur_bullet_lines = []
        cur_bullet_is_top = None

    for line in lines:
        m_version = re.match(r"^## \[(.+?)\](.*)$", line)
        m_sub = re.match(r"^### (.+)$", line)
        m_nested = re.match(r"^  - (.*)$", line)
        m_top = re.match(r"^- (.*)$", line)

        if m_version:
            flush_bullet()
            key = f"{m_version.group(1)}{m_version.group(2)}".strip()
            cur_version = VersionSection(key=key)
            versions.append(cur_version)
            cur_sub = None
            continue
        if m_sub:
            flush_bullet()
            cur_sub = Subsection(name=m_sub.group(1).strip())
            if cur_version is not None:
                cur_version.subsections.append(cur_sub)
            continue
        if m_nested:
            flush_bullet()
            cur_bullet_lines = [m_nested.group(1)]
            cur_bullet_is_top = False
            continue
        if m_top:
            flush_bullet()
            cur_bullet_lines = [m_top.group(1)]
            cur_bullet_is_top = True
            continue

        stripped = line.strip()
        if stripped and cur_bullet_lines:
            cur_bullet_lines.append(stripped)

    flush_bullet()
    return versions


def parse_rst(path: Path) -> list:
    lines = path.read_text().splitlines()
    versions = []
    cur_version = None
    cur_sub = None
    cur_bullet_lines = []
    cur_bullet_is_top = None

    def flush_bullet():
        nonlocal cur_bullet_lines, cur_bullet_is_top
        if cur_sub is not None and cur_bullet_lines:
            text = " ".join(cur_bullet_lines)
            if cur_bullet_is_top:
                cur_sub.top_bullets.append(fingerprint(text))
            else:
                cur_sub.nested_count += 1
        cur_bullet_lines = []
        cur_bullet_is_top = None

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        nxt = lines[i + 1].strip() if i + 1 < n else ""
        title = line.strip()

        if title and re.fullmatch(r"-{3,}", nxt) and len(nxt) >= len(title):
            flush_bullet()
            cur_version = VersionSection(key=title)
            versions.append(cur_version)
            cur_sub = None
            i += 2
            continue
        if title and re.fullmatch(r"~{3,}", nxt) and len(nxt) >= len(title):
            flush_bullet()
            cur_sub = Subsection(name=title)
            if cur_version is not None:
                cur_version.subsections.append(cur_sub)
            i += 2
            continue

        m_nested = re.match(r"^  \* (.*)$", line)
        m_top = re.match(r"^\* (.*)$", line)
        if m_nested:
            flush_bullet()
            cur_bullet_lines = [m_nested.group(1)]
            cur_bullet_is_top = False
            i += 1
            continue
        if m_top:
            flush_bullet()
            cur_bullet_lines = [m_top.group(1)]
            cur_bullet_is_top = True
            i += 1
            continue

        stripped = line.strip()
        if stripped and cur_bullet_lines:
            cur_bullet_lines.append(stripped)
        i += 1

    flush_bullet()
    return versions


def compare(md_versions: list, rst_versions: list) -> list:
    errors = []
    md_keys = [v.key for v in md_versions]
    rst_keys = [v.key for v in rst_versions]
    if md_keys != rst_keys:
        errors.append(
            "Version heading sequence differs:\n"
            f"  CHANGELOG.md:  {md_keys}\n"
            f"  changelog.rst: {rst_keys}"
        )
        return errors

    for md_v, rst_v in zip(md_versions, rst_versions):
        md_sub_names = [s.name for s in md_v.subsections]
        rst_sub_names = [s.name for s in rst_v.subsections]
        if md_sub_names != rst_sub_names:
            errors.append(
                f"[{md_v.key}] subsection sequence differs:\n"
                f"  CHANGELOG.md:  {md_sub_names}\n"
                f"  changelog.rst: {rst_sub_names}"
            )
            continue

        for md_s, rst_s in zip(md_v.subsections, rst_v.subsections):
            if len(md_s.top_bullets) != len(rst_s.top_bullets):
                errors.append(
                    f"[{md_v.key}] / {md_s.name}: top-level bullet count differs "
                    f"(CHANGELOG.md={len(md_s.top_bullets)}, changelog.rst={len(rst_s.top_bullets)})"
                )
                continue
            if md_s.nested_count != rst_s.nested_count:
                errors.append(
                    f"[{md_v.key}] / {md_s.name}: nested bullet count differs "
                    f"(CHANGELOG.md={md_s.nested_count}, changelog.rst={rst_s.nested_count})"
                )
            for idx, (md_fp, rst_fp) in enumerate(zip(md_s.top_bullets, rst_s.top_bullets), start=1):
                if md_fp != rst_fp:
                    errors.append(
                        f"[{md_v.key}] / {md_s.name}, bullet #{idx}: opening text diverges\n"
                        f"  CHANGELOG.md:  {md_fp!r}\n"
                        f"  changelog.rst: {rst_fp!r}"
                    )
    return errors


def main() -> int:
    md_versions = parse_markdown(MD_PATH)
    rst_versions = parse_rst(RST_PATH)
    errors = compare(md_versions, rst_versions)

    if errors:
        print("CHANGELOG.md and docs/source/changelog.rst have drifted out of sync:\n")
        for e in errors:
            print(f"- {e}\n")
        print(
            "These two files are kept in structural sync by hand. See the module "
            "docstring of scripts/check_changelog_sync.py for exactly what is and "
            "isn't checked. To fix: update docs/source/changelog.rst to mirror the "
            "new/changed entries in CHANGELOG.md, re-adding its RST-only "
            "enhancements (.. warning:: admonitions, :ref:/:doc: cross-references, "
            "double-backtick code spans) by hand, the same way every prior entry "
            "in that file was written."
        )
        return 1

    print("CHANGELOG.md and docs/source/changelog.rst are in structural sync.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
