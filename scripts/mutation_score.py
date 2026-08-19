"""
Mutation score for the C++ engine.

A test count is not evidence. 733 tests could all pass against an engine that
returns zero. What a suite is worth is how much of the behaviour it PINS, and the
way to measure that is to break the behaviour on purpose and see whether anything
notices.

Python mutation tools mutate Python. The accounting, margin, settlement and spread
logic all live in the header-only C++ engine, so a Python score would measure the
thin translation layer and report a number about the wrong code. This harness
mutates the headers, rebuilds, and runs the suite.

A mutant is KILLED if the suite fails on it and SURVIVED if the suite still passes.
A mutant that does not COMPILE is neither: it is invalid, and it is excluded from the
denominator rather than counted as killed. Counting compile failures as kills is the
easiest way to report a score that flatters a suite -- most relational mutations in
template and constexpr context simply do not build, and the compiler rejecting them
says nothing about whether a test would have.

Survivors are the output that matters: each one is a specific change to the engine
that no test objects to.

Cost: a build is ~15 s and the suite ~20 s, so roughly 35 s per mutant. Run it
deliberately, not on every push.

    uv run python scripts/mutation_score.py                 # the default catalogue
    uv run python scripts/mutation_score.py --limit 20      # a quick sample
    uv run python scripts/mutation_score.py --target margin.h
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "engine"

# Files whose behaviour the suite is supposed to constrain. bindings.cpp is
# excluded: it is a translation layer, and mutating it mostly produces compile
# errors, which inflate the score without saying anything.
TARGETS = ("types.h", "position.h", "ledger.h", "contract.h",
           "margin.h", "spread.h", "engine.h", "dividend.h")


@dataclass(frozen=True)
class Operator:
    """One textual mutation, applied to whole-token matches only."""
    name: str
    pattern: str
    replacement: str

    def sites(self, text: str) -> list[re.Match]:
        return list(re.finditer(self.pattern, text))


# Deliberately conservative: each of these produces a plausible off-by-one or
# inverted-condition defect rather than obvious nonsense, which is what a suite
# should be expected to catch.
OPERATORS = (
    Operator("relational <  -> <=", r"(?<![<>=!])<(?![<=])", "<="),
    Operator("relational >  -> >=", r"(?<![<>=!-])>(?![>=])", ">="),
    Operator("relational <= -> <", r"<=", "<"),
    Operator("relational >= -> >", r">=", ">"),
    Operator("equality  == -> !=", r"==", "!="),
    Operator("logical  && -> ||", r"&&", "||"),
    Operator("logical  || -> &&", r"\|\|", "&&"),
    Operator("arithmetic + -> -", r"(?<![+\-=<>!*/])\+(?![+=])", "-"),
    Operator("arithmetic - -> +", r"(?<![+\-=<>!*/(,\s])-(?![-=>])", "+"),
    Operator("boolean true -> false", r"\btrue\b", "false"),
    Operator("boolean false -> true", r"\bfalse\b", "true"),
    Operator("min -> max", r"\bmin_money\b", "max_money"),
    Operator("max -> min", r"\bmax_money\b", "min_money"),
    Operator("std::min -> std::max", r"std::min\b", "std::max"),
    Operator("std::max -> std::min", r"std::max\b", "std::min"),
)

# Lines that are comment or preprocessor text: mutating them changes nothing
# observable and would only pad the denominator.
_SKIP = re.compile(r"^\s*(//|#|\*|/\*)")


@dataclass
class Mutant:
    file: str
    line: int
    operator: str
    before: str
    after: str
    # None until run; True killed, False survived. `invalid` is separate, because a
    # mutant that does not compile is not evidence either way.
    killed: bool | None = None
    invalid: bool = False
    how: str = ""


@dataclass
class Result:
    mutants: list[Mutant] = field(default_factory=list)

    @property
    def killed(self) -> list[Mutant]:
        return [m for m in self.mutants if m.killed and not m.invalid]

    @property
    def survived(self) -> list[Mutant]:
        return [m for m in self.mutants if m.killed is False and not m.invalid]

    @property
    def invalid(self) -> list[Mutant]:
        return [m for m in self.mutants if m.invalid]

    @property
    def score(self) -> float:
        scored = len(self.killed) + len(self.survived)
        return len(self.killed) / scored if scored else 0.0


def candidate_mutants(targets: tuple[str, ...]) -> list[tuple[Path, int, Operator, str]]:
    out = []
    for name in targets:
        path = ENGINE / name
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text().split("\n"), start=1):
            if not line.strip() or _SKIP.match(line):
                continue
            for op in OPERATORS:
                for _ in op.sites(line):
                    out.append((path, lineno, op, line))
                    break      # one mutant per (line, operator)
    return out


def apply_mutation(path: Path, lineno: int, op: Operator, backup: Path) -> tuple[str, str]:
    lines = path.read_text().split("\n")
    original = lines[lineno - 1]
    mutated = re.sub(op.pattern, op.replacement, original, count=1)
    lines[lineno - 1] = mutated
    shutil.copy(path, backup)
    path.write_text("\n".join(lines))
    return original.strip(), mutated.strip()


def build() -> bool:
    (ENGINE / "bindings.cpp").touch()
    done = subprocess.run([sys.executable, str(ENGINE / "build.py")],
                          cwd=ROOT, capture_output=True, check=False)
    return done.returncode == 0


def tests_pass() -> bool:
    # -x stops at the first failure: a mutant is killed by one test as surely as by
    # a hundred, and stopping early cuts the run time roughly in half.
    done = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "-x",
                           "--no-header", "-p", "no:cacheprovider"],
                          cwd=ROOT, capture_output=True, check=False)
    return done.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0,
                        help="sample this many mutants at random (0 = all)")
    parser.add_argument("--target", action="append", default=[],
                        help="restrict to one or more engine headers")
    parser.add_argument("--seed", type=int, default=7,
                        help="sampling seed, so a run is repeatable")
    args = parser.parse_args()

    targets = tuple(args.target) if args.target else TARGETS
    candidates = candidate_mutants(targets)
    if args.limit and args.limit < len(candidates):
        candidates = random.Random(args.seed).sample(candidates, args.limit)
    print(f"{len(candidates)} mutant(s) across {len(targets)} file(s). "
          f"~35 s each, so about {len(candidates) * 35 / 60:.0f} min.\n")

    # Establish that the suite passes BEFORE mutating, or every mutant would read
    # as killed and the score would be meaningless.
    if not build() or not tests_pass():
        print("FAIL: the suite does not pass unmutated. Nothing measured.")
        return 2

    result = Result()
    backup = Path(tempfile.mkdtemp()) / "backup.h"
    started = time.perf_counter()

    for i, (path, lineno, op, _line) in enumerate(candidates, start=1):
        before, after = apply_mutation(path, lineno, op, backup)
        mutant = Mutant(path.name, lineno, op.name, before, after)
        try:
            if not build():
                mutant.invalid, mutant.how = True, "did not compile"
            elif not tests_pass():
                mutant.killed, mutant.how = True, "tests failed"
            else:
                mutant.killed, mutant.how = False, "SURVIVED"
        finally:
            shutil.copy(backup, path)
        result.mutants.append(mutant)
        elapsed = time.perf_counter() - started
        print(f"  [{i}/{len(candidates)}] {path.name}:{lineno} {op.name:<24}"
              f"{mutant.how:<16} score {result.score:.0%}"
              f"  ({elapsed / i:.0f}s/mutant)")

    # Leave the tree as it was found.
    build()

    scored = len(result.killed) + len(result.survived)
    print(f"\nMutation score: {len(result.killed)}/{scored} = {result.score:.1%}"
          f"   ({len(result.invalid)} invalid mutant(s) excluded: they did not compile)")
    if result.survived:
        print(f"\n{len(result.survived)} survivor(s) -- each is a change to the engine "
              "no test objects to:")
        for m in result.survived:
            print(f"  {m.file}:{m.line}  [{m.operator}]")
            print(f"      was: {m.before}")
            print(f"      now: {m.after}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
