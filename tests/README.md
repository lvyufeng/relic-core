# The test suite

`tests/` holds two different things and pytest collects only one of them.

- **`test_*.py`** — the suite. Run it from the repository root, because there is no `conftest.py`
  and no pytest configuration: modules import from the root and nothing puts it on `sys.path`.
- **`bench_*.py`, `profile_*.py`** — measurement scripts. pytest does not collect them; they print
  kernel timings and are run by hand.

```bash
python -m pytest tests/ -q          # the whole suite
python -m pytest tests/test_x.py -q # one module
```

There is no `conftest.py` by design rather than by accident: a root `conftest.py` would change how
every module is imported, and several of these modules are run as scripts directly (see each
module's docstring) often enough that the two entry points have to agree. Shared fixtures are
defined in the module that uses them.

**The tests need the built extension.** They reach the kernels through
`relic_core.kernels.cuda_loader.load_cuda_kernel()`, which finds the `.so` under `build/extensions/`.
Build first, with the *same* interpreter and torch you run the tests under, or the loader silently
returns `None` and every test reads as a skip:

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

A skip here therefore has two quite different meanings — "this host has no card" and "the extension
was not built for this interpreter" — and only the second is a mistake.

## The baseline, and why it is a set

The failures that are known are recorded, one per line by node id, in
[`baseline_failures.txt`](baseline_failures.txt), and
[`scripts/check_test_baseline.py`](../scripts/check_test_baseline.py) is what diffs a run against it:

```bash
python scripts/check_test_baseline.py                 # run the suite, diff, exit 1 on anything new
python scripts/check_test_baseline.py --observed FILE # diff a run recorded earlier
python scripts/check_test_baseline.py --update        # rewrite the baseline from a fresh run
python scripts/check_test_baseline.py --fail-on-fixed # also fail when an entry stops failing
```

It is a **set of node ids and not a count**, and that distinction is the whole point. Three tests
fixed and one broken is a *net improvement* in a count and a regression in the tree, and reporting
that as "-2" hides the one that matters. It is also symmetric in the other direction: a test that
used to fail and now passes silently is reported on its own line, because a fixed test and a test
that has started skipping itself look identical from one run.

`--fail-on-fixed` is off by default because those two really are indistinguishable from here: this
suite skips itself for want of a card or a built extension, and a failure recorded on this host may
legitimately become a skip on another. Deciding otherwise is a decision about the host, so it is a
flag rather than the default.

The recorder itself, [`scripts/_baseline_recorder.py`](../scripts/_baseline_recorder.py), is loaded
with `-p` by that script and never by the suite, so a plain `pytest` run is unaffected by any of
this.

## Skips are not passes

A module that needs a GPU or a built extension skips itself — `pytest.skip`, `pytest.importorskip`
or a fixture that skips — and **a skip is not a pass**. It says the test could not run here, which is
a different claim from the test having run and succeeded.

That is why skips are absent from the baseline. An entry in `baseline_failures.txt` is a claim that
the test *runs on this host and fails*; a test that cannot run here belongs nowhere in that file, and
a skip that was reported as a new failure would make the check useless on any machine but the one it
was recorded on.

The skipped set is a coverage claim, so it is worth reading before trusting a green run:

```bash
python -m pytest tests/ -q -rs        # show why each test was skipped
```

## What CI runs

**Nothing yet.** This repository has no workflow that runs pytest, so
`python scripts/check_test_baseline.py` is a manual step and the baseline is a record rather than a
gate.