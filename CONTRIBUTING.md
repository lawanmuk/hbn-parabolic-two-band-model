# Contributing

## Setup

```bash
git clone https://github.com/lawanmuk/hbn-parabolic-two-band-model.git
cd hbn-parabolic-two-band-model
python -m venv .venv
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff format src tests
ruff check src tests
pytest
```

All three must pass; CI runs the same checks.

## Guidelines

- Put reusable logic in `src/twoband/`, not in notebooks, and add a test for it in `tests/`.
- Use atomic units inside the package; convert to eV or fs only for plotting.
- For a bug fix, add a test that fails before the fix and passes after it.
- Use short commit messages with a type prefix: `feat:`, `fix:`, `test:`, `docs:`, `ci:`.
- Add a line to `CHANGELOG.md` under "Unreleased".
