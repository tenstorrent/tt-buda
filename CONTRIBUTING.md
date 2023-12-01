# Contributing to model_demos

Thank you for your interest in this project.

If you are interested in making a contribution, then please familiarize
yourself with our technical contribution standards as set forth in this guide.

Next, please request appropriate write permissions by [opening an
issue](https://github.com/tenstorrent/tt-buda/issues/new/choose) for
GitHub permissions.

All contributions require:

- an issue
  - Your issue should be filed under an appropriate project. Please file a
    feature support request or bug report under Issues to get help with finding
    an appropriate project to get a maintainer's attention.
- a pull request (PR).
  - Your PR must be approved by appropriate reviewers. We do not accept PRs
    from forked repositories.

## Setting up environment

Install all dependencies from [model_demos/requriements.txt](model_demos/requirements.txt) and install pre-commit hooks in a Python environment with PyBUDA installed.

```bash
pip install -r model_demos/requirements.txt
pre-commit install
```

## Developing model_demos

### Cleaning the dev environment with `make clean` and `make clean_tt`

`make clean` and `make clean_tt` clears out model and build artifacts. Please make sure no artifacts are being pushed.

### Running pre-commit hooks

You must run hooks before you commit something.

```bash
make style
```

### Adding tests

For new model demos, please include a test case under `tests/` with the naming format of `test_[framework]_[model].py`.

Also include a pytest marker for each model family and update the markers list in [pyproject.toml](pyproject.toml).

For example,

```python
model_demos/tests/test_pytorch_bert.py

@pytest.mark.bert
def test_bert_masked_lm_pytorch(clear_pybuda):
    run_bert_masked_lm_pytorch()
```

### Updating Models Table

For new model demos, please include an entry in the [Models Table](model_demos/README.md/#models-table) along with the supported hardware.

## Contribution standards

### Code reviews

- A PR must be opened for any code change with the following criteria:
  - Be approved, by a maintaining team member and any codeowners whose modules
    are relevant for the PR.
  - Run pre-commit hooks.
  - Pass any acceptance criteria mandated in the original issue.
  - Pass any testing criteria mandated by codeowners whose modules are relevant
    for the PR.

## Hardware troubleshooting

### Resetting an accelerator board

If a Tenstorrent chip seems to hang and/or is producing unexpected behaviour,
you may try a software reset of the board.

For Grayskull: `tt-smi -tr all`

If the software reset does not work, unfortunately you will have to power cycle
the board. This usually means rebooting the host of a board.