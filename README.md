# Jetbrains test project.

A Word2Vec realization in pure NumPy.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Installation

### Install `uv`

On macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or with Homebrew:

```bash
brew install uv
```

### Install project dependencies

Sync the project dependencies from `pyproject.toml`:

```bash
uv sync
```

It is also recommended to install the development dependencies:

```bash
uv sync --dev
```

## Project structure

```text
.
├── README.md
├── notebooks
│   └── test.ipynb
├── pyproject.toml
├── src
│   └── numpyword2vec
│       ├── __init__.py
│       └── word2vec.py
└── uv.lock
```

## Code location

The Word2Vec implementation is located in:

```text
./src/numpyword2vec/word2vec.py
```

A simple test notebook is available at:

```text
./notebooks/test.ipynb
```

The notebook uses the `text8` dataset loaded from Hugging Face.

## Running the notebook

To start Jupyter Lab with `uv`, run:

```bash
uv run jupyter lab
```

Then open `notebooks/test.ipynb` in the Jupyter interface.
