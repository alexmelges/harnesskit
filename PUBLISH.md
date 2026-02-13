# Publishing to PyPI

## Prerequisites

```bash
pip install build twine
```

## Build

```bash
python3 -m build
```

This creates `dist/harnesskit-X.Y.Z.tar.gz` and `dist/harnesskit-X.Y.Z-py3-none-any.whl`.

## Upload

### First time: Create a PyPI API token

1. Go to https://pypi.org/manage/account/token/
2. Create a token scoped to the `harnesskit` project (or all projects for first upload)
3. Store it securely

### Upload to PyPI

```bash
python3 -m twine upload dist/*
```

You'll be prompted for credentials:
- **Username:** `__token__`
- **Password:** your API token (starts with `pypi-`)

### Upload to Test PyPI (recommended first)

```bash
python3 -m twine upload --repository testpypi dist/*
```

## Version Bump Checklist

1. Update `version` in `pyproject.toml`
2. `rm -rf dist/ && python3 -m build`
3. `python3 -m twine upload dist/*`
4. `git tag vX.Y.Z && git push --tags`
