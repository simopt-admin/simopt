pair-notebooks:
    cd notebooks && find . -maxdepth 1 -type f -name "*.ipynb" -exec jupytext --set-formats ipynb,py:percent {} \;

sync-notebooks:
    cd notebooks && find . -maxdepth 1 -type f -name "*.ipynb" -exec jupytext --sync {} \;

bump version:
    sed -i 's/^version: .*/version: {{ version }}/' CITATION.cff && \
    sed -i 's/^version = .*/version = "{{ version }}"/' pyproject.toml && \
    sed -i 's/^release = .*/release = "{{ version }}"/' docs/source/conf.py
