pair-notebooks:
    cd notebooks && find . -maxdepth 1 -type f -name "*.ipynb" -exec jupytext --set-formats ipynb,py:percent {} \;

sync-notebooks:
    cd notebooks && find . -maxdepth 1 -type f -name "*.ipynb" -exec jupytext --sync {} \;
