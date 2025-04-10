# Sphinx Makefile with catch-all support and apidocs generation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build

.PHONY: help clean apidocs Makefile

# Show Sphinx options
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Clean build artifacts
clean:
	rm -rf "$(BUILDDIR)"/*

# Generate .rst files from the simopt package
# NOTE: need one for each module
apidocs:
	PYTHONPATH=.. sphinx-apidoc -o source ../simopt -f --module-first

# Catch-all: routes all other targets to sphinx-build -M
%:
	@$(SPHINXBUILD) -M $@ $(SOURCEDIR) $(BUILDDIR) -E -a $(SPHINXOPTS)
