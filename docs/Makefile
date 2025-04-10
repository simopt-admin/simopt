# Sphinx Makefile with catch-all support and apidocs generation

SPHINXOPTS    ?= -j auto # Use parallel build
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = _build
TOPLEVEL      = ..
IGNORED_DIRS  = $(TOPLEVEL)/dev_tools

.PHONY: help clean apidocs Makefile

# Show Sphinx options
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

# Clean build artifacts
clean:
	rm -rf "$(BUILDDIR)"/*

# Generate .rst files from the simopt package
apidocs:
	sphinx-apidoc -f --module-first -o "$(SOURCEDIR)" "$(TOPLEVEL)" "$(IGNORED_DIRS)"

# Catch-all: routes all other targets to sphinx-build -M
%:
	@$(SPHINXBUILD) -M $@ $(SOURCEDIR) $(BUILDDIR) -E -a $(SPHINXOPTS)
