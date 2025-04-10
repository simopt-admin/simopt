"""Generate API documentation using Sphinx apidoc."""

from pathlib import Path

from sphinx.ext.apidoc import main

output_dir = Path(__file__).resolve().parent / "source"
source_dir = Path(__file__).resolve().parent.parent / "simopt"

main(["-o", str(output_dir), str(source_dir), "-f"])
