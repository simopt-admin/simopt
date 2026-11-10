These baselines contain the numerical data emitted by the legacy plotting tests
before `simopt.plots` was removed. The manifest identifies the
source revision, dependencies, test IDs, artifact names, and SHA-256 checksums.

Each Zstandard-compressed JSON file stores one test's logged arrays, plotted curve
coordinates, and/or violin input data. The regression tests load these recorded
results independently of the current analysis implementation. Missing or modified
baselines fail validation; running tests never regenerates them.
