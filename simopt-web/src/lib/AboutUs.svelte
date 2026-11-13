<script lang="ts">
  const papers = [
    {
      cite: "Eckman et al. (2024)",
      href: "https://ieeexplore.ieee.org/document/10408734",
      note: "Feasibility metrics for stochastically constrained simulation-optimization problems.",
    },
    {
      cite: "Shashaani et al. (2024)",
      href: "https://dl.acm.org/doi/10.1145/3680282",
      note: "A large data-farming experiment relating solver factor settings to finite-time performance.",
    },
    {
      cite: "Eckman et al. (2023)",
      href: "https://pubsonline.informs.org/doi/10.1287/ijoc.2023.1273",
      note: "The current code architecture and how users interact with the library.",
    },
    {
      cite: "Eckman et al. (2023)",
      href: "https://pubsonline.informs.org/doi/10.1287/ijoc.2022.1261",
      note: "The design of experiments for comparing solvers, including macroreplications, post-processing, and solvability plots.",
    },
    {
      cite: "Eckman et al. (2019)",
      href: "https://www.informs-sim.org/wsc19papers/374.pdf",
      note: "Architecture changes to the MATLAB version and control of random number streams.",
    },
    {
      cite: "Dong et al. (2017)",
      href: "https://www.informs-sim.org/wsc17papers/includes/files/179.pdf",
      note: "An experimental comparison of several SimOpt solvers.",
    },
    {
      cite: "Pasupathy and Henderson (2011)",
      href: "https://www.informs-sim.org/wsc11papers/363.pdf",
      note: "An earlier interface for MATLAB implementations of problems and solvers.",
    },
    {
      cite: "Pasupathy and Henderson (2006)",
      href: "https://www.informs-sim.org/wsc06papers/028.pdf",
      note: "The original motivation for the testbed.",
    },
  ];
</script>

<div class="page-narrow">
  <section class="card page-card">
    <header class="card-head">
      <h2 class="card-title">About SimOpt</h2>
      <p class="lede">A testbed of simulation-optimization problems and solvers.</p>
    </header>
    <p>
      SimOpt exists to encourage the development and constructive comparison of simulation-optimization solvers. The
      focus is on how solvers perform in finite time, rather than the asymptotic results common in the literature.
    </p>
    <p>
      Here, simulation means estimating statistical measures of a complex system by modeling it as if the probability
      distributions of its random inputs were known, drawing realizations from those distributions, and aggregating the
      responses over many replications. Each replication is one noisy evaluation of the objective function or of the
      stochastic constraints.
    </p>
    <div class="links">
      <a class="btn" href="https://github.com/simopt-admin/simopt" target="_blank" rel="noreferrer">Source on GitHub</a>
      <a class="btn" href="https://simopt.readthedocs.io/en/latest/index.html" target="_blank" rel="noreferrer">Documentation</a>
    </div>
  </section>

  <section class="card page-card">
    <header class="card-head">
      <h2 class="card-title">Publications</h2>
      <p class="lede">Papers on the development of SimOpt and experiments run on it, newest first.</p>
    </header>
    <ul class="papers">
      {#each papers as paper (paper.href)}
        <li>
          <a href={paper.href} target="_blank" rel="noreferrer">{paper.cite}</a>
          <span>{paper.note}</span>
        </li>
      {/each}
    </ul>
  </section>

  <section class="card page-card">
    <header class="card-head">
      <h2 class="card-title">Acknowledgments</h2>
    </header>
    <p class="small">
      An earlier SimOpt website was developed through work supported by the National Science Foundation under grant nos.
      DMI-0400287, CMMI-0800688 and CMMI-1200315. Recent development has been supported by the National Science
      Foundation under grant nos. DGE-1650441, CMMI-1537394, CMMI-1254298, CMMI-1536895, IIS-1247696, and TRIPODS+X
      DMS-1839346, by the Air Force Office of Scientific Research under grant nos. FA9550-12-1-0200, FA9550-15-1-0038,
      and FA9550-16-1-0046, and by the Army Research Office under grant no. W911NF-17-1-0094.
    </p>
    <p class="small muted">
      Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and
      do not necessarily reflect the views of the National Science Foundation (NSF).
    </p>
  </section>
</div>

<style>
  p {
    margin: 0 0 0.75rem;
    color: var(--text);
    line-height: 1.6;
    max-width: 72ch;
  }
  p:last-child {
    margin-bottom: 0;
  }
  p.small {
    font-size: 13px;
  }
  p.muted {
    color: var(--muted);
  }

  .links {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 1.25rem;
  }
  .links .btn {
    display: inline-flex;
    align-items: center;
  }

  .papers {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .papers li {
    display: grid;
    grid-template-columns: minmax(12rem, 16rem) minmax(0, 1fr);
    gap: 1.25rem;
    padding: 0.7rem 0;
    border-top: 1px solid var(--border);
    line-height: 1.5;
  }
  .papers li:first-child {
    border-top: none;
    padding-top: 0;
  }
  .papers span {
    color: var(--text);
  }

  @media (max-width: 640px) {
    .papers li {
      grid-template-columns: minmax(0, 1fr);
      gap: 0.15rem;
    }
  }
</style>
