<script lang="ts">
  const steps = [
    {
      title: "Add solvers",
      body: "Open the Solvers tab and choose a solver. Adjust any parameters (hover the i next to a name for its description), then select Add Solver. Added solvers appear in the experiment panel, where you can expand one to review or edit its settings.",
    },
    {
      title: "Add problems",
      body: "On the Problems tab, choose a problem and adjust its factors. A problem's list includes both problem factors and the factors of the simulation model underneath it.",
    },
    {
      title: "Choose plots",
      body: "On the Plots tab, pick a plot type, set its options, and add it. Tick specific solvers or problems to limit what the plot covers, or leave them all unticked to include everything. An experiment needs at least one plot.",
    },
    {
      title: "Check compatibility",
      body: "The compatibility grid in the experiment panel has one square per solver and problem pair. A red square means the solver cannot handle that problem, for example because of its constraint types or number of objectives. Open the matrix and hover a red cell to see why, then remove the solver or problem before running.",
    },
    {
      title: "Review post-processing settings",
      body: "The Settings tab sets how many post-replications to take at each recommended solution and at the initial and optimal solutions, and whether to use common random numbers for them. The defaults suit most experiments.",
    },
    {
      title: "Run the experiment",
      body: "Set the number of macroreplications and select Run Experiment. Results open in a new tab showing the run status, a live output log, and each plot as it is saved. If you run again with the same solvers, problems, and settings, only the plots are regenerated.",
    },
  ];

  const terms = [
    ["Macroreplication", "One independent run of a solver on a problem. More macroreplications give tighter estimates of a solver's typical performance."],
    ["Post-replication", "Extra simulation replications taken at the solutions a solver recommends, so their objective values are estimated independently of the solver's own estimates."],
    ["Post-normalization", "Estimating the objective at the initial and best-known solutions so progress curves can be compared across problems on a common 0 to 1 scale."],
    ["Common random numbers (CRN)", "Reusing the same random number streams across solutions so that differences between them reflect the solutions, not sampling noise."],
    ["Design point", "One combination of factor values in a data-farming design. Each point added to an experiment becomes its own solver or problem entry."],
  ];
</script>

<div class="page-narrow">
  <section class="card page-card">
    <header class="card-head">
      <h2 class="card-title">User guide</h2>
      <p class="lede">Compare simulation-optimization solvers in six steps, from picking solvers to reading the plots.</p>
    </header>

    <ol class="steps">
      {#each steps as step, i (step.title)}
        <li>
          <span class="step-num" aria-hidden="true">{i + 1}</span>
          <div>
            <h3>{step.title}</h3>
            <p>{step.body}</p>
          </div>
        </li>
      {/each}
    </ol>
  </section>

  <section class="card page-card">
    <header class="card-head">
      <h2 class="card-title">Data farming</h2>
      <p class="lede">Explore how factor settings change performance by running a designed experiment over them.</p>
    </header>
    <p>
      Choose a solver, problem, or model, tick the factors to vary, and give each one a range. SimOpt builds a nearly
      orthogonal Latin hypercube over those ranges; add stacks for more design points. For solvers and problems, Add to
      experiment sends every design point to the Simulator as its own entry. For models, run the design directly and
      download the raw results as CSV.
    </p>
    <p>Save a design to reuse it later. Saved designs can be loaded back into the editor, exported as CSV, or deleted.</p>
  </section>

  <section class="card page-card">
    <header class="card-head">
      <h2 class="card-title">Terms</h2>
    </header>
    <dl class="terms">
      {#each terms as [term, def] (term)}
        <dt>{term}</dt>
        <dd>{def}</dd>
      {/each}
    </dl>
    <p class="more">
      For the full experimental design behind these terms, see
      <a href="https://pubsonline.informs.org/doi/10.1287/ijoc.2022.1261" target="_blank" rel="noreferrer">Eckman et al. (2023)</a>.
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

  .steps {
    list-style: none;
    margin: 0;
    padding: 0;
  }
  .steps li {
    display: grid;
    grid-template-columns: 28px minmax(0, 1fr);
    gap: 1rem;
    padding: 1rem 0;
    border-top: 1px solid var(--border);
  }
  .steps li:first-child {
    border-top: none;
    padding-top: 0.25rem;
  }
  .step-num {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--primary-tint);
    color: var(--primary);
    font-weight: 600;
    font-size: 13px;
    display: grid;
    place-items: center;
    font-variant-numeric: tabular-nums;
  }
  h3 {
    margin: 0.2rem 0 0.3rem;
    font-size: 15px;
    font-weight: 600;
    color: var(--ink);
  }
  .steps p {
    margin: 0;
  }

  .terms {
    display: grid;
    grid-template-columns: minmax(10rem, 14rem) minmax(0, 1fr);
    margin: 0;
  }
  .terms dt,
  .terms dd {
    padding: 0.7rem 0;
    border-top: 1px solid var(--border);
    margin: 0;
    line-height: 1.55;
  }
  .terms dt {
    font-weight: 600;
    color: var(--ink);
    padding-right: 1.25rem;
  }
  .terms dd {
    color: var(--text);
  }
  .terms dt:first-of-type,
  .terms dt:first-of-type + dd {
    border-top: none;
    padding-top: 0;
  }
  .more {
    margin-top: 1rem;
    font-size: 13px;
    color: var(--muted);
  }

  @media (max-width: 640px) {
    .terms {
      grid-template-columns: minmax(0, 1fr);
    }
    .terms dd {
      border-top: none;
      padding-top: 0;
    }
    .terms dt:first-of-type + dd {
      padding-top: 0;
    }
  }
</style>
