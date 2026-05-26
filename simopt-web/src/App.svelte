<script lang="ts">
  import { onMount } from 'svelte';
  import type {
    Compatibility,
    EditMode,
    FixedSchema,
    FormValues,
    Page,
    Param,
    ParamsResponse,
    PlotSummaryEntry,
    SummaryEntry,
    SummaryKind,
  } from './types';

  let currentPage: Page = "Simulator";
  const navigate = (p: Page) => (currentPage = p);

  let allSolvers: string[] = [];
  let allProblems: string[] = [];

  let selectedSolverName = "";     // blank by default
  let selectedProblemName = "";    // blank by default
  let solverParams: Param[] = [];           // [{name, description, default, value}]
  let problemParams: Param[] = [];

  let summarySolvers: SummaryEntry[] = [];         // [{name, params:[...], expanded?:bool}]
  let summaryProblems: SummaryEntry[] = [];

  // { kind: 'solver'|'problem', index: number } | null
  let editMode: EditMode | null = null;

  let macroreps = 10;
  let showPostProcess = false;
  let showPostNormalize = false;

  let showConfirm = false;
  let confirmKind: SummaryKind | null = null;
  let confirmIndex: number | null = null;

  function clickIsOnBackdrop(event: MouseEvent): boolean {
    return event.target === event.currentTarget;
  }

  function keydownIsBackdropDismiss(event: KeyboardEvent): boolean {
    return event.target === event.currentTarget && (event.key === 'Escape' || event.key === 'Enter' || event.key === ' ');
  }

  function onConfirmBackdropClick(event: MouseEvent): void {
    if (clickIsOnBackdrop(event)) closeConfirm();
  }

  function onConfirmBackdropKeydown(event: KeyboardEvent): void {
    if (!keydownIsBackdropDismiss(event)) return;
    event.preventDefault();
    closeConfirm();
  }

  function onCompatBackdropClick(event: MouseEvent): void {
    if (clickIsOnBackdrop(event)) closeCompatModal();
  }

  function onCompatBackdropKeydown(event: KeyboardEvent): void {
    if (!keydownIsBackdropDismiss(event)) return;
    event.preventDefault();
    closeCompatModal();
  }

  let allPlots: string[] = [];
  let selectedPlotName = "";
  let plotParams: Param[] = []; // [{name, description, default, value}]
  let selectedPlotSolvers: string[] = []; // Array of solver names selected for this plot
  let selectedPlotProblems: string[] = []; // Array of problem names selected for this plot

  let lastRunId: string | null = null;

  function toDisplayString(val: unknown): string {
    if (val === null || val === undefined) return "";
    if (typeof val === "string") return val;
    try { return JSON.stringify(val); } catch { return String(val); }
  }

  function inputValue(event: Event): string {
    return (event.currentTarget as HTMLInputElement | HTMLSelectElement).value;
  }

  async function fetchPlotParams(name: string): Promise<Param[]> {
    if (!name) return [];
    const res = await fetch(`http://localhost:8000/plot_params/${encodeURIComponent(name)}`);
    const data: ParamsResponse = await res.json();
    return (data.parameters || []).map(p => ({
      name: p.name ?? "",
      description: p.description || "",
      default: p.default,
      value: toDisplayString(p.default),
    }));
  }

  async function onPlotChange(name: string): Promise<void> {
    selectedPlotName = name;
    editMode = null;
    plotParams = name ? await fetchPlotParams(name) : [];
    selectedPlotSolvers = [];
    selectedPlotProblems = [];
  }

  function deepCopyParams(arr: Param[]): Param[] {
    return (arr || []).map(p => ({ ...p }));
  }

  function abbrev(name: string): string {
    if (!name) return "";
    const clean = String(name).replace(/[^A-Za-z0-9]/g, "");
    return clean.slice(-4).toUpperCase();   // last 4 chars
  }

  async function fetchSolverParams(name: string): Promise<Param[]> {
    if (!name) return [];
    const res = await fetch(`http://localhost:8000/solver_params/${encodeURIComponent(name)}`);
    const data: ParamsResponse = await res.json();
    return (data.parameters || []).map(p => ({
      name: p.name ?? "",
      description: p.description || "",
      default: p.default,
      value: toDisplayString(p.default)
    }));
  }

  async function fetchProblemParams(name: string): Promise<Param[]> {
    if (!name) return [];
    const res = await fetch(`http://localhost:8000/problem_params/${encodeURIComponent(name)}`);
    const data: ParamsResponse = await res.json();
    return (data.parameters || []).map(p => ({
      name: p.name ?? "",
      description: p.description || "",
      default: p.default,
      value: toDisplayString(p.default)
    }));
  }

  async function onSolverChange(name: string): Promise<void> {
    selectedSolverName = name;
    solverParams = name ? await fetchSolverParams(name) : [];
  }
  async function onProblemChange(name: string): Promise<void> {
    selectedProblemName = name;
    problemParams = name ? await fetchProblemParams(name) : [];
  }

  function resetSolverEditor() { selectedSolverName = ""; solverParams = []; editMode = null; }
  function resetProblemEditor() { selectedProblemName = ""; problemParams = []; editMode = null; }

  function addSolverToSummary() {
    if (!selectedSolverName) return;
    const entry = { name: selectedSolverName, params: deepCopyParams(solverParams), expanded: false };

    if (editMode?.kind === 'solver') {
      summarySolvers[editMode.index] = entry;
      summarySolvers = [...summarySolvers];
      resetSolverEditor();
    } else {
      summarySolvers = [...summarySolvers, entry];
      resetSolverEditor();
    }
  }

  function addProblemToSummary() {
    if (!selectedProblemName) return;
    const entry = { name: selectedProblemName, params: deepCopyParams(problemParams), expanded: false };

    if (editMode?.kind === 'problem') {
      summaryProblems[editMode.index] = entry;
      summaryProblems = [...summaryProblems];
      resetProblemEditor();
    } else {
      summaryProblems = [...summaryProblems, entry];
      resetProblemEditor();
    }
  }

  const removeSummarySolver = (i: number) => (summarySolvers = summarySolvers.filter((_, idx) => idx !== i));
  const removeSummaryProblem = (i: number) => (summaryProblems = summaryProblems.filter((_, idx) => idx !== i));

  let summaryPlots: PlotSummaryEntry[] = []; // [{ name, params:[{name, description, default, value}], solvers:[], problems:[], expanded?:bool }]

  function resetPlotEditor() {
    selectedPlotName = "";
    plotParams = [];
    selectedPlotSolvers = [];
    selectedPlotProblems = [];
    editMode = null;
  }

  function addPlotToSummary() {
    if (!selectedPlotName) return;
    const entry = {
      name: selectedPlotName,
      params: deepCopyParams(plotParams),
      solvers: [...selectedPlotSolvers], // Copy the selected solvers
      problems: [...selectedPlotProblems], // Copy the selected problems
      expanded: false
    };
    if (editMode?.kind === 'plot') {
        summaryPlots[editMode.index] = entry;
        summaryPlots = [...summaryPlots];
        resetPlotEditor();
    } else {
        summaryPlots = [...summaryPlots, entry];
        resetPlotEditor();
    }
  }

  const removeSummaryPlot = (i: number) =>
    (summaryPlots = summaryPlots.filter((_, idx) => idx !== i));

  function requestEdit(kind: SummaryKind, index: number) {
    const occupied = (kind === 'solver'  && selectedSolverName) ||
                     (kind === 'problem' && selectedProblemName) ||
                     (kind === 'plot' && selectedPlotName);
    if (occupied) {
      confirmKind = kind;
      confirmIndex = index;
      showConfirm = true;
    } else {
      startEdit(kind, index);
    }
  }

  function closeConfirm() {
    showConfirm = false;
    confirmKind = null;
    confirmIndex = null;
  }

  function confirmProceed() {
    if (confirmKind != null && confirmIndex != null) {
      startEdit(confirmKind, confirmIndex);
    }
    closeConfirm();
  }

  // --- Post-replicate / Post-normalize fixed forms ---
  let prSchema: FixedSchema = { params: [] };      // fetched schema for post-replicate
  let pnSchema: FixedSchema = { params: [] };      // fetched schema for post-normalize
  let prValues: FormValues = {};                  // current values bound to the form
  let pnValues: FormValues = {};                  // current values bound to the form

  function initValuesFromSchema(schema: FixedSchema, target: FormValues): FormValues {
    const t: FormValues = {};
    for (const p of schema.params || []) t[p.name] = p.default;
    return Object.assign(target, t);
  }

  // grab schemas on mount (you already have an onMount—just add these fetches there)
  async function loadPostFixedForms() {
    try {
      const r1 = await fetch("http://localhost:8000/postreplicate_schema");
      prSchema = await r1.json();
      prValues = initValuesFromSchema(prSchema, {});
    } catch (e) { console.error("postreplicate_schema", e); }

    try {
      const r2 = await fetch("http://localhost:8000/postnormalize_schema");
      pnSchema = await r2.json();
      pnValues = initValuesFromSchema(pnSchema, {});
    } catch (e) { console.error("postnormalize_schema", e); }
  }

  async function startEdit(kind: SummaryKind, index: number): Promise<void> {
    if (kind === 'solver') {
      const s = summarySolvers[index];
      selectedSolverName = s.name;
      solverParams = deepCopyParams(s.params);
      editMode = { kind: 'solver', index };
    } else if (kind === 'problem') {
      const p = summaryProblems[index];
      selectedProblemName = p.name;
      problemParams = deepCopyParams(p.params);
      editMode = { kind: 'problem', index };
    } else {
        const pl = summaryPlots[index];
        selectedPlotName = pl.name;
        plotParams = await fetchPlotParams(pl.name);
        plotParams = plotParams.map((p, i) => ({
            ...p,
            value: pl.params[i]?.value ?? p.value
        }));
        selectedPlotSolvers = [...(pl.solvers || [])];
        selectedPlotProblems = [...(pl.problems || [])];
        editMode = { kind: 'plot', index };
    }
  }

  async function runExperiment(): Promise<void> {
    // Validate that we have at least one solver and problem
    if (summarySolvers.length === 0) {
      alert("Please add at least one solver before running the experiment.");
      return;
    }
    if (summaryProblems.length === 0) {
      alert("Please add at least one problem before running the experiment.");
      return;
    }

    // Parse parameter values (handle JSON strings)
    const parseValue = (val: unknown): unknown => {
      if (val === null || val === undefined || val === "") return null;
      
      // If it's already not a string, return as-is
      if (typeof val !== 'string') return val;
      
      // Try to parse as JSON first (handles arrays, objects, booleans, numbers)
      try {
        return JSON.parse(val);
      } catch {
        // If JSON parse fails, return the string value
        return val;
      }
    };

    // Build the experiment payload
    const payload = {
      last_run_id: lastRunId,
      experiment_params: {
        num_macroreps: macroreps,
        num_postreps: prValues.num_post_reps || 100,
        num_postnorms: pnValues.num_post_reps_init_opt || 100
      },
      problems: summaryProblems.map(p => ({
        name: p.name,
        rename: p.name,
        fixed_factors: p.params.reduce<Record<string, unknown>>((acc, param) => {
          const parsed = parseValue(param.value);
          if (parsed !== null) acc[param.name] = parsed;
          return acc;
        }, {}),
        model_fixed_factors: {}
      })),
      solvers: summarySolvers.map(s => ({
        name: s.name,
        rename: s.name,
        fixed_factors: s.params.reduce<Record<string, unknown>>((acc, param) => {
          const parsed = parseValue(param.value);
          if (parsed !== null) acc[param.name] = parsed;
          return acc;
        }, {})
      })),
      plots: summaryPlots.map(pl => ({
        plot_type: pl.name,
        params: pl.params.reduce<Record<string, unknown>>((acc, param) => {
          const parsed = parseValue(param.value);
          if (parsed !== null) acc[param.name] = parsed;
          return acc;
        }, {}),
        solvers: pl.solvers && pl.solvers.length > 0 ? pl.solvers : null, // null means "all solvers"
        problems: pl.problems && pl.problems.length > 0 ? pl.problems : null, // null means "all problems"
      }))
    };

    try {
      const res = await fetch("http://localhost:8000/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        alert("Failed to start experiment");
        return;
      }

      const data: { id: string } = await res.json();
      lastRunId = data.id;
      
      // Open results page in new window
      window.open(`/results/${data.id}/index.html`, '_blank');
    } catch (error) {
      console.error("Error running experiment:", error);
      alert("Failed to start experiment: " + (error instanceof Error ? error.message : String(error)));
    }
  }

  onMount(async () => {
    await loadPostFixedForms();
    try {
      const sRes = await fetch('http://localhost:8000/solvers');
      const data: { solvers?: string[] } = await sRes.json();
      allSolvers = data.solvers || [];
    } catch (e) { console.error('Failed to fetch solvers', e); }

    try {
      const pRes = await fetch('http://localhost:8000/problems');
      const data: { problems?: string[] } = await pRes.json();
      allProblems = data.problems || [];
    } catch (e) { console.error('Failed to fetch problems', e); }

    try {
      const r = await fetch('http://localhost:8000/plots');
      const data: { plots?: string[] } = await r.json();
      allPlots = data.plots || [];
    } catch (e) {
      console.error('Failed to fetch plots:', e);
      allPlots = [];
    }
  });

  let showCompatModal = false;

  function openCompatModal() {
    if (summarySolvers.length && summaryProblems.length) showCompatModal = true;
  }
  function closeCompatModal() {
    showCompatModal = false;
  }

  let compatibility: Compatibility = {};
  async function checkCompatibility(solvers: SummaryEntry[], problems: SummaryEntry[]): Promise<void> {
    if (solvers.length === 0 || problems.length === 0) {
      compatibility = {};
      return;
    }
    const payload = {
      solvers: solvers.map(s => s.name),
      problems: problems.map(p => p.name)
    };
    try {
      const res = await fetch("http://localhost:8000/check_compatibility", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data: { compatibility?: Compatibility } = await res.json();
      compatibility = data.compatibility || {};
    } catch (e) {
      console.error("Error checking compatibility:", e);
    }
  }
  $: checkCompatibility(summarySolvers, summaryProblems);

  $: hasIncompatibility =
    summarySolvers.length > 0 &&
    summaryProblems.length > 0 &&
    Object.values(compatibility).some(row =>
      Object.values(row || {}).some(cell => cell && cell.compatible === false)
    );
  
  let totalCompatCells: number;
  let greenPct: number;
  let redPct: number;

  $: {
    let g = 0, r = 0;
    const solverNames = summarySolvers?.map(s => s.name) ?? [];
    const problemNames = summaryProblems?.map(p => p.name) ?? [];

    for (const s of solverNames) {
      for (const p of problemNames) {
        const cell = compatibility?.[s]?.[p];
        // count only cells that explicitly report compatibility (skip neutral/missing)
        if (cell && typeof cell.compatible === 'boolean') {
          if (cell.compatible) g++; else r++;
        }
      }
    }
    totalCompatCells = g + r;

    if (totalCompatCells > 0) {
      greenPct = Math.round((g * 1000) / totalCompatCells) / 10; // 1 decimal
      redPct   = Math.round((r * 1000) / totalCompatCells) / 10;
      // Keep them summing to 100.0 in case of rounding drift
      const drift = 100 - (greenPct + redPct);
      if (Math.abs(drift) >= 0.1) redPct = Math.max(0, +(redPct + drift).toFixed(1));
    } else {
      greenPct = 0;
      redPct = 0;
    }
  }
</script>

<nav>
  <div class="nav-left">
    <span class="title">SimOpt Web Interface</span>
  </div>
  <div class="nav-right">
    <ul>
      <li>
        <a
          href="#simulator"
          class:active={currentPage === "Simulator"}
          on:click|preventDefault={() => navigate("Simulator")}
        >Simulator</a>
      </li>
      <li>
        <a
          href="#user-guide"
          class:active={currentPage === "User Guide"}
          on:click|preventDefault={() => navigate("User Guide")}
        >User Guide</a>
      </li>
      <li>
        <a
          href="#about-us"
          class:active={currentPage === "About Us"}
          on:click|preventDefault={() => navigate("About Us")}
        >About Us</a>
      </li>
    </ul>
  </div>
</nav>

<main>
  {#if currentPage === "Simulator"}
    <div class="row-3col">
      <!-- ===== Left column: Solver + Post-replicate + Choose Plot ===== -->
      <div class="col-stack">
        <!-- === Choose Solver === -->
        <div class="card column">
          <h2>Choose Solver</h2>

          {#if selectedSolverName}
            <button
              class="secondary-outline"
              style="margin-bottom:0.75rem;"
              on:click={addSolverToSummary}
            >
              {editMode?.kind === 'solver' ? 'Apply Changes' : '+ Add Solver'}
            </button>
          {/if}

          <div class="block-header">
            <select bind:value={selectedSolverName} on:change={(e) => onSolverChange(inputValue(e))}>
              <option value="">— Select a Solver —</option>
              {#each allSolvers as option (option)}
                <option value={option}>{option}</option>
              {/each}
            </select>
          </div>

          {#if selectedSolverName}
            <div class="param-box">
              <p class="param-title">Solver Parameters</p>
              {#each solverParams as param, idx (param.name)}
                <label>
                    <div style="display:flex;align-items:center;gap:0.4rem;">
                        <span>{param.name}</span>
                        {#if param.description}
                            <span class="info-wrapper" aria-hidden="true">
                                <span class="info-icon">ℹ</span>
                                <div class="tooltip">{param.description}</div>
                            </span>
                        {/if}
                    </div>
                    {#if typeof param.default === 'boolean'}
                        <select
                            value={param.value}
                            on:change={(e) => { solverParams[idx].value = inputValue(e); solverParams = [...solverParams]; }}
                        >
                            <option value="true">True</option>
                            <option value="false">False</option>
                        </select>
                    {:else}
                        <input
                            type="text"
                            bind:value={param.value}
                            on:input={(e) => (solverParams[idx].value = inputValue(e))}
                        />
                    {/if}
                </label>
              {/each}
            </div>
          {/if}
        </div>

        <!-- === POST-REPLICATE === -->
        <div class="card">
          <button class="dropdown" on:click={() => (showPostProcess = !showPostProcess)}>
            {showPostProcess ? "▼" : "▶"} Post-replicate
          </button>

          {#if showPostProcess}
            <div class="dropdown-content">
              <div class="param-box">
                <p class="param-title">Options for Post-replication</p>

                {#each prSchema.params as p (p.name)}
                  <label>
                    <span>{p.label}</span>

                    {#if p.type === "bool"}
                      <select bind:value={prValues[p.name]}>
                        <option value={true}>Yes</option>
                        <option value={false}>No</option>
                      </select>
                    {:else if p.type === "int"}
                      <input type="number" bind:value={prValues[p.name]} min="0" step="1" />
                    {:else if p.type === "float"}
                      <input type="number" bind:value={prValues[p.name]} step="0.01" />
                    {:else}
                      <input type="text" bind:value={prValues[p.name]} />
                    {/if}
                  </label>
                {/each}
              </div>
            </div>
          {/if}
        </div>

        <!-- Choose Plot -->
        <div class="card column">
          <h2>Choose Plot</h2>

          {#if selectedPlotName}
            <button
              class="secondary-outline"
              style="margin-bottom:0.75rem;"
              on:click={addPlotToSummary}
            >
              {editMode?.kind === 'plot' ? 'Apply Changes' : '+ Add Plot'}
            </button>
          {/if}

          <div class="block-header">
            <select bind:value={selectedPlotName} on:change={(e) => onPlotChange(inputValue(e))}>
              <option value="">— Select a Plot —</option>
              {#each allPlots as plot (plot)}
                <option value={plot}>{plot}</option>
              {/each}
            </select>
          </div>

          {#if plotParams.length}
            <div class="param-box" style="margin-top:.5rem;">
              <p class="param-title">Plot Parameters ({selectedPlotName})</p>
              {#each plotParams as p, i (p.name)}
                <label>
                    <div style="display:flex;align-items:center;gap:0.4rem;">
                        <span>{p.name}</span>
                        {#if p.description}
                            <span class="info-wrapper" aria-hidden="true">
                                <span class="info-icon">℩</span>
                                <div class="tooltip">{p.description}</div>
                            </span>
                        {/if}
                    </div>
                    {#if p.name === 'ref_solver'}
                        <select
                            value={p.value}
                            on:change={(e) => { plotParams[i].value = inputValue(e); plotParams = [...plotParams]; }}
                        >
                            <option value="">— None —</option>
                            {#each summarySolvers as solver (solver.name)}
                                <option value={solver.name}>{solver.name}</option>
                            {/each}
                        </select>
                    {:else if typeof p.default === 'boolean'}
                        <select
                            value={p.value}
                            on:change={(e) => { plotParams[i].value = inputValue(e); plotParams = [...plotParams]; }}
                        >
                            <option value="true">True</option>
                            <option value="false">False</option>
                        </select>
                    {:else}
                        <input
                            type="text"
                            bind:value={p.value}
                            on:input={(e) => (plotParams[i].value = inputValue(e))}
                        />
                    {/if}
                </label>
              {/each}
            </div>
          {/if}

          {#if selectedPlotName}
            <div class="param-box" style="margin-top:.5rem;">
              <p class="param-title">Select Solvers (leave empty for all)</p>
              {#if summarySolvers.length === 0}
                <p style="color:#6b7280;font-size:0.9rem;margin-top:0.25rem;">No solvers added yet</p>
              {:else}
                {#each summarySolvers as solver (solver.name)}
                  <label style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.25rem;">
                    <input
                      type="checkbox"
                      value={solver.name}
                      bind:group={selectedPlotSolvers}
                    />
                    <span style="flex:1;">{solver.name}</span>
                  </label>
                {/each}
              {/if}
            </div>

            <div class="param-box" style="margin-top:.5rem;">
              <p class="param-title">Select Problems (leave empty for all)</p>
              {#if summaryProblems.length === 0}
                <p style="color:#6b7280;font-size:0.9rem;margin-top:0.25rem;">No problems added yet</p>
              {:else}
                {#each summaryProblems as problem (problem.name)}
                  <label style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.25rem;">
                    <input
                      type="checkbox"
                      value={problem.name}
                      bind:group={selectedPlotProblems}
                    />
                    <span style="flex:1;">{problem.name}</span>
                  </label>
                {/each}
              {/if}
            </div>
          {/if}
        </div>
      </div>

      <!-- ===== Middle column: Problem + Post-normalize ===== -->
      <div class="col-stack">
        <!-- === Choose Problem === -->
        <div class="card column">
          <h2>Choose Problem</h2>

          {#if selectedProblemName}
            <button
              class="secondary-outline"
              style="margin-bottom:0.75rem;"
              on:click={addProblemToSummary}
            >
              {editMode?.kind === 'problem' ? 'Apply Changes' : '+ Add Problem'}
            </button>
          {/if}

          <div class="block-header">
            <select bind:value={selectedProblemName} on:change={(e) => onProblemChange(inputValue(e))}>
              <option value="">— Select a Problem —</option>
              {#each allProblems as option (option)}
                <option value={option}>{option}</option>
              {/each}
            </select>
          </div>

          {#if selectedProblemName}
            <div class="param-box">
              <p class="param-title">Problem Parameters</p>
              {#each problemParams as param, idx (param.name)}
                <label>
                    <div style="display:flex;align-items:center;gap:0.4rem;">
                        <span>{param.name}</span>
                        {#if param.description}
                            <span class="info-wrapper" aria-hidden="true">
                                <span class="info-icon">ℹ</span>
                                <div class="tooltip">{param.description}</div>
                            </span>
                        {/if}
                    </div>
                    {#if typeof param.default === 'boolean'}
                        <select
                            value={param.value}
                            on:change={(e) => { problemParams[idx].value = inputValue(e); problemParams = [...problemParams]; }}
                        >
                            <option value="true">True</option>
                            <option value="false">False</option>
                        </select>
                    {:else}
                        <input
                            type="text"
                            bind:value={param.value}
                            on:input={(e) => (problemParams[idx].value = inputValue(e))}
                        />
                    {/if}
                </label>
              {/each}
            </div>
          {/if}
        </div>

        <!-- === POST-NORMALIZE (moved under problem) === -->
        <div class="card">
          <button class="dropdown" on:click={() => (showPostNormalize = !showPostNormalize)}>
            {showPostNormalize ? "▼" : "▶"} Post-normalize
          </button>

          {#if showPostNormalize}
            <div class="dropdown-content">
              <div class="param-box">
                <p class="param-title">Options for Post-normalize</p>

                {#each pnSchema.params as p (p.name)}
                  <label>
                    <span>{p.label}</span>

                    {#if p.type === "bool"}
                      <select bind:value={pnValues[p.name]}>
                        <option value={true}>Yes</option>
                        <option value={false}>No</option>
                      </select>
                    {:else if p.type === "int"}
                      <input type="number" bind:value={pnValues[p.name]} min="0" step="1" />
                    {:else if p.type === "float"}
                      <input type="number" bind:value={pnValues[p.name]} step="0.01" />
                    {:else}
                      <input type="text" bind:value={pnValues[p.name]} />
                    {/if}
                  </label>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      </div>

      <!-- ===== Right column: Summary + Compatibility ===== -->
      <div class="right-column">
        <div class="summary card">
          <h3>Summary</h3>

          <!-- Solvers -->
          <div class="summary-section">
            <p><strong>Solvers</strong></p>
            {#if summarySolvers.length === 0}
              <p style="color:#6b7280;">No solvers added.</p>
            {/if}
            {#each summarySolvers as s, i (`${s.name}-${i}`)}
              <div class="summary-item">
                <button
                  class="summary-toggle pill"
                  on:click={() => { s.expanded = !s.expanded; summarySolvers = [...summarySolvers]; }}
                  title={s.name}
                >
                  <span class="pill-text">{s.name}</span>
                  <span class="pill-right">
                    <span class="pill-chevron">{s.expanded ? "▼" : "▶"}</span>
                    <a
                      href="#remove-solver"
                      class="pill-close"
                      title="Remove"
                      on:click|preventDefault|stopPropagation={() => removeSummarySolver(i)}
                    >×</a>
                  </span>
                </button>
                {#if s.expanded}
                  <ul class="param-list" style="margin:.5rem 0;">
                    {#each s.params as p (p.name)}<li><strong>{p.name}:</strong> {p.value ?? p.default ?? ""}</li>{/each}
                  </ul>
                  <button class="secondary-outline" on:click={() => requestEdit('solver', i)}>Edit</button>
                {/if}
              </div>
            {/each}
          </div>

          <!-- Problems -->
          <div class="summary-section" style="margin-top:1rem;">
            <p><strong>Problems</strong></p>
            {#if summaryProblems.length === 0}
              <p style="color:#6b7280;">No problems added.</p>
            {/if}
            {#each summaryProblems as p, i (`${p.name}-${i}`)}
              <div class="summary-item">
                <button
                  class="summary-toggle pill"
                  on:click={() => { p.expanded = !p.expanded; summaryProblems = [...summaryProblems]; }}
                  title={p.name}
                >
                  <span class="pill-text">{p.name}</span>
                  <span class="pill-right">
                    <span class="pill-chevron">{p.expanded ? "▼" : "▶"}</span>
                    <a
                      href="#remove-problem"
                      class="pill-close"
                      title="Remove"
                      on:click|preventDefault|stopPropagation={() => removeSummaryProblem(i)}
                    >×</a>
                  </span>
                </button>
                {#if p.expanded}
                  <ul class="param-list" style="margin:.5rem 0;">
                    {#each p.params as q (q.name)}<li><strong>{q.name}:</strong> {q.value ?? q.default ?? ""}</li>{/each}
                  </ul>
                  <button class="secondary-outline" on:click={() => requestEdit('problem', i)}>Edit</button>
                {/if}
              </div>
            {/each}
          </div>

          <!-- Plots -->
          <div class="summary-section" style="margin-top:1rem;">
            <p><strong>Plots</strong></p>
            {#if summaryPlots.length === 0}
              <p style="color:#6b7280;">No plots added.</p>
            {/if}

            {#each summaryPlots as pl, i (`${pl.name}-${i}`)}
              <div class="summary-item">
                <button
                  class="summary-toggle pill"
                  on:click={() => {
                    pl.expanded = !pl.expanded;
                    summaryPlots = [...summaryPlots];
                  }}
                  title={pl.name}
                >
                  <span class="pill-text">{pl.name}</span>
                  <span class="pill-right">
                    <span class="pill-chevron">{pl.expanded ? "▼" : "▶"}</span>
                    <a
                      href="#remove-plot"
                      class="pill-close"
                      title="Remove"
                      on:click|preventDefault|stopPropagation={() => removeSummaryPlot(i)}
                    >×</a>
                  </span>
                </button>

                {#if pl.expanded}
                  {#if pl.params && pl.params.length}
                      <ul class="param-list" style="margin:.5rem 0;">
                          {#each pl.params as p (p.name)}
                              <li><strong>{p.name}:</strong> {p.value ?? p.default ?? ""}</li>
                          {/each}
                      </ul>
                  {:else}
                      <p style="margin:.5rem 0;color:#6b7280;">No parameters.</p>
                  {/if}
                  <button class="secondary-outline" on:click={() => requestEdit('plot', i)}>Edit</button>
              {/if}
              </div>
            {/each}
          </div>
        </div>

        <!-- Compatibility + progress + full-width trigger -->
        {#if summarySolvers.length && summaryProblems.length}
          <div class="card compatibility-section compact">
            <div class="compat-header">
              <h3>Compatibility</h3>

              {#if totalCompatCells > 0}
                <span class="compat-header-pct">
                  {greenPct}% compatible
                </span>
              {/if}
            </div>

            {#if totalCompatCells > 0}
              <div class="compat-progress" aria-label="Compatibility summary">
                <div
                  class="compat-bar"
                  role="progressbar"
                  aria-valuemin="0"
                  aria-valuemax="100"
                  aria-valuenow={greenPct}
                  aria-label={`${greenPct}% compatible, ${redPct}% incompatible`}
                >
                  <div class="bar-green" style="width:{greenPct}%;"></div>
                  <div class="bar-red"   style="width:{redPct}%;"></div>
                </div>
              </div>
            {:else}
              <p class="compat-progress-empty">Add at least one solver and one problem to see compatibility.</p>
            {/if}

            <button
              class="compat-trigger"
              class:red-btn={hasIncompatibility}
              class:green-btn={!hasIncompatibility}
              on:click={openCompatModal}
              style="display:block;width:100%;margin-top:.75rem;"
            >
              Open matrix
            </button>
          </div>
        {/if}
      </div>
    </div>

    <div class="button-row">
      <button class="primary" on:click={runExperiment}>Run Experiment</button>
    </div>

    <!-- === Modals (unchanged) === -->
    {#if showConfirm}
      <div
        class="modal-backdrop"
        role="button"
        tabindex="-1"
        aria-label="Close confirmation dialog"
        on:click={onConfirmBackdropClick}
        on:keydown={onConfirmBackdropKeydown}
      >
        <div class="modal" role="dialog" aria-modal="true" aria-labelledby="confirm-dialog-title" tabindex="-1">
          <h3 id="confirm-dialog-title">Replace current editor?</h3>
          <p>You already have a {confirmKind === 'solver' ? 'solver' : confirmKind === 'problem' ? 'problem' : 'plot'} open in the editor. If you continue, the current selection and any unsaved parameter changes will be replaced.</p>
          <div class="modal-actions">
            <button class="btn" on:click={closeConfirm}>Cancel</button>
            <button class="btn btn-primary" on:click={confirmProceed}>Replace</button>
          </div>
        </div>
      </div>
    {/if}

    {#if showCompatModal}
      <div
        class="modal-backdrop"
        role="button"
        tabindex="-1"
        aria-label="Close compatibility matrix"
        on:click={onCompatBackdropClick}
        on:keydown={onCompatBackdropKeydown}
      >
        <div class="modal" role="dialog" aria-modal="true" aria-labelledby="compat-dialog-title" tabindex="-1">
          <h3 id="compat-dialog-title" style="margin-top:0">Solver–Problem Compatibility</h3>
          <table class="compatibility-table compact" aria-label="Solver–Problem compatibility matrix">
            <thead>
              <tr>
                <th scope="col">S \ P</th>
                {#each summaryProblems as p (p.name)}
                  <th scope="col" title={p.name}>{abbrev(p.name)}</th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each summarySolvers as s (s.name)}
                <tr>
                  <th class="solver-name" scope="row" title={s.name}>{abbrev(s.name)}</th>
                  {#each summaryProblems as p (p.name)}
                    <td
                      class={compatibility[s.name]?.[p.name] ? (compatibility[s.name][p.name].compatible ? 'compat-cell ok' : 'compat-cell bad') : 'compat-cell neutral'}
                      title={compatibility[s.name]?.[p.name] && !compatibility[s.name][p.name].compatible && compatibility[s.name][p.name].message ? `${s.name} × ${p.name}: ${compatibility[s.name][p.name].message}` : ''}
                    >&nbsp;</td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
          <div class="modal-actions" style="margin-top:0.75rem;">
            <button class="btn" on:click={closeCompatModal}>Close</button>
          </div>
        </div>
      </div>
    {/if}
  {/if}
</main>

<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  :global(body), main, input, button, select, label {
    font-family: 'Inter', sans-serif;
    font-size: 15px;
  }

  /* === NAVBAR === */
  nav {
    background: #e5e7eb;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 1.5rem;
    width: 100%;
    box-sizing: border-box;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 1000;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  .nav-left { flex-shrink: 0; }

  .title {
    font-size: 2rem;
    font-weight: 700;
    color: #0f172a;
    white-space: nowrap;
  }

  .nav-right ul {
    display: flex;
    gap: 1.25rem;
    margin: 0;
    padding: 0;
    list-style: none;
  }

  .nav-right a {
    cursor: pointer;
    font-weight: 500;
    color: #374151;
    white-space: nowrap;
    text-decoration: none;
  }

  .nav-right a.active {
    color: #0f172a;
    border-bottom: 2px solid #14b8a6;
  }

  /* === MAIN LAYOUT === */
  main {
    font-family: 'Inter', sans-serif;
    margin: 100px 20px 20px;
  }

  h2 {
    color: #2563eb;
    margin-top: 0;
    margin-bottom: 0.75rem;
    font-size: 1.2em;
    font-weight: 600;
  }

  /* Three-column layout: solvers | problems | right column (summary+compat) */
  .row-3col {
    display: grid;
    grid-template-columns: 0.7fr 0.7fr 0.45fr;
    gap: 1.75rem;
    margin-bottom: 1.5rem;
    align-items: start;
  }

  /* Right column stacks Summary + Compatibility */
  .right-column {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    min-width: 0;
  }

  /* === SUMMARY PANEL === */
  .summary {
    max-height: 75vh;
    overflow-y: auto;
    overflow-x: hidden;
    white-space: normal;
    word-wrap: break-word;
    background: #fafafa;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 1rem;
  }

  .summary h3 {
    margin-top: 0;
    color: #0f172a;
    font-size: 1.1em;
  }

  .summary ul { padding-left: 1rem; margin: 0.25rem 0 1rem; }

  .summary li {
    font-size: 14px;
    color: #374151;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* Spacing between summary items */
  .summary-section { margin-bottom: 1rem; }
  .summary-item + .summary-item { margin-top: 0.75rem; }

  /* Outlined toggle buttons inside summary */
  .summary-toggle {
    display: block;
    width: 100%;
    text-align: left;
    padding: 0.65rem 0.9rem;
    background: #ffffff;              /* white fill */
    color: #1e3a8a;                   /* blue-ish text */
    border: 1.5px solid #2563eb;      /* blue outline */
    border-radius: 10px;
    font-weight: 600;
    line-height: 1.2;
    cursor: pointer;
    transition: background-color .15s ease, box-shadow .15s ease, border-color .15s ease;
  }
  .summary-toggle:hover {
    background: #eff6ff;              /* light blue hover */
    box-shadow: 0 1px 4px rgba(37, 99, 235, 0.15);
  }
  .summary-toggle:active { background: #dbeafe; }
  .summary-toggle:focus-visible {
    outline: 3px solid rgba(37, 99, 235, .35);
    outline-offset: 2px;
  }

  /* === CARDS === */
  .card {
    background: #ffffff;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    margin-bottom: 1.5rem;
  }

  .block-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  /* === INPUTS & SELECTS === */
  select,
  input[type="number"],
  input[type="text"] {
    margin: 0.5rem 0;
    padding: 0.5rem;
    width: 100%;
    max-width: 300px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    font-size: 15px;
    box-sizing: border-box;
  }

  /* === PARAMETER BOX === */
  .param-box {
    border: 1px solid #93c5fd;
    background: #eff6ff;
    padding: 0.75rem 1rem;
    border-radius: 6px;
    margin-top: 0.5rem;
    position: relative;
    overflow: visible !important; /* tooltips shouldn't be clipped */
  }

  .param-title {
    margin: 0 0 0.5rem;
    font-weight: 600;
    color: #1d4ed8;
  }

  /* Label row layout */
  .param-box label {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 0.5rem; /* tight gap between name and input */
    margin-bottom: 0.4rem;
  }

  /* Label text column */
  .param-box label span {
    flex: 0 0 140px; /* tidy left column */
    text-align: left;
  }

  /* Input */
  .param-box input[type="text"],
  .param-box input[type="number"] {
    flex: 1;
    max-width: 200px;   /* longer inputs but still align right edge */
    text-align: right;
    margin-left: 8px;   /* small space from label */
  }

  /* === BUTTONS === */
  button {
    font-size: 15px;
    font-weight: 500;
    border-radius: 6px;
    cursor: pointer;
  }

  .secondary-outline {
    background: white;
    border: 1px solid #2563eb;
    color: #2563eb;
    padding: 0.4rem 0.8rem;
    margin-top: 0.5rem;
  }
  .secondary-outline:hover { background: #eff6ff; }

  .button-row { display: flex; justify-content: flex-start; margin: 1rem 0; }

  /* === DROPDOWN SECTIONS === */
  .dropdown {
    background: #d0e2ff;
    color: #1e3a8a;
    padding: 0.6rem 1rem;
    border-radius: 6px;
    border: 1px solid #a6c8ff;
    width: 100%;
    font-weight: 500;
    cursor: pointer;
    text-align: left;
  }
  .dropdown:hover { background: #a6c8ff; }

  .dropdown-content {
    border: 1px solid #c7d2fe;
    background: #f9fafb;
    padding: 0.75rem;
    margin-top: 0.5rem;
    border-radius: 6px;
  }

  /* === INFO ICON + TOOLTIP === */
  .info-wrapper {
    position: relative;
    display: inline-block;
    margin-left: 4px;
    vertical-align: text-top;
    z-index: 1000;
  }

  .info-icon {
    font-size: 0.8rem;
    color: #2563eb;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 15px;
    height: 15px;
    background-color: #e0f2fe;
    border: 1px solid #93c5fd;
    cursor: help;
    transform: translateY(-2px);
  }

  .tooltip {
    position: absolute;
    bottom: 130%;
    left: 0;
    transform: translateX(-10%);
    background-color: #111827;
    color: #f9fafb;
    text-align: left;
    border-radius: 6px;
    padding: 0.4rem 0.6rem;
    white-space: normal;
    width: max-content;
    max-width: 280px;
    font-size: 0.8rem;
    line-height: 1.3;
    z-index: 3000;
    visibility: hidden;
    opacity: 0;
    transition: opacity 0.2s ease;
    overflow-wrap: break-word;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  }

  .tooltip::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 12px;
    border-width: 5px;
    border-style: solid;
    border-color: #111827 transparent transparent transparent;
  }

  .info-wrapper:hover .tooltip { visibility: visible; opacity: 1; }

  /* === COMPATIBILITY MATRIX (colored cells) === */
  .compatibility-section { padding: 1rem; border: 1px solid #e5e7eb; border-radius: 8px; background: #fafafa; }
  .compatibility-section.compact { padding: 0.75rem; }

  .compatibility-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 0.75rem;
    text-align: center;
    table-layout: fixed;
  }
  .compatibility-table.compact { font-size: 12px; }

  .compatibility-table th,
  .compatibility-table td {
    border: 1px solid #d1d5db;
    padding: 0.55rem;
    font-size: 0.95rem;
  }
  .compatibility-table.compact th,
  .compatibility-table.compact td { padding: 0.35rem; }

  .compatibility-table thead th {
    background: #dbeafe;
    color: #1e3a8a;
    font-weight: 600;
  }

  .solver-name {
    font-weight: 600;
    background: #eff6ff;
    text-align: left;
    padding-left: 0.5rem;
    width: auto; /* shrink to acronym */
  }

  .compat-cell {
    transition: background-color 0.15s ease, color 0.15s ease;
    font-weight: 600;
    min-width: 28px;
    height: 24px;
    line-height: 1;
  }

  .compat-cell.ok   { background: #dcfce7; color: #166534; }
  .compat-cell.bad  { background: #fee2e2; color: #991b1b; }
  .compat-cell.neutral { background: #f3f4f6; color: #6b7280; }

  /* Modal */
  .modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(15, 23, 42, 0.4);
    display: grid;
    place-items: center;
    z-index: 2000;
  }
  .modal {
    background: #ffffff;
    width: min(520px, 92vw);
    border-radius: 12px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.18);
    padding: 1.25rem 1.25rem 1rem;
  }
  .modal h3 {
    margin: 0 0 0.5rem;
    font-size: 1.1rem;
    color: #0f172a;
  }
  .modal p {
    margin: 0 0 1rem;
    color: #374151;
    line-height: 1.4;
  }
  .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
  }
  .btn {
    border: 1px solid #cbd5e1;
    background: #fff;
    color: #0f172a;
    padding: 0.45rem 0.9rem;
    border-radius: 8px;
    cursor: pointer;
  }
  .btn:hover { background: #f8fafc; }
  .btn-primary {
    border-color: #2563eb;
    background: #2563eb;
    color: #fff;
  }
  .btn-primary:hover { background: #1e40af; }

  /* Summary pill with inline close (×) */
  .summary-toggle.pill {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: .5rem;
    width: 100%;
    background: #fff;
    color: #1e3a8a;
    border: 1.5px solid #2563eb;
    border-radius: 10px;
    padding: .55rem .6rem .55rem .75rem;
    text-align: left;
  }
  .summary-toggle.pill:hover { background: #eff6ff; }
  .pill-text {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .pill-right {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
  }
  .pill-chevron { font-weight: 700; font-size: .9rem; color: #1e3a8a; }
  .pill-close {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px;
    height: 22px;
    border-radius: 6px;
    border: 1px solid #e5e7eb;
    color: #6b7280;
    cursor: pointer;
    text-decoration: none;
  }
  .pill-close:hover { background: #f9fafb; color: #111827; border-color: #d1d5db; }

  /* space out items more */
  .summary-item + .summary-item { margin-top: .9rem; }
  .compat-trigger {
    border: 1.5px solid #2563eb;
    background: #ffffff;
    color: #1e3a8a;
    padding: 0.45rem 0.9rem;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    transition: background-color .15s ease, color .15s ease, border-color .15s ease, box-shadow .15s ease;
  }

  /* Green variant (no incompatibilities) */
  .compat-trigger.green-btn {
    border-color: #16a34a;
    color: #14532d;
    background: #dcfce7;
  }
  .compat-trigger.green-btn:hover {
    background: #bbf7d0;
    box-shadow: 0 1px 4px rgba(22,163,74,.2);
  }

  /* Red variant (>=1 incompatibility) */
  .compat-trigger.red-btn {
    border-color: #dc2626;
    color: #7f1d1d;
    background: #fee2e2;
  }
  .compat-trigger.red-btn:hover {
    background: #fecaca;
    box-shadow: 0 1px 4px rgba(220,38,38,.2);
  }

  /* Keyboard focus */
  .compat-trigger:focus-visible {
    outline: 3px solid rgba(37,99,235,.35);
    outline-offset: 2px;
  }

  /* --- Compatibility progress bar --- */
  .compat-progress { margin-bottom: 0.6rem; }

  .compat-bar {
    display: flex;            /* lays green and red side-by-side */
    width: 100%;
    height: 10px;
    background: #f3f4f6;      /* neutral track */
    border-radius: 9999px;
    overflow: hidden;
    box-shadow: inset 0 0 0 1px #e5e7eb;
  }

  .bar-green {
    height: 100%;
    background: #22c55e;      /* green-500 */
  }

  .bar-red {
    height: 100%;
    background: #ef4444;      /* red-500 */
  }

  .compat-progress-empty {
    margin: 0 0 0.6rem 0;
    color: #6b7280;
    font-size: 0.9rem;
  }

  .col-stack {
    display: flex;
    flex-direction: column;
    gap: 1rem;        /* small consistent spacing between cards */
    min-width: 0;
  }
</style>
