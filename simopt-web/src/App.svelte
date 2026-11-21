<script lang="ts">
  import { onMount, tick } from 'svelte';
  import AboutUs from './lib/AboutUs.svelte';
  import DataFarming from './lib/DataFarming.svelte';
  import UserGuide from './lib/UserGuide.svelte';
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

  function addDesignToExperiment(kind: "solver" | "problem", entries: SummaryEntry[]): void {
    if (kind === "solver") summarySolvers = [...summarySolvers, ...entries];
    else summaryProblems = [...summaryProblems, ...entries];
    navigate("Simulator");
  }

  let allSolvers: string[] = [];
  let allProblems: string[] = [];

  let selectedSolverName = "";     // blank by default
  let selectedProblemName = "";    // blank by default
  let solverParams: Param[] = [];           // [{name, description, default, value}]
  let problemParams: Param[] = [];

  let summarySolvers: SummaryEntry[] = [];         // [{name, params:[...], expanded?:bool}]
  let summaryProblems: SummaryEntry[] = [];
  // Design points share a class name; pickers and the matrix use class names once each.
  $: uniqueSolverNames = [...new Set(summarySolvers.map(s => s.name))];
  $: uniqueProblemNames = [...new Set(summaryProblems.map(p => p.name))];

  // { kind: 'solver'|'problem', index: number } | null
  let editMode: EditMode | null = null;

  let macroreps = 10;
  type Tab = 'solvers' | 'problems' | 'plots' | 'settings';
  let activeTab: Tab = 'solvers';
  const TABS: Tab[] = ['solvers', 'problems', 'plots', 'settings'];
  $: if (activeTab) {
    tick().then(() => document.getElementById(`tab-${activeTab}`)?.scrollIntoView({ block: 'nearest', inline: 'nearest' }));
  }
  function onTabKeydown(e: KeyboardEvent, i: number): void {
    if (!['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(e.key)) return;
    e.preventDefault();
    const n = e.key === 'Home' ? 0
      : e.key === 'End' ? TABS.length - 1
      : (i + (e.key === 'ArrowRight' ? 1 : TABS.length - 1)) % TABS.length;
    activeTab = TABS[n];
    (document.getElementById(`tab-${TABS[n]}`) as HTMLElement | null)?.focus();
  }

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
    activeTab = kind === 'solver' ? 'solvers' : kind === 'problem' ? 'problems' : 'plots';
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
    if (summaryPlots.length === 0) {
      alert("Please add at least one plot before running the experiment.");
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
      problems: summaryProblems.map(p => {
        const collect = (isModel: boolean) =>
          p.params.reduce<Record<string, unknown>>((acc, param) => {
            if ((param.source === "model") !== isModel) return acc;
            const parsed = parseValue(param.value);
            if (parsed !== null) acc[param.name] = parsed;
            return acc;
          }, {});
        return {
          name: p.name,
          rename: p.rename ?? p.name,
          fixed_factors: collect(false),
          model_fixed_factors: collect(true)
        };
      }),
      solvers: summarySolvers.map(s => ({
        name: s.name,
        rename: s.rename ?? s.name,
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
      solvers: [...new Set(solvers.map(s => s.name))],
      problems: [...new Set(problems.map(p => p.name))]
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
  
  $: macrorepsValid = Number.isInteger(macroreps) && macroreps >= 1;
  $: canRun = summarySolvers.length > 0 && summaryProblems.length > 0 && macrorepsValid;

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

{#snippet tip(desc: string | undefined, id: string)}
  {#if desc}
    <span class="info-wrapper">
      <span class="info-icon" role="button" aria-label="Description" tabindex="0" aria-describedby={id}>i</span>
      <span class="tooltip" role="tooltip" {id}>{desc}</span>
    </span>
  {/if}
{/snippet}

<nav>
  <a class="brand" href="#simulator" on:click|preventDefault={() => navigate("Simulator")}>
    <svg class="brand-mark" viewBox="0 0 24 24" aria-hidden="true"><path d="M3 4 L8 13 L12 9 L16 17 L21 19" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="21" cy="19" r="2.2" fill="currentColor"/></svg>
    <span class="title">SimOpt</span>
  </a>
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
        href="#data-farming"
        class:active={currentPage === "Data Farming"}
        on:click|preventDefault={() => navigate("Data Farming")}
      >Data Farming</a>
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
</nav>

<main>
  {#if currentPage === "Simulator"}
    <div class="layout">
      <!-- ===== Left: tabbed editor ===== -->
      <section class="editor card">
        <div class="tabs" role="tablist" aria-label="Experiment editor">
          {#each TABS as tab, ti (tab)}
            <button
              type="button"
              role="tab"
              id={`tab-${tab}`}
              class="tab"
              class:active={activeTab === tab}
              aria-selected={activeTab === tab}
              aria-controls="tabpanel"
              tabindex={activeTab === tab ? 0 : -1}
              on:click={() => (activeTab = tab)}
              on:keydown={(e) => onTabKeydown(e, ti)}
            >
              {tab === 'solvers' ? 'Solvers' : tab === 'problems' ? 'Problems' : tab === 'plots' ? 'Plots' : 'Settings'}
              {#if tab !== 'settings'}
                <span class="count">{tab === 'solvers' ? summarySolvers.length : tab === 'problems' ? summaryProblems.length : summaryPlots.length}</span>
              {/if}
            </button>
          {/each}
        </div>

        <div class="tab-body" id="tabpanel" role="tabpanel" aria-labelledby={`tab-${activeTab}`}>
          {#if activeTab === 'solvers'}
            <div class="picker">
              <select aria-label="Solver" bind:value={selectedSolverName} on:change={(e) => onSolverChange(inputValue(e))}>
                <option value="">Choose a solver…</option>
                {#each allSolvers as option (option)}
                  <option value={option}>{option}</option>
                {/each}
              </select>
              {#if selectedSolverName}
                <button class="btn-primary-sm" on:click={addSolverToSummary}>
                  {editMode?.kind === 'solver' ? 'Apply Changes' : '+ Add Solver'}
                </button>
              {/if}
            </div>

            {#if selectedSolverName}
              <h4 class="section-title">Solver parameters</h4>
              <div class="param-grid">
                {#each solverParams as param, idx (param.name)}
                  <label class="param-row">
                    <span class="param-name"><span class="param-name-text" title={param.name}>{param.name}</span>{@render tip(param.description, `tip-solver-${param.name}`)}</span>
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
            {:else}
              <p class="muted">Pick a solver to configure its parameters, then add it to the experiment.</p>
            {/if}

          {:else if activeTab === 'problems'}
            <div class="picker">
              <select aria-label="Problem" bind:value={selectedProblemName} on:change={(e) => onProblemChange(inputValue(e))}>
                <option value="">Choose a problem…</option>
                {#each allProblems as option (option)}
                  <option value={option}>{option}</option>
                {/each}
              </select>
              {#if selectedProblemName}
                <button class="btn-primary-sm" on:click={addProblemToSummary}>
                  {editMode?.kind === 'problem' ? 'Apply Changes' : '+ Add Problem'}
                </button>
              {/if}
            </div>

            {#if selectedProblemName}
              <h4 class="section-title">Problem parameters</h4>
              <div class="param-grid">
                {#each problemParams as param, idx (param.name)}
                  <label class="param-row">
                    <span class="param-name"><span class="param-name-text" title={param.name}>{param.name}</span>{@render tip(param.description, `tip-problem-${param.name}`)}</span>
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
            {:else}
              <p class="muted">Pick a problem to configure its parameters, then add it to the experiment.</p>
            {/if}

          {:else if activeTab === 'plots'}
            <div class="picker">
              <select aria-label="Plot" bind:value={selectedPlotName} on:change={(e) => onPlotChange(inputValue(e))}>
                <option value="">Choose a plot type…</option>
                {#each allPlots as plot (plot)}
                  <option value={plot}>{plot}</option>
                {/each}
              </select>
              {#if selectedPlotName}
                <button class="btn-primary-sm" on:click={addPlotToSummary}>
                  {editMode?.kind === 'plot' ? 'Apply Changes' : '+ Add Plot'}
                </button>
              {/if}
            </div>

            {#if plotParams.length}
              <h4 class="section-title">Plot parameters</h4>
              <div class="param-grid">
                {#each plotParams as p, i (p.name)}
                  <label class="param-row">
                    <span class="param-name"><span class="param-name-text" title={p.name}>{p.name}</span>{@render tip(p.description, `tip-plot-${p.name}`)}</span>
                    {#if p.name === 'ref_solver'}
                      <select
                        value={p.value}
                        on:change={(e) => { plotParams[i].value = inputValue(e); plotParams = [...plotParams]; }}
                      >
                        <option value="">— None —</option>
                        {#each uniqueSolverNames as name (name)}
                          <option value={name}>{name}</option>
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
              <div class="check-cols">
                <div>
                  <h4 class="section-title">Solvers <span class="hint">Leave all unchecked to include every solver</span></h4>
                  {#if summarySolvers.length === 0}
                    <p class="muted">No solvers in the experiment yet.</p>
                  {:else}
                    {#each uniqueSolverNames as name (name)}
                      <label class="check-row" title={name}>
                        <input type="checkbox" value={name} bind:group={selectedPlotSolvers} />
                        <span>{name}</span>
                      </label>
                    {/each}
                  {/if}
                </div>
                <div>
                  <h4 class="section-title">Problems <span class="hint">Leave all unchecked to include every problem</span></h4>
                  {#if summaryProblems.length === 0}
                    <p class="muted">No problems in the experiment yet.</p>
                  {:else}
                    {#each uniqueProblemNames as name (name)}
                      <label class="check-row" title={name}>
                        <input type="checkbox" value={name} bind:group={selectedPlotProblems} />
                        <span>{name}</span>
                      </label>
                    {/each}
                  {/if}
                </div>
              </div>
            {:else}
              <p class="muted">Pick a plot type to configure it and choose which solvers and problems it covers.</p>
            {/if}

          {:else}
            <h4 class="section-title">Post-replicate</h4>
            <div class="param-grid wide">
              {#each prSchema.params as p (p.name)}
                <label class="param-row">
                  <span class="param-name"><span class="param-name-text" title={p.label}>{p.label}</span></span>
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

            <h4 class="section-title">Post-normalize</h4>
            <div class="param-grid wide">
              {#each pnSchema.params as p (p.name)}
                <label class="param-row">
                  <span class="param-name"><span class="param-name-text" title={p.label}>{p.label}</span></span>
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
            <p class="muted">Macroreplications are set in the experiment panel, above the run button.</p>
          {/if}
        </div>
      </section>

      <!-- ===== Right: sticky experiment panel ===== -->
      <aside class="panel card">
        <h3>Experiment</h3>

        <div class="group">
          <div class="group-head"><span>Solvers</span><span class="count">{summarySolvers.length}</span></div>
          {#if summarySolvers.length === 0}<p class="muted empty">None yet. Add one from the Solvers tab.</p>{/if}
          {#each summarySolvers as s, i (`${s.name}-${i}`)}
            <div class="item">
              <div class="chip">
                <button type="button" class="chip-main" title={s.rename ?? s.name} aria-expanded={!!s.expanded}
                  on:click={() => { s.expanded = !s.expanded; summarySolvers = [...summarySolvers]; }}>
                  <span class="chev" class:open={s.expanded} aria-hidden="true"></span>
                  <span class="chip-text">{s.rename ?? s.name}</span>
                </button>
                <button type="button" class="chip-x" title="Remove" aria-label="Remove solver" on:click={() => removeSummarySolver(i)}>×</button>
              </div>
              {#if s.expanded}
                <ul class="param-list">
                  {#each s.params as p (p.name)}<li title={`${p.name}: ${p.value ?? p.default ?? ""}`}><strong>{p.name}:</strong> {p.value ?? p.default ?? ""}</li>{/each}
                </ul>
                <button class="btn-link" on:click={() => requestEdit('solver', i)}>Edit</button>
              {/if}
            </div>
          {/each}
        </div>

        <div class="group">
          <div class="group-head"><span>Problems</span><span class="count">{summaryProblems.length}</span></div>
          {#if summaryProblems.length === 0}<p class="muted empty">None yet. Add one from the Problems tab.</p>{/if}
          {#each summaryProblems as p, i (`${p.name}-${i}`)}
            <div class="item">
              <div class="chip">
                <button type="button" class="chip-main" title={p.rename ?? p.name} aria-expanded={!!p.expanded}
                  on:click={() => { p.expanded = !p.expanded; summaryProblems = [...summaryProblems]; }}>
                  <span class="chev" class:open={p.expanded} aria-hidden="true"></span>
                  <span class="chip-text">{p.rename ?? p.name}</span>
                </button>
                <button type="button" class="chip-x" title="Remove" aria-label="Remove problem" on:click={() => removeSummaryProblem(i)}>×</button>
              </div>
              {#if p.expanded}
                <ul class="param-list">
                  {#each p.params as q (q.name)}<li title={`${q.name}: ${q.value ?? q.default ?? ""}`}><strong>{q.name}:</strong> {q.value ?? q.default ?? ""}</li>{/each}
                </ul>
                <button class="btn-link" on:click={() => requestEdit('problem', i)}>Edit</button>
              {/if}
            </div>
          {/each}
        </div>

        <div class="group">
          <div class="group-head"><span>Plots</span><span class="count">{summaryPlots.length}</span></div>
          {#if summaryPlots.length === 0}<p class="muted empty">None yet. Add one from the Plots tab.</p>{/if}
          {#each summaryPlots as pl, i (`${pl.name}-${i}`)}
            <div class="item">
              <div class="chip">
                <button type="button" class="chip-main" title={pl.name} aria-expanded={!!pl.expanded}
                  on:click={() => { pl.expanded = !pl.expanded; summaryPlots = [...summaryPlots]; }}>
                  <span class="chev" class:open={pl.expanded} aria-hidden="true"></span>
                  <span class="chip-text">{pl.name}</span>
                </button>
                <button type="button" class="chip-x" title="Remove" aria-label="Remove plot" on:click={() => removeSummaryPlot(i)}>×</button>
              </div>
              {#if pl.expanded}
                {#if pl.params && pl.params.length}
                  <ul class="param-list">
                    {#each pl.params as p (p.name)}
                      <li title={`${p.name}: ${p.value ?? p.default ?? ""}`}><strong>{p.name}:</strong> {p.value ?? p.default ?? ""}</li>
                    {/each}
                  </ul>
                {:else}
                  <p class="muted">No parameters.</p>
                {/if}
                <button class="btn-link" on:click={() => requestEdit('plot', i)}>Edit</button>
              {/if}
            </div>
          {/each}
        </div>

        <div class="group compat">
          <div class="group-head">
            <span>Compatibility</span>
            {#if totalCompatCells > 0}<span class="compat-pct" class:has-bad={hasIncompatibility}>{greenPct}% compatible</span>{/if}
          </div>
          {#if canRun && totalCompatCells > 0}
            <button
              type="button"
              class="compat-grid"
              style="--cols:{uniqueProblemNames.length}"
              on:click={openCompatModal}
              aria-label={`${greenPct}% compatible, ${redPct}% incompatible. Open the compatibility matrix.`}
            >
              {#each uniqueSolverNames as sn (sn)}
                {#each uniqueProblemNames as pn (pn)}
                  <span class="dot {compatibility[sn]?.[pn] ? (compatibility[sn][pn].compatible ? 'ok' : 'bad') : 'neutral'}"></span>
                {/each}
              {/each}
            </button>
            <p class="compat-caption">
              {uniqueSolverNames.length} solver{uniqueSolverNames.length === 1 ? '' : 's'} × {uniqueProblemNames.length} problem{uniqueProblemNames.length === 1 ? '' : 's'}.
              <button type="button" class="btn-link inline" on:click={openCompatModal}>Open matrix</button>
            </p>
          {:else}
            <p class="muted empty">Add a solver and a problem to check which pairs can run together.</p>
          {/if}
        </div>

        <div class="group run-group">
          <label class="param-row macroreps">
            <span class="param-name"><span class="param-name-text">Macroreplications</span></span>
            <input type="number" min="1" step="1" bind:value={macroreps} class:invalid={!macrorepsValid} aria-invalid={!macrorepsValid} />
          </label>
          <button class="run-btn" disabled={!canRun || summaryPlots.length === 0} on:click={runExperiment}>
            <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M4 2.5v11l9-5.5z" fill="currentColor"/></svg>
            Run Experiment
          </button>
          {#if !macrorepsValid}
            <p class="muted run-hint panel-invalid">Macroreps must be a whole number ≥ 1.</p>
          {/if}
          {#if summarySolvers.length === 0 || summaryProblems.length === 0}
            <p class="muted run-hint">
              Add at least one {summarySolvers.length === 0 ? 'solver' : ''}{summarySolvers.length === 0 && summaryProblems.length === 0 ? ' and one ' : ''}{summaryProblems.length === 0 ? 'problem' : ''} to run.
            </p>
          {/if}
          {#if summaryPlots.length === 0}
            <p class="muted run-hint">Add at least one plot to run.</p>
          {/if}
        </div>
      </aside>
    </div>

    <!-- === Modals === -->
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
          <h3 id="compat-dialog-title">Solver–problem compatibility</h3>
          <p class="modal-sub">Hover a red cell to see why the solver cannot run on that problem.</p>
          <table class="compatibility-table compact" aria-label="Solver–Problem compatibility matrix">
            <thead>
              <tr>
                <th scope="col" class="corner">Solver \ Problem</th>
                {#each uniqueProblemNames as pn (pn)}
                  <th scope="col" title={pn}>{abbrev(pn)}</th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each uniqueSolverNames as sn (sn)}
                <tr>
                  <th class="solver-name" scope="row" title={sn}>{abbrev(sn)}</th>
                  {#each uniqueProblemNames as pn (pn)}
                    <td
                      class={compatibility[sn]?.[pn] ? (compatibility[sn][pn].compatible ? 'compat-cell ok' : 'compat-cell bad') : 'compat-cell neutral'}
                      title={compatibility[sn]?.[pn] && !compatibility[sn][pn].compatible && compatibility[sn][pn].message ? `${sn} × ${pn}: ${compatibility[sn][pn].message}` : ''}
                    >&nbsp;</td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
          <div class="modal-actions">
            <button class="btn" on:click={closeCompatModal}>Close</button>
          </div>
        </div>
      </div>
    {/if}
  {/if}
  <!-- Kept mounted (just hidden) so the form survives page switches. -->
  <div hidden={currentPage !== "Data Farming"}>
    <DataFarming onAddToExperiment={addDesignToExperiment} />
  </div>
  {#if currentPage === "User Guide"}
    <UserGuide />
  {:else if currentPage === "About Us"}
    <AboutUs />
  {/if}
</main>

<style>
  /* === HEADER === */
  nav {
    background: var(--nav-bg);
    color: var(--nav-ink);
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    gap: 1.5rem;
    height: var(--nav-h);
    padding: 0 1.5rem;
    box-sizing: border-box;
    position: sticky;
    top: 0;
    z-index: 1000;
  }

  .brand {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    color: #fff;
    font-weight: 600;
  }
  .brand:hover { color: #fff; }
  .brand-mark { width: 22px; height: 22px; color: var(--brand-accent); }
  .title {
    font-size: 17px;
    font-weight: 600;
    letter-spacing: -0.01em;
    white-space: nowrap;
  }

  nav ul {
    display: flex;
    gap: 0.25rem;
    margin: 0;
    padding: 0;
    list-style: none;
  }
  nav li { display: flex; }

  nav ul a {
    display: flex;
    align-items: center;
    padding: 0 0.85rem;
    font-weight: 500;
    font-size: 14px;
    color: var(--nav-ink);
    white-space: nowrap;
    border-bottom: 2px solid transparent;
    border-top: 2px solid transparent;
    transition: color 0.15s;
  }
  nav ul a:hover { color: #fff; }
  nav ul a.active { color: #fff; border-bottom-color: var(--brand-accent); }

  /* === MAIN LAYOUT === */
  main {
    margin: 0 auto;
    padding: 1.5rem 1.5rem 3rem;
    max-width: 1480px;
    box-sizing: border-box;
  }

  .layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 352px;
    gap: 1.5rem;
    align-items: start;
  }

  /* === TABS === */
  .tabs {
    display: flex;
    gap: 0.25rem;
    padding: 0 1rem;
    box-shadow: inset 0 -1px 0 var(--border);
    overflow-x: auto;
    overflow-y: hidden;
    scrollbar-width: none;
  }

  .tab {
    background: transparent;
    color: var(--muted);
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    padding: 0.95rem 0.8rem 0.85rem;
    font-weight: 500;
    font-size: 14px;
    white-space: nowrap;
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
  }
  .tab:hover { background: transparent; color: var(--ink); }
  .tab.active { color: var(--ink); font-weight: 600; border-bottom-color: var(--primary); }
  .tab:focus-visible { outline: 2px solid var(--focus); outline-offset: -4px; border-radius: 6px; }

  .count {
    display: inline-block;
    min-width: 1.5em;
    padding: 0 0.4em;
    border-radius: 999px;
    background: var(--chip-bg);
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
    text-align: center;
    line-height: 1.6;
    font-variant-numeric: tabular-nums;
  }
  .tab.active .count { background: var(--primary-tint); color: var(--primary); }

  .tab-body { padding: 1.25rem 1.5rem 1.5rem; }

  .picker {
    display: flex;
    gap: 0.6rem;
    align-items: center;
    flex-wrap: wrap;
  }
  .picker select { flex: 1 1 280px; height: 38px; font-size: 14px; font-weight: 500; }

  .section-title {
    margin: 1.5rem 0 0.6rem;
    font-size: 13px;
    font-weight: 600;
    color: var(--ink);
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    flex-wrap: wrap;
  }
  .tab-body > .section-title:first-child { margin-top: 0; }
  .hint { font-weight: 400; color: var(--muted); font-size: 12px; }

  .muted { color: var(--muted); font-size: 13px; margin: 0.75rem 0 0; }

  /* === PARAMETER GRID === */
  .param-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(min(100%, 270px), 1fr));
    gap: 0.5rem 1.75rem;
  }
  .param-grid.wide { grid-template-columns: repeat(auto-fill, minmax(min(100%, 360px), 1fr)); }
  .wide .param-name { flex-basis: 55%; }
  .wide .param-name-text { font-family: var(--font-sans); font-size: 13.5px; }

  .param-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    min-width: 0;
    min-height: 34px;
  }

  .param-name {
    flex: 0 1 50%;
    min-width: 0;
    display: flex;
    align-items: center;
    gap: 0.35rem;
    color: var(--text);
  }
  .param-name-text {
    min-width: 0;
    font-family: var(--font-mono);
    font-size: 12.5px;
    white-space: normal;
    overflow-wrap: anywhere;
    line-height: 1.3;
  }

  select,
  input[type="number"],
  input[type="text"] {
    flex: 1 1 0;
    min-width: 0;
    width: 100%;
    margin: 0;
    padding: 0.35rem 0.6rem;
    height: 34px;
    border: 1px solid var(--border-strong);
    border-radius: var(--radius);
    background-color: var(--surface);
    color: var(--ink);
    font-size: 13.5px;
    box-sizing: border-box;
    transition: border-color 0.15s, box-shadow 0.15s;
  }
  .param-row input[type="text"],
  .param-row input[type="number"] { font-family: var(--font-mono); font-size: 13px; }
  select { text-overflow: ellipsis; cursor: pointer; }
  select:hover, input:hover { border-color: var(--muted-2); }
  select:focus, input:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px var(--focus-ring);
  }

  .check-cols {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr));
    gap: 0 1.75rem;
  }
  .check-row {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    min-height: 30px;
    min-width: 0;
    padding: 0 0.4rem;
    margin: 0 -0.4rem;
    border-radius: 6px;
    cursor: pointer;
  }
  .check-row:hover { background: var(--canvas); }
  .check-row span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  input[type="checkbox"] { accent-color: var(--primary); width: 15px; height: 15px; margin: 0; flex: none; }

  /* === BUTTONS === */
  .btn-primary-sm {
    padding: 0 1rem;
    height: 38px;
    box-sizing: border-box;
    white-space: nowrap;
  }

  .btn-link {
    background: transparent;
    color: var(--primary);
    border: none;
    padding: 0.15rem 0;
    font-size: 13px;
    font-weight: 500;
    margin: 0.2rem 0 0.1rem 1.6rem;
  }
  .btn-link:hover { background: transparent; text-decoration: underline; color: var(--primary-hover); }
  .btn-link.inline { margin: 0 0 0 0.25rem; font-size: 12.5px; }

  /* === EXPERIMENT PANEL === */
  .panel {
    position: sticky;
    top: calc(var(--nav-h) + 1.5rem);
    max-height: calc(100vh - var(--nav-h) - 3rem);
    overflow-y: auto;
    padding: 0;
    display: flex;
    flex-direction: column;
  }
  .panel h3 {
    margin: 0;
    padding: 1rem 1.25rem 0.9rem;
    font-size: 15px;
    font-weight: 600;
    color: var(--ink);
    border-bottom: 1px solid var(--border);
  }

  .group { padding: 0.9rem 1.25rem; border-top: 1px solid var(--border); }
  .panel h3 + .group { border-top: none; }
  .group-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 13px;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 0.45rem;
  }
  .group .muted { margin: 0.2rem 0; }
  .muted.empty { font-size: 12.5px; }

  .item + .item { margin-top: 2px; }

  .chip {
    display: flex;
    align-items: stretch;
    border-radius: var(--radius);
    overflow: hidden;
  }
  .chip:hover { background: var(--canvas); }
  .chip-main {
    flex: 1;
    min-width: 0;
    display: flex;
    align-items: flex-start;
    gap: 0.4rem;
    text-align: left;
    background: transparent;
    color: var(--ink);
    border: none;
    border-radius: 0;
    padding: 0.4rem 0.45rem;
    font-size: 13.5px;
    font-weight: 500;
    line-height: 1.35;
  }
  .chip-main:hover { background: transparent; }
  .chip-main:focus-visible, .chip-x:focus-visible { outline: 2px solid var(--focus); outline-offset: -2px; border-radius: 6px; }
  .chev {
    flex: none;
    width: 0.9em;
    height: 1.35em;
    position: relative;
  }
  .chev::before {
    content: "";
    position: absolute;
    left: 2px;
    top: 50%;
    width: 5px;
    height: 5px;
    border-right: 1.5px solid var(--muted);
    border-bottom: 1.5px solid var(--muted);
    transform: translateY(-65%) rotate(-45deg);
    transition: transform 0.15s;
  }
  .chev.open::before { transform: translateY(-75%) rotate(45deg); }
  .chip-text {
    min-width: 0;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow-wrap: anywhere;
  }
  .chip-x {
    flex: none;
    width: 30px;
    padding: 0;
    background: transparent;
    color: var(--muted-2);
    border: none;
    border-radius: 0;
    font-size: 17px;
    line-height: 1;
    opacity: 0;
  }
  .chip:hover .chip-x, .chip-x:focus-visible { opacity: 1; }
  .chip-x:hover { background: var(--bad-tint); color: var(--bad); }
  @media (hover: none) { .chip-x { opacity: 1; } }

  .param-list {
    list-style: none;
    margin: 0.15rem 0 0.1rem 1.6rem;
    padding: 0.15rem 0 0.15rem 0.7rem;
    border-left: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text);
  }
  .param-list li { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; line-height: 1.65; }
  .param-list strong { font-weight: 500; color: var(--muted); }

  /* Compatibility dot grid */
  .compat-pct { font-weight: 500; font-size: 12.5px; color: var(--ok); font-variant-numeric: tabular-nums; }
  .compat-pct.has-bad { color: var(--bad); }

  .compat-grid {
    display: grid;
    grid-template-columns: repeat(var(--cols), minmax(0, 18px));
    gap: 3px;
    width: fit-content;
    max-width: 100%;
    box-sizing: border-box;
    max-height: 168px;
    overflow: hidden;
    margin: 0.35rem 0 0;
    padding: 0.6rem;
    background: var(--canvas);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    color: inherit;
  }
  .compat-grid:hover { background: var(--canvas); border-color: var(--muted-2); }
  .dot { aspect-ratio: 1; border-radius: 3px; }
  .dot.ok { background: var(--ok); }
  .dot.bad { background: var(--bad); }
  .dot.neutral { background: var(--border-strong); }
  .compat-caption { margin: 0.5rem 0 0; font-size: 12.5px; color: var(--muted); }

  .run-group {
    position: sticky;
    bottom: 0;
    margin-top: auto;
    background: var(--surface);
    padding-bottom: 1.25rem;
    box-shadow: 0 -8px 12px -10px rgba(23, 34, 58, 0.18);
  }
  .macroreps .param-name-text { font-family: var(--font-sans); font-size: 13.5px; }
  .macroreps input { flex: 0 0 96px; text-align: right; }
  input.invalid { border-color: var(--bad); background: var(--bad-tint); }
  .run-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    width: 100%;
    margin-top: 0.85rem;
    height: 44px;
    padding: 0 1rem;
    font-size: 15px;
    font-weight: 600;
    border-radius: var(--radius);
    box-shadow: 0 1px 2px rgba(11, 100, 119, 0.25);
  }
  .run-btn svg { width: 14px; height: 14px; }
  .run-btn:disabled {
    background: var(--chip-bg);
    color: var(--muted-2);
    box-shadow: none;
    cursor: not-allowed;
  }
  .run-hint { text-align: center; font-size: 12.5px; }
  .panel-invalid { color: var(--bad); }

  /* === INFO ICON + TOOLTIP === */
  .info-wrapper {
    position: relative;
    display: inline-flex;
    flex: none;
  }

  .info-icon {
    font-family: var(--font-sans);
    font-size: 10px;
    font-weight: 700;
    font-style: italic;
    color: var(--muted);
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    border: 1px solid var(--border-strong);
    cursor: help;
  }
  .info-wrapper:hover .info-icon { color: var(--primary); border-color: var(--primary); }

  .tooltip {
    position: absolute;
    bottom: calc(100% + 8px);
    left: -6px;
    background-color: var(--ink);
    color: #F3F5F8;
    font-family: var(--font-sans);
    text-align: left;
    border-radius: 6px;
    padding: 0.45rem 0.65rem;
    white-space: normal;
    width: max-content;
    max-width: min(300px, 70vw);
    font-size: 12.5px;
    font-weight: 400;
    line-height: 1.4;
    z-index: 3000;
    display: none;
    pointer-events: none;
    overflow-wrap: break-word;
    box-shadow: var(--shadow-lg);
  }
  .tooltip::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 9px;
    border-width: 5px;
    border-style: solid;
    border-color: var(--ink) transparent transparent transparent;
  }
  .info-wrapper:hover .tooltip,
  .info-wrapper:focus-within .tooltip { display: block; }
  .info-icon:focus-visible { outline: 2px solid var(--focus); outline-offset: 1px; }

  /* === COMPATIBILITY MATRIX === */
  .compatibility-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 3px;
    margin: 0.5rem 0 0;
    text-align: center;
    table-layout: fixed;
    font-size: 12px;
  }
  .compatibility-table th {
    font-family: var(--font-mono);
    font-weight: 500;
    font-size: 11.5px;
    color: var(--muted);
    padding: 0.25rem;
  }
  .compatibility-table th.corner { font-family: var(--font-sans); text-align: left; color: var(--muted-2); width: 7.5rem; }
  .solver-name { text-align: left; color: var(--ink) !important; }
  .compat-cell {
    height: 26px;
    border-radius: 4px;
    padding: 0;
  }
  .compat-cell.ok { background: var(--ok); }
  .compat-cell.bad { background: var(--bad); cursor: help; }
  .compat-cell.neutral { background: var(--chip-bg); }

  /* === MODAL === */
  .modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(23, 34, 58, 0.45);
    display: grid;
    place-items: center;
    z-index: 2000;
    padding: 1rem;
  }
  .modal {
    background: var(--surface);
    width: min(540px, 100%);
    max-height: 85vh;
    overflow: auto;
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-lg);
    padding: 1.4rem 1.5rem 1.25rem;
    box-sizing: border-box;
  }
  .modal h3 { margin: 0 0 0.4rem; font-size: 16px; font-weight: 600; color: var(--ink); }
  .modal p { margin: 0 0 1.1rem; color: var(--text); line-height: 1.5; }
  .modal p.modal-sub { font-size: 13px; color: var(--muted); margin-bottom: 0.5rem; }
  .modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: 0.5rem;
    margin-top: 1rem;
  }

  @media (max-width: 960px) {
    .layout { grid-template-columns: minmax(0, 1fr); }
    .panel { position: static; max-height: none; overflow: visible; }
    .run-group { position: static; }
  }
  @media (max-width: 640px) {
    nav { padding: 0 0.75rem; gap: 0.5rem; overflow-x: auto; }
    nav ul a { padding: 0 0.5rem; font-size: 13px; }
    .title { display: none; }
    main { padding: 1rem 0.75rem 2rem; }
    .tab { gap: 0.3rem; font-size: 13px; padding: 0.85rem 0.5rem; }
    .tabs { padding: 0 0.5rem; gap: 0; }
    .tab-body { padding: 1rem; }
  }
</style>
