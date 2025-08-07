<script>
  import {
    getProblems,
    getSolvers,
    addProblem,
    addSolver,
    removeProblem,
    removeSolver,
    getWorkspace,
    getExperiments,
    clearWorkspace,
    createExperiment
  } from '../lib/api.ts';

  let problems = [];
  let solvers = [];
  let workspaceProblems = [];
  let workspaceSolvers = [];
  let experiments = [];

  let selectedProblem = '';
  let selectedSolver = '';

  let designType = 'nolhs';
  let numStacks = 1;
  let experimentName = 'Experiment';

  async function loadAll() {
    problems = await getProblems();
    solvers = await getSolvers();
    const ws = await getWorkspace();
    workspaceProblems = ws.problems || [];
    workspaceSolvers = ws.solvers || [];
    experiments = await getExperiments();
  }

  loadAll();

  async function onAddProblem() {
    if (!selectedProblem) return;
    const res = await addProblem(selectedProblem);
    workspaceProblems = res.problems || [];
    selectedProblem = '';
  }

  async function onAddSolver() {
    if (!selectedSolver) return;
    const res = await addSolver(selectedSolver);
    workspaceSolvers = res.solvers || [];
    selectedSolver = '';
  }

  async function onRemoveProblem(key) {
    const res = await removeProblem(key);
    workspaceProblems = res.problems || [];
  }

  async function onRemoveSolver(key) {
    const res = await removeSolver(key);
    workspaceSolvers = res.solvers || [];
  }

  async function onClearWorkspace() {
    await clearWorkspace();
    const ws = await getWorkspace();
    workspaceProblems = ws.problems || [];
    workspaceSolvers = ws.solvers || [];
  }

  async function onCreateExperiment() {
    const payload = {
      experiment_name: experimentName,
      design_type: designType,
      num_stacks: Number(numStacks)
    };
    const res = await createExperiment(payload);
    if (res.success) {
      experiments = await getExperiments();
    }
  }
</script>

<div class="py-2">
  <h2 class="text-3xl font-semibold tracking-tight text-slate-900">Optimization Experiments</h2>
  <p class="text-slate-600 mt-1">Design and execute experiments to compare solver performance.</p>
</div>

<div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
  <!-- Left column: Experiments list -->
  <section class="lg:col-span-1">
    <div class="card">
      <div class="px-4 py-3 border-b border-slate-200/60 font-medium">Created Experiments</div>
      <div class="p-4 space-y-3 max-h-[60vh] overflow-auto">
        {#if experiments.length === 0}
          <div class="text-slate-500">No experiments created yet.</div>
        {:else}
          {#each experiments as exp}
            <div class="border border-slate-200/60 rounded-xl p-3 bg-white/60">
              <div class="font-medium text-slate-900">{exp.name}</div>
              <div class="text-sm text-slate-600 mt-1">{exp.problems.length} problem(s), {exp.solvers.length} solver(s)</div>
              <div class="text-xs text-slate-500 mt-1">{exp.design_type}, {exp.num_stacks} stack(s)</div>
            </div>
          {/each}
        {/if}
      </div>
    </div>
  </section>

  <!-- Right column: Config -->
  <section class="lg:col-span-2 space-y-6">
    <!-- Problem selection -->
    <div class="card">
      <div class="px-4 py-3 border-b border-slate-200/60 font-medium">Problem Selection</div>
      <div class="p-4 grid md:grid-cols-3 gap-4">
        <div class="md:col-span-2">
          <label class="block text-sm font-medium text-slate-700 mb-1">Add Problem</label>
          <div class="flex gap-2">
            <select class="w-full field" bind:value={selectedProblem}>
              <option value="">Select a problem...</option>
              {#each problems as p}
                <option value={p.key}>{p.key}</option>
              {/each}
            </select>
            <button class="btn btn-primary" on:click={onAddProblem}>Add</button>
          </div>
        </div>
        <div class="text-sm text-slate-600">
          Objective: Single|Multiple<br />
          Constraint: Unconstrained|Constrained<br />
          Variable: Discrete|Continuous|Mixed<br />
          Gradient: True|False
        </div>
      </div>
    </div>

    <!-- Solver selection -->
    <div class="card">
      <div class="px-4 py-3 border-b border-slate-200/60 font-medium">Solver Selection</div>
      <div class="p-4 grid md:grid-cols-3 gap-4">
        <div class="md:col-span-2">
          <label class="block text-sm font-medium text-slate-700 mb-1">Add Solver</label>
          <div class="flex gap-2">
            <select class="w-full field" bind:value={selectedSolver}>
              <option value="">Select a solver...</option>
              {#each solvers as s}
                <option value={s.key}>{s.key}</option>
              {/each}
            </select>
            <button class="btn bg-emerald-600 text-white hover:bg-emerald-700" on:click={onAddSolver}>Add</button>
          </div>
        </div>
        <div class="text-sm text-slate-600">
          Incompatible problems/solvers will be disabled in a full UI.
        </div>
      </div>
    </div>

    <!-- Workspace -->
    <div class="card">
      <div class="px-4 py-3 border-b border-slate-200/60 font-medium">Workspace</div>
      <div class="p-4 grid md:grid-cols-2 gap-6">
        <div>
          <h4 class="font-semibold mb-2">Problems</h4>
          {#if workspaceProblems.length === 0}
            <div class="text-slate-500">No problems added yet.</div>
          {:else}
            <ul class="space-y-2">
              {#each workspaceProblems as p}
                <li class="flex items-center justify-between border border-slate-200/60 rounded-xl p-2 bg-white/60">
                  <div>
                    <div class="font-medium">{p.key}</div>
                    <div class="text-xs text-slate-500">{p.name}</div>
                  </div>
                  <button class="btn bg-rose-600 text-white hover:bg-rose-700 px-3 py-1" on:click={() => onRemoveProblem(p.key)}>Remove</button>
                </li>
              {/each}
            </ul>
          {/if}
        </div>
        <div>
          <h4 class="font-semibold mb-2">Solvers</h4>
          {#if workspaceSolvers.length === 0}
            <div class="text-slate-500">No solvers added yet.</div>
          {:else}
            <ul class="space-y-2">
              {#each workspaceSolvers as s}
                <li class="flex items-center justify-between border border-slate-200/60 rounded-xl p-2 bg-white/60">
                  <div>
                    <div class="font-medium">{s.key}</div>
                    <div class="text-xs text-slate-500">{s.name}</div>
                  </div>
                  <button class="btn bg-rose-600 text-white hover:bg-rose-700 px-3 py-1" on:click={() => onRemoveSolver(s.key)}>Remove</button>
                </li>
              {/each}
            </ul>
          {/if}
        </div>
        <div class="md:col-span-2">
          <button class="btn btn-ghost" on:click={onClearWorkspace}>Clear Problem/Solver Lists</button>
        </div>
      </div>
    </div>

    <!-- Design options -->
    <div class="card">
      <div class="px-4 py-3 border-b border-slate-200/60 font-medium">Design Options</div>
      <div class="p-4 grid md:grid-cols-3 gap-4">
        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">Design Type</label>
          <select class="w-full field" bind:value={designType}>
            <option value="nolhs">No Latin Hypercube</option>
            <option value="lhs">Latin Hypercube</option>
          </select>
        </div>
        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1"># of Stacks</label>
          <input class="w-full field" type="number" min="1" bind:value={numStacks} />
        </div>
        <div>
          <label class="block text-sm font-medium text-slate-700 mb-1">Experiment Name</label>
          <input class="w-full field" type="text" bind:value={experimentName} />
        </div>
        <div class="md:col-span-3">
          <button class="btn bg-emerald-600 text-white hover:bg-emerald-700" on:click={onCreateExperiment}>Create Experiment</button>
        </div>
      </div>
    </div>
  </section>
</div>
