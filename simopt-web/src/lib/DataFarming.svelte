<script lang="ts">
  import { onMount } from "svelte";
  import type { SummaryEntry } from "../types";

  type Kind = "solver" | "problem" | "model";
  type Factor = {
    name: string;
    type: string;
    default: unknown;
    description: string;
    datafarmable: boolean;
    source: string;
  };
  type Range = { min: string | number; max: string | number; decimals: string | number };
  type Row = Record<string, unknown>;
  type Table = { columns: string[]; rows: Row[] };
  type SavedDesign = { name: string; kind: string; target: string; n_points: number };

  const API = "http://localhost:8000";

  let { onAddToExperiment }: { onAddToExperiment: (kind: "solver" | "problem", entries: SummaryEntry[]) => void } =
    $props();

  // Target choices: {value: what the API expects, label: what is shown}.
  let options = $state<Record<Kind, { value: string; label: string }[]>>({ solver: [], problem: [], model: [] });
  let kind = $state<Kind>("solver");
  let target = $state("");
  let factors = $state<Factor[]>([]);
  let vary = $state<Record<string, boolean>>({});
  let values = $state<Record<string, string>>({});
  let ranges = $state<Record<string, Range>>({});
  let nStacks = $state(1);
  let designName = $state("");
  let nReps = $state(10);
  let crn = $state(true);

  let design = $state<Table | null>(null);
  let designSpec = $state<ReturnType<typeof buildSpec> | null>(null);
  let error = $state("");
  let busy = $state(false);
  let running = $state(false);
  let results = $state<(Table & { run_id: string }) | null>(null);
  let saved = $state<SavedDesign[]>([]);

  const show = (v: unknown): string => (typeof v === "string" ? v : JSON.stringify(v) ?? "");
  const parse = (text: string, type: string): unknown => {
    if (type === "str") return text;
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  };

  async function getJson(url: string, init?: RequestInit) {
    const res = await fetch(`${API}${url}`, init);
    const body = await res.json().catch(() => ({}));
    return { res, body };
  }
  const detail = (body: { detail?: unknown }) =>
    typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? "Request failed");

  async function loadOptions() {
    try {
      const [s, p, m] = await Promise.all([getJson("/solvers"), getJson("/problems"), getJson("/models")]);
      options = {
        solver: (s.body.solvers ?? []).map((n: string) => ({ value: n, label: n })),
        problem: (p.body.problems ?? []).map((n: string) => ({ value: n, label: n })),
        model: (m.body.models ?? []).map((x: { name: string; display: string }) => ({
          value: x.name,
          label: x.display,
        })),
      };
    } catch {
      error = "Could not reach the API.";
    }
  }

  async function loadSaved() {
    try {
      saved = (await getJson("/df/designs")).body.designs ?? [];
    } catch {
      saved = [];
    }
  }

  async function loadFactors() {
    factors = [];
    design = null;
    results = null;
    error = "";
    vary = {};
    values = {};
    ranges = {};
    if (!target) return;
    designName = `${target.split(" ")[0]}_design`;
    const { res, body } = await getJson(`/df/factors/${kind}/${encodeURIComponent(target)}`);
    if (!res.ok) {
      error = detail(body);
      return;
    }
    factors = body.factors;
    for (const f of factors) {
      vary[f.name] = false;
      values[f.name] = show(f.default);
      const d = typeof f.default === "number" ? f.default : 0;
      ranges[f.name] = {
        min: String(d),
        max: String(d === 0 ? 1 : f.type === "int" ? Math.round(d * 2) : d * 2),
        decimals: f.type === "int" ? "0" : "2",
      };
    }
  }

  function changeKind(k: Kind) {
    kind = k;
    target = "";
    loadFactors();
  }

  function buildSpec() {
    const varied: Record<string, { min: number; max: number; decimals: number }> = {};
    const crossed: Record<string, boolean[]> = {};
    const fixed: Record<string, unknown> = {};
    for (const f of factors) {
      if (vary[f.name]) {
        if (f.type === "bool") crossed[f.name] = [true, false];
        else {
          const r = ranges[f.name];
          varied[f.name] = {
            min: Number(r.min),
            max: Number(r.max),
            decimals: f.type === "int" ? 0 : Number(r.decimals),
          };
        }
      } else if (values[f.name] !== show(f.default)) {
        fixed[f.name] = parse(values[f.name], f.type);
      }
    }
    return { kind, name: target, varied, crossed, fixed, design_type: "nolhs", n_stacks: Number(nStacks) };
  }

  async function post(url: string, payload: unknown, method = "POST") {
    return getJson(url, {
      method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  async function generate() {
    error = "";
    design = null;
    results = null;
    busy = true;
    try {
      const spec = buildSpec();
      const { res, body } = await post("/df/preview", spec);
      if (!res.ok) error = detail(body);
      else {
        design = { columns: body.columns, rows: body.rows };
        designSpec = spec;
      }
    } catch {
      error = "Could not reach the API.";
    } finally {
      busy = false;
    }
  }

  async function save() {
    if (!designSpec || !designName) return;
    error = "";
    const url = `/df/designs/${encodeURIComponent(designName)}`;
    let { res, body } = await post(`${url}?overwrite=false`, designSpec, "PUT");
    if (res.status === 409 && confirm(`Design "${designName}" already exists. Overwrite it?`)) {
      ({ res, body } = await post(`${url}?overwrite=true`, designSpec, "PUT"));
    } else if (res.status === 409) return;
    if (!res.ok) error = detail(body);
    else await loadSaved();
  }

  async function loadDesign(name: string) {
    error = "";
    results = null;
    const { res, body } = await getJson(`/df/designs/${encodeURIComponent(name)}`);
    if (!res.ok) {
      error = detail(body);
      return;
    }
    const spec = body.spec;
    kind = spec.kind;
    // Saved specs hold the abbreviated name; the dropdown may use a display name starting with it.
    const match = options[kind as Kind].find((o) => o.value === spec.name || o.value.startsWith(`${spec.name} (`));
    target = match?.value ?? spec.name;
    await loadFactors();
    designName = name;
    nStacks = spec.n_stacks;
    for (const f of factors) {
      if (f.name in spec.varied) {
        vary[f.name] = true;
        const r = spec.varied[f.name];
        ranges[f.name] = { min: String(r.min), max: String(r.max), decimals: String(r.decimals) };
      } else if (f.name in spec.crossed) vary[f.name] = true;
      else if (f.name in spec.fixed) values[f.name] = show(spec.fixed[f.name]);
    }
    designSpec = spec;
    design = { columns: Object.keys(body.rows[0] ?? {}), rows: body.rows };
  }

  async function remove(name: string) {
    if (!confirm(`Delete design "${name}"?`)) return;
    await getJson(`/df/designs/${encodeURIComponent(name)}`, { method: "DELETE" });
    await loadSaved();
  }

  function addToExperiment() {
    if (!design || kind === "model") return;
    const entries: SummaryEntry[] = design.rows.map((row, i) => ({
      name: target,
      rename: `${designName}_dp${i}`,
      expanded: false,
      params: factors.map((f) => ({
        name: f.name,
        description: f.description,
        default: f.default,
        value: JSON.stringify(row[f.name] ?? f.default),
        ...(kind === "problem" ? { source: f.source } : {}),
      })),
    }));
    onAddToExperiment(kind, entries);
  }

  async function runModel() {
    if (!designSpec) return;
    error = "";
    results = null;
    running = true;
    try {
      const { res, body } = await post("/df/run_model", {
        spec: designSpec,
        n_reps: Number(nReps),
        crn_across_design_pts: crn,
      });
      if (!res.ok) error = detail(body);
      else results = body;
    } catch {
      error = "Could not reach the API.";
    } finally {
      running = false;
    }
  }

  let highlighted = $derived(new Set([...Object.keys(designSpec?.varied ?? {}), ...Object.keys(designSpec?.crossed ?? {})]));

  onMount(async () => {
    await loadOptions();
    loadSaved();
  });
</script>

<div class="card">
  <h2>Data Farming</h2>

  <div class="row">
    <label>
      Kind
      <select value={kind} onchange={(e) => changeKind(e.currentTarget.value as Kind)}>
        <option value="solver">Solver</option>
        <option value="problem">Problem</option>
        <option value="model">Model</option>
      </select>
    </label>
    <label>
      Target
      <select value={target} onchange={(e) => { target = e.currentTarget.value; loadFactors(); }}>
        <option value="">— Select —</option>
        {#each options[kind] as o (o.value)}
          <option value={o.value}>{o.label}</option>
        {/each}
      </select>
    </label>
  </div>

  {#if factors.length}
    <div class="scroll">
      <table>
        <thead>
          <tr>
            <th>Factor</th><th>Type</th>
            {#if kind === "problem"}<th>Source</th>{/if}
            <th>Description</th><th>Vary</th><th>Value</th><th>Min</th><th>Max</th><th>Decimals</th>
          </tr>
        </thead>
        <tbody>
          {#each factors as f (f.source + f.name)}
            <tr>
              <td>{f.name}</td>
              <td>{f.type}</td>
              {#if kind === "problem"}<td>{f.source}</td>{/if}
              <td>{f.description}</td>
              <td><input type="checkbox" bind:checked={vary[f.name]} disabled={!f.datafarmable} /></td>
              <td>
                {#if vary[f.name]}
                  {f.type === "bool" ? "true, false" : ""}
                {:else}
                  <input type="text" bind:value={values[f.name]} />
                {/if}
              </td>
              {#if vary[f.name] && f.type !== "bool"}
                <td><input type="number" bind:value={ranges[f.name].min} /></td>
                <td><input type="number" bind:value={ranges[f.name].max} /></td>
                <td>
                  {#if f.type === "float"}
                    <input type="number" min="0" step="1" bind:value={ranges[f.name].decimals} />
                  {:else}0{/if}
                </td>
              {:else}
                <td></td><td></td><td></td>
              {/if}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>

    <div class="row">
      <label>Number of stacks <input type="number" min="1" step="1" bind:value={nStacks} /></label>
      <label>Design name <input type="text" bind:value={designName} /></label>
      <button class="btn btn-primary" onclick={generate} disabled={busy}>
        {busy ? "Generating…" : "Generate"}
      </button>
    </div>
  {/if}

  {#if error}<p class="error">{error}</p>{/if}

  {#if design}
    <p><strong>{design.rows.length}</strong> design points</p>
    <div class="scroll tall">
      <table>
        <thead>
          <tr>
            {#each design.columns as c (c)}<th class:varied={highlighted.has(c)}>{c}</th>{/each}
          </tr>
        </thead>
        <tbody>
          {#each design.rows as row, i (i)}
            <tr>
              {#each design.columns as c (c)}<td class:varied={highlighted.has(c)}>{show(row[c])}</td>{/each}
            </tr>
          {/each}
        </tbody>
      </table>
    </div>

    <div class="row">
      <button class="btn" onclick={save} disabled={!designName}>Save</button>
      {#if kind !== "model"}
        <button class="btn btn-primary" onclick={addToExperiment}>Add to experiment</button>
      {:else}
        <label>Replications <input type="number" min="1" step="1" bind:value={nReps} /></label>
        <label class="inline"><input type="checkbox" bind:checked={crn} /> CRN across design points</label>
        <button class="btn btn-primary" onclick={runModel} disabled={running}>Run</button>
      {/if}
    </div>
  {/if}

  {#if running}<p>Running…</p>{/if}
  {#if results}
    <p>
      {results.rows.length} result rows ·
      <a href={`${API}/df/results/${results.run_id}/raw_results.csv`}>Download raw_results.csv</a>
    </p>
    <div class="scroll tall">
      <table>
        <thead><tr>{#each results.columns as c (c)}<th>{c}</th>{/each}</tr></thead>
        <tbody>
          {#each results.rows as row, i (i)}
            <tr>{#each results.columns as c (c)}<td>{show(row[c])}</td>{/each}</tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</div>

<div class="card">
  <h2>Saved designs</h2>
  {#if saved.length === 0}
    <p class="muted">No saved designs.</p>
  {:else}
    <table>
      <thead><tr><th>Name</th><th>Kind</th><th>Target</th><th>Points</th><th></th></tr></thead>
      <tbody>
        {#each saved as d (d.name)}
          <tr>
            <td>{d.name}</td><td>{d.kind}</td><td>{d.target}</td><td>{d.n_points}</td>
            <td class="actions">
              <button class="btn" onclick={() => loadDesign(d.name)}>Load</button>
              <button class="btn" onclick={() => remove(d.name)}>Delete</button>
              <a href={`${API}/df/designs/${encodeURIComponent(d.name)}/design.csv`}>Export CSV</a>
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</div>

<style>
  .card {
    background: #ffffff;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    margin-bottom: 1.5rem;
  }
  h2 {
    color: #2563eb;
    margin-top: 0;
    margin-bottom: 0.75rem;
  }
  .row {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: 1rem;
    margin: 0.75rem 0;
  }
  label {
    display: flex;
    flex-direction: column;
    font-size: 14px;
    gap: 0.25rem;
  }
  label.inline {
    flex-direction: row;
    align-items: center;
  }
  select,
  input[type="number"],
  input[type="text"] {
    padding: 0.4rem;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    font-size: 14px;
    box-sizing: border-box;
    width: 100%;
    max-width: 300px;
  }
  td input[type="number"] {
    width: 6rem;
  }
  .scroll {
    overflow: auto;
  }
  .scroll.tall {
    max-height: 350px;
  }
  table {
    border-collapse: collapse;
    font-size: 14px;
    width: 100%;
  }
  th,
  td {
    border: 1px solid #e5e7eb;
    padding: 0.3rem 0.5rem;
    text-align: left;
    vertical-align: top;
  }
  th {
    background: #f3f4f6;
    position: sticky;
    top: 0;
  }
  .varied {
    background: #eff6ff;
  }
  th.varied {
    background: #dbeafe;
  }
  .error {
    color: #b91c1c;
  }
  .muted {
    color: #6b7280;
  }
  .actions {
    white-space: nowrap;
  }
  .btn {
    border: 1px solid #cbd5e1;
    background: #fff;
    color: #0f172a;
    padding: 0.45rem 0.9rem;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 500;
    cursor: pointer;
  }
  .btn:hover {
    background: #f8fafc;
  }
  .btn:disabled {
    opacity: 0.5;
    cursor: default;
  }
  .btn-primary {
    border-color: #2563eb;
    background: #2563eb;
    color: #fff;
  }
  .btn-primary:hover {
    background: #1e40af;
  }
</style>
