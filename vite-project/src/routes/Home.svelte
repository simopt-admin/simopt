<script>
  function go(to) {
    window.location.hash = to;
  }
</script>

<section class="py-16">
  <div class="grid gap-10 md:grid-cols-2 items-center">
    <div>
      <h1 class="text-4xl md:text-5xl font-semibold tracking-tight text-slate-900">Simulation Optimization,<br />Refined.</h1>
      <p class="mt-4 text-slate-600 max-w-prose">Design and run clean, comparable experiments on problems and solvers — with a calm, focused interface.</p>
      <div class="mt-8 flex gap-3">
        <button class="btn btn-primary shadow-sm" on:click={() => go('/experiments')}>Start Experiments</button>
        <button class="btn btn-ghost" on:click={() => go('/data-farming')}>Data Farming</button>
      </div>
    </div>
    <div class="card p-8 fade-in">
      <h2 class="font-semibold text-slate-900">Getting Started</h2>
      <ul class="mt-3 text-slate-600 list-disc pl-5 space-y-1">
        <li>Add problems and solvers in Experiments.</li>
        <li>Choose a design and create an experiment.</li>
        <li>Manage your workspace with clarity.</li>
      </ul>
    </div>
  </div>
</section>
