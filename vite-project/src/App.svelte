
<script>
  import Home from './routes/Home.svelte';
  import Experiments from './routes/Experiments.svelte';
  import DataFarming from './routes/DataFarming.svelte';

  let route = window.location.hash.slice(1) || '/';
  function navigate(to) {
    window.location.hash = to;
  }
  function onHashChange() {
    route = window.location.hash.slice(1) || '/';
  }
  window.addEventListener('hashchange', onHashChange);
</script>

<header class="sticky top-0 z-10 bg-white/70 backdrop-blur border-b border-slate-200/60">
  <div class="container-page flex items-center justify-between py-4">
    <div class="font-semibold text-lg tracking-tight text-slate-900">SimOpt</div>
    <nav class="flex gap-6 text-sm">
      <a class="hover:text-blue-600 {route==='/' ? 'text-slate-900' : 'text-slate-600'}" href="#/" on:click|preventDefault={() => navigate('/')}>Home</a>
      <a class="hover:text-blue-600 {route==='/experiments' ? 'text-slate-900' : 'text-slate-600'}" href="#/experiments" on:click|preventDefault={() => navigate('/experiments')}>Experiments</a>
      <a class="hover:text-blue-600 {route==='/data-farming' ? 'text-slate-900' : 'text-slate-600'}" href="#/data-farming" on:click|preventDefault={() => navigate('/data-farming')}>Data Farming</a>
    </nav>
  </div>
</header>

<main class="container-page py-10">

{#if route === '/' }
  <Home />
{:else if route === '/experiments'}
  <Experiments />
{:else if route === '/data-farming'}
  <DataFarming />
{:else}
  <div class="text-center text-slate-500">Not found</div>
{/if}

</main>
