{"payload":{"allShortcutsEnabled":true,"fileTree":{"":{"items":[{"name":"data_farming_experiments","path":"data_farming_experiments","contentType":"directory"},{"name":"demo","path":"demo","contentType":"directory"},{"name":"dist","path":"dist","contentType":"directory"},{"name":"docs","path":"docs","contentType":"directory"},{"name":"experiments","path":"experiments","contentType":"directory"},{"name":"notebooks","path":"notebooks","contentType":"directory"},{"name":"simopt","path":"simopt","contentType":"directory"},{"name":"simoptlib.egg-info","path":"simoptlib.egg-info","contentType":"directory"},{"name":"workshop","path":"workshop","contentType":"directory"},{"name":".gitignore","path":".gitignore","contentType":"file"},{"name":"LICENSE","path":"LICENSE","contentType":"file"},{"name":"README.md","path":"README.md","contentType":"file"},{"name":"demo_radom_model.py","path":"demo_radom_model.py","contentType":"file"},{"name":"demo_random_problem.py","path":"demo_random_problem.py","contentType":"file"},{"name":"demo_random_problem_solver.py","path":"demo_random_problem_solver.py","contentType":"file"},{"name":"demo_user.py","path":"demo_user.py","contentType":"file"},{"name":"pyproject.toml","path":"pyproject.toml","contentType":"file"}],"totalCount":17}},"fileTreeProcessingTime":3.607858,"foldersToFetch":[],"reducedMotionEnabled":"system","repo":{"id":194012165,"defaultBranch":"master","name":"simopt","ownerLogin":"simopt-admin","currentUserCanPush":true,"isFork":false,"isEmpty":false,"createdAt":"2019-06-26T22:55:30.000-04:00","ownerAvatar":"https://avatars.githubusercontent.com/u/52267122?v=4","public":true,"private":false,"isOrgOwned":false},"symbolsExpanded":false,"treeExpanded":true,"refInfo":{"name":"python_dev_litong","listCacheKey":"v0:1692137230.0","canEdit":true,"refType":"branch","currentOid":"382561d40918dac6fcfb54e7c1f873bdca0f46e9"},"path":"demo_random_problem_solver.py","currentUser":{"id":46491025,"login":"liulitong-Jessie","userEmail":"118010185@link.cuhk.edu.cn"},"blob":{"rawLines":["\"\"\"","This script is intended to help with debugging random problems and solvers.","It create a problem-solver pairing by importing problems and runs multiple","macroreplications of the solver on the problem.","\"\"\"","","import sys","import os.path as o","import numpy as np","import os","sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), \"..\")))","","# Import the ProblemSolver class and other useful functions","from simopt.experiment_base import ProblemSolver, read_experiment_results, post_normalize, plot_progress_curves, plot_solvability_cdfs","from rng.mrg32k3a import MRG32k3a","from simopt.models.san_2 import SANLongestPath, SANLongestPathConstr","","# !! When testing a new solver/problem, first go to directory.py.","# See directory.py for more details.","# Specify the names of the solver to test.","","# -----------------------------------------------","solver_name = \"RNDSRCH\"  # Random search solver","# -----------------------------------------------","","","def rebase(random_rng, n):","    new_rngs = []","    for rng in random_rng:","        stream_index = rng.s_ss_sss_index[0]","        substream_index = rng.s_ss_sss_index[1]","        subsubstream_index = rng.s_ss_sss_index[2]","        new_rngs.append(MRG32k3a(s_ss_sss_index=[stream_index, substream_index + n, subsubstream_index]))","    random_rng = new_rngs","    return random_rng","","def strtobool(t):","    t = t.lower()","    if t == \"t\":","        return True","    else:","        return False","","n_inst = int(input('Please enter the number of instance you want to generate: '))","rand = input('Please decide whether you want to generate random instances or determinent instances (T/F): ')","rand = strtobool(rand)","","model_fixed_factors = {}  # Override model factors","","myproblem = SANLongestPathConstr(random=True, model_fixed_factors=model_fixed_factors)","","random_rng = [MRG32k3a(s_ss_sss_index=[2, 4, ss]) for ss in range(myproblem.model.n_random, myproblem.model.n_random + myproblem.n_rngs)]","rng_list2 = [MRG32k3a(s_ss_sss_index=[2, 4, ss]) for ss in range(myproblem.model.n_random)]","","# Generate 5 random problem instances","for i in range(n_inst):","    random_rng = rebase(random_rng, 1)","    rng_list2 = rebase(rng_list2, 1)","    myproblem = SANLongestPathConstr(random=rand, random_rng=rng_list2, model_fixed_factors=model_fixed_factors)","    myproblem.attach_rngs(random_rng)","    problem_name = myproblem.model.name + str(i)","    print('-------------------------------------------------------')","    print(f\"Testing solver {solver_name} on problem {problem_name}.\")","","    # Specify file path name for storing experiment outputs in .pickle file.","    file_name_path = \"experiments/outputs/\" + solver_name + \"_on_\" + problem_name + \".pickle\"","    print(f\"Results will be stored as {file_name_path}.\")","","    # Initialize an instance of the experiment class.","    myexperiment = ProblemSolver(solver_name=solver_name, problem=myproblem)","","    # Run a fixed number of macroreplications of the solver on the problem.","    myexperiment.run(n_macroreps=100)","","    # If the solver runs have already been performed, uncomment the","    # following pair of lines (and uncommmen the myexperiment.run(...)","    # line above) to read in results from a .pickle file.","    # myexperiment = read_experiment_results(file_name_path)","","    print(\"Post-processing results.\")","    # Run a fixed number of postreplications at all recommended solutions.","    myexperiment.post_replicate(n_postreps=1) #200, 10","    # Find an optimal solution x* for normalization.","    post_normalize([myexperiment], n_postreps_init_opt=1) #200, 5","","    # Log results.","    myexperiment.log_experiment_results()","","    print(\"Optimal solution: \",np.array(myexperiment.xstar))","    print(\"Optimal Value: \", myexperiment.all_est_objectives[0])","","    print(\"Plotting results.\")","    # Produce basic plots of the solver on the problem.","    plot_progress_curves(experiments=[myexperiment], plot_type=\"all\", normalize=False)","    plot_progress_curves(experiments=[myexperiment], plot_type=\"mean\", normalize=False)","    plot_progress_curves(experiments=[myexperiment], plot_type=\"quantile\", beta=0.90, normalize=False)","    plot_solvability_cdfs(experiments=[myexperiment], solve_tol=0.1)","","    # Plots will be saved in the folder experiments/plots.","    print(\"Finished. Plots can be found in experiments/plots folder.\")"],"stylingDirectives":[[{"start":0,"end":3,"cssClass":"pl-s"}],[{"start":0,"end":75,"cssClass":"pl-s"}],[{"start":0,"end":74,"cssClass":"pl-s"}],[{"start":0,"end":47,"cssClass":"pl-s"}],[{"start":0,"end":3,"cssClass":"pl-s"}],[],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":10,"cssClass":"pl-s1"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":9,"cssClass":"pl-s1"},{"start":10,"end":14,"cssClass":"pl-s1"},{"start":15,"end":17,"cssClass":"pl-k"},{"start":18,"end":19,"cssClass":"pl-s1"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":15,"cssClass":"pl-k"},{"start":16,"end":18,"cssClass":"pl-s1"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":9,"cssClass":"pl-s1"}],[{"start":0,"end":3,"cssClass":"pl-s1"},{"start":4,"end":8,"cssClass":"pl-s1"},{"start":9,"end":15,"cssClass":"pl-en"},{"start":16,"end":17,"cssClass":"pl-s1"},{"start":18,"end":25,"cssClass":"pl-en"},{"start":26,"end":27,"cssClass":"pl-s1"},{"start":28,"end":32,"cssClass":"pl-en"},{"start":33,"end":34,"cssClass":"pl-s1"},{"start":35,"end":42,"cssClass":"pl-en"},{"start":43,"end":46,"cssClass":"pl-s1"},{"start":47,"end":54,"cssClass":"pl-s1"},{"start":55,"end":63,"cssClass":"pl-s1"},{"start":65,"end":73,"cssClass":"pl-s1"},{"start":76,"end":80,"cssClass":"pl-s"}],[],[{"start":0,"end":59,"cssClass":"pl-c"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":11,"cssClass":"pl-s1"},{"start":12,"end":27,"cssClass":"pl-s1"},{"start":28,"end":34,"cssClass":"pl-k"},{"start":35,"end":48,"cssClass":"pl-v"},{"start":50,"end":73,"cssClass":"pl-s1"},{"start":75,"end":89,"cssClass":"pl-s1"},{"start":91,"end":111,"cssClass":"pl-s1"},{"start":113,"end":134,"cssClass":"pl-s1"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":8,"cssClass":"pl-s1"},{"start":9,"end":17,"cssClass":"pl-s1"},{"start":18,"end":24,"cssClass":"pl-k"},{"start":25,"end":33,"cssClass":"pl-v"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":11,"cssClass":"pl-s1"},{"start":12,"end":18,"cssClass":"pl-s1"},{"start":19,"end":24,"cssClass":"pl-s1"},{"start":25,"end":31,"cssClass":"pl-k"},{"start":32,"end":46,"cssClass":"pl-v"},{"start":48,"end":68,"cssClass":"pl-v"}],[],[{"start":0,"end":65,"cssClass":"pl-c"}],[{"start":0,"end":36,"cssClass":"pl-c"}],[{"start":0,"end":42,"cssClass":"pl-c"}],[],[{"start":0,"end":49,"cssClass":"pl-c"}],[{"start":0,"end":11,"cssClass":"pl-s1"},{"start":12,"end":13,"cssClass":"pl-c1"},{"start":14,"end":23,"cssClass":"pl-s"},{"start":25,"end":47,"cssClass":"pl-c"}],[{"start":0,"end":49,"cssClass":"pl-c"}],[],[],[{"start":0,"end":3,"cssClass":"pl-k"},{"start":4,"end":10,"cssClass":"pl-en"},{"start":11,"end":21,"cssClass":"pl-s1"},{"start":23,"end":24,"cssClass":"pl-s1"}],[{"start":4,"end":12,"cssClass":"pl-s1"},{"start":13,"end":14,"cssClass":"pl-c1"}],[{"start":4,"end":7,"cssClass":"pl-k"},{"start":8,"end":11,"cssClass":"pl-s1"},{"start":12,"end":14,"cssClass":"pl-c1"},{"start":15,"end":25,"cssClass":"pl-s1"}],[{"start":8,"end":20,"cssClass":"pl-s1"},{"start":21,"end":22,"cssClass":"pl-c1"},{"start":23,"end":26,"cssClass":"pl-s1"},{"start":27,"end":41,"cssClass":"pl-s1"},{"start":42,"end":43,"cssClass":"pl-c1"}],[{"start":8,"end":23,"cssClass":"pl-s1"},{"start":24,"end":25,"cssClass":"pl-c1"},{"start":26,"end":29,"cssClass":"pl-s1"},{"start":30,"end":44,"cssClass":"pl-s1"},{"start":45,"end":46,"cssClass":"pl-c1"}],[{"start":8,"end":26,"cssClass":"pl-s1"},{"start":27,"end":28,"cssClass":"pl-c1"},{"start":29,"end":32,"cssClass":"pl-s1"},{"start":33,"end":47,"cssClass":"pl-s1"},{"start":48,"end":49,"cssClass":"pl-c1"}],[{"start":8,"end":16,"cssClass":"pl-s1"},{"start":17,"end":23,"cssClass":"pl-en"},{"start":24,"end":32,"cssClass":"pl-v"},{"start":33,"end":47,"cssClass":"pl-s1"},{"start":47,"end":48,"cssClass":"pl-c1"},{"start":49,"end":61,"cssClass":"pl-s1"},{"start":63,"end":78,"cssClass":"pl-s1"},{"start":79,"end":80,"cssClass":"pl-c1"},{"start":81,"end":82,"cssClass":"pl-s1"},{"start":84,"end":102,"cssClass":"pl-s1"}],[{"start":4,"end":14,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":17,"end":25,"cssClass":"pl-s1"}],[{"start":4,"end":10,"cssClass":"pl-k"},{"start":11,"end":21,"cssClass":"pl-s1"}],[],[{"start":0,"end":3,"cssClass":"pl-k"},{"start":4,"end":13,"cssClass":"pl-en"},{"start":14,"end":15,"cssClass":"pl-s1"}],[{"start":4,"end":5,"cssClass":"pl-s1"},{"start":6,"end":7,"cssClass":"pl-c1"},{"start":8,"end":9,"cssClass":"pl-s1"},{"start":10,"end":15,"cssClass":"pl-en"}],[{"start":4,"end":6,"cssClass":"pl-k"},{"start":7,"end":8,"cssClass":"pl-s1"},{"start":9,"end":11,"cssClass":"pl-c1"},{"start":12,"end":15,"cssClass":"pl-s"}],[{"start":8,"end":14,"cssClass":"pl-k"},{"start":15,"end":19,"cssClass":"pl-c1"}],[{"start":4,"end":8,"cssClass":"pl-k"}],[{"start":8,"end":14,"cssClass":"pl-k"},{"start":15,"end":20,"cssClass":"pl-c1"}],[],[{"start":0,"end":6,"cssClass":"pl-s1"},{"start":7,"end":8,"cssClass":"pl-c1"},{"start":9,"end":12,"cssClass":"pl-en"},{"start":13,"end":18,"cssClass":"pl-en"},{"start":19,"end":79,"cssClass":"pl-s"}],[{"start":0,"end":4,"cssClass":"pl-s1"},{"start":5,"end":6,"cssClass":"pl-c1"},{"start":7,"end":12,"cssClass":"pl-en"},{"start":13,"end":107,"cssClass":"pl-s"}],[{"start":0,"end":4,"cssClass":"pl-s1"},{"start":5,"end":6,"cssClass":"pl-c1"},{"start":7,"end":16,"cssClass":"pl-en"},{"start":17,"end":21,"cssClass":"pl-s1"}],[],[{"start":0,"end":19,"cssClass":"pl-s1"},{"start":20,"end":21,"cssClass":"pl-c1"},{"start":26,"end":50,"cssClass":"pl-c"}],[],[{"start":0,"end":9,"cssClass":"pl-s1"},{"start":10,"end":11,"cssClass":"pl-c1"},{"start":12,"end":32,"cssClass":"pl-v"},{"start":33,"end":39,"cssClass":"pl-s1"},{"start":39,"end":40,"cssClass":"pl-c1"},{"start":40,"end":44,"cssClass":"pl-c1"},{"start":46,"end":65,"cssClass":"pl-s1"},{"start":65,"end":66,"cssClass":"pl-c1"},{"start":66,"end":85,"cssClass":"pl-s1"}],[],[{"start":0,"end":10,"cssClass":"pl-s1"},{"start":11,"end":12,"cssClass":"pl-c1"},{"start":14,"end":22,"cssClass":"pl-v"},{"start":23,"end":37,"cssClass":"pl-s1"},{"start":37,"end":38,"cssClass":"pl-c1"},{"start":39,"end":40,"cssClass":"pl-c1"},{"start":42,"end":43,"cssClass":"pl-c1"},{"start":45,"end":47,"cssClass":"pl-s1"},{"start":50,"end":53,"cssClass":"pl-k"},{"start":54,"end":56,"cssClass":"pl-s1"},{"start":57,"end":59,"cssClass":"pl-c1"},{"start":60,"end":65,"cssClass":"pl-en"},{"start":66,"end":75,"cssClass":"pl-s1"},{"start":76,"end":81,"cssClass":"pl-s1"},{"start":82,"end":90,"cssClass":"pl-s1"},{"start":92,"end":101,"cssClass":"pl-s1"},{"start":102,"end":107,"cssClass":"pl-s1"},{"start":108,"end":116,"cssClass":"pl-s1"},{"start":117,"end":118,"cssClass":"pl-c1"},{"start":119,"end":128,"cssClass":"pl-s1"},{"start":129,"end":135,"cssClass":"pl-s1"}],[{"start":0,"end":9,"cssClass":"pl-s1"},{"start":10,"end":11,"cssClass":"pl-c1"},{"start":13,"end":21,"cssClass":"pl-v"},{"start":22,"end":36,"cssClass":"pl-s1"},{"start":36,"end":37,"cssClass":"pl-c1"},{"start":38,"end":39,"cssClass":"pl-c1"},{"start":41,"end":42,"cssClass":"pl-c1"},{"start":44,"end":46,"cssClass":"pl-s1"},{"start":49,"end":52,"cssClass":"pl-k"},{"start":53,"end":55,"cssClass":"pl-s1"},{"start":56,"end":58,"cssClass":"pl-c1"},{"start":59,"end":64,"cssClass":"pl-en"},{"start":65,"end":74,"cssClass":"pl-s1"},{"start":75,"end":80,"cssClass":"pl-s1"},{"start":81,"end":89,"cssClass":"pl-s1"}],[],[{"start":0,"end":37,"cssClass":"pl-c"}],[{"start":0,"end":3,"cssClass":"pl-k"},{"start":4,"end":5,"cssClass":"pl-s1"},{"start":6,"end":8,"cssClass":"pl-c1"},{"start":9,"end":14,"cssClass":"pl-en"},{"start":15,"end":21,"cssClass":"pl-s1"}],[{"start":4,"end":14,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":17,"end":23,"cssClass":"pl-en"},{"start":24,"end":34,"cssClass":"pl-s1"},{"start":36,"end":37,"cssClass":"pl-c1"}],[{"start":4,"end":13,"cssClass":"pl-s1"},{"start":14,"end":15,"cssClass":"pl-c1"},{"start":16,"end":22,"cssClass":"pl-en"},{"start":23,"end":32,"cssClass":"pl-s1"},{"start":34,"end":35,"cssClass":"pl-c1"}],[{"start":4,"end":13,"cssClass":"pl-s1"},{"start":14,"end":15,"cssClass":"pl-c1"},{"start":16,"end":36,"cssClass":"pl-v"},{"start":37,"end":43,"cssClass":"pl-s1"},{"start":43,"end":44,"cssClass":"pl-c1"},{"start":44,"end":48,"cssClass":"pl-s1"},{"start":50,"end":60,"cssClass":"pl-s1"},{"start":60,"end":61,"cssClass":"pl-c1"},{"start":61,"end":70,"cssClass":"pl-s1"},{"start":72,"end":91,"cssClass":"pl-s1"},{"start":91,"end":92,"cssClass":"pl-c1"},{"start":92,"end":111,"cssClass":"pl-s1"}],[{"start":4,"end":13,"cssClass":"pl-s1"},{"start":14,"end":25,"cssClass":"pl-en"},{"start":26,"end":36,"cssClass":"pl-s1"}],[{"start":4,"end":16,"cssClass":"pl-s1"},{"start":17,"end":18,"cssClass":"pl-c1"},{"start":19,"end":28,"cssClass":"pl-s1"},{"start":29,"end":34,"cssClass":"pl-s1"},{"start":35,"end":39,"cssClass":"pl-s1"},{"start":40,"end":41,"cssClass":"pl-c1"},{"start":42,"end":45,"cssClass":"pl-en"},{"start":46,"end":47,"cssClass":"pl-s1"}],[{"start":4,"end":9,"cssClass":"pl-en"},{"start":10,"end":67,"cssClass":"pl-s"}],[{"start":4,"end":9,"cssClass":"pl-en"},{"start":10,"end":68,"cssClass":"pl-s"},{"start":27,"end":40,"cssClass":"pl-s1"},{"start":27,"end":28,"cssClass":"pl-kos"},{"start":28,"end":39,"cssClass":"pl-s1"},{"start":39,"end":40,"cssClass":"pl-kos"},{"start":52,"end":66,"cssClass":"pl-s1"},{"start":52,"end":53,"cssClass":"pl-kos"},{"start":53,"end":65,"cssClass":"pl-s1"},{"start":65,"end":66,"cssClass":"pl-kos"}],[],[{"start":4,"end":76,"cssClass":"pl-c"}],[{"start":4,"end":18,"cssClass":"pl-s1"},{"start":19,"end":20,"cssClass":"pl-c1"},{"start":21,"end":43,"cssClass":"pl-s"},{"start":44,"end":45,"cssClass":"pl-c1"},{"start":46,"end":57,"cssClass":"pl-s1"},{"start":58,"end":59,"cssClass":"pl-c1"},{"start":60,"end":66,"cssClass":"pl-s"},{"start":67,"end":68,"cssClass":"pl-c1"},{"start":69,"end":81,"cssClass":"pl-s1"},{"start":82,"end":83,"cssClass":"pl-c1"},{"start":84,"end":93,"cssClass":"pl-s"}],[{"start":4,"end":9,"cssClass":"pl-en"},{"start":10,"end":56,"cssClass":"pl-s"},{"start":38,"end":54,"cssClass":"pl-s1"},{"start":38,"end":39,"cssClass":"pl-kos"},{"start":39,"end":53,"cssClass":"pl-s1"},{"start":53,"end":54,"cssClass":"pl-kos"}],[],[{"start":4,"end":53,"cssClass":"pl-c"}],[{"start":4,"end":16,"cssClass":"pl-s1"},{"start":17,"end":18,"cssClass":"pl-c1"},{"start":19,"end":32,"cssClass":"pl-v"},{"start":33,"end":44,"cssClass":"pl-s1"},{"start":44,"end":45,"cssClass":"pl-c1"},{"start":45,"end":56,"cssClass":"pl-s1"},{"start":58,"end":65,"cssClass":"pl-s1"},{"start":65,"end":66,"cssClass":"pl-c1"},{"start":66,"end":75,"cssClass":"pl-s1"}],[],[{"start":4,"end":75,"cssClass":"pl-c"}],[{"start":4,"end":16,"cssClass":"pl-s1"},{"start":17,"end":20,"cssClass":"pl-en"},{"start":21,"end":32,"cssClass":"pl-s1"},{"start":32,"end":33,"cssClass":"pl-c1"},{"start":33,"end":36,"cssClass":"pl-c1"}],[],[{"start":4,"end":67,"cssClass":"pl-c"}],[{"start":4,"end":70,"cssClass":"pl-c"}],[{"start":4,"end":57,"cssClass":"pl-c"}],[{"start":4,"end":60,"cssClass":"pl-c"}],[],[{"start":4,"end":9,"cssClass":"pl-en"},{"start":10,"end":36,"cssClass":"pl-s"}],[{"start":4,"end":74,"cssClass":"pl-c"}],[{"start":4,"end":16,"cssClass":"pl-s1"},{"start":17,"end":31,"cssClass":"pl-en"},{"start":32,"end":42,"cssClass":"pl-s1"},{"start":42,"end":43,"cssClass":"pl-c1"},{"start":43,"end":44,"cssClass":"pl-c1"},{"start":46,"end":54,"cssClass":"pl-c"}],[{"start":4,"end":52,"cssClass":"pl-c"}],[{"start":4,"end":18,"cssClass":"pl-en"},{"start":20,"end":32,"cssClass":"pl-s1"},{"start":35,"end":54,"cssClass":"pl-s1"},{"start":54,"end":55,"cssClass":"pl-c1"},{"start":55,"end":56,"cssClass":"pl-c1"},{"start":58,"end":65,"cssClass":"pl-c"}],[],[{"start":4,"end":18,"cssClass":"pl-c"}],[{"start":4,"end":16,"cssClass":"pl-s1"},{"start":17,"end":39,"cssClass":"pl-en"}],[],[{"start":4,"end":9,"cssClass":"pl-en"},{"start":10,"end":30,"cssClass":"pl-s"},{"start":31,"end":33,"cssClass":"pl-s1"},{"start":34,"end":39,"cssClass":"pl-en"},{"start":40,"end":52,"cssClass":"pl-s1"},{"start":53,"end":58,"cssClass":"pl-s1"}],[{"start":4,"end":9,"cssClass":"pl-en"},{"start":10,"end":27,"cssClass":"pl-s"},{"start":29,"end":41,"cssClass":"pl-s1"},{"start":42,"end":60,"cssClass":"pl-s1"},{"start":61,"end":62,"cssClass":"pl-c1"}],[],[{"start":4,"end":9,"cssClass":"pl-en"},{"start":10,"end":29,"cssClass":"pl-s"}],[{"start":4,"end":55,"cssClass":"pl-c"}],[{"start":4,"end":24,"cssClass":"pl-en"},{"start":25,"end":36,"cssClass":"pl-s1"},{"start":36,"end":37,"cssClass":"pl-c1"},{"start":38,"end":50,"cssClass":"pl-s1"},{"start":53,"end":62,"cssClass":"pl-s1"},{"start":62,"end":63,"cssClass":"pl-c1"},{"start":63,"end":68,"cssClass":"pl-s"},{"start":70,"end":79,"cssClass":"pl-s1"},{"start":79,"end":80,"cssClass":"pl-c1"},{"start":80,"end":85,"cssClass":"pl-c1"}],[{"start":4,"end":24,"cssClass":"pl-en"},{"start":25,"end":36,"cssClass":"pl-s1"},{"start":36,"end":37,"cssClass":"pl-c1"},{"start":38,"end":50,"cssClass":"pl-s1"},{"start":53,"end":62,"cssClass":"pl-s1"},{"start":62,"end":63,"cssClass":"pl-c1"},{"start":63,"end":69,"cssClass":"pl-s"},{"start":71,"end":80,"cssClass":"pl-s1"},{"start":80,"end":81,"cssClass":"pl-c1"},{"start":81,"end":86,"cssClass":"pl-c1"}],[{"start":4,"end":24,"cssClass":"pl-en"},{"start":25,"end":36,"cssClass":"pl-s1"},{"start":36,"end":37,"cssClass":"pl-c1"},{"start":38,"end":50,"cssClass":"pl-s1"},{"start":53,"end":62,"cssClass":"pl-s1"},{"start":62,"end":63,"cssClass":"pl-c1"},{"start":63,"end":73,"cssClass":"pl-s"},{"start":75,"end":79,"cssClass":"pl-s1"},{"start":79,"end":80,"cssClass":"pl-c1"},{"start":80,"end":84,"cssClass":"pl-c1"},{"start":86,"end":95,"cssClass":"pl-s1"},{"start":95,"end":96,"cssClass":"pl-c1"},{"start":96,"end":101,"cssClass":"pl-c1"}],[{"start":4,"end":25,"cssClass":"pl-en"},{"start":26,"end":37,"cssClass":"pl-s1"},{"start":37,"end":38,"cssClass":"pl-c1"},{"start":39,"end":51,"cssClass":"pl-s1"},{"start":54,"end":63,"cssClass":"pl-s1"},{"start":63,"end":64,"cssClass":"pl-c1"},{"start":64,"end":67,"cssClass":"pl-c1"}],[],[{"start":4,"end":58,"cssClass":"pl-c"}],[{"start":4,"end":9,"cssClass":"pl-en"},{"start":10,"end":69,"cssClass":"pl-s"}]],"csv":null,"csvError":null,"dependabotInfo":{"showConfigurationBanner":null,"configFilePath":null,"networkDependabotPath":"/simopt-admin/simopt/network/updates","dismissConfigurationNoticePath":"/settings/dismiss-notice/dependabot_configuration_notice","configurationNoticeDismissed":false,"repoAlertsPath":"/simopt-admin/simopt/security/dependabot","repoSecurityAndAnalysisPath":"/simopt-admin/simopt/settings/security_analysis","repoOwnerIsOrg":false,"currentUserCanAdminRepo":false},"displayName":"demo_random_problem_solver.py","displayUrl":"https://github.com/simopt-admin/simopt/blob/python_dev_litong/demo_random_problem_solver.py?raw=true","headerInfo":{"blobSize":"4.17 KB","deleteInfo":{"deleteTooltip":"Delete this file"},"editInfo":{"editTooltip":"Edit this file"},"ghDesktopPath":"x-github-client://openRepo/https://github.com/simopt-admin/simopt?branch=python_dev_litong&filepath=demo_random_problem_solver.py","gitLfsPath":null,"onBranch":true,"shortPath":"dc37c07","siteNavLoginPath":"/login?return_to=https%3A%2F%2Fgithub.com%2Fsimopt-admin%2Fsimopt%2Fblob%2Fpython_dev_litong%2Fdemo_random_problem_solver.py","isCSV":false,"isRichtext":false,"toc":null,"lineInfo":{"truncatedLoc":"100","truncatedSloc":"79"},"mode":"file"},"image":false,"isCodeownersFile":null,"isPlain":false,"isValidLegacyIssueTemplate":false,"issueTemplateHelpUrl":"https://docs.github.com/articles/about-issue-and-pull-request-templates","issueTemplate":null,"discussionTemplate":null,"language":"Python","languageID":303,"large":false,"loggedIn":true,"newDiscussionPath":"/simopt-admin/simopt/discussions/new","newIssuePath":"/simopt-admin/simopt/issues/new","planSupportInfo":{"repoIsFork":null,"repoOwnedByCurrentUser":null,"requestFullPath":"/simopt-admin/simopt/blob/python_dev_litong/demo_random_problem_solver.py","showFreeOrgGatedFeatureMessage":null,"showPlanSupportBanner":null,"upgradeDataAttributes":null,"upgradePath":null},"publishBannersInfo":{"dismissActionNoticePath":"/settings/dismiss-notice/publish_action_from_dockerfile","dismissStackNoticePath":"/settings/dismiss-notice/publish_stack_from_file","releasePath":"/simopt-admin/simopt/releases/new?marketplace=true","showPublishActionBanner":false,"showPublishStackBanner":false},"renderImageOrRaw":false,"richText":null,"renderedFileInfo":null,"shortPath":null,"tabSize":8,"topBannersInfo":{"overridingGlobalFundingFile":false,"globalPreferredFundingPath":null,"repoOwner":"simopt-admin","repoName":"simopt","showInvalidCitationWarning":false,"citationHelpUrl":"https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files","showDependabotConfigurationBanner":null,"actionsOnboardingTip":null},"truncated":false,"viewable":true,"workflowRedirectUrl":null,"symbols":{"timedOut":false,"notAnalyzed":false,"symbols":[{"name":"solver_name","kind":"constant","identStart":849,"identEnd":860,"extentStart":849,"extentEnd":872,"fullyQualifiedName":"solver_name","identUtf16":{"start":{"lineNumber":22,"utf16Col":0},"end":{"lineNumber":22,"utf16Col":11}},"extentUtf16":{"start":{"lineNumber":22,"utf16Col":0},"end":{"lineNumber":22,"utf16Col":23}}},{"name":"rebase","kind":"function","identStart":953,"identEnd":959,"extentStart":949,"extentEnd":1318,"fullyQualifiedName":"rebase","identUtf16":{"start":{"lineNumber":26,"utf16Col":4},"end":{"lineNumber":26,"utf16Col":10}},"extentUtf16":{"start":{"lineNumber":26,"utf16Col":0},"end":{"lineNumber":34,"utf16Col":21}}},{"name":"strtobool","kind":"function","identStart":1324,"identEnd":1333,"extentStart":1320,"extentEnd":1423,"fullyQualifiedName":"strtobool","identUtf16":{"start":{"lineNumber":36,"utf16Col":4},"end":{"lineNumber":36,"utf16Col":13}},"extentUtf16":{"start":{"lineNumber":36,"utf16Col":0},"end":{"lineNumber":41,"utf16Col":20}}},{"name":"n_inst","kind":"constant","identStart":1425,"identEnd":1431,"extentStart":1425,"extentEnd":1506,"fullyQualifiedName":"n_inst","identUtf16":{"start":{"lineNumber":43,"utf16Col":0},"end":{"lineNumber":43,"utf16Col":6}},"extentUtf16":{"start":{"lineNumber":43,"utf16Col":0},"end":{"lineNumber":43,"utf16Col":81}}},{"name":"rand","kind":"constant","identStart":1507,"identEnd":1511,"extentStart":1507,"extentEnd":1615,"fullyQualifiedName":"rand","identUtf16":{"start":{"lineNumber":44,"utf16Col":0},"end":{"lineNumber":44,"utf16Col":4}},"extentUtf16":{"start":{"lineNumber":44,"utf16Col":0},"end":{"lineNumber":44,"utf16Col":108}}},{"name":"rand","kind":"constant","identStart":1616,"identEnd":1620,"extentStart":1616,"extentEnd":1638,"fullyQualifiedName":"rand","identUtf16":{"start":{"lineNumber":45,"utf16Col":0},"end":{"lineNumber":45,"utf16Col":4}},"extentUtf16":{"start":{"lineNumber":45,"utf16Col":0},"end":{"lineNumber":45,"utf16Col":22}}},{"name":"model_fixed_factors","kind":"constant","identStart":1640,"identEnd":1659,"extentStart":1640,"extentEnd":1664,"fullyQualifiedName":"model_fixed_factors","identUtf16":{"start":{"lineNumber":47,"utf16Col":0},"end":{"lineNumber":47,"utf16Col":19}},"extentUtf16":{"start":{"lineNumber":47,"utf16Col":0},"end":{"lineNumber":47,"utf16Col":24}}},{"name":"myproblem","kind":"constant","identStart":1692,"identEnd":1701,"extentStart":1692,"extentEnd":1778,"fullyQualifiedName":"myproblem","identUtf16":{"start":{"lineNumber":49,"utf16Col":0},"end":{"lineNumber":49,"utf16Col":9}},"extentUtf16":{"start":{"lineNumber":49,"utf16Col":0},"end":{"lineNumber":49,"utf16Col":86}}},{"name":"random_rng","kind":"constant","identStart":1780,"identEnd":1790,"extentStart":1780,"extentEnd":1917,"fullyQualifiedName":"random_rng","identUtf16":{"start":{"lineNumber":51,"utf16Col":0},"end":{"lineNumber":51,"utf16Col":10}},"extentUtf16":{"start":{"lineNumber":51,"utf16Col":0},"end":{"lineNumber":51,"utf16Col":137}}},{"name":"rng_list2","kind":"constant","identStart":1918,"identEnd":1927,"extentStart":1918,"extentEnd":2009,"fullyQualifiedName":"rng_list2","identUtf16":{"start":{"lineNumber":52,"utf16Col":0},"end":{"lineNumber":52,"utf16Col":9}},"extentUtf16":{"start":{"lineNumber":52,"utf16Col":0},"end":{"lineNumber":52,"utf16Col":91}}}]}},"copilotInfo":{"notices":{"codeViewPopover":{"dismissed":false,"dismissPath":"/settings/dismiss-notice/code_view_copilot_popover"}},"userAccess":{"accessAllowed":false,"hasSubscriptionEnded":false,"orgHasCFBAccess":false,"userHasCFIAccess":false,"userHasOrgs":false,"userIsOrgAdmin":false,"userIsOrgMember":false,"business":null,"featureRequestInfo":null}},"csrf_tokens":{"/simopt-admin/simopt/branches":{"post":"bo50Ta0pqezuFqdyhIuxc4A08ugUj2YisoOlcEleflvGgbDCQyrX6x2QrXdyQL1Gx5s7jQXbLyjUL5oqWBfTZA"},"/repos/preferences":{"post":"wcttIkjLR6fY-zNC2Fo7-UJslO5L-3dymWrHjxwpae5VIvim4U7WKfi5aFRtkY0DfMXteLVfrWvlUETm6eCsag"}}},"title":"simopt/demo_random_problem_solver.py at python_dev_litong · simopt-admin/simopt"}