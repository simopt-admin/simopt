{"payload":{"allShortcutsEnabled":true,"fileTree":{"":{"items":[{"name":"data_farming_experiments","path":"data_farming_experiments","contentType":"directory"},{"name":"demo","path":"demo","contentType":"directory"},{"name":"dist","path":"dist","contentType":"directory"},{"name":"docs","path":"docs","contentType":"directory"},{"name":"experiments","path":"experiments","contentType":"directory"},{"name":"notebooks","path":"notebooks","contentType":"directory"},{"name":"simopt","path":"simopt","contentType":"directory"},{"name":"simoptlib.egg-info","path":"simoptlib.egg-info","contentType":"directory"},{"name":"workshop","path":"workshop","contentType":"directory"},{"name":".gitignore","path":".gitignore","contentType":"file"},{"name":"LICENSE","path":"LICENSE","contentType":"file"},{"name":"README.md","path":"README.md","contentType":"file"},{"name":"demo_radom_model.py","path":"demo_radom_model.py","contentType":"file"},{"name":"demo_random_problem.py","path":"demo_random_problem.py","contentType":"file"},{"name":"demo_random_problem_solver.py","path":"demo_random_problem_solver.py","contentType":"file"},{"name":"demo_user.py","path":"demo_user.py","contentType":"file"},{"name":"pyproject.toml","path":"pyproject.toml","contentType":"file"}],"totalCount":17}},"fileTreeProcessingTime":2.910284,"foldersToFetch":[],"reducedMotionEnabled":"system","repo":{"id":194012165,"defaultBranch":"master","name":"simopt","ownerLogin":"simopt-admin","currentUserCanPush":true,"isFork":false,"isEmpty":false,"createdAt":"2019-06-26T22:55:30.000-04:00","ownerAvatar":"https://avatars.githubusercontent.com/u/52267122?v=4","public":true,"private":false,"isOrgOwned":false},"symbolsExpanded":false,"treeExpanded":true,"refInfo":{"name":"python_dev_litong","listCacheKey":"v0:1692137230.0","canEdit":true,"refType":"branch","currentOid":"382561d40918dac6fcfb54e7c1f873bdca0f46e9"},"path":"demo_user.py","currentUser":{"id":46491025,"login":"liulitong-Jessie","userEmail":"118010185@link.cuhk.edu.cn"},"blob":{"rawLines":["\"\"\"","This script is the user interface for generating multiple random problem instances and","solve them by specified solvers.","It create problem-solver groups and runs multiple","macroreplications of each problem-solver pair. To run the file, user need","to import the solver and probelm they want to build random instances at the beginning,","and also provide an input file, which include the information needed to ","build random instances (the name of problem, number of random instances to ","generate, and some overriding factors).","\"\"\"","","import sys","import os.path as o","import os","import re","sys.path.append(o.abspath(o.join(o.dirname(sys.modules[__name__].__file__), \"..\")))","","# Import the ProblemsSolvers class and other useful functions","from simopt.experiment_base import ProblemsSolvers, plot_solvability_profiles","from rng.mrg32k3a import MRG32k3a","from simopt.base import Solution","from simopt.models.smf import SMF_Max","from simopt.models.rmitd import RMITDMaxRevenue","from simopt.models.san_2 import SANLongestPath, SANLongestPathConstr","from simopt.models.mm1queue import MM1MinMeanSojournTime","","","# !! When testing a new solver/problem, first import problems from the random code file,","# Then create a test_input.txt file in your computer.","# There you should add the import statement and an entry in the file","# You need to specify name of solvers and problems you want to test in the file by 'solver_name'","# And specify the problem related informations by problem = [...]","# All lines start with '#' will be counted as commend and will not be implemented","# See the following example for more details.","","# Ex:","# To create two random instance of SAN and three random instances of SMF:","# In the demo_user.py, modify:","# from simopt.models.smf import SMF_Max","# from simopt.models.san_2 import SANLongestPath","# In the input information file (test_input.txt), include the following lines:","# solver_names = [\"RNDSRCH\", \"ASTRODF\", \"NELDMD\"]","# problem1 = [SANLongestPath, 2, {'num_nodes':8, 'num_arcs':12}]","# problem2 = [SMF_Max, 3, {'num_nodes':7, 'num_arcs':16}]","","# Grab information from the input file","def get_info(path):","    L = []","    with open(path) as file:","        lines = [line.rstrip() for line in file]","        for line in lines:","            if not line.startswith(\"#\") and line:","                L.append(line)","    lines = L","    command_lines = []","    problem_sets = []","    for line in lines:","        if 'import' in line:","            command_lines.append(line)","        elif 'solver_names' in line:","            solver_names = line","        else:","            problem_sets.append(line)","","    for i in command_lines:","        exec(i)","    ","    problems = []","    solver_names = eval(re.findall(r'\\[.*?\\]', solver_names)[0])","    for l in problem_sets:","        o = re.findall(r'\\[.*?\\]', l)[0]","        problems.append(eval(o))","    ","    problem_sets = [p[0] for p in problems]","    L_num = [p[1] for p in problems]","    L_para = [p[2] for p in problems]","    ","    return solver_names, problem_sets, L_num, L_para","","# Read input file and process information","path = input('Please input the path of the input file: ')","if \"'\" in path:  # If the input path already has quotation marks","    path = path.replace(\"'\", \"\")","    ","solver_names, problem_set, L_num, L_para = get_info(path)","rands = [True for i in range(len(problem_set))]","","# Check whether the input file is valid","if len(L_num) != len(problem_set) or len(L_para) != len(problem_set):","    print('Invalid input. The input number of random instances does not match with the number of problems you want.')","    print('Please check your input file')","","def rebase(random_rng, n):","    new_rngs = []","    for rng in random_rng:","        stream_index = rng.s_ss_sss_index[0]","        substream_index = rng.s_ss_sss_index[1]","        subsubstream_index = rng.s_ss_sss_index[2]","        new_rngs.append(MRG32k3a(s_ss_sss_index=[stream_index, substream_index + n, subsubstream_index]))","    random_rng = new_rngs","    return random_rng","","myproblems = problem_set","","# Check whether the problem is random","for i in range(len(problem_set)):","    if L_num[i] == 0:","        L_num[i] = 1","        rands[i] = False","    else:","        rands[i] = True","","problems = []","problem_names = []","","def generate_problem(i, myproblems, rands, problems, L_num, L_para):","    print('For problem ', myproblems[i]().name, ':')  ","    model_fixed_factors = L_para[i]","    ","    name = myproblems[i]","    myproblem = name(model_fixed_factors=model_fixed_factors, random=rands[i])","    random_rng = [MRG32k3a(s_ss_sss_index=[2, 4 + L_num[i], ss]) for ss in range(myproblem.n_rngs)]","    rng_list2 = [MRG32k3a(s_ss_sss_index=[2, 4, ss]) for ss in range(myproblem.model.n_random)]","    ","    if rands[i] == False:  # Determinant case","        problems.append(myproblem)","        myproblem.name = str(myproblem.model.name) + str(0)","        problem_names.append(myproblem.name)","        print('')","    ","    else:","        for j in range(L_num[i]):","            random_rng = rebase(random_rng, 1)  # Advance the substream for different instances","            rng_list2 = rebase(rng_list2, 1)","            name = myproblems[i]","            myproblem = name(model_fixed_factors=model_fixed_factors, random=rands[i], random_rng=rng_list2)","            myproblem.attach_rngs(random_rng)","            # myproblem.name = str(myproblem.model.name) + str(j)","            myproblem.name = str(myproblem.name) + '-' + str(j)","            problems.append(myproblem)","            problem_names.append(myproblem.name)","            print('')","    ","    return problems, problem_names","   ","# Generate problems","for i in range(len(L_num)):","        problems, problem_names = generate_problem(i, myproblems, rands, problems, L_num, L_para)","","# Initialize an instance of the experiment class.","mymetaexperiment = ProblemsSolvers(solver_names=solver_names, problems = problems)","","# Run a fixed number of macroreplications of each solver on each problem.","mymetaexperiment.run(n_macroreps=3)","","print(\"Post-processing results.\")","# Run a fixed number of postreplications at all recommended solutions.","mymetaexperiment.post_replicate(n_postreps=20)","# Find an optimal solution x* for normalization.","mymetaexperiment.post_normalize(n_postreps_init_opt=20)","","print(\"Plotting results.\")","# Produce basic plots of the solvers on the problems.","plot_solvability_profiles(experiments=mymetaexperiment.experiments, plot_type=\"cdf_solvability\")","","# Plots will be saved in the folder experiments/plots.","print(\"Finished. Plots can be found in experiments/plots folder.\")"],"stylingDirectives":[[{"start":0,"end":3,"cssClass":"pl-s"}],[{"start":0,"end":86,"cssClass":"pl-s"}],[{"start":0,"end":32,"cssClass":"pl-s"}],[{"start":0,"end":49,"cssClass":"pl-s"}],[{"start":0,"end":73,"cssClass":"pl-s"}],[{"start":0,"end":86,"cssClass":"pl-s"}],[{"start":0,"end":72,"cssClass":"pl-s"}],[{"start":0,"end":75,"cssClass":"pl-s"}],[{"start":0,"end":39,"cssClass":"pl-s"}],[{"start":0,"end":3,"cssClass":"pl-s"}],[],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":10,"cssClass":"pl-s1"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":9,"cssClass":"pl-s1"},{"start":10,"end":14,"cssClass":"pl-s1"},{"start":15,"end":17,"cssClass":"pl-k"},{"start":18,"end":19,"cssClass":"pl-s1"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":9,"cssClass":"pl-s1"}],[{"start":0,"end":6,"cssClass":"pl-k"},{"start":7,"end":9,"cssClass":"pl-s1"}],[{"start":0,"end":3,"cssClass":"pl-s1"},{"start":4,"end":8,"cssClass":"pl-s1"},{"start":9,"end":15,"cssClass":"pl-en"},{"start":16,"end":17,"cssClass":"pl-s1"},{"start":18,"end":25,"cssClass":"pl-en"},{"start":26,"end":27,"cssClass":"pl-s1"},{"start":28,"end":32,"cssClass":"pl-en"},{"start":33,"end":34,"cssClass":"pl-s1"},{"start":35,"end":42,"cssClass":"pl-en"},{"start":43,"end":46,"cssClass":"pl-s1"},{"start":47,"end":54,"cssClass":"pl-s1"},{"start":55,"end":63,"cssClass":"pl-s1"},{"start":65,"end":73,"cssClass":"pl-s1"},{"start":76,"end":80,"cssClass":"pl-s"}],[],[{"start":0,"end":61,"cssClass":"pl-c"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":11,"cssClass":"pl-s1"},{"start":12,"end":27,"cssClass":"pl-s1"},{"start":28,"end":34,"cssClass":"pl-k"},{"start":35,"end":50,"cssClass":"pl-v"},{"start":52,"end":77,"cssClass":"pl-s1"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":8,"cssClass":"pl-s1"},{"start":9,"end":17,"cssClass":"pl-s1"},{"start":18,"end":24,"cssClass":"pl-k"},{"start":25,"end":33,"cssClass":"pl-v"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":11,"cssClass":"pl-s1"},{"start":12,"end":16,"cssClass":"pl-s1"},{"start":17,"end":23,"cssClass":"pl-k"},{"start":24,"end":32,"cssClass":"pl-v"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":11,"cssClass":"pl-s1"},{"start":12,"end":18,"cssClass":"pl-s1"},{"start":19,"end":22,"cssClass":"pl-s1"},{"start":23,"end":29,"cssClass":"pl-k"},{"start":30,"end":37,"cssClass":"pl-v"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":11,"cssClass":"pl-s1"},{"start":12,"end":18,"cssClass":"pl-s1"},{"start":19,"end":24,"cssClass":"pl-s1"},{"start":25,"end":31,"cssClass":"pl-k"},{"start":32,"end":47,"cssClass":"pl-v"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":11,"cssClass":"pl-s1"},{"start":12,"end":18,"cssClass":"pl-s1"},{"start":19,"end":24,"cssClass":"pl-s1"},{"start":25,"end":31,"cssClass":"pl-k"},{"start":32,"end":46,"cssClass":"pl-v"},{"start":48,"end":68,"cssClass":"pl-v"}],[{"start":0,"end":4,"cssClass":"pl-k"},{"start":5,"end":11,"cssClass":"pl-s1"},{"start":12,"end":18,"cssClass":"pl-s1"},{"start":19,"end":27,"cssClass":"pl-s1"},{"start":28,"end":34,"cssClass":"pl-k"},{"start":35,"end":56,"cssClass":"pl-v"}],[],[],[{"start":0,"end":88,"cssClass":"pl-c"}],[{"start":0,"end":53,"cssClass":"pl-c"}],[{"start":0,"end":68,"cssClass":"pl-c"}],[{"start":0,"end":96,"cssClass":"pl-c"}],[{"start":0,"end":65,"cssClass":"pl-c"}],[{"start":0,"end":81,"cssClass":"pl-c"}],[{"start":0,"end":45,"cssClass":"pl-c"}],[],[{"start":0,"end":5,"cssClass":"pl-c"}],[{"start":0,"end":73,"cssClass":"pl-c"}],[{"start":0,"end":30,"cssClass":"pl-c"}],[{"start":0,"end":39,"cssClass":"pl-c"}],[{"start":0,"end":48,"cssClass":"pl-c"}],[{"start":0,"end":78,"cssClass":"pl-c"}],[{"start":0,"end":49,"cssClass":"pl-c"}],[{"start":0,"end":64,"cssClass":"pl-c"}],[{"start":0,"end":57,"cssClass":"pl-c"}],[],[{"start":0,"end":38,"cssClass":"pl-c"}],[{"start":0,"end":3,"cssClass":"pl-k"},{"start":4,"end":12,"cssClass":"pl-en"},{"start":13,"end":17,"cssClass":"pl-s1"}],[{"start":4,"end":5,"cssClass":"pl-v"},{"start":6,"end":7,"cssClass":"pl-c1"}],[{"start":4,"end":8,"cssClass":"pl-k"},{"start":9,"end":13,"cssClass":"pl-en"},{"start":14,"end":18,"cssClass":"pl-s1"},{"start":20,"end":22,"cssClass":"pl-k"},{"start":23,"end":27,"cssClass":"pl-s1"}],[{"start":8,"end":13,"cssClass":"pl-s1"},{"start":14,"end":15,"cssClass":"pl-c1"},{"start":17,"end":21,"cssClass":"pl-s1"},{"start":22,"end":28,"cssClass":"pl-en"},{"start":31,"end":34,"cssClass":"pl-k"},{"start":35,"end":39,"cssClass":"pl-s1"},{"start":40,"end":42,"cssClass":"pl-c1"},{"start":43,"end":47,"cssClass":"pl-s1"}],[{"start":8,"end":11,"cssClass":"pl-k"},{"start":12,"end":16,"cssClass":"pl-s1"},{"start":17,"end":19,"cssClass":"pl-c1"},{"start":20,"end":25,"cssClass":"pl-s1"}],[{"start":12,"end":14,"cssClass":"pl-k"},{"start":15,"end":18,"cssClass":"pl-c1"},{"start":19,"end":23,"cssClass":"pl-s1"},{"start":24,"end":34,"cssClass":"pl-en"},{"start":35,"end":38,"cssClass":"pl-s"},{"start":40,"end":43,"cssClass":"pl-c1"},{"start":44,"end":48,"cssClass":"pl-s1"}],[{"start":16,"end":17,"cssClass":"pl-v"},{"start":18,"end":24,"cssClass":"pl-en"},{"start":25,"end":29,"cssClass":"pl-s1"}],[{"start":4,"end":9,"cssClass":"pl-s1"},{"start":10,"end":11,"cssClass":"pl-c1"},{"start":12,"end":13,"cssClass":"pl-v"}],[{"start":4,"end":17,"cssClass":"pl-s1"},{"start":18,"end":19,"cssClass":"pl-c1"}],[{"start":4,"end":16,"cssClass":"pl-s1"},{"start":17,"end":18,"cssClass":"pl-c1"}],[{"start":4,"end":7,"cssClass":"pl-k"},{"start":8,"end":12,"cssClass":"pl-s1"},{"start":13,"end":15,"cssClass":"pl-c1"},{"start":16,"end":21,"cssClass":"pl-s1"}],[{"start":8,"end":10,"cssClass":"pl-k"},{"start":11,"end":19,"cssClass":"pl-s"},{"start":20,"end":22,"cssClass":"pl-c1"},{"start":23,"end":27,"cssClass":"pl-s1"}],[{"start":12,"end":25,"cssClass":"pl-s1"},{"start":26,"end":32,"cssClass":"pl-en"},{"start":33,"end":37,"cssClass":"pl-s1"}],[{"start":8,"end":12,"cssClass":"pl-k"},{"start":13,"end":27,"cssClass":"pl-s"},{"start":28,"end":30,"cssClass":"pl-c1"},{"start":31,"end":35,"cssClass":"pl-s1"}],[{"start":12,"end":24,"cssClass":"pl-s1"},{"start":25,"end":26,"cssClass":"pl-c1"},{"start":27,"end":31,"cssClass":"pl-s1"}],[{"start":8,"end":12,"cssClass":"pl-k"}],[{"start":12,"end":24,"cssClass":"pl-s1"},{"start":25,"end":31,"cssClass":"pl-en"},{"start":32,"end":36,"cssClass":"pl-s1"}],[],[{"start":4,"end":7,"cssClass":"pl-k"},{"start":8,"end":9,"cssClass":"pl-s1"},{"start":10,"end":12,"cssClass":"pl-c1"},{"start":13,"end":26,"cssClass":"pl-s1"}],[{"start":8,"end":12,"cssClass":"pl-en"},{"start":13,"end":14,"cssClass":"pl-s1"}],[],[{"start":4,"end":12,"cssClass":"pl-s1"},{"start":13,"end":14,"cssClass":"pl-c1"}],[{"start":4,"end":16,"cssClass":"pl-s1"},{"start":17,"end":18,"cssClass":"pl-c1"},{"start":19,"end":23,"cssClass":"pl-en"},{"start":24,"end":26,"cssClass":"pl-s1"},{"start":27,"end":34,"cssClass":"pl-en"},{"start":35,"end":45,"cssClass":"pl-s"},{"start":47,"end":59,"cssClass":"pl-s1"},{"start":61,"end":62,"cssClass":"pl-c1"}],[{"start":4,"end":7,"cssClass":"pl-k"},{"start":8,"end":9,"cssClass":"pl-s1"},{"start":10,"end":12,"cssClass":"pl-c1"},{"start":13,"end":25,"cssClass":"pl-s1"}],[{"start":8,"end":9,"cssClass":"pl-s1"},{"start":10,"end":11,"cssClass":"pl-c1"},{"start":12,"end":14,"cssClass":"pl-s1"},{"start":15,"end":22,"cssClass":"pl-en"},{"start":23,"end":33,"cssClass":"pl-s"},{"start":35,"end":36,"cssClass":"pl-s1"},{"start":38,"end":39,"cssClass":"pl-c1"}],[{"start":8,"end":16,"cssClass":"pl-s1"},{"start":17,"end":23,"cssClass":"pl-en"},{"start":24,"end":28,"cssClass":"pl-en"},{"start":29,"end":30,"cssClass":"pl-s1"}],[],[{"start":4,"end":16,"cssClass":"pl-s1"},{"start":17,"end":18,"cssClass":"pl-c1"},{"start":20,"end":21,"cssClass":"pl-s1"},{"start":22,"end":23,"cssClass":"pl-c1"},{"start":25,"end":28,"cssClass":"pl-k"},{"start":29,"end":30,"cssClass":"pl-s1"},{"start":31,"end":33,"cssClass":"pl-c1"},{"start":34,"end":42,"cssClass":"pl-s1"}],[{"start":4,"end":9,"cssClass":"pl-v"},{"start":10,"end":11,"cssClass":"pl-c1"},{"start":13,"end":14,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":18,"end":21,"cssClass":"pl-k"},{"start":22,"end":23,"cssClass":"pl-s1"},{"start":24,"end":26,"cssClass":"pl-c1"},{"start":27,"end":35,"cssClass":"pl-s1"}],[{"start":4,"end":10,"cssClass":"pl-v"},{"start":11,"end":12,"cssClass":"pl-c1"},{"start":14,"end":15,"cssClass":"pl-s1"},{"start":16,"end":17,"cssClass":"pl-c1"},{"start":19,"end":22,"cssClass":"pl-k"},{"start":23,"end":24,"cssClass":"pl-s1"},{"start":25,"end":27,"cssClass":"pl-c1"},{"start":28,"end":36,"cssClass":"pl-s1"}],[],[{"start":4,"end":10,"cssClass":"pl-k"},{"start":11,"end":23,"cssClass":"pl-s1"},{"start":25,"end":37,"cssClass":"pl-s1"},{"start":39,"end":44,"cssClass":"pl-v"},{"start":46,"end":52,"cssClass":"pl-v"}],[],[{"start":0,"end":41,"cssClass":"pl-c"}],[{"start":0,"end":4,"cssClass":"pl-s1"},{"start":5,"end":6,"cssClass":"pl-c1"},{"start":7,"end":12,"cssClass":"pl-en"},{"start":13,"end":56,"cssClass":"pl-s"}],[{"start":0,"end":2,"cssClass":"pl-k"},{"start":3,"end":6,"cssClass":"pl-s"},{"start":7,"end":9,"cssClass":"pl-c1"},{"start":10,"end":14,"cssClass":"pl-s1"},{"start":17,"end":64,"cssClass":"pl-c"}],[{"start":4,"end":8,"cssClass":"pl-s1"},{"start":9,"end":10,"cssClass":"pl-c1"},{"start":11,"end":15,"cssClass":"pl-s1"},{"start":16,"end":23,"cssClass":"pl-en"},{"start":24,"end":27,"cssClass":"pl-s"},{"start":29,"end":31,"cssClass":"pl-s"}],[],[{"start":0,"end":12,"cssClass":"pl-s1"},{"start":14,"end":25,"cssClass":"pl-s1"},{"start":27,"end":32,"cssClass":"pl-v"},{"start":34,"end":40,"cssClass":"pl-v"},{"start":41,"end":42,"cssClass":"pl-c1"},{"start":43,"end":51,"cssClass":"pl-en"},{"start":52,"end":56,"cssClass":"pl-s1"}],[{"start":0,"end":5,"cssClass":"pl-s1"},{"start":6,"end":7,"cssClass":"pl-c1"},{"start":9,"end":13,"cssClass":"pl-c1"},{"start":14,"end":17,"cssClass":"pl-k"},{"start":18,"end":19,"cssClass":"pl-s1"},{"start":20,"end":22,"cssClass":"pl-c1"},{"start":23,"end":28,"cssClass":"pl-en"},{"start":29,"end":32,"cssClass":"pl-en"},{"start":33,"end":44,"cssClass":"pl-s1"}],[],[{"start":0,"end":39,"cssClass":"pl-c"}],[{"start":0,"end":2,"cssClass":"pl-k"},{"start":3,"end":6,"cssClass":"pl-en"},{"start":7,"end":12,"cssClass":"pl-v"},{"start":14,"end":16,"cssClass":"pl-c1"},{"start":17,"end":20,"cssClass":"pl-en"},{"start":21,"end":32,"cssClass":"pl-s1"},{"start":34,"end":36,"cssClass":"pl-c1"},{"start":37,"end":40,"cssClass":"pl-en"},{"start":41,"end":47,"cssClass":"pl-v"},{"start":49,"end":51,"cssClass":"pl-c1"},{"start":52,"end":55,"cssClass":"pl-en"},{"start":56,"end":67,"cssClass":"pl-s1"}],[{"start":4,"end":9,"cssClass":"pl-en"},{"start":10,"end":116,"cssClass":"pl-s"}],[{"start":4,"end":9,"cssClass":"pl-en"},{"start":10,"end":40,"cssClass":"pl-s"}],[],[{"start":0,"end":3,"cssClass":"pl-k"},{"start":4,"end":10,"cssClass":"pl-en"},{"start":11,"end":21,"cssClass":"pl-s1"},{"start":23,"end":24,"cssClass":"pl-s1"}],[{"start":4,"end":12,"cssClass":"pl-s1"},{"start":13,"end":14,"cssClass":"pl-c1"}],[{"start":4,"end":7,"cssClass":"pl-k"},{"start":8,"end":11,"cssClass":"pl-s1"},{"start":12,"end":14,"cssClass":"pl-c1"},{"start":15,"end":25,"cssClass":"pl-s1"}],[{"start":8,"end":20,"cssClass":"pl-s1"},{"start":21,"end":22,"cssClass":"pl-c1"},{"start":23,"end":26,"cssClass":"pl-s1"},{"start":27,"end":41,"cssClass":"pl-s1"},{"start":42,"end":43,"cssClass":"pl-c1"}],[{"start":8,"end":23,"cssClass":"pl-s1"},{"start":24,"end":25,"cssClass":"pl-c1"},{"start":26,"end":29,"cssClass":"pl-s1"},{"start":30,"end":44,"cssClass":"pl-s1"},{"start":45,"end":46,"cssClass":"pl-c1"}],[{"start":8,"end":26,"cssClass":"pl-s1"},{"start":27,"end":28,"cssClass":"pl-c1"},{"start":29,"end":32,"cssClass":"pl-s1"},{"start":33,"end":47,"cssClass":"pl-s1"},{"start":48,"end":49,"cssClass":"pl-c1"}],[{"start":8,"end":16,"cssClass":"pl-s1"},{"start":17,"end":23,"cssClass":"pl-en"},{"start":24,"end":32,"cssClass":"pl-v"},{"start":33,"end":47,"cssClass":"pl-s1"},{"start":47,"end":48,"cssClass":"pl-c1"},{"start":49,"end":61,"cssClass":"pl-s1"},{"start":63,"end":78,"cssClass":"pl-s1"},{"start":79,"end":80,"cssClass":"pl-c1"},{"start":81,"end":82,"cssClass":"pl-s1"},{"start":84,"end":102,"cssClass":"pl-s1"}],[{"start":4,"end":14,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":17,"end":25,"cssClass":"pl-s1"}],[{"start":4,"end":10,"cssClass":"pl-k"},{"start":11,"end":21,"cssClass":"pl-s1"}],[],[{"start":0,"end":10,"cssClass":"pl-s1"},{"start":11,"end":12,"cssClass":"pl-c1"},{"start":13,"end":24,"cssClass":"pl-s1"}],[],[{"start":0,"end":37,"cssClass":"pl-c"}],[{"start":0,"end":3,"cssClass":"pl-k"},{"start":4,"end":5,"cssClass":"pl-s1"},{"start":6,"end":8,"cssClass":"pl-c1"},{"start":9,"end":14,"cssClass":"pl-en"},{"start":15,"end":18,"cssClass":"pl-en"},{"start":19,"end":30,"cssClass":"pl-s1"}],[{"start":4,"end":6,"cssClass":"pl-k"},{"start":7,"end":12,"cssClass":"pl-v"},{"start":13,"end":14,"cssClass":"pl-s1"},{"start":16,"end":18,"cssClass":"pl-c1"},{"start":19,"end":20,"cssClass":"pl-c1"}],[{"start":8,"end":13,"cssClass":"pl-v"},{"start":14,"end":15,"cssClass":"pl-s1"},{"start":17,"end":18,"cssClass":"pl-c1"},{"start":19,"end":20,"cssClass":"pl-c1"}],[{"start":8,"end":13,"cssClass":"pl-s1"},{"start":14,"end":15,"cssClass":"pl-s1"},{"start":17,"end":18,"cssClass":"pl-c1"},{"start":19,"end":24,"cssClass":"pl-c1"}],[{"start":4,"end":8,"cssClass":"pl-k"}],[{"start":8,"end":13,"cssClass":"pl-s1"},{"start":14,"end":15,"cssClass":"pl-s1"},{"start":17,"end":18,"cssClass":"pl-c1"},{"start":19,"end":23,"cssClass":"pl-c1"}],[],[{"start":0,"end":8,"cssClass":"pl-s1"},{"start":9,"end":10,"cssClass":"pl-c1"}],[{"start":0,"end":13,"cssClass":"pl-s1"},{"start":14,"end":15,"cssClass":"pl-c1"}],[],[{"start":0,"end":3,"cssClass":"pl-k"},{"start":4,"end":20,"cssClass":"pl-en"},{"start":21,"end":22,"cssClass":"pl-s1"},{"start":24,"end":34,"cssClass":"pl-s1"},{"start":36,"end":41,"cssClass":"pl-s1"},{"start":43,"end":51,"cssClass":"pl-s1"},{"start":53,"end":58,"cssClass":"pl-v"},{"start":60,"end":66,"cssClass":"pl-v"}],[{"start":4,"end":9,"cssClass":"pl-en"},{"start":10,"end":24,"cssClass":"pl-s"},{"start":26,"end":36,"cssClass":"pl-s1"},{"start":37,"end":38,"cssClass":"pl-s1"},{"start":42,"end":46,"cssClass":"pl-s1"},{"start":48,"end":51,"cssClass":"pl-s"}],[{"start":4,"end":23,"cssClass":"pl-s1"},{"start":24,"end":25,"cssClass":"pl-c1"},{"start":26,"end":32,"cssClass":"pl-v"},{"start":33,"end":34,"cssClass":"pl-s1"}],[],[{"start":4,"end":8,"cssClass":"pl-s1"},{"start":9,"end":10,"cssClass":"pl-c1"},{"start":11,"end":21,"cssClass":"pl-s1"},{"start":22,"end":23,"cssClass":"pl-s1"}],[{"start":4,"end":13,"cssClass":"pl-s1"},{"start":14,"end":15,"cssClass":"pl-c1"},{"start":16,"end":20,"cssClass":"pl-en"},{"start":21,"end":40,"cssClass":"pl-s1"},{"start":40,"end":41,"cssClass":"pl-c1"},{"start":41,"end":60,"cssClass":"pl-s1"},{"start":62,"end":68,"cssClass":"pl-s1"},{"start":68,"end":69,"cssClass":"pl-c1"},{"start":69,"end":74,"cssClass":"pl-s1"},{"start":75,"end":76,"cssClass":"pl-s1"}],[{"start":4,"end":14,"cssClass":"pl-s1"},{"start":15,"end":16,"cssClass":"pl-c1"},{"start":18,"end":26,"cssClass":"pl-v"},{"start":27,"end":41,"cssClass":"pl-s1"},{"start":41,"end":42,"cssClass":"pl-c1"},{"start":43,"end":44,"cssClass":"pl-c1"},{"start":46,"end":47,"cssClass":"pl-c1"},{"start":48,"end":49,"cssClass":"pl-c1"},{"start":50,"end":55,"cssClass":"pl-v"},{"start":56,"end":57,"cssClass":"pl-s1"},{"start":60,"end":62,"cssClass":"pl-s1"},{"start":65,"end":68,"cssClass":"pl-k"},{"start":69,"end":71,"cssClass":"pl-s1"},{"start":72,"end":74,"cssClass":"pl-c1"},{"start":75,"end":80,"cssClass":"pl-en"},{"start":81,"end":90,"cssClass":"pl-s1"},{"start":91,"end":97,"cssClass":"pl-s1"}],[{"start":4,"end":13,"cssClass":"pl-s1"},{"start":14,"end":15,"cssClass":"pl-c1"},{"start":17,"end":25,"cssClass":"pl-v"},{"start":26,"end":40,"cssClass":"pl-s1"},{"start":40,"end":41,"cssClass":"pl-c1"},{"start":42,"end":43,"cssClass":"pl-c1"},{"start":45,"end":46,"cssClass":"pl-c1"},{"start":48,"end":50,"cssClass":"pl-s1"},{"start":53,"end":56,"cssClass":"pl-k"},{"start":57,"end":59,"cssClass":"pl-s1"},{"start":60,"end":62,"cssClass":"pl-c1"},{"start":63,"end":68,"cssClass":"pl-en"},{"start":69,"end":78,"cssClass":"pl-s1"},{"start":79,"end":84,"cssClass":"pl-s1"},{"start":85,"end":93,"cssClass":"pl-s1"}],[],[{"start":4,"end":6,"cssClass":"pl-k"},{"start":7,"end":12,"cssClass":"pl-s1"},{"start":13,"end":14,"cssClass":"pl-s1"},{"start":16,"end":18,"cssClass":"pl-c1"},{"start":19,"end":24,"cssClass":"pl-c1"},{"start":27,"end":45,"cssClass":"pl-c"}],[{"start":8,"end":16,"cssClass":"pl-s1"},{"start":17,"end":23,"cssClass":"pl-en"},{"start":24,"end":33,"cssClass":"pl-s1"}],[{"start":8,"end":17,"cssClass":"pl-s1"},{"start":18,"end":22,"cssClass":"pl-s1"},{"start":23,"end":24,"cssClass":"pl-c1"},{"start":25,"end":28,"cssClass":"pl-en"},{"start":29,"end":38,"cssClass":"pl-s1"},{"start":39,"end":44,"cssClass":"pl-s1"},{"start":45,"end":49,"cssClass":"pl-s1"},{"start":51,"end":52,"cssClass":"pl-c1"},{"start":53,"end":56,"cssClass":"pl-en"},{"start":57,"end":58,"cssClass":"pl-c1"}],[{"start":8,"end":21,"cssClass":"pl-s1"},{"start":22,"end":28,"cssClass":"pl-en"},{"start":29,"end":38,"cssClass":"pl-s1"},{"start":39,"end":43,"cssClass":"pl-s1"}],[{"start":8,"end":13,"cssClass":"pl-en"},{"start":14,"end":16,"cssClass":"pl-s"}],[],[{"start":4,"end":8,"cssClass":"pl-k"}],[{"start":8,"end":11,"cssClass":"pl-k"},{"start":12,"end":13,"cssClass":"pl-s1"},{"start":14,"end":16,"cssClass":"pl-c1"},{"start":17,"end":22,"cssClass":"pl-en"},{"start":23,"end":28,"cssClass":"pl-v"},{"start":29,"end":30,"cssClass":"pl-s1"}],[{"start":12,"end":22,"cssClass":"pl-s1"},{"start":23,"end":24,"cssClass":"pl-c1"},{"start":25,"end":31,"cssClass":"pl-en"},{"start":32,"end":42,"cssClass":"pl-s1"},{"start":44,"end":45,"cssClass":"pl-c1"},{"start":48,"end":95,"cssClass":"pl-c"}],[{"start":12,"end":21,"cssClass":"pl-s1"},{"start":22,"end":23,"cssClass":"pl-c1"},{"start":24,"end":30,"cssClass":"pl-en"},{"start":31,"end":40,"cssClass":"pl-s1"},{"start":42,"end":43,"cssClass":"pl-c1"}],[{"start":12,"end":16,"cssClass":"pl-s1"},{"start":17,"end":18,"cssClass":"pl-c1"},{"start":19,"end":29,"cssClass":"pl-s1"},{"start":30,"end":31,"cssClass":"pl-s1"}],[{"start":12,"end":21,"cssClass":"pl-s1"},{"start":22,"end":23,"cssClass":"pl-c1"},{"start":24,"end":28,"cssClass":"pl-en"},{"start":29,"end":48,"cssClass":"pl-s1"},{"start":48,"end":49,"cssClass":"pl-c1"},{"start":49,"end":68,"cssClass":"pl-s1"},{"start":70,"end":76,"cssClass":"pl-s1"},{"start":76,"end":77,"cssClass":"pl-c1"},{"start":77,"end":82,"cssClass":"pl-s1"},{"start":83,"end":84,"cssClass":"pl-s1"},{"start":87,"end":97,"cssClass":"pl-s1"},{"start":97,"end":98,"cssClass":"pl-c1"},{"start":98,"end":107,"cssClass":"pl-s1"}],[{"start":12,"end":21,"cssClass":"pl-s1"},{"start":22,"end":33,"cssClass":"pl-en"},{"start":34,"end":44,"cssClass":"pl-s1"}],[{"start":12,"end":65,"cssClass":"pl-c"}],[{"start":12,"end":21,"cssClass":"pl-s1"},{"start":22,"end":26,"cssClass":"pl-s1"},{"start":27,"end":28,"cssClass":"pl-c1"},{"start":29,"end":32,"cssClass":"pl-en"},{"start":33,"end":42,"cssClass":"pl-s1"},{"start":43,"end":47,"cssClass":"pl-s1"},{"start":49,"end":50,"cssClass":"pl-c1"},{"start":51,"end":54,"cssClass":"pl-s"},{"start":55,"end":56,"cssClass":"pl-c1"},{"start":57,"end":60,"cssClass":"pl-en"},{"start":61,"end":62,"cssClass":"pl-s1"}],[{"start":12,"end":20,"cssClass":"pl-s1"},{"start":21,"end":27,"cssClass":"pl-en"},{"start":28,"end":37,"cssClass":"pl-s1"}],[{"start":12,"end":25,"cssClass":"pl-s1"},{"start":26,"end":32,"cssClass":"pl-en"},{"start":33,"end":42,"cssClass":"pl-s1"},{"start":43,"end":47,"cssClass":"pl-s1"}],[{"start":12,"end":17,"cssClass":"pl-en"},{"start":18,"end":20,"cssClass":"pl-s"}],[],[{"start":4,"end":10,"cssClass":"pl-k"},{"start":11,"end":19,"cssClass":"pl-s1"},{"start":21,"end":34,"cssClass":"pl-s1"}],[],[{"start":0,"end":19,"cssClass":"pl-c"}],[{"start":0,"end":3,"cssClass":"pl-k"},{"start":4,"end":5,"cssClass":"pl-s1"},{"start":6,"end":8,"cssClass":"pl-c1"},{"start":9,"end":14,"cssClass":"pl-en"},{"start":15,"end":18,"cssClass":"pl-en"},{"start":19,"end":24,"cssClass":"pl-v"}],[{"start":8,"end":16,"cssClass":"pl-s1"},{"start":18,"end":31,"cssClass":"pl-s1"},{"start":32,"end":33,"cssClass":"pl-c1"},{"start":34,"end":50,"cssClass":"pl-en"},{"start":51,"end":52,"cssClass":"pl-s1"},{"start":54,"end":64,"cssClass":"pl-s1"},{"start":66,"end":71,"cssClass":"pl-s1"},{"start":73,"end":81,"cssClass":"pl-s1"},{"start":83,"end":88,"cssClass":"pl-v"},{"start":90,"end":96,"cssClass":"pl-v"}],[],[{"start":0,"end":49,"cssClass":"pl-c"}],[{"start":0,"end":16,"cssClass":"pl-s1"},{"start":17,"end":18,"cssClass":"pl-c1"},{"start":19,"end":34,"cssClass":"pl-v"},{"start":35,"end":47,"cssClass":"pl-s1"},{"start":47,"end":48,"cssClass":"pl-c1"},{"start":48,"end":60,"cssClass":"pl-s1"},{"start":62,"end":70,"cssClass":"pl-s1"},{"start":71,"end":72,"cssClass":"pl-c1"},{"start":73,"end":81,"cssClass":"pl-s1"}],[],[{"start":0,"end":73,"cssClass":"pl-c"}],[{"start":0,"end":16,"cssClass":"pl-s1"},{"start":17,"end":20,"cssClass":"pl-en"},{"start":21,"end":32,"cssClass":"pl-s1"},{"start":32,"end":33,"cssClass":"pl-c1"},{"start":33,"end":34,"cssClass":"pl-c1"}],[],[{"start":0,"end":5,"cssClass":"pl-en"},{"start":6,"end":32,"cssClass":"pl-s"}],[{"start":0,"end":70,"cssClass":"pl-c"}],[{"start":0,"end":16,"cssClass":"pl-s1"},{"start":17,"end":31,"cssClass":"pl-en"},{"start":32,"end":42,"cssClass":"pl-s1"},{"start":42,"end":43,"cssClass":"pl-c1"},{"start":43,"end":45,"cssClass":"pl-c1"}],[{"start":0,"end":48,"cssClass":"pl-c"}],[{"start":0,"end":16,"cssClass":"pl-s1"},{"start":17,"end":31,"cssClass":"pl-en"},{"start":32,"end":51,"cssClass":"pl-s1"},{"start":51,"end":52,"cssClass":"pl-c1"},{"start":52,"end":54,"cssClass":"pl-c1"}],[],[{"start":0,"end":5,"cssClass":"pl-en"},{"start":6,"end":25,"cssClass":"pl-s"}],[{"start":0,"end":53,"cssClass":"pl-c"}],[{"start":0,"end":25,"cssClass":"pl-en"},{"start":26,"end":37,"cssClass":"pl-s1"},{"start":37,"end":38,"cssClass":"pl-c1"},{"start":38,"end":54,"cssClass":"pl-s1"},{"start":55,"end":66,"cssClass":"pl-s1"},{"start":68,"end":77,"cssClass":"pl-s1"},{"start":77,"end":78,"cssClass":"pl-c1"},{"start":78,"end":95,"cssClass":"pl-s"}],[],[{"start":0,"end":54,"cssClass":"pl-c"}],[{"start":0,"end":5,"cssClass":"pl-en"},{"start":6,"end":65,"cssClass":"pl-s"}]],"csv":null,"csvError":null,"dependabotInfo":{"showConfigurationBanner":null,"configFilePath":null,"networkDependabotPath":"/simopt-admin/simopt/network/updates","dismissConfigurationNoticePath":"/settings/dismiss-notice/dependabot_configuration_notice","configurationNoticeDismissed":false,"repoAlertsPath":"/simopt-admin/simopt/security/dependabot","repoSecurityAndAnalysisPath":"/simopt-admin/simopt/settings/security_analysis","repoOwnerIsOrg":false,"currentUserCanAdminRepo":false},"displayName":"demo_user.py","displayUrl":"https://github.com/simopt-admin/simopt/blob/python_dev_litong/demo_user.py?raw=true","headerInfo":{"blobSize":"6.31 KB","deleteInfo":{"deleteTooltip":"Delete this file"},"editInfo":{"editTooltip":"Edit this file"},"ghDesktopPath":"x-github-client://openRepo/https://github.com/simopt-admin/simopt?branch=python_dev_litong&filepath=demo_user.py","gitLfsPath":null,"onBranch":true,"shortPath":"1872aa0","siteNavLoginPath":"/login?return_to=https%3A%2F%2Fgithub.com%2Fsimopt-admin%2Fsimopt%2Fblob%2Fpython_dev_litong%2Fdemo_user.py","isCSV":false,"isRichtext":false,"toc":null,"lineInfo":{"truncatedLoc":"167","truncatedSloc":"139"},"mode":"file"},"image":false,"isCodeownersFile":null,"isPlain":false,"isValidLegacyIssueTemplate":false,"issueTemplateHelpUrl":"https://docs.github.com/articles/about-issue-and-pull-request-templates","issueTemplate":null,"discussionTemplate":null,"language":"Python","languageID":303,"large":false,"loggedIn":true,"newDiscussionPath":"/simopt-admin/simopt/discussions/new","newIssuePath":"/simopt-admin/simopt/issues/new","planSupportInfo":{"repoIsFork":null,"repoOwnedByCurrentUser":null,"requestFullPath":"/simopt-admin/simopt/blob/python_dev_litong/demo_user.py","showFreeOrgGatedFeatureMessage":null,"showPlanSupportBanner":null,"upgradeDataAttributes":null,"upgradePath":null},"publishBannersInfo":{"dismissActionNoticePath":"/settings/dismiss-notice/publish_action_from_dockerfile","dismissStackNoticePath":"/settings/dismiss-notice/publish_stack_from_file","releasePath":"/simopt-admin/simopt/releases/new?marketplace=true","showPublishActionBanner":false,"showPublishStackBanner":false},"renderImageOrRaw":false,"richText":null,"renderedFileInfo":null,"shortPath":null,"tabSize":8,"topBannersInfo":{"overridingGlobalFundingFile":false,"globalPreferredFundingPath":null,"repoOwner":"simopt-admin","repoName":"simopt","showInvalidCitationWarning":false,"citationHelpUrl":"https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/about-citation-files","showDependabotConfigurationBanner":null,"actionsOnboardingTip":null},"truncated":false,"viewable":true,"workflowRedirectUrl":null,"symbols":{"timedOut":false,"notAnalyzed":false,"symbols":[{"name":"get_info","kind":"function","identStart":2086,"identEnd":2094,"extentStart":2082,"extentEnd":2985,"fullyQualifiedName":"get_info","identUtf16":{"start":{"lineNumber":46,"utf16Col":4},"end":{"lineNumber":46,"utf16Col":12}},"extentUtf16":{"start":{"lineNumber":46,"utf16Col":0},"end":{"lineNumber":77,"utf16Col":52}}},{"name":"path","kind":"constant","identStart":3029,"identEnd":3033,"extentStart":3029,"extentEnd":3086,"fullyQualifiedName":"path","identUtf16":{"start":{"lineNumber":80,"utf16Col":0},"end":{"lineNumber":80,"utf16Col":4}},"extentUtf16":{"start":{"lineNumber":80,"utf16Col":0},"end":{"lineNumber":80,"utf16Col":57}}},{"name":"rands","kind":"constant","identStart":3248,"identEnd":3253,"extentStart":3248,"extentEnd":3295,"fullyQualifiedName":"rands","identUtf16":{"start":{"lineNumber":85,"utf16Col":0},"end":{"lineNumber":85,"utf16Col":5}},"extentUtf16":{"start":{"lineNumber":85,"utf16Col":0},"end":{"lineNumber":85,"utf16Col":47}}},{"name":"rebase","kind":"function","identStart":3572,"identEnd":3578,"extentStart":3568,"extentEnd":3937,"fullyQualifiedName":"rebase","identUtf16":{"start":{"lineNumber":92,"utf16Col":4},"end":{"lineNumber":92,"utf16Col":10}},"extentUtf16":{"start":{"lineNumber":92,"utf16Col":0},"end":{"lineNumber":100,"utf16Col":21}}},{"name":"myproblems","kind":"constant","identStart":3939,"identEnd":3949,"extentStart":3939,"extentEnd":3963,"fullyQualifiedName":"myproblems","identUtf16":{"start":{"lineNumber":102,"utf16Col":0},"end":{"lineNumber":102,"utf16Col":10}},"extentUtf16":{"start":{"lineNumber":102,"utf16Col":0},"end":{"lineNumber":102,"utf16Col":24}}},{"name":"problems","kind":"constant","identStart":4140,"identEnd":4148,"extentStart":4140,"extentEnd":4153,"fullyQualifiedName":"problems","identUtf16":{"start":{"lineNumber":112,"utf16Col":0},"end":{"lineNumber":112,"utf16Col":8}},"extentUtf16":{"start":{"lineNumber":112,"utf16Col":0},"end":{"lineNumber":112,"utf16Col":13}}},{"name":"problem_names","kind":"constant","identStart":4154,"identEnd":4167,"extentStart":4154,"extentEnd":4172,"fullyQualifiedName":"problem_names","identUtf16":{"start":{"lineNumber":113,"utf16Col":0},"end":{"lineNumber":113,"utf16Col":13}},"extentUtf16":{"start":{"lineNumber":113,"utf16Col":0},"end":{"lineNumber":113,"utf16Col":18}}},{"name":"generate_problem","kind":"function","identStart":4178,"identEnd":4194,"extentStart":4174,"extentEnd":5505,"fullyQualifiedName":"generate_problem","identUtf16":{"start":{"lineNumber":115,"utf16Col":4},"end":{"lineNumber":115,"utf16Col":20}},"extentUtf16":{"start":{"lineNumber":115,"utf16Col":0},"end":{"lineNumber":143,"utf16Col":34}}},{"name":"mymetaexperiment","kind":"constant","identStart":5707,"identEnd":5723,"extentStart":5707,"extentEnd":5789,"fullyQualifiedName":"mymetaexperiment","identUtf16":{"start":{"lineNumber":150,"utf16Col":0},"end":{"lineNumber":150,"utf16Col":16}},"extentUtf16":{"start":{"lineNumber":150,"utf16Col":0},"end":{"lineNumber":150,"utf16Col":82}}}]}},"copilotInfo":{"notices":{"codeViewPopover":{"dismissed":false,"dismissPath":"/settings/dismiss-notice/code_view_copilot_popover"}},"userAccess":{"accessAllowed":false,"hasSubscriptionEnded":false,"orgHasCFBAccess":false,"userHasCFIAccess":false,"userHasOrgs":false,"userIsOrgAdmin":false,"userIsOrgMember":false,"business":null,"featureRequestInfo":null}},"csrf_tokens":{"/simopt-admin/simopt/branches":{"post":"gb0BMxr-fXyz6kun91_9a83dZUM5fg4_5o-QJh56D-QpssW89P0De0BsQaIBlPFeinKsJigqRzWAI698DzOi2w"},"/repos/preferences":{"post":"EIlhdtGF6ei2TTLwdZupeDLV6yHP0AEgzvpCLVPSf8WEYPTyeAB4ZpYPaebAUB-CDHyStzF02zmywMFEphu6QQ"}}},"title":"simopt/demo_user.py at python_dev_litong · simopt-admin/simopt"}