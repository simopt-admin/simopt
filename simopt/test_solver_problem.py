from wrapper_base import Experiment, read_experiment_results

solver_name = "RNDSRCH" # random search solver
problem_name = "SSCONT-1" # mm1 queueing problem
myexperiment = Experiment(solver_name, problem_name, solver_fixed_factors={"sample_size": 10})
myexperiment.run(n_macroreps=10, crn_across_solns=True)
print("Here")
myexperiment = read_experiment_results(solver_name + "_on_" + problem_name)
myexperiment.post_replicate(n_postreps=50, n_postreps_init_opt=100, crn_across_budget=True, crn_across_macroreps=False)
# print("Now here.")
myexperiment.plot_progress_curves(plot_type="all", normalize=False)
myexperiment.plot_progress_curves(plot_type="all", normalize=True)
# print("Finally here.")
#myexperiment.plot_progress_curves(plot_type="mean", normalize=False)
#myexperiment.plot_progress_curves(plot_type="mean", normalize=True)

# Testing solver.simulate_up_to()

# from rng.mrg32k3a import MRG32k3a
# from solvers.randomsearch import RandomSearch
# from problems.cntnv_max_profit import CntNVMaxProfit

# mysolver = RandomSearch()
# myproblem = CntNVMaxProfit()
# mysolver.solution_progenitor_rngs = [MRG32k3a()]

# crn_across_solns = False
# soln1 = mysolver.create_new_solution(x=(1,), problem=myproblem, crn_across_solns=crn_across_solns)
# soln2 = mysolver.create_new_solution(x=(2,), problem=myproblem, crn_across_solns=crn_across_solns)
# soln3 = mysolver.create_new_solution(x=(3,), problem=myproblem, crn_across_solns=crn_across_solns)

# myproblem.simulate(soln1, m=1)
# print(soln1.rng_list[0].s_ss_sss_index)
# myproblem.simulate(soln2, m=2)
# print(soln2.rng_list[0].s_ss_sss_index)
# myproblem.simulate(soln3, m=3)
# print(soln3.rng_list[0].s_ss_sss_index)

# myproblem.simulate_up_to(solutions={soln1, soln2, soln3}, n_reps = 10)
# print(soln1.rng_list[0].s_ss_sss_index)
# print(soln2.rng_list[0].s_ss_sss_index)
# print(soln3.rng_list[0].s_ss_sss_index)