import pickle
import sys

import numpy as np
import zstandard as zstd

from simopt.experiment_base import (
    ProblemSolver,
    post_normalize,
)
from simopt.models.san import SANLongestPathStochastic
from simopt.solvers.fcsa import FCSA

initial = (5,) * 13  # starting mean for each arc
constraint_nodes = [6, 8]  # nodes with corresponding stochastic constraints
max_length_to_node = [5, 5]  # max expected length to each constraint node
budget = 2000  # number of simmulation replications ran by solver
problem_factors = {
    "constraint_nodes": constraint_nodes,
    "length_to_node_constraint": max_length_to_node,
    "initial_solution": initial,
    "budget": budget,
}
problem = SANLongestPathStochastic(fixed_factors=problem_factors)
problems = [problem]

csa_factors = {
    "search_direction": "CSA",
    "normalize_grads": False,
    "report_all_solns": True,
    "crn_across_solns": False,
}
csa = FCSA(fixed_factors=csa_factors, name="CSA")
csa_n_factors = {
    "search_direction": "CSA",
    "normalize_grads": True,
    "report_all_solns": True,
    "crn_across_solns": False,
}
csa_n = FCSA(fixed_factors=csa_n_factors, name="CSA-N")
fcsa_factors = {
    "search_direction": "FCSA",
    "normalize_grads": True,
    "report_all_solns": True,
    "crn_across_solns": False,
}
fcsa = FCSA(fixed_factors=fcsa_factors, name="FCSA")
solver = csa
solvers = [csa, csa_n, fcsa]


e1 = ProblemSolver(solver=csa, problem=problem)
e2 = ProblemSolver(solver=csa_n, problem=problem)
e3 = ProblemSolver(solver=fcsa, problem=problem)
e1.run(n_macroreps=10)
e2.run(n_macroreps=10)
e3.run(n_macroreps=10)
e1.post_replicate(n_postreps=100)
e2.post_replicate(n_postreps=100)
e3.post_replicate(n_postreps=100)
post_normalize([e1, e2, e3], 100)


def main():
    assert len(sys.argv) == 2
    print(sys.argv)

    if sys.argv[1] == "dump":
        with open("test-data.pkl", "wb") as f:
            data = pickle.dumps([e1, e2, e3])
            f.write(zstd.compress(data))
    elif sys.argv[1] == "check":
        with open("test-data.pkl", "rb") as f:
            data = zstd.decompress(f.read())
            e1_saved, e2_saved, e3_saved = pickle.loads(data)

            for e, e_saved in zip(
                [e1, e2, e3], [e1_saved, e2_saved, e3_saved], strict=False
            ):
                for a, b in zip(
                    e.all_recommended_xs, e_saved.all_recommended_xs, strict=False
                ):
                    assert np.allclose(a, b)

                for a, b in zip(
                    e.all_est_objectives, e_saved.all_est_objectives, strict=False
                ):
                    assert np.allclose(a, b)

                for a, b in zip(e.all_est_lhs, e_saved.all_est_lhs, strict=False):
                    assert np.allclose(a, b)


if __name__ == "__main__":
    main()
