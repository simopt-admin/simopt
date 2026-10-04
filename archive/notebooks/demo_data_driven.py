# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys

sys.path.append("..")

# %%
import random

import matplotlib.pyplot as plt
import numpy as np

from simopt.experiment_base import PlotType, ProblemSolver, post_normalize
from simopt.analysis.progress_curve import ProgressCurve
from simopt.experiment.plot import save_plots
from simopt.options import ConfidenceIntervalOptions
from simopt.input_models import InputModel


# %%
class DemandInputModel(InputModel):
    def random(self, rng: random.Random, burr_c: float, burr_k: float) -> float:
        mean = 10
        std = 0.5
        return rng.normalvariate(mean, std)


# %%
class FileInputModel(InputModel):
    def __init__(self, filename):
        self.data = np.load(filename)

    def random(self, rng: random.Random, burr_c: float, burr_k: float) -> float:
        return np.random.choice(self.data, size=1, replace=True)[0]


# %%
class Experiment(ProblemSolver):
    def model_created(self, model):
        # model.demand_model = DemandInputModel()
        model.demand_model = FileInputModel("demand.npy")


# %%
# Run 10 macroreplications of ASTRO-DF on the continuous newsvendor problem.
experiment = Experiment("ASTRODF", "CNTNEWS-1")
experiment.run(n_macroreps=10)

# Post-process the results.
experiment.post_replicate(n_postreps=200)
post_normalize(experiments=[experiment], n_postreps_init_opt=200)

# Record the results and plot the mean progress curve.
experiment.log_experiment_results()
save_plots(
    [experiment],
    ProgressCurve(
        agg=PlotType.MEAN.value,
        normalize=False,
        beta=0.5,
        ci_options=ConfidenceIntervalOptions(
            n_bootstraps=100, confidence_level=0.95, bias_correction=True
        ),
    ),
    print_max_hw=True,
    use_experiment_crn=True,
)
plt.show()
