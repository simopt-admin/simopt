from experiment_test import ExperimentTest, ExperimentTestMixin  # type: ignore


class TestProblemSolver(ExperimentTest, ExperimentTestMixin):
    def setUp(self) -> None:
        # Set the name of the experiment file
        self.file = "{{FILE}}"
        # Let the parent class set up and run the experiment
        super().setUp()
