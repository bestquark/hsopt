"""One shared protocol; the host supplies sealed per-molecule starting parameters."""


class ProductFormula:
    baseline_parameters = {}

    def __init__(self, problem):
        self.r, self.epsilon, self.beta = self.baseline_parameters[problem["name"]]

    def repetitions(self):
        return self.r

    def build_step(self, terms, step_time):
        ordered = sorted(terms, key=lambda term: term["label"])
        half = [(term, 0.5 * term["coefficient"] * step_time) for term in ordered[:-1]]
        center = [(ordered[-1], ordered[-1]["coefficient"] * step_time)]
        return half + center + list(reversed(half))

    def synthesis_epsilon(self, term, angle):
        # Starting value only: mutate this into a term/angle-dependent precision scheme.
        return self.epsilon
