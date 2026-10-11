"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline: Reiher et al. Eq. (19)/(34) rigorous PF2, adapted to 1 kcal/mol.
Calibration SHA-256: 5bcc82f3f94b3fefe69beeb0ebfe4b5fbdc689b471ce59511b3e315d1710fcf1
Calibration SHA-256: bb0ba00af0d90ed51a0d6a9b7967d96bd5723a0f5711e522deab7a6f0ad1d767
Full bound/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with CUDA-Q.
"""


class ProductFormula:
    parameters = {
        "Reaction seed hcn": (83, 4.5679320461379546e-12),
        "Reaction seed hnc": (83, 4.856839181519122e-12),
    }

    def __init__(self, problem):
        self.n_steps, self.epsilon = self.parameters[str(problem["name"])]

    def repetitions(self):
        return self.n_steps

    def build_step(self, terms, step_time):
        ordered = sorted(terms, key=lambda term: term["label"])
        half = [(term, 0.5 * float(term["coefficient"]) * step_time) for term in ordered[:-1]]
        center = [(ordered[-1], float(ordered[-1]["coefficient"]) * step_time)]
        return half + center + list(reversed(half))

    def synthesis_epsilon(self, term, angle):
        _ = (term, angle)
        return self.epsilon
