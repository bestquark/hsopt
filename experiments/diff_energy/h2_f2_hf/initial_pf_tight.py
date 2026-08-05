"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline variant: tight
Baseline: Reiher et al. LIQUi|>-style numeric-beta/Eq. (34) PF2 (tight), adapted to 1 kcal/mol.
Calibration SHA-256: 716feaf21bed6eca4415289c16f3b396c779462be33f6d00bb29a32aece8f3cb
Calibration SHA-256: 27b6ba0028a689b273bafd6bde5735cb7d4a666964844cc760df357ed09a7823
Calibration SHA-256: 8c4c936a9054c7d9b2c63391743c4b3c1fc54f5ad76da52ad3c1a3c2d4851f81
Full beta/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with PyGridSynth.
"""


class ProductFormula:
    parameters = {
        "Reaction seed f2": (2, 1.3601767112112144e-10),
        "Reaction seed h2": (4, 2.107157487687332e-08),
        "Reaction seed hf": (17, 7.313623298533612e-12),
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
