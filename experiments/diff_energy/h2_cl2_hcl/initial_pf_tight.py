"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline variant: tight
Baseline: Reiher et al. LIQUi|>-style numeric-beta/Eq. (34) PF2 (tight), adapted to 1 kcal/mol.
Calibration SHA-256: 40b1045a670cc7013f96b7e6d75f5a0482eb827d44ee62edda7d3c56c052c0dd
Calibration SHA-256: 27b6ba0028a689b273bafd6bde5735cb7d4a666964844cc760df357ed09a7823
Calibration SHA-256: e046400f62a46f74d164db070d80fb3d21dcdb8939296331d9caac68f4bf5961
Full beta/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with CUDA-Q.
"""


class ProductFormula:
    parameters = {
        "Reaction seed cl2": (4, 8.96714588919966e-11),
        "Reaction seed h2": (4, 2.107157487687332e-08),
        "Reaction seed hcl": (11, 1.9308080985900095e-11),
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
