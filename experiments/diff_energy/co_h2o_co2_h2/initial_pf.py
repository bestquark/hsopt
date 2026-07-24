"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline: Reiher et al. Eq. (19)/(34) rigorous PF2, adapted to 1 kcal/mol.
Calibration SHA-256: 54c2ec1a1920ca3bcf4312ba563fa4c60aef3a891fe866620f12ea30fb879b41
Calibration SHA-256: e11a3110730d91aff7db7f5f4aad0570e92cc8bb1e7f876da9c452c8f47dd21d
Calibration SHA-256: f4a02811d32ef062fab0760c4d3fb62b94cab0f8dbba19c262ece5ef0ffab5c7
Calibration SHA-256: 380459e0917ba69111ad7645463ad3021de3ff36e884ea94afe2df6b364a8f59
Full bound/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with PyGridSynth.
"""


class ProductFormula:
    parameters = {
        "Reaction seed co": (7337, 1.388226831886365e-12),
        "Reaction seed co2": (4881, 4.864806163475133e-12),
        "Reaction seed h2": (474, 8.798268690635403e-10),
        "Reaction seed h2o": (9600, 1.7988423693035088e-12),
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
