"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline: Reiher et al. Eq. (19)/(34) rigorous PF2, adapted to 1 kcal/mol.
Calibration SHA-256: 824d9587adb641cdbc7af4477f2c499af03ce07ce71b55be89c350a9eaf7586f
Calibration SHA-256: f4a02811d32ef062fab0760c4d3fb62b94cab0f8dbba19c262ece5ef0ffab5c7
Calibration SHA-256: 54d82a7c27c5b45e5619f7bc3d1457fecb095337384be8cde96ec505eb0ffed1
Full bound/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with PyGridSynth.
"""


class ProductFormula:
    parameters = {
        "Reaction seed cl2": (3903, 3.575114741579249e-12),
        "Reaction seed h2": (411, 1.3703794022127365e-09),
        "Reaction seed hcl": (7547, 1.4354341322483844e-12),
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
