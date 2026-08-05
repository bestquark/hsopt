"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline: Reiher et al. Eq. (19)/(34) rigorous PF2, adapted to 1 kcal/mol.
Calibration SHA-256: 3360c6bc1a6241cd2a2f8d37592f97f22863fbe9f9b63322313ab3c0f34ef7d1
Calibration SHA-256: 8a203636f38c708b40b9a3e78987ea42698170ee5c688fac0dce59c4159ce5b5
Calibration SHA-256: 15ebed832e3ada0019e0aae2efbda168b9effb9e9caddfb9abbdf8b5216bc55b
Full bound/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with PyGridSynth.
"""


class ProductFormula:
    parameters = {
        "Reaction seed h2": (94, 2.5085831835241464e-10),
        "Reaction seed n2": (82, 3.547888883337231e-12),
        "Reaction seed nh3": (161, 7.116493542666714e-13),
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
