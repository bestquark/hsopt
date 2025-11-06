"""
Usage:
CombinedMetric = 0.5 * GPUMemoryEval + 0.3 * TrotterErrorEval + 0.2 * SimulatedMemoryEval

evaluator = CombinedMetric(my_ham, program="candidate.py", trials=5)
result = evaluator.evaluate()
print(json.dumps(result, indent=2))
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from trovolut.eval.base import BaseEval

class CombinedEvalMeta(type):
    """Enable syntax like 0.5 * GPUMemoryEval + 0.3 * TrotterErrorEval."""

    def __rmul__(cls, weight: float):
        return CombinedComponent(weight, cls)

    def __add__(cls, other):
        if isinstance(other, CombinedComponent):
            return CombinedSum([CombinedComponent(1.0, cls), other])
        elif isinstance(other, CombinedSum):
            return CombinedSum([CombinedComponent(1.0, cls)] + other.components)
        elif isinstance(other, type):
            return CombinedSum([CombinedComponent(1.0, cls), CombinedComponent(1.0, other)])
        raise TypeError("Can only add Evaluator classes or CombinedComponent instances.")


class CombinedComponent:
    def __init__(self, weight: float, evaluator_cls: type):
        self.weight = float(weight)
        self.evaluator_cls = evaluator_cls

    def __add__(self, other):
        if isinstance(other, CombinedComponent):
            return CombinedSum([self, other])
        elif isinstance(other, CombinedSum):
            return CombinedSum([self] + other.components)
        elif isinstance(other, type):
            return CombinedSum([self, CombinedComponent(1.0, other)])
        raise TypeError("Invalid addition for CombinedComponent")

    def __rmul__(self, weight: float):
        return CombinedComponent(weight * self.weight, self.evaluator_cls)


class CombinedSum:
    def __init__(self, components):
        self.components = components  # list[CombinedComponent]

    def __add__(self, other):
        if isinstance(other, CombinedComponent):
            return CombinedSum(self.components + [other])
        elif isinstance(other, CombinedSum):
            return CombinedSum(self.components + other.components)
        elif isinstance(other, type):
            return CombinedSum(self.components + [CombinedComponent(1.0, other)])
        raise TypeError("Invalid addition for CombinedSum")

    def __rmul__(self, weight: float):
        return CombinedSum([CombinedComponent(weight * c.weight, c.evaluator_cls) for c in self.components])

    def __call__(self, *args, **kwargs):
        return CombinedEval(self.components, *args, **kwargs)


class CombinedEval(BaseEval):
    """
    Runs multiple evaluators and returns a weighted score combination.
    Each component evaluator is instantiated with the same args.
    """

    def __init__(self, components, *args, **kwargs):
        # components: list[CombinedComponent]
        if not components:
            raise ValueError("CombinedEval needs at least one component.")
        self.components = components
        # Initialize base BaseEval with first evaluator’s config (just for shared behavior)
        first_cls = components[0].evaluator_cls
        super().__init__(*args, **kwargs)
        self._args = args
        self._kwargs = kwargs

    def evaluate(self, program: Optional[str] = None) -> Dict[str, Any]:
        results = []
        total_weight = sum(c.weight for c in self.components)
        combined_score = 0.0

        per_evaluator = {}

        for comp in self.components:
            ev = comp.evaluator_cls(*self._args, **self._kwargs)
            res = ev.evaluate(program)
            score = res.get("score", -1e9)
            combined_score += comp.weight * score
            per_evaluator[comp.evaluator_cls.__name__] = res
            results.append(res)

        combined_score /= (total_weight if total_weight else 1.0)

        return {
            "combined_score": float(combined_score),
            "weights": {c.evaluator_cls.__name__: c.weight for c in self.components},
            "subresults": per_evaluator,
        }
    
