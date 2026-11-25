"""A small, portable expression language for symbolic synthesis tolerances."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np

from ..evaluation import compiled_unitary as judge
from ..campaign.protocol import bind_policy, suite_at, within

FEATURES = ["c", "w", "a", "d", "xy", "lc"]


def expression_source(expression: str) -> str:
    """Parse arithmetic, never arbitrary SymPy/Python/Julia source."""
    if not isinstance(expression, str) or len(expression) > 4096:
        raise ValueError("symbolic expression is too long")
    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 200:
        raise ValueError("symbolic expression is too complex")

    def convert(node):
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float) or not math.isfinite(node.value):
                raise ValueError("constants must be finite numbers")
        elif isinstance(node, ast.Name):
            if node.id not in FEATURES:
                raise ValueError(f"unknown symbolic feature: {node.id}")
        elif isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            node.left, node.right = convert(node.left), convert(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            node.operand = convert(node.operand)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if (
                node.func.id not in {"sqrt", "abs", "negate"}
                or len(node.args) != 1
                or node.keywords
            ):
                raise ValueError("unsupported symbolic function")
            value = convert(node.args[0])
            if node.func.id == "negate":
                return ast.UnaryOp(op=ast.USub(), operand=value)
            if node.func.id == "sqrt":
                return ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="math", ctx=ast.Load()), attr="sqrt", ctx=ast.Load()
                    ),
                    args=[value],
                    keywords=[],
                )
            node.args = [value]
        else:
            raise ValueError("only bounded arithmetic expressions are supported")
        return node

    return ast.unparse(ast.fix_missing_locations(convert(tree.body)))


def tolerance_policy(base_source: str, expression: str) -> str:
    """Preserve the proposed splitting/r; replace only its tolerance rule."""
    delta = expression_source(expression)
    tree = ast.parse(base_source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ProductFormula")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    names = {"symbolic_c_ref", "symbolic_w_ref", "symbolic_target_time"}
    init.body = [
        n
        for n in init.body
        if not (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Attribute) and t.attr in names for t in n.targets)
        )
    ]
    init.body[:0] = ast.parse("""
self.symbolic_c_ref = max(max([abs(t["coefficient"]) for t in problem["terms"]]), 1e-12)
self.symbolic_w_ref = max(1, max([sum([ch != "I" for ch in t["label"]]) for t in problem["terms"]]))
self.symbolic_target_time = problem["target_time"]
""").body
    method = ast.parse(f"""
def synthesis_epsilon(self, term, angle):
    label = term["label"]
    c = max(abs(term["coefficient"]) / self.symbolic_c_ref, 1e-12)
    w = sum([ch != "I" for ch in label]) / self.symbolic_w_ref
    a = abs(angle) * self.repetitions() / (self.symbolic_c_ref * self.symbolic_target_time)
    d = 1.0 if not any([ch in "XY" for ch in label]) else 0.0
    xy = sum([ch in "XY" for ch in label]) / self.symbolic_w_ref
    lc = math.log(c)
    delta = {delta}
    return min(1.0 - 1e-12, max(1e-15, self.epsilon * math.exp(max(-8.0, min(16.0, delta)))))
""").body[0]
    cls.body = [
        method if isinstance(n, ast.FunctionDef) and n.name == "synthesis_epsilon" else n
        for n in cls.body
    ]
    if not any(isinstance(n, ast.Import) and n.names[0].name == "math" for n in tree.body):
        tree.body.insert(0, ast.Import(names=[ast.alias(name="math")]))
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def feature_scales(problem):
    c_ref = max(max(abs(t["coefficient"]) for t in problem["terms"]), 1e-12)
    w_ref = max(1, max(sum(ch != "I" for ch in t["label"]) for t in problem["terms"]))
    return c_ref, w_ref


def feature_vector(
    problem: dict, term: dict, angle: float, repetitions: int, *, scales=None
) -> list[float]:
    c_ref, w_ref = feature_scales(problem) if scales is None else scales
    label = term["label"]
    c = max(abs(term["coefficient"]) / c_ref, 1e-12)
    return [
        c,
        sum(ch != "I" for ch in label) / w_ref,
        abs(angle) * repetitions / (c_ref * problem["target_time"]),
        float(not any(ch in "XY" for ch in label)),
        sum(ch in "XY" for ch in label) / w_ref,
        math.log(c),
    ]


def training_features(run: Path, base: Path, output: Path) -> list[list[float]]:
    """Only the sealed training suite is visible, including every reaction species."""
    suite = suite_at(run)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in suite["cases"]:
        bound = output / f"{case['id']}.py"
        bind_policy(base, case["parameters"], bound)
        module = judge.load_policy(bound)
        source = within(run, case["source"])
        sources = (
            [source]
            if suite["kind"] == "absolute"
            else [
                source.parent / item["problem"] for item in json.loads(source.read_text())["terms"]
            ]
        )
        for path in sources:
            problem = judge.load_problem(path)
            pf, rotations, _ = judge._product_formula_schedule(problem, module)
            scales = feature_scales(problem)
            repetitions = pf.repetitions()
            rows.extend(
                feature_vector(problem, term, angle, repetitions, scales=scales)
                for term, angle in rotations
                if angle
            )
    if not rows or not np.isfinite(np.asarray(rows)).all():
        raise ValueError("the proposed schedule has no finite symbolic-search features")
    return rows
