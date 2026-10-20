"""Julia-only child process: ask the host to score each symbolic expression.

No scientific/compiler imports here: Julia and CUDA-Q may load incompatible LLVMs.
The parent owns inputs, budgets, GPU requests, results and acceptance.
"""

import json
import os
import sys
import time

# Julia initializes its own standard streams. Keep dedicated descriptors so its
# stdin watcher/redirection cannot consume or close the host's fitness channel.
RPC_INPUT = os.fdopen(os.dup(sys.stdin.fileno()), "r")
RPC_OUTPUT = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)


def search(config):
    import numpy as np
    from pysr import PySRRegressor, jl

    # Julia/libuv marks inherited pipe descriptions nonblocking, including dup'd
    # descriptors. Python TextIO otherwise mistakes EAGAIN for an EOF before the
    # host finishes the physical evaluation. Julia's stdin watcher is disabled.
    os.set_blocking(RPC_INPUT.fileno(), True)
    os.set_blocking(RPC_OUTPUT.fileno(), True)
    print(
        "HSOPT_READY "
        + json.dumps(
            {
                "pysr_version": __import__("pysr").__version__,
                "julia_version": str(jl.seval("VERSION")),
                "python_version": sys.version.split()[0],
            }
        ),
        file=RPC_OUTPUT,
        flush=True,
    )

    def fitness(prediction, expression):
        print(
            "HSOPT_FITNESS " + json.dumps({"expression": str(expression)}),
            file=RPC_OUTPUT,
            flush=True,
        )
        line = RPC_INPUT.readline()
        if not line:
            raise RuntimeError("symbolic fitness host disconnected")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return float(reply["loss"])

    jl.hsopt_fitness = fitness
    jl.seval("""
        using PythonCall
        function hsopt_symbolic_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
            prediction, completed = eval_tree_array(tree, dataset.X, options)
            completed && all(isfinite, prediction) || return L(1e12)
            expression = string_tree(tree, options; variable_names=dataset.variable_names, pretty=false)
            PythonCall.GIL.@lock begin
                return pyconvert(L, hsopt_fitness(prediction, expression))
            end
        end
    """)
    operators = config.get("operators", ["+", "-", "*", "/", "abs", "sqrt"])
    binary = [op for op in operators if op in {"+", "-", "*", "/"}]
    unary = [op for op in operators if op in {"abs", "sqrt"}]
    model = PySRRegressor(
        niterations=10000,
        populations=3,
        population_size=16,
        ncycles_per_iteration=3,
        max_evals=config["max_evals"],
        timeout_in_seconds=config["seconds"],
        binary_operators=binary,
        unary_operators=unary,
        maxsize=config.get("maxsize", 12),
        maxdepth=6,
        loss_function="hsopt_symbolic_loss",
        loss_scale="linear",
        model_selection="accuracy",
        precision=64,
        parallelism="serial",
        deterministic=True,
        random_state=config["seed"],
        should_optimize_constants=False,
        guesses=config.get("guesses", ["0.0"]),
        batching=False,
        parsimony=0.0,
        progress=False,
        verbosity=0,
        print_precision=17,
        input_stream="devnull",
        output_directory=config["output_directory"],
        run_id="equations",
    )
    X = np.asarray(config["features"], dtype=float)
    start = time.monotonic()
    model.fit(X, np.ones(len(X)), variable_names=config["feature_names"])
    return {
        "search_seconds": time.monotonic() - start,
        "pysr_version": __import__("pysr").__version__,
        "julia_version": str(jl.seval("VERSION")),
        "loss": float(model.get_best()["loss"]),
    }


if __name__ == "__main__":
    try:
        result = search(json.loads(RPC_INPUT.readline()))
        print("HSOPT_DONE " + json.dumps(result), file=RPC_OUTPUT, flush=True)
    except Exception as exc:
        print(
            "HSOPT_ERROR " + json.dumps({"error": f"{type(exc).__name__}: {exc}"}),
            file=RPC_OUTPUT,
            flush=True,
        )
        sys.exit(1)
