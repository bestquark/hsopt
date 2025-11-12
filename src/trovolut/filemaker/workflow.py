import inspect
import textwrap
from pathlib import Path




class Comment:
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return self.name


EVO_START = Comment("EVOLVE-BLOCK-START")
EVO_END = Comment("EVOLVE-BLOCK-END")


import inspect
import textwrap
import ast
from pathlib import Path

# ---------- helpers ----------

def _find_prefixed_attr_functions(func, root_name="trv"):
    """
    Find attribute chains in `func` that start with `root_name`, e.g.
    trv.qm.dynamics.product_formula, and resolve them to callables.
    Returns dict[dotted_name -> function_obj].
    """
    src = inspect.getsource(func)
    src = textwrap.dedent(src)
    tree = ast.parse(src)

    chains = []

    class AttrFinder(ast.NodeVisitor):
        def visit_Attribute(self, node):
            names = []
            cur = node
            while isinstance(cur, ast.Attribute):
                names.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                names.append(cur.id)
                names_rev = list(reversed(names))
                if names_rev[0] == root_name:  # starts with trv
                    full = ".".join(names_rev)
                    chains.append(full)
            self.generic_visit(node)

    AttrFinder().visit(tree)

    out = {}
    globs = func.__globals__
    for dotted in set(chains):
        try:
            obj = eval(dotted, globs)
        except Exception:
            continue
        if inspect.isfunction(obj):
            out[dotted] = obj
    return out


def get_source_with_deps(func, visited=None, root_name="trv"):
    """
    Return source for `func` plus all helper functions it calls:
    - same-module global functions
    - any functions referenced as attributes under `root_name` (e.g. trv.*)
    All deps are emitted before users.
    """
    if visited is None:
        visited = set()

    out = []

    def walk(f):
        if f in visited:
            return
        visited.add(f)

        module = inspect.getmodule(f)
        code = f.__code__
        globs = f.__globals__

        # 1) attribute-based deps from trv.*
        attr_funcs = _find_prefixed_attr_functions(f, root_name=root_name)
        for _, dep_f in attr_funcs.items():
            walk(dep_f)

        # 2) same-module global function deps
        for name in code.co_names:
            obj = globs.get(name, None)
            if inspect.isfunction(obj) and inspect.getmodule(obj) is module:
                walk(obj)

        # 3) add source for f itself, rewriting trv.* calls to local names
        src = inspect.getsource(f)
        src = textwrap.dedent(src).rstrip()
        for dotted, dep_f in attr_funcs.items():
            src = src.replace(dotted, dep_f.__name__)
        out.append(src)

    walk(func)
    return "\n\n".join(out)


# ---------- Workflow ----------

class Workflow:
    def __init__(self, parts, out_path="initial_program.py", imports=None, main_func=None):
        """
        parts: list of callables, comments, or code strings
        main_func: callable (optional)
        """
        self.parts = parts
        self.out_path = Path(out_path)
        self.imports = imports or [
            "import numpy as np",
            "import scipy",
            "import trovolut as trv",
        ]
        self.main_func = main_func

    def build(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        with self.out_path.open("w", encoding="utf-8") as f:
            # header / imports
            for line in self.imports:
                f.write(line + "\n")
            f.write("\n")

            visited = set()

            # code sections
            for part in self.parts:
                if isinstance(part, Comment):
                    f.write(f"# {part.name}\n\n")
                elif callable(part):
                    src = get_source_with_deps(part, visited=visited, root_name="trv")
                    f.write(src + "\n\n")
                else:
                    f.write(str(part).rstrip() + "\n\n")

            # optional main block
            if self.main_func is not None:
                f.write('if __name__ == "__main__":\n')
                src = inspect.getsource(self.main_func)
                src = textwrap.dedent(src)
                lines = src.splitlines()
                body = "\n".join(lines[1:])
                body = textwrap.indent(body, "    ")
                f.write(body.rstrip() + "\n\n")

        return self.out_path
    
# class Workflow:
#     def __init__(self, parts, out_path="initial_program.py", imports=None, main_func=None):
#         """
#         parts: list of callables, comments, or code strings
#         main_func: callable (optional)
#         """
#         self.parts = parts
#         self.out_path = Path(out_path)
#         self.imports = imports or [
#             "import numpy as np",
#             "import scipy",
#             "import trovolut as trv",
#         ]
#         self.main_func = main_func

#     def build(self):
#         # ensure folder exists
#         self.out_path.parent.mkdir(parents=True, exist_ok=True)

#         with self.out_path.open("w") as f:
#             # header / imports
#             for line in self.imports:
#                 f.write(line + "\n")
#             f.write("\n")

#             visited = set()

#             # code sections
#             for part in self.parts:
#                 if isinstance(part, Comment):
#                     f.write(f"# {part.name}\n\n")
#                 elif callable(part):
#                     src = get_source_with_deps(part, visited=visited)
#                     f.write(src + "\n\n")
#                 else:
#                     f.write(str(part).rstrip() + "\n\n")

#             # optional main block
#             if self.main_func is not None:
#                 f.write('if __name__ == "__main__":\n')
#                 src = inspect.getsource(self.main_func)
#                 src = textwrap.dedent(src)
#                 lines = src.splitlines()
#                 body = "\n".join(lines[1:])
#                 body = textwrap.indent(body, "    ")
#                 f.write(body.rstrip() + "\n\n")

#         return self.out_path