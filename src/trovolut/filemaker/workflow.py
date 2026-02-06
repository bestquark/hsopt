import inspect
import textwrap
import ast
import re
from pathlib import Path


class Comment:
    def __init__(self, name: str):
        self.name = name
    def __repr__(self):
        return self.name


EVO_START = Comment("EVOLVE-BLOCK-START")
EVO_END = Comment("EVOLVE-BLOCK-END")

# Registry of all Comment markers so we can recognise them in source code.
# Maps every way a marker can appear in source → the comment text to emit.
_MARKER_REGISTRY: dict[str, str] = {}


def _register_marker(obj: Comment, *aliases: str) -> None:
    """Register source-level patterns that should be replaced with ``# obj.name``."""
    for alias in aliases:
        _MARKER_REGISTRY[alias] = f"# {obj.name}"


_register_marker(
    EVO_START,
    "EVO_START",
    "trv.filemaker.EVO_START",
    "trovolut.filemaker.EVO_START",
    "trovolut.filemaker.workflow.EVO_START",
)
_register_marker(
    EVO_END,
    "EVO_END",
    "trv.filemaker.EVO_END",
    "trovolut.filemaker.EVO_END",
    "trovolut.filemaker.workflow.EVO_END",
)

# Pre-compile a regex that matches any registered marker as a standalone
# expression on its own line (preserving leading whitespace).
_MARKER_RE = re.compile(
    r"^(?P<indent>\s*)(?P<marker>"
    + "|".join(re.escape(k) for k in sorted(_MARKER_REGISTRY, key=len, reverse=True))
    + r")\s*$",
    re.MULTILINE,
)


def _replace_inline_markers(source: str) -> str:
    """Replace bare marker expressions in source with comment lines.

    A line like ``    trv.filemaker.EVO_START`` becomes
    ``    # EVOLVE-BLOCK-START`` (preserving indentation).
    """
    def _sub(m: re.Match) -> str:
        indent = m.group("indent")
        marker = m.group("marker")
        return f"{indent}{_MARKER_REGISTRY[marker]}"
    return _MARKER_RE.sub(_sub, source)


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


def _extract_imports_from_function(func):
    """
    Extract import statements from a function's source code.
    
    Returns:
        Set of import statement strings
    """
    try:
        src = inspect.getsource(func)
        src = textwrap.dedent(src)
        tree = ast.parse(src)
        
        imports = set()
        
        class ImportFinder(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    if alias.asname:
                        imports.add(f"import {alias.name} as {alias.asname}")
                    else:
                        imports.add(f"import {alias.name}")
            
            def visit_ImportFrom(self, node):
                module = node.module or ""
                for alias in node.names:
                    if alias.name == '*':
                        imports.add(f"from {module} import *")
                    elif alias.asname:
                        imports.add(f"from {module} import {alias.name} as {alias.asname}")
                    else:
                        imports.add(f"from {module} import {alias.name}")
        
        ImportFinder().visit(tree)
        return imports
    except Exception:
        return set()


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
        # 4) convert inline EVO_START / EVO_END expressions to comments
        src = _replace_inline_markers(src)
        out.append(src)

    walk(func)
    return "\n\n".join(out)


# ---------- Workflow ----------

class Workflow:
    def __init__(self, parts, out_path="initial_program.py", imports=None, main_func=None, auto_imports=True):
        """
        Build standalone Python files from functions and objects.
        
        Args:
            parts: list of callables, comments, or code strings
            out_path: output file path
            imports: list of import statements (if None, auto-detected if auto_imports=True)
            main_func: callable (optional)
            auto_imports: if True and imports=None, automatically extract imports from functions
        
        Example:
            workflow = Workflow(
                [EVO_START, run_dynamics, EVO_END],
                out_path="program.py",
            )
            workflow.build()  # Auto-detects imports from run_dynamics function
        """
        self.parts = parts
        self.out_path = Path(out_path)
        self.main_func = main_func
        
        # Auto-detect imports if not provided
        if imports is None and auto_imports:
            self.imports = self._auto_detect_imports()
        else:
            self.imports = imports or [
                "import numpy as np",
                "import scipy",
                "import trovolut as trv",
            ]
    
    def _auto_detect_imports(self):
        """Auto-detect imports from all callable parts and main_func."""
        detected = set()
        
        # Always include core imports
        detected.add("import numpy as np")
        detected.add("import scipy")
        detected.add("import trovolut as trv")
        
        # Extract from all callable parts
        for part in self.parts:
            if callable(part):
                detected.update(_extract_imports_from_function(part))
        
        # Extract from main function
        if self.main_func is not None:
            detected.update(_extract_imports_from_function(self.main_func))
        
        # Sort for consistency (stdlib first, then third-party, then trovolut)
        imports_list = sorted(detected, key=lambda x: (
            0 if x.startswith('import sys') or x.startswith('import os') or x.startswith('import time') else
            1 if x.startswith('import numpy') or x.startswith('import scipy') else
            2 if x.startswith('import trovolut') or x.startswith('from trovolut') else
            3
        ))
        
        return imports_list

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
