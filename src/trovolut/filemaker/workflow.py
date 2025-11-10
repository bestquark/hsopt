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
        # ensure folder exists
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

        with self.out_path.open("w") as f:
            # header / imports
            for line in self.imports:
                f.write(line + "\n")
            f.write("\n")

            # code sections
            for part in self.parts:
                if isinstance(part, Comment):
                    f.write(f"# {part.name}\n\n")
                elif callable(part):
                    src = inspect.getsource(part)
                    src = textwrap.dedent(src)
                    f.write(src.rstrip() + "\n\n")
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
