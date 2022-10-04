from typing import Any, Dict, List, Optional

from sphinx.application import Sphinx
from sphinx.ext.autodoc import Options


def process_docstring(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Optional[Options],
    lines: List[str],
) -> None:
    print("--------------------------------------------------")
    print("--------------------------------------------------")
    print(app)
    print(what)
    print(name)
    print(obj)
    print(options)
    for line in lines:
        print("*", line)
    print("--------------------------------------------------")
    print("--------------------------------------------------")


def setup(app: Sphinx) -> Dict[str, Any]:
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print("Wat???")
    app.connect("autodoc-process-docstring", process_docstring)
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    return {"parallel_read_safe": True}
