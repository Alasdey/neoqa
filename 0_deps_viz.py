#!/usr/bin/env python3
"""
Build an import graph for a Python project.

Outputs a Graphviz DOT file (and optionally a PNG/SVG if Graphviz is installed).
Usage:
  python import_graph.py /path/to/project -o graph.dot
  python import_graph.py . -o graph.dot --format png   # also writes graph.png
  python import_graph.py src -o deps.dot --exclude tests,venv,.venv,build,dist

Requires: networkx, pydot (or pygraphviz) to write DOT.
    pip install networkx pydot

For PNG/SVG rendering you also need Graphviz installed (the `dot` command).
"""
import argparse
import ast
import sys
from pathlib import Path
from typing import Iterable, Set, Tuple, Dict, Optional

# For graph building / DOT writing
import networkx as nx
from networkx.drawing.nx_pydot import write_dot  # writes Graphviz DOT


DEFAULT_EXCLUDES = {
    "__pycache__", ".git", ".hg", ".svn",
    "build", "dist", ".mypy_cache", ".pytest_cache",
    "venv", ".venv", ".tox",
    ".idea", ".vscode",
    "site-packages", "egg-info",
    # tests often import across the tree; include them if you want
    "tests",
}


def is_python_file(p: Path) -> bool:
    return p.suffix == ".py" and p.name != "__main__.py"


def iter_python_files(root: Path, excludes: Set[str]) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        # skip excluded dirs anywhere in path
        parts = set(p.parts)
        if parts & excludes:
            continue
        yield p


def module_name_from_path(project_root: Path, file_path: Path) -> str:
    """
    Convert a file path to a dotted module name relative to project root.
    e.g., src/pkg/foo/bar.py -> pkg.foo.bar
          src/pkg/foo/__init__.py -> pkg.foo
    """
    rel = file_path.relative_to(project_root)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def resolve_from_import(current_module: str, imported_module: Optional[str], level: int) -> str:
    """
    Resolve a 'from ... import ...' to an absolute dotted module, given
    the current module and relative level.

    current_module: e.g., 'pkg.sub.module' or 'pkg.sub' (for __init__.py)
    imported_module: e.g., 'utils' in 'from . import utils', or 'x.y' in 'from ..x.y import z'
    level: number of leading dots in the import (0 for absolute)
    """
    if level == 0:
        return imported_module or ""
    base_parts = current_module.split(".")
    # If current_module refers to a module (file) not a package, trim last part to get its package
    # Heuristic: if the current_module had no __init__.py name, we treat it as a module; drop last segment
    # This works with our module_name_from_path that omits '__init__'
    if current_module:
        base_parts = base_parts[:-1] if base_parts else base_parts
    # Walk up 'level - 1' additional parents after moving to the package
    up = max(level - 1, 0)
    if up:
        base_parts = base_parts[:-up] if up <= len(base_parts) else []
    if imported_module:
        return ".".join([p for p in base_parts if p] + imported_module.split("."))
    return ".".join([p for p in base_parts if p])


def collect_imports_for_file(py_file: Path, project_root: Path) -> Set[str]:
    """
    Return a set of (absolute) module names imported by this file.
    Only returns the top-level module path delivered by the import statement
    (e.g., importing 'pkg.sub.mod' yields 'pkg.sub.mod').
    """
    text = py_file.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(py_file))
    except SyntaxError:
        return set()

    current_module = module_name_from_path(project_root, py_file)
    found: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # import a.b as x, import c
            for alias in node.names:
                if alias.name:
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # from .a.b import c, from .. import d
            mod = node.module  # may be None for 'from . import x'
            level = getattr(node, "level", 0) or 0
            abs_mod = resolve_from_import(current_module, mod, level)
            if abs_mod:
                found.add(abs_mod)
            # also consider specific names as submodules (from pkg import submodule)
            for alias in node.names:
                if alias.name == "*":
                    continue
                name_as_module = abs_mod + "." + alias.name if abs_mod else alias.name
                # Add both the container module and the potential submodule path
                found.add(name_as_module)

    return found


def normalize_to_project(mod: str, project_modules: Set[str]) -> Optional[str]:
    """
    If `mod` refers to a module/package under the project, return the
    *deepest* matching project module prefix; else None.
    """
    # Try longest-prefix match so 'mypkg.sub.mod' matches 'mypkg.sub.mod' first, then 'mypkg.sub', then 'mypkg'
    candidates = sorted(project_modules, key=len, reverse=True)
    for cand in candidates:
        if mod == cand or mod.startswith(cand + "."):
            return mod  # keep full path inside the project
    return None


def build_graph(project_root: Path, excludes: Set[str]) -> Tuple[nx.DiGraph, Set[str]]:
    """
    Build a DiGraph of internal module -> internal module edges.
    Returns graph and the set of discovered project module names.
    """
    project_root = project_root.resolve()
    files = list(iter_python_files(project_root, excludes))
    module_of: Dict[Path, str] = {f: module_name_from_path(project_root, f) for f in files}

    # Project modules are every module we found (files and packages)
    project_modules: Set[str] = set(module_of.values())

    G = nx.DiGraph()
    G.add_nodes_from(project_modules)

    for f, modname in module_of.items():
        imports = collect_imports_for_file(f, project_root)
        for target in imports:
            internal = normalize_to_project(target, project_modules)
            if internal and internal != modname:
                G.add_edge(modname, internal)
    return G, project_modules


def render_with_graphviz(dot_path: Path, fmt: str) -> Optional[Path]:
    """
    If Graphviz 'dot' is installed, render to the requested format (png/svg/pdf).
    """
    import shutil
    if shutil.which("dot") is None:
        return None
    out = dot_path.with_suffix("." + fmt.lower())
    import subprocess
    subprocess.run(
        ["dot", f"-T{fmt}", str(dot_path), "-o", str(out)],
        check=False
    )
    return out if out.exists() else None


def main(argv: Iterable[str]) -> int:
    ap = argparse.ArgumentParser(description="Generate an import graph for a Python project.")
    ap.add_argument("project_root", type=Path, help="Path to the project root (package dir or repo root).")
    ap.add_argument("-o", "--output", type=Path, default=Path("import_graph.dot"),
                    help="DOT output file (default: import_graph.dot)")
    ap.add_argument("--format", choices=["png", "svg", "pdf"], default=None,
                    help="Also render to this format if Graphviz is installed.")
    ap.add_argument("--exclude", type=str, default="",
                    help="Comma-separated names of directories to exclude (in addition to sensible defaults).")
    args = ap.parse_args(list(argv))

    excludes = set(DEFAULT_EXCLUDES)
    if args.exclude:
        excludes |= {e.strip() for e in args.exclude.split(",") if e.strip()}

    root = args.project_root.resolve()
    if not root.exists():
        print(f"Error: {root} does not exist.", file=sys.stderr)
        return 2

    G, project_modules = build_graph(root, excludes)

    # Basic styling hints (Graphviz will ignore unknown attrs in DOT)
    for n in G.nodes:
        G.nodes[n]["shape"] = "box"
        G.nodes[n]["style"] = "rounded"

    write_dot(G, str(args.output))

    print(f"Wrote DOT file: {args.output}")
    if args.format:
        rendered = render_with_graphviz(args.output, args.format)
        if rendered:
            print(f"Wrote {args.format.upper()} file: {rendered}")
        else:
            print("Graphviz 'dot' not found; install Graphviz to render images.", file=sys.stderr)
            print("Get started with DOT here: https://graphviz.org/doc/info/lang.html", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
