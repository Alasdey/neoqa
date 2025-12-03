#!/usr/bin/env python3
import os
import re
import sys

"""
Usage:
  python filter_dot_project.py <input.dot> <output.dot> <project_root>

Filters a Graphviz DOT file: keeps only nodes whose tooltip
path is under project_root, and excludes nodes whose label or tooltip
match certain unwanted patterns (e.g. '<frozen importlib').
Edges to/from excluded nodes are removed.
"""

if len(sys.argv) != 4:
    print("Usage: filter_dot_project.py <input.dot> <output.dot> <project_root>")
    sys.exit(1)

in_dot, out_dot, project_root = sys.argv[1], sys.argv[2], os.path.abspath(sys.argv[3])

# regex to match node definitions with tooltip
node_re = re.compile(r'^\s*(?P<id>\d+)\s+\[.*tooltip="(?P<tooltip>[^"]+)"(?:.*label="(?P<label>[^"]+)")?')
edge_re = re.compile(r'^\s*(?P<src>\d+)\s*->\s*(?P<dst>\d+)')

# Patterns for exclusion: e.g. frozen importlib, builtins, etc.
EXCLUDE_PATTERNS = [
    re.compile(r'<frozen importlib'),   # exclude nodes from frozen importlib
    re.compile(r'\~'),   # exclude nodes from method
    # add more patterns here if needed
]

keep_nodes = {}
raw_edges = []
header = []
footer = []
in_body = False

with open(in_dot, 'r') as f:
    lines = f.readlines()

# separate header (before first node/edge), body (nodes & edges), footer (if any)
for line in lines:
    if not in_body:
        if node_re.match(line) or edge_re.match(line):
            in_body = True
            # but also this line belongs to body; continue processing it below
        else:
            header.append(line)
            continue
    if in_body:
        # collect everything; we'll split later
        footer.append(line)

# Actually split body into node and edge lines
node_lines = []
edge_lines = []
other_lines = []

for line in footer:
    if node_re.match(line):
        node_lines.append(line)
    elif edge_re.match(line):
        edge_lines.append(line)
    else:
        other_lines.append(line)

# Process nodes: decide which to keep
for nl in node_lines:
    m = node_re.match(nl)
    if not m:
        continue
    node_id = m.group('id')
    tooltip = m.group('tooltip')
    label = m.group('label') or ""
    # normalize tooltip path
    abs_path = tooltip
    if not os.path.isabs(abs_path):
        abs_path = os.path.abspath(abs_path)
    # check under project root
    under_project = abs_path.startswith(project_root)
    # check exclusion patterns
    excluded = any(p.search(label) or p.search(tooltip) for p in EXCLUDE_PATTERNS)
    if under_project and (not excluded):
        keep_nodes[node_id] = nl

# Filter edges: only those where both src and dst are kept
filtered_edges = []
for el in edge_lines:
    m = edge_re.match(el)
    if not m:
        continue
    src, dst = m.group('src'), m.group('dst')
    if src in keep_nodes and dst in keep_nodes:
        filtered_edges.append(el)

# Write filtered DOT
with open(out_dot, 'w') as f:
    for l in header:
        f.write(l)
    # write kept nodes
    for node_id, nl in keep_nodes.items():
        f.write(nl)
    # write edges
    for e in filtered_edges:
        f.write(e)
    # write other (non-node/edge) lines if any
    for l in other_lines:
        f.write(l)

print(f"Filtered DOT written to {out_dot}")
print(f"Nodes kept: {len(keep_nodes)}, edges kept: {len(filtered_edges)}")
