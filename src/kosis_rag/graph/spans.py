"""Per-node timing of the search graph (T2-41 S4: graph.graph_nodes' inner `timed`, and spans_of for SearchOutput).

Every node writes `trace[name] = {"seconds": ..., **extra}`; spans_of reads them back in node order.
"""

from __future__ import annotations

import time
from typing import Any


def timed(name: str, trace: dict[str, dict[str, Any]], started: float, **extra: Any) -> dict[str, dict[str, Any]]:
    """The trace with this node's entry added.

    In: name = node name; trace = the state's trace so far; started = perf_counter at node start; extra = fields to
        record. Out: {**trace, name: {"seconds": ..., **extra}}. Side effects: none. Node: every node.
    """
    return {**trace, name: {"seconds": time.perf_counter() - started, **extra}}


def spans_of(trace: dict[str, dict[str, Any]], skip: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """The trace as spans in node order: [{"node": name, **entry}] (SearchOutput.spans).

    In: trace = a final state's trace (insertion order = the order the nodes finished); skip = node names left out.
    Out: one dict per node. Side effects: none. Node: none (SearchModule.search reads it).
    """
    return [{"node": name, **entry} for name, entry in trace.items() if name not in skip]
