"""
Break cycles in an agr_gender grouped graph by "splitting out" culprit heads.

This script keeps the *clustering criterion* (dominant signature) unchanged, but
refines grouping by detaching a small number of heads that are most responsible
for creating bidirectional group connections inside SCCs (cycles).

Inputs:
  - --graph-json: head-level graph JSON (nodes/edge_list), e.g. filtered/graph_thr0_001.json
  - --dominant-soft: *_soft.json from extract_dominant_patterns_agr_gender.py (per_head.signature)
  - --edge-weights-json: (optional) joint_p_qkv.json for delta_logit_diff weights

Outputs:
  - --output-json: refined grouped graph JSON (same schema as group_graph_agr_gender.py output)
  - --output-png: (optional) rendered PNG
  - --report-json: (optional) detailed iteration report (SCCs, split heads, scores)

Typical usage:
  python tests/experiments/break_cycles_by_splitting_heads.py \\
    --graph-json results/agr_gender_gpt2_new/filtered/graph_thr0_001.json \\
    --dominant-soft results/agr_gender_gpt2_new/dominant/head_patterns_thr0_001_w1_allpos_thr75_soft.json \\
    --edge-weights-json results/agr_gender_gpt2_new/joint_p_qkv.json \\
    --min-edge-abs-score 0.001 \\
    --split-per-iter 3 \\
    --max-iters 10 \\
    --output-json results/agr_gender_gpt2_new/group_graph_thr0_001_thr75_dagish.json \\
    --output-png results/agr_gender_gpt2_new/group_graph_thr0_001_thr75_dagish.png \\
    --report-json results/agr_gender_gpt2_new/group_graph_thr0_001_thr75_dagish_report.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pygraphviz as pgv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Break cycles in grouped graph by splitting culprit heads into singleton groups."
    )
    p.add_argument("--graph-json", type=Path, required=True)
    p.add_argument("--dominant-soft", type=Path, required=True)
    p.add_argument("--edge-weights-json", type=Path, default=None)
    p.add_argument("--min-edge-abs-score", type=float, default=0.0, help="Filter head-edge weights by |delta|")
    p.add_argument(
        "--layer-buckets",
        type=str,
        default="",
        help="Optional: further split groups by layer buckets, e.g. '0-4,5-8,9-11'.",
    )

    p.add_argument("--max-iters", type=int, default=10)
    p.add_argument("--split-per-iter", type=int, default=3)
    p.add_argument(
        "--score-mode",
        choices=["frequency", "abs_weighted"],
        default="frequency",
        help="How to score culprit heads: frequency or abs(delta) weighted.",
    )
    p.add_argument(
        "--exclude-groups",
        type=str,
        default="input,logits",
        help="Comma-separated group ids to exclude from SCC/cycle handling.",
    )
    p.add_argument(
        "--delete-weakest-edges",
        action="store_true",
        help="After splitting heads, greedily delete weakest *group edges* inside SCCs until acyclic.",
    )
    p.add_argument(
        "--edge-delete-score",
        choices=["signed_mean", "abs_max", "count"],
        default="signed_mean",
        help="Which group-edge field to use when choosing the weakest edge (by absolute value).",
    )
    p.add_argument(
        "--max-edge-deletions",
        type=int,
        default=1000,
        help="Safety cap on number of group edges deleted when --delete-weakest-edges is enabled.",
    )

    p.add_argument("--max-heads-in-label", type=int, default=12)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-png", type=Path, default=None)
    p.add_argument("--layout", type=str, default="dot")
    p.add_argument("--report-json", type=Path, default=None)
    return p.parse_args()


def load_graph(path: Path) -> Tuple[List[str], List[Tuple[str, str]]]:
    data = json.loads(path.read_text())
    nodes = data.get("nodes")
    if isinstance(nodes, dict):
        nodes = [n for n, info in nodes.items() if (info or {}).get("in_graph", False)]
    if not isinstance(nodes, list):
        raise ValueError("graph-json nodes 格式不正确")
    edges = data.get("edge_list") or data.get("edges") or []
    if not isinstance(edges, list):
        raise ValueError("graph-json edge_list 格式不正确")
    return nodes, [(a, b) for a, b in edges]


def head_key(node_name: str) -> str:
    # aL.hH -> "L.H"
    if isinstance(node_name, str) and node_name.startswith("a") and ".h" in node_name:
        left, right = node_name.split(".h", 1)
        return f"{int(left[1:])}.{int(right)}"
    return node_name


def head_layer(node_name: str) -> Optional[int]:
    if isinstance(node_name, str) and node_name.startswith("a") and ".h" in node_name:
        left = node_name.split(".h", 1)[0]
        try:
            return int(left[1:])
        except Exception:
            return None
    return None


def parse_layer_buckets(spec: str) -> List[Tuple[int, int, str]]:
    spec = (spec or "").strip()
    if not spec:
        return []
    buckets: List[Tuple[int, int, str]] = []
    for chunk in spec.split(","):
        c = chunk.strip()
        if not c:
            continue
        if "-" not in c:
            raise ValueError(f"Invalid layer bucket: {c} (expected 'lo-hi')")
        lo_s, hi_s = [x.strip() for x in c.split("-", 1)]
        lo, hi = int(lo_s), int(hi_s)
        if lo > hi:
            lo, hi = hi, lo
        buckets.append((lo, hi, f"{lo}-{hi}"))
    return buckets


def bucket_for_layer(layer: Optional[int], buckets: List[Tuple[int, int, str]]) -> str:
    if layer is None or not buckets:
        return ""
    for lo, hi, label in buckets:
        if lo <= layer <= hi:
            return label
    return "other"


def load_signatures(dominant_soft: Path) -> Dict[str, str]:
    data = json.loads(dominant_soft.read_text())
    per_head = data.get("per_head", {})
    if not isinstance(per_head, dict):
        raise ValueError("dominant-soft 缺少 per_head dict")
    sigs: Dict[str, str] = {}
    for h, info in per_head.items():
        sig = (info or {}).get("signature") if isinstance(info, dict) else None
        if not isinstance(sig, str) or not sig:
            sig = "none"
        sigs[str(h)] = sig
    return sigs


def load_edge_weights(
    path: Optional[Path],
    min_abs: float,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Return head-edge weight lookup keyed by (src,dst):
      - abs_max: max |delta_logit_diff| among records for that edge
      - signed_at_abs_max: signed delta at abs_max
    """
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    records = payload.get("edges", [])
    if not isinstance(records, list):
        raise ValueError("edge-weights-json 缺少 edges 列表")

    best: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for e in records:
        src = e.get("src")
        dst = e.get("dst") or "logits"
        if not isinstance(src, str) or not isinstance(dst, str):
            continue
        delta = e.get("delta_logit_diff")
        if delta is None:
            continue
        d = float(delta)
        a = abs(d)
        if a < min_abs:
            continue
        key = (src, dst)
        prev = best.get(key)
        if prev is None or a > prev[0]:
            best[key] = (a, d)
    return {k: {"abs_max": v[0], "signed_at_abs_max": v[1]} for k, v in best.items()}


def render_group_graph(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], out_png: Path, layout: str) -> None:
    g = pgv.AGraph(directed=True, strict=True, splines="true", overlap="false", layout=layout)
    for n in nodes:
        g.add_node(
            n["id"],
            label=n["label"],
            shape="box",
            style="rounded,filled",
            fillcolor=n.get("fillcolor", "#E8F0FE"),
            fontname="Helvetica",
        )
    for e in edges:
        attrs: Dict[str, str] = {}
        label = e.get("label")
        if label:
            attrs["label"] = label
            attrs["fontsize"] = "10"
        g.add_edge(e["src"], e["dst"], **attrs)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    g.draw(out_png, prog=layout)


def tarjan_scc(nodes: Iterable[str], edges: Iterable[Tuple[str, str]]) -> List[List[str]]:
    """
    Tarjan SCC algorithm. Returns list of SCCs (each SCC is a list of node ids).
    """
    graph: Dict[str, List[str]] = defaultdict(list)
    node_set = set(nodes)
    for a, b in edges:
        if a in node_set and b in node_set:
            graph[a].append(b)

    index = 0
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    out: List[List[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        for w in graph.get(v, []):
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp: List[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            out.append(comp)

    for v in node_set:
        if v not in indices:
            strongconnect(v)
    return out


@dataclass(frozen=True)
class GroupGraph:
    group_members: Dict[str, List[str]]
    group_edges: Dict[Tuple[str, str], List[Tuple[str, str]]]
    group_nodes: Set[str]


def greedy_delete_edges_to_break_cycles(
    gg: GroupGraph,
    weights_by_group_edge: Dict[Tuple[str, str], Dict[str, float]],
    exclude_groups: Set[str],
    score_mode: str,
    max_deletions: int,
) -> Tuple[GroupGraph, List[Dict[str, Any]]]:
    """
    Delete weakest group edges inside SCCs until the group graph becomes acyclic.

    score_mode:
      - signed_mean: abs(weight_signed_mean)
      - abs_max: weight_abs_max
      - count: count
    """
    group_members = gg.group_members
    group_edges = dict(gg.group_edges)  # mutable copy: (gs,gd) -> list[(head_s,head_d)]
    group_nodes = set(gg.group_nodes)

    deletions: List[Dict[str, Any]] = []

    def edge_score(pair: Tuple[str, str]) -> float:
        w = weights_by_group_edge.get(pair) or {}
        if score_mode == "signed_mean":
            v = float(w.get("weight_signed_mean", 0.0))
            return abs(v)
        if score_mode == "abs_max":
            return float(w.get("weight_abs_max", 0.0))
        return float(w.get("count", 0.0))

    for _ in range(max_deletions):
        nodes2 = [n for n in group_nodes if n not in exclude_groups]
        edges2 = [(a, b) for (a, b) in group_edges.keys() if a in nodes2 and b in nodes2]
        sccs = tarjan_scc(nodes2, edges2)
        cyclic = [c for c in sccs if len(c) >= 2]
        if not cyclic:
            break

        # Collect all edges fully inside any cyclic SCC
        cyc_sets = [set(c) for c in cyclic]
        candidates: List[Tuple[float, Tuple[str, str]]] = []
        for pair in list(group_edges.keys()):
            a, b = pair
            for s in cyc_sets:
                if a in s and b in s:
                    candidates.append((edge_score(pair), pair))
                    break
        if not candidates:
            break

        candidates.sort(key=lambda x: (x[0], x[1][0], x[1][1]))
        weakest_score, weakest_pair = candidates[0]
        # record deletion info
        w = weights_by_group_edge.get(weakest_pair) or {}
        deletions.append(
            {
                "src": weakest_pair[0],
                "dst": weakest_pair[1],
                "score_abs": weakest_score,
                "weight_signed_mean": w.get("weight_signed_mean"),
                "weight_abs_max": w.get("weight_abs_max"),
                "count": w.get("count"),
            }
        )
        group_edges.pop(weakest_pair, None)

    gg2 = GroupGraph(group_members=group_members, group_edges=group_edges, group_nodes=group_nodes)
    return gg2, deletions


def build_group_graph(
    base_nodes: List[str],
    base_edges: List[Tuple[str, str]],
    node_to_group: Dict[str, str],
) -> GroupGraph:
    group_members: Dict[str, List[str]] = defaultdict(list)
    for h in base_nodes:
        g = node_to_group[h]
        group_members[g].append(h)
    for g in group_members:
        group_members[g].sort()

    group_edges: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)
    group_nodes: Set[str] = set(group_members.keys())

    for s, d in base_edges:
        if s not in node_to_group or d not in node_to_group:
            continue
        gs, gd = node_to_group[s], node_to_group[d]
        if gs == gd:
            continue
        group_edges[(gs, gd)].append((s, d))
        group_nodes.add(gs)
        group_nodes.add(gd)

    return GroupGraph(group_members=dict(group_members), group_edges=dict(group_edges), group_nodes=group_nodes)


def culprit_scores_for_cycles(
    gg: GroupGraph,
    node_to_group: Dict[str, str],
    weights: Dict[Tuple[str, str], Dict[str, float]],
    scc_nodes: Set[str],
    score_mode: str,
) -> Dict[str, float]:
    """
    Compute culprit score per head from bidirectional group connections inside the SCC node set.
    """
    # Build quick lookup: group pair -> head edge list
    edges_by_pair = gg.group_edges
    scores: Dict[str, float] = defaultdict(float)

    # Determine bidirectional group pairs within SCC
    pairs = [(a, b) for (a, b) in edges_by_pair.keys() if a in scc_nodes and b in scc_nodes]
    pair_set = set(pairs)
    bidir_pairs = [(a, b) for (a, b) in pairs if (b, a) in pair_set]

    # For each bidirectional pair, count heads participating in either direction evidence
    for a, b in bidir_pairs:
        for (s, d) in edges_by_pair.get((a, b), []):
            w = weights.get((s, d))
            inc = float(w["abs_max"]) if (score_mode == "abs_weighted" and w) else 1.0
            scores[s] += inc
            scores[d] += inc
        for (s, d) in edges_by_pair.get((b, a), []):
            w = weights.get((s, d))
            inc = float(w["abs_max"]) if (score_mode == "abs_weighted" and w) else 1.0
            scores[s] += inc
            scores[d] += inc

    # Keep only heads within the SCC groups
    allowed_heads = {h for h, g in node_to_group.items() if g in scc_nodes}
    return {h: sc for h, sc in scores.items() if h in allowed_heads and sc > 0}


def materialize_output(
    args: argparse.Namespace,
    gg: GroupGraph,
    weights: Dict[Tuple[str, str], Dict[str, float]],
) -> Dict[str, Any]:
    # build group node labels
    groups_out: List[Dict[str, Any]] = []
    for grp, members in sorted(gg.group_members.items(), key=lambda x: x[0]):
        show_n = args.max_heads_in_label
        if show_n == 0 or len(members) <= show_n:
            heads_line = ", ".join(members)
        else:
            heads_line = ", ".join(members[:show_n]) + ", …"
        label = f"{grp}\\nheads: {len(members)}\\n{heads_line}"
        fill = "#E8F0FE"
        if grp.startswith("solo:") or "::solo::" in grp:
            fill = "#FFF7E6"
        if grp in {"input", "logits"}:
            fill = "#EEEEEE"
        groups_out.append({"id": grp, "label": label, "fillcolor": fill})

    edges_out: List[Dict[str, Any]] = []
    for (gs, gd), head_edges in sorted(gg.group_edges.items(), key=lambda x: (x[0][0], x[0][1])):
        abs_vals = []
        signed_vals = []
        for s, d in head_edges:
            w = weights.get((s, d))
            if not w:
                continue
            abs_vals.append(float(w["abs_max"]))
            signed_vals.append(float(w["signed_at_abs_max"]))
        payload: Dict[str, Any] = {"src": gs, "dst": gd, "count": len(head_edges)}
        if abs_vals:
            payload["weight_abs_max"] = max(abs_vals)
            payload["weight_signed_mean"] = mean(signed_vals) if signed_vals else 0.0
        edges_out.append(payload)

    out = {
        "meta": {
            "source_graph": str(args.graph_json),
            "dominant_soft": str(args.dominant_soft),
            "edge_weights": str(args.edge_weights_json) if args.edge_weights_json else None,
            "min_edge_abs_score": args.min_edge_abs_score,
            "method": "split_culprit_heads",
            "score_mode": args.score_mode,
            "max_iters": args.max_iters,
            "split_per_iter": args.split_per_iter,
        },
        "groups": groups_out,
        "edges": edges_out,
        "group_members": gg.group_members,
    }
    return out


def main() -> None:
    args = parse_args()
    exclude_groups_scc = {s.strip() for s in args.exclude_groups.split(",") if s.strip()}
    buckets = parse_layer_buckets(args.layer_buckets)

    nodes, edge_list = load_graph(args.graph_json)
    sigs = load_signatures(args.dominant_soft)
    weights = load_edge_weights(args.edge_weights_json, args.min_edge_abs_score)

    head_nodes = [n for n in nodes if isinstance(n, str) and n.startswith("a") and ".h" in n]
    special_nodes = [n for n in nodes if n in {"input", "logits"}]
    base_nodes = head_nodes + special_nodes
    node_to_group: Dict[str, str] = {}
    for h in head_nodes:
        k = head_key(h)
        sig = sigs.get(k, "none")
        if buckets:
            b = bucket_for_layer(head_layer(h), buckets)
            node_to_group[h] = f"{sig}::L{b}"
        else:
            node_to_group[h] = sig
    for n in special_nodes:
        node_to_group[n] = n

    report: Dict[str, Any] = {"iterations": []}
    already_split: Set[str] = set()

    for it in range(1, args.max_iters + 1):
        gg = build_group_graph(base_nodes, edge_list, node_to_group)
        group_nodes = [g for g in gg.group_nodes if g not in exclude_groups_scc]
        group_edges = list(gg.group_edges.keys())

        sccs = tarjan_scc(group_nodes, group_edges)
        cyclic_sccs = [c for c in sccs if len(c) >= 2]

        report_it: Dict[str, Any] = {
            "iter": it,
            "n_groups": len(gg.group_members),
            "n_group_edges": len(gg.group_edges),
            "n_cyclic_scc": len(cyclic_sccs),
            "cyclic_scc_sizes": sorted([len(c) for c in cyclic_sccs], reverse=True),
            "splits": [],
        }

        if not cyclic_sccs:
            report["iterations"].append(report_it)
            break

        # Score heads within cyclic SCCs
        total_scores: Dict[str, float] = defaultdict(float)
        for comp in cyclic_sccs:
            comp_set = set(comp)
            sc = culprit_scores_for_cycles(gg, node_to_group, weights, comp_set, args.score_mode)
            for h, v in sc.items():
                total_scores[h] += float(v)

        # Pick heads to split this iteration
        candidates = [(h, s) for h, s in total_scores.items() if h not in already_split]
        candidates.sort(key=lambda x: (-x[1], x[0]))
        chosen = candidates[: max(0, args.split_per_iter)]

        if not chosen:
            report_it["note"] = "no culprit heads found to split; stop"
            report["iterations"].append(report_it)
            break

        for h, sc in chosen:
            old = node_to_group[h]
            new = f"solo:{h}|from:{old}"
            node_to_group[h] = new
            already_split.add(h)
            report_it["splits"].append({"head": h, "score": sc, "from_group": old, "new_group": new})

        report["iterations"].append(report_it)

    # Final grouped graph
    gg_final = build_group_graph(base_nodes, edge_list, node_to_group)

    # If requested, delete weakest *group edges* inside SCCs until acyclic.
    deletions: List[Dict[str, Any]] = []
    if args.delete_weakest_edges:
        # Precompute group-edge weights from current head-edge lists (using weights dict).
        weights_by_group_edge: Dict[Tuple[str, str], Dict[str, float]] = {}
        for (gs, gd), head_edges in gg_final.group_edges.items():
            signed_vals = []
            abs_vals = []
            for s, d in head_edges:
                w = weights.get((s, d))
                if not w:
                    continue
                abs_vals.append(float(w["abs_max"]))
                signed_vals.append(float(w["signed_at_abs_max"]))
            payload: Dict[str, float] = {"count": float(len(head_edges))}
            if abs_vals:
                payload["weight_abs_max"] = float(max(abs_vals))
                payload["weight_signed_mean"] = float(mean(signed_vals) if signed_vals else 0.0)
            weights_by_group_edge[(gs, gd)] = payload

        gg_final, deletions = greedy_delete_edges_to_break_cycles(
            gg_final,
            weights_by_group_edge,
            exclude_groups=exclude_groups_scc,
            score_mode=args.edge_delete_score,
            max_deletions=args.max_edge_deletions,
        )

    out = materialize_output(args, gg_final, weights)
    if deletions:
        out["meta"]["edge_deletions"] = {
            "enabled": True,
            "mode": args.edge_delete_score,
            "n_deleted": len(deletions),
        }
        out["meta"]["edge_deletions_report_path"] = str(args.report_json) if args.report_json else None

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"✅ wrote DAG-ish grouped graph json: {args.output_json}")

    if args.output_png:
        edges_for_draw = []
        for e in out["edges"]:
            lbl = f"{e.get('count', 0)}"
            if "weight_abs_max" in e:
                lbl += f"\\n|maxΔ|={float(e['weight_abs_max']):.3g}"
            edges_for_draw.append({"src": e["src"], "dst": e["dst"], "label": lbl})
        render_group_graph(out["groups"], edges_for_draw, args.output_png, args.layout)
        print(f"✅ wrote DAG-ish grouped graph png: {args.output_png}")

    if args.report_json:
        report["edge_deletions"] = deletions
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"✅ wrote cycle-breaking report: {args.report_json}")


if __name__ == "__main__":
    main()
