import argparse, json, hashlib
from pathlib import Path
from typing import Dict, Tuple, Sequence
import pygraphviz as pgv

ROLE_COLOR = {"IO":"#4CAF50","S1":"#FB8C00","S2":"#E53935","mixed":"#BDBDBD","NA":"#EEEEEE"}
LABEL_COLOR = {"name_mover_leaning":"#4CAF50","s_inhibition_leaning":"#E53935","mixed_nm_inh":"#8E24AA","weak_signal":"#FB8C00","uncertain":"#BDBDBD"}

def load_graph(graph_json: Path) -> Tuple[Sequence[str], Sequence[Tuple[str,str]]]:
    d = json.loads(graph_json.read_text()); return d.get("nodes") or [], d.get("edge_list") or []

def load_soft(soft_json: Path) -> Dict[str, Dict]:
    return json.loads(soft_json.read_text())

def map_head_name(a_name: str) -> str:
    if isinstance(a_name, str) and a_name.startswith("a") and ".h" in a_name:
        Ls, Hs = a_name.split("."); return f"{int(Ls[1:])}.{int(Hs[1:])}"
    return ""

def main():
    ap = argparse.ArgumentParser(description="把分类/模式标注到图上")
    ap.add_argument("--graph-json", type=Path, required=True)
    ap.add_argument("--soft-json", type=Path, required=True, help="classify.py 输出 JSON（含 patterns）")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--layout", type=str, default="dot")
    ap.add_argument("--annotate-scores", action="store_true")
    ap.add_argument("--color-mode", type=str, default="pattern", choices=["pattern","pos_roles","soft_label"],
                    help="pattern: 统一灰底，仅标注模式签名；pos_roles/soft_label：沿用旧配色")
    args = ap.parse_args()

    nodes, edges = load_graph(args.graph_json)
    soft_all = load_soft(args.soft_json)
    soft_labels = soft_all.get("soft_labels", {})
    group_labels = soft_all.get("group_labels", {})
    patterns = soft_all.get("patterns", {})
    per_head_sig = (patterns.get("per_head") or soft_all.get("head_patterns") or {})

    def signature_color(sig: str) -> str:
        if not sig:
            return "#EEEEEE"
        h = hashlib.md5(sig.encode("utf-8")).hexdigest()
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
        b = int(h[4:6], 16)
        # soften颜色避免过暗
        r = (r // 2) + 96
        g = (g // 2) + 96
        b = (b // 2) + 96
        return f"#{r:02X}{g:02X}{b:02X}"

    G = pgv.AGraph(directed=True, strict=True, splines="true", overlap="false", layout=args.layout)
    for n in nodes:
        head_key = map_head_name(n)
        label = "uncertain"; nm = inh = None
        end_tgt = s2_tgt = "NA"
        sig = per_head_sig.get(head_key, {}).get("signature", "")
        path_patch = per_head_sig.get(head_key, {}).get("path_patch") if head_key else None
        border_color = "#555555"
        penwidth = 1.5
        if path_patch:
            cls = path_patch.get("classification")
            if cls == "positive":
                border_color = "#2E7D32"
                penwidth = 2.5
            elif cls == "negative":
                border_color = "#C62828"
                penwidth = 2.5
            else:
                border_color = "#616161"
                penwidth = 2.0

        if head_key and head_key in soft_labels:
            label = soft_labels[head_key]["label"]
            nm = soft_labels[head_key]["scores"]["name_mover"]
            inh = soft_labels[head_key]["scores"]["s_inhibition"]
        if head_key and head_key in group_labels:
            end_tgt = group_labels[head_key].get("END", "mixed")
            s2_tgt = group_labels[head_key].get("S2", "mixed")

        if args.color_mode == "pattern":
            fillcolor = signature_color(sig)
            label_lines = [n]
            if sig:
                label_lines.append(sig)
            patterns_info = per_head_sig.get(head_key, {}).get("positions", {}) if head_key else {}
            for role in ("END", "S2", "S1", "IO"):
                role_list = patterns_info.get(role)
                if role_list:
                    first = role_list[0]
                    pat = first.get("pattern", "")
                    cnt = first.get("count")
                    tot = first.get("total")
                    if pat:
                        if cnt and tot:
                            label_lines.append(f"{role}: {pat} ({cnt}/{tot})")
                        else:
                            label_lines.append(f"{role}: {pat}")
            if path_patch:
                metric = path_patch.get("metric")
                cls = path_patch.get("classification") or "unknown"
                if metric is not None and cls in ("positive", "negative"):
                    label_lines.append(f"Path: {cls} ({metric:+.2f})")
                elif cls:
                    label_lines.append(f"Path: {cls}")
            node_label = "\n".join(label_lines)
            G.add_node(n, shape="box", style="rounded,filled", fillcolor=fillcolor, color=border_color, penwidth=penwidth,
                       fontname="Helvetica", label=node_label)
        elif args.color_mode == "pos_roles":
            fillcolor = ROLE_COLOR.get(end_tgt, ROLE_COLOR["NA"])
            border_color = ROLE_COLOR.get(s2_tgt, ROLE_COLOR["NA"])
            node_label = n
            if args.annotate_scores and nm is not None:  # 防止属性名中连字符误用
                pass
            node_label = f"{n}\nEND->{end_tgt} | S2->{s2_tgt}\nNM:{nm:.2f} INH:{inh:.2f}" if (args.annotate_scores and nm is not None) else f"{n}\nEND->{end_tgt} | S2->{s2_tgt}"
            G.add_node(n, shape="box", style="rounded,filled", fillcolor=fillcolor, color=border_color, penwidth=3,
                       fontname="Helvetica", label=node_label)
        else:
            fillcolor = LABEL_COLOR.get(label, LABEL_COLOR["uncertain"])
            node_label = f"{n}\nNM:{nm:.2f} INH:{inh:.2f}" if (args.annotate_scores and nm is not None) else n
            G.add_node(n, shape="box", style="rounded,filled", fillcolor=fillcolor, fontname="Helvetica", label=node_label)

    for s, d in edges: G.add_edge(s, d)
    G.draw(args.output, prog=args.layout)
    print(f"保存: {args.output}")

if __name__ == "__main__":
    main()