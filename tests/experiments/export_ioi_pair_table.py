#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from xml.sax.saxutils import escape


FAMILY_CONFIG = {
    "NMH": {"family_dir": "Name_Mover_Head", "reference_file": "NMH.txt", "heads": ["9.6", "9.9", "10.0"]},
    "NNMH": {"family_dir": "Negative_Name_Mover_Head", "reference_file": "NNMH.txt", "heads": ["10.7", "11.10"]},
    "SIH": {"family_dir": "SIH", "reference_file": "SIH.txt", "heads": ["7.9", "8.6", "8.10"]},
    "DTH": {"family_dir": "DTH", "reference_file": "DTH.txt", "heads": ["0.1", "0.10", "3.0"]},
}

ROW_ORDER = [
    ("all", "initial", "all_initial"),
    ("all", "final", "all_final"),
    ("causal", "initial", "causal_initial"),
    ("causal", "final", "causal_final"),
    ("att", "initial", "att_initial"),
    ("att", "final", "att_final"),
]


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest(results_root: Path, family_dir: str, head: str, mode: str, filename: str) -> Optional[Path]:
    pattern = f"{head}_*_{mode}/{filename}"
    matches = sorted(
        (results_root / "hypothesis" / family_dir).glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _load_reference_text(reference_root: Path, ref_file: str) -> str:
    p = reference_root / ref_file
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def _extract_final(test_json: Optional[dict]) -> tuple[str, float, float, float]:
    if not isinstance(test_json, dict):
        return "", 0.0, 0.0, 0.0
    hyp = str(test_json.get("hypothesis", "") or "")
    scores = test_json.get("validation_scores", {}) if isinstance(test_json.get("validation_scores", {}), dict) else {}
    c = float(scores.get("causal_f1", 0.0) or 0.0)
    a = float(scores.get("attention_f1", scores.get("direct_attention_f1", 0.0)) or 0.0)
    comp = math.sqrt(c * a) if c > 0 and a > 0 else 0.0
    return hyp, c, a, comp


def _extract_initial(init_json: Optional[dict]) -> tuple[str, float, float, float]:
    if not isinstance(init_json, dict):
        return "", 0.0, 0.0, 0.0
    hyp = str(init_json.get("hypothesis", "") or "")
    scores = init_json.get("validation_scores", {}) if isinstance(init_json.get("validation_scores", {}), dict) else {}
    c = float(scores.get("causal_f1", 0.0) or 0.0)
    a = float(scores.get("attention_f1", scores.get("direct_attention_f1", 0.0)) or 0.0)
    comp = math.sqrt(c * a) if c > 0 and a > 0 else 0.0
    return hyp, c, a, comp


def _extract_pair(all_run_dir: Optional[Path], head: str, tag: str) -> tuple[str, str, str, str, str]:
    if all_run_dir is None:
        return "", "", "", "", ""
    p = all_run_dir / f"pair_{head}_{tag}_vs_all_final.json"
    data = _load_json(p)
    if not isinstance(data, dict):
        return "", "", "", "", ""
    ev = data.get("llm_evaluation", {}) if isinstance(data.get("llm_evaluation", {}), dict) else {}
    return (
        str(ev.get("winner", "") or ""),
        str(ev.get("confidence", "") or ""),
        str(ev.get("a_score", "") or ""),
        str(ev.get("b_score", "") or ""),
        str(ev.get("comments", "") or ""),
    )


def build_rows(results_root: Path, reference_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for family_label, cfg in FAMILY_CONFIG.items():
        family_dir = cfg["family_dir"]
        ref_text = _load_reference_text(reference_root, cfg["reference_file"])
        for head in cfg["heads"]:
            all_final_path = _latest(results_root, family_dir, head, "all", "test_results.json")
            all_run_dir = all_final_path.parent if all_final_path else None
            for mode, phase, pair_tag in ROW_ORDER:
                if phase == "final":
                    path = _latest(results_root, family_dir, head, mode, "test_results.json")
                    hyp, c, a, comp = _extract_final(_load_json(path) if path else None)
                    run_dir = str(path.parent) if path else ""
                else:
                    path = _latest(results_root, family_dir, head, mode, "initial_test_results.json")
                    hyp, c, a, comp = _extract_initial(_load_json(path) if path else None)
                    run_dir = str(path.parent) if path else ""

                winner = conf = a_score = b_score = comments = ""
                if pair_tag != "all_final":
                    winner, conf, a_score, b_score, comments = _extract_pair(all_run_dir, head, pair_tag)

                rows.append(
                    {
                        "head_type": family_label,
                        "standard_explanation": ref_text,
                        "head": head,
                        "type": f"{mode}.{phase}",
                        "hypothesis": hyp,
                        "causal_f1": c,
                        "att_f1": a,
                        "composite_score": comp,
                        "llm_winner_vs_all_final": winner,
                        "llm_confidence": conf,
                        "llm_a_score": a_score,
                        "llm_b_score": b_score,
                        "llm_comments": comments,
                        "run_dir": run_dir,
                    }
                )
    return rows


def write_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "head_type", "standard_explanation", "head", "type", "hypothesis",
        "causal_f1", "att_f1", "composite_score",
        "llm_winner_vs_all_final", "llm_confidence", "llm_a_score", "llm_b_score", "llm_comments",
        "run_dir",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _col_name(idx: int) -> str:
    s = ""
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        s = chr(65 + rem) + s
    return s


def _xlsx_cell(cell_ref: str, value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    text = escape("" if value is None else str(value))
    return f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'


def write_xlsx(rows: List[Dict[str, object]], out_xlsx: Path) -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "head_type", "standard_explanation", "head", "type", "hypothesis",
        "causal_f1", "att_f1", "composite_score",
        "llm_winner_vs_all_final", "llm_confidence", "llm_a_score", "llm_b_score", "llm_comments",
        "run_dir",
    ]
    all_rows = [headers] + [[r.get(h, "") for h in headers] for r in rows]

    # merge by family and by head
    merge_ranges: List[str] = []
    col = {h: i + 1 for i, h in enumerate(headers)}
    ch = _col_name(col["head_type"])
    cs = _col_name(col["standard_explanation"])
    chead = _col_name(col["head"])
    start_data = 2
    i = 0
    n = len(rows)
    while i < n:
        fam = str(rows[i]["head_type"])
        j = i
        while j < n and str(rows[j]["head_type"]) == fam:
            j += 1
        if j - i > 1:
            merge_ranges.append(f"{ch}{start_data+i}:{ch}{start_data+j-1}")
            merge_ranges.append(f"{cs}{start_data+i}:{cs}{start_data+j-1}")
        i = j
    i = 0
    while i < n:
        fam = str(rows[i]["head_type"])
        hd = str(rows[i]["head"])
        j = i
        while j < n and str(rows[j]["head_type"]) == fam and str(rows[j]["head"]) == hd:
            j += 1
        if j - i > 1:
            merge_ranges.append(f"{chead}{start_data+i}:{chead}{start_data+j-1}")
        i = j

    row_xml = []
    for r_idx, row in enumerate(all_rows, start=1):
        cells = "".join(_xlsx_cell(f"{_col_name(c_idx)}{r_idx}", v) for c_idx, v in enumerate(row, start=1))
        row_xml.append(f'<row r="{r_idx}">{cells}</row>')

    merge_xml = ""
    if merge_ranges:
        merge_xml = f'<mergeCells count="{len(merge_ranges)}">' + "".join(
            f'<mergeCell ref="{escape(m)}"/>' for m in merge_ranges
        ) + "</mergeCells>"

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        "<sheetData>" + "".join(row_xml) + "</sheetData>" + merge_xml + "</worksheet>"
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="pair_summary" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    wb_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )
    with zipfile.ZipFile(out_xlsx, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export pair-comparison summary table.")
    p.add_argument("--results-root", default="/data31/private/wangziran/eap_auto/results/ioi_0305")
    p.add_argument("--reference-root", default="/data31/private/wangziran/eap_auto/results/ioi_0305/answer")
    p.add_argument("--out-csv", default="")
    p.add_argument("--out-xlsx", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.results_root)
    ref = Path(args.reference_root)
    out_csv = Path(args.out_csv) if args.out_csv else root / "summary" / "ioi_pair_summary.csv"
    out_xlsx = Path(args.out_xlsx) if args.out_xlsx else root / "summary" / "ioi_pair_summary.xlsx"
    rows = build_rows(root, ref)
    write_csv(rows, out_csv)
    write_xlsx(rows, out_xlsx)
    print(f"✅ CSV written: {out_csv}")
    print(f"✅ XLSX written: {out_xlsx}")


if __name__ == "__main__":
    main()
