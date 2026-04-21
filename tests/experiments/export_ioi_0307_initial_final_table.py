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


FAMILY_ORDER = [
    ("NMH", "Name_Mover_Head", "NMH.txt", ["9.6", "9.9", "10.0"]),
    ("NNMH", "Negative_Name_Mover_Head", "NNMH.txt", ["10.7", "11.10"]),
    ("SIH", "SIH", "SIH.txt", ["7.9", "8.6", "8.10"]),
    ("DTH", "DTH", "DTH.txt", ["0.1", "0.10", "3.0"]),
]


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_all_run(results_root: Path, family_dir: str, head: str) -> Optional[Path]:
    matches = sorted(
        (results_root / "hypothesis" / family_dir).glob(f"{head}_*_all"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _score_from_validation_scores(scores: dict) -> tuple[float, float, float]:
    causal = float(scores.get("causal_f1", 0.0) or 0.0)
    att = float(scores.get("attention_f1", scores.get("direct_attention_f1", 0.0)) or 0.0)
    comp = math.sqrt(causal * att) if causal > 0 and att > 0 else 0.0
    return causal, att, comp


def _extract_initial(run_dir: Path) -> tuple[str, float, float, float]:
    # 优先使用 initial_test_results（与 final 同口径，均为 test 集）
    init_test = _load_json(run_dir / "initial_test_results.json")
    if isinstance(init_test, dict):
        hyp = str(init_test.get("hypothesis", "") or "")
        scores = init_test.get("validation_scores", {}) if isinstance(init_test.get("validation_scores"), dict) else {}
        c, a, comp = _score_from_validation_scores(scores)
        return hyp, c, a, comp

    # 兜底：老目录可能没有 initial_test_results
    val0 = _load_json(run_dir / "validation_results" / "validation_epoch_0_initial.json") or {}
    hyp = str(val0.get("hypothesis", "") or "")
    scores = val0.get("validation_scores", {}) if isinstance(val0.get("validation_scores"), dict) else {}
    c, a, comp = _score_from_validation_scores(scores)
    return hyp, c, a, comp


def _extract_final(run_dir: Path) -> tuple[str, float, float, float]:
    test = _load_json(run_dir / "test_results.json") or {}
    hyp = str(test.get("hypothesis", "") or "")
    scores = test.get("validation_scores", {}) if isinstance(test.get("validation_scores"), dict) else {}
    c, a, comp = _score_from_validation_scores(scores)
    return hyp, c, a, comp


def _extract_pair(compare_root: Path, family_dir: str, head: str) -> tuple[str, float, float, float, str]:
    p = compare_root / family_dir / head / "initial_vs_final_pair.json"
    data = _load_json(p) or {}
    ev = data.get("llm_evaluation", {}) if isinstance(data.get("llm_evaluation"), dict) else {}
    winner = str(ev.get("winner", "") or "")
    conf = float(ev.get("confidence", 0.0) or 0.0)
    a_score = float(ev.get("a_score", 0.0) or 0.0)
    b_score = float(ev.get("b_score", 0.0) or 0.0)
    comments = str(ev.get("comments", "") or "")
    return winner, conf, a_score, b_score, comments


def _load_reference(reference_root: Path, filename: str) -> str:
    p = reference_root / filename
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def build_rows(results_root: Path, reference_root: Path, compare_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for fam_label, fam_dir, ref_file, heads in FAMILY_ORDER:
        ref_text = _load_reference(reference_root, ref_file)
        for head in heads:
            run_dir = _latest_all_run(results_root, fam_dir, head)
            if run_dir is None:
                continue
            init_h, init_c, init_a, init_comp = _extract_initial(run_dir)
            fin_h, fin_c, fin_a, fin_comp = _extract_final(run_dir)
            winner, conf, a_score, b_score, comments = _extract_pair(compare_root, fam_dir, head)
            rows.append(
                {
                    "head_type": fam_label,
                    "standard_explanation": ref_text,
                    "head": head,
                    "initial_hypothesis": init_h,
                    "initial_causal_f1": init_c,
                    "initial_att_f1": init_a,
                    "initial_composite_f1": init_comp,
                    "final_hypothesis": fin_h,
                    "final_causal_f1": fin_c,
                    "final_att_f1": fin_a,
                    "final_composite_f1": fin_comp,
                    "llm_winner_initial_vs_final": winner,
                    "llm_confidence": conf,
                    "llm_a_score_initial": a_score,
                    "llm_b_score_final": b_score,
                    "llm_comments": comments,
                    "run_dir": str(run_dir),
                }
            )
    return rows


def write_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    fields = [
        "head_type",
        "standard_explanation",
        "head",
        "initial_hypothesis",
        "initial_causal_f1",
        "initial_att_f1",
        "initial_composite_f1",
        "final_hypothesis",
        "final_causal_f1",
        "final_att_f1",
        "final_composite_f1",
        "llm_winner_initial_vs_final",
        "llm_confidence",
        "llm_a_score_initial",
        "llm_b_score_final",
        "llm_comments",
        "run_dir",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
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
    headers = [
        "head_type",
        "standard_explanation",
        "head",
        "initial_hypothesis",
        "initial_causal_f1",
        "initial_att_f1",
        "initial_composite_f1",
        "final_hypothesis",
        "final_causal_f1",
        "final_att_f1",
        "final_composite_f1",
        "llm_winner_initial_vs_final",
        "llm_confidence",
        "llm_a_score_initial",
        "llm_b_score_final",
        "llm_comments",
        "run_dir",
    ]
    all_rows = [headers] + [[r.get(h, "") for h in headers] for r in rows]

    merge_ranges: List[str] = []
    col = {h: i + 1 for i, h in enumerate(headers)}
    ch = _col_name(col["head_type"])
    cs = _col_name(col["standard_explanation"])
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
        '<sheets><sheet name="initial_final" sheetId="1" r:id="rId1"/></sheets></workbook>'
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
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_xlsx, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export ioi_0307 initial-vs-final summary table.")
    p.add_argument("--results-root", default="/home/wangziran/eap_auto/results/ioi_0307")
    p.add_argument("--reference-root", default="/home/wangziran/eap_auto/results/ioi_0126/answer")
    p.add_argument("--compare-root", default="/home/wangziran/eap_auto/results/ioi_0307/compare_initial_vs_final")
    p.add_argument("--out-csv", default="")
    p.add_argument("--out-xlsx", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    reference_root = Path(args.reference_root)
    compare_root = Path(args.compare_root)
    out_csv = Path(args.out_csv) if args.out_csv else results_root / "summary" / "ioi_0307_initial_final_summary.csv"
    out_xlsx = Path(args.out_xlsx) if args.out_xlsx else results_root / "summary" / "ioi_0307_initial_final_summary.xlsx"
    rows = build_rows(results_root, reference_root, compare_root)
    write_csv(rows, out_csv)
    write_xlsx(rows, out_xlsx)
    print(f"✅ CSV written: {out_csv}")
    print(f"✅ XLSX written: {out_xlsx}")


if __name__ == "__main__":
    main()
