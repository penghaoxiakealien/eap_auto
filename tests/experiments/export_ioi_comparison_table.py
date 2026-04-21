#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from xml.sax.saxutils import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple


FAMILY_CONFIG = {
    "NMH": {
        "family_dir": "Name_Mover_Head",
        "reference_file": "NMH.txt",
        "heads": ["9.6", "9.9", "10.0"],
    },
    "NNMH": {
        "family_dir": "Negative_Name_Mover_Head",
        "reference_file": "NNMH.txt",
        "heads": ["10.7", "11.10"],
    },
    "SIH": {
        "family_dir": "SIH",
        "reference_file": "SIH.txt",
        "heads": ["7.9", "8.6", "8.10"],
    },
    "DTH": {
        "family_dir": "DTH",
        "reference_file": "DTH.txt",
        "heads": ["0.1", "0.10", "3.0"],
    },
}

MODES = ["all", "causal", "att"]


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_test_result(results_root: Path, family_dir: str, head: str, mode: str) -> Optional[Path]:
    pattern = f"{head}_*_{mode}/test_results.json"
    matches = sorted(
        (results_root / "hypothesis" / family_dir).glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None

def _latest_iteration1(results_root: Path, family_dir: str, head: str, mode: str) -> Optional[Path]:
    pattern = f"{head}_*_{mode}/iteration_results/iteration_1.json"
    matches = sorted(
        (results_root / "hypothesis" / family_dir).glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None

def _extract_initial_hypothesis(iter1_json: dict) -> str:
    if not isinstance(iter1_json, dict):
        return ""
    # NMH uses "hypothesis"; auto_s_att uses "hypothesis_before".
    hyp = iter1_json.get("hypothesis_before") or iter1_json.get("hypothesis") or ""
    return str(hyp or "")


def _extract_scores(test_json: dict) -> Tuple[float, float, float]:
    scores = test_json.get("validation_scores", {}) if isinstance(test_json, dict) else {}
    causal = scores.get("causal_f1", 0.0) or 0.0
    att = scores.get("attention_f1", scores.get("direct_attention_f1", 0.0)) or 0.0
    # Force unified composite definition across all modes.
    comp = math.sqrt(causal * att) if causal > 0 and att > 0 else 0.0
    return float(causal), float(att), float(comp)


def _load_reference_text(reference_root: Path, ref_file: str) -> str:
    p = reference_root / ref_file
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8").strip()


def _load_compare_json_for_mode(all_test_result_path: Optional[Path], mode: str) -> Optional[dict]:
    if not all_test_result_path or mode == "all":
        return None
    all_run_dir = all_test_result_path.parent
    compare_file = all_run_dir / f"compare_{mode}_vs_all.json"
    return _load_json(compare_file)


def build_rows(results_root: Path, reference_root: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for family_label, cfg in FAMILY_CONFIG.items():
        family_dir = cfg["family_dir"]
        reference_text = _load_reference_text(reference_root, cfg["reference_file"])

        for head in cfg["heads"]:
            all_test_path = _latest_test_result(results_root, family_dir, head, "all")
            all_iter1_path = _latest_iteration1(results_root, family_dir, head, "all")
            all_iter1_json = _load_json(all_iter1_path) if all_iter1_path else None
            initial_hypothesis = _extract_initial_hypothesis(all_iter1_json) if all_iter1_json else ""
            initial_test_json = _load_json((all_iter1_path.parent.parent / "initial_test_results.json")) if all_iter1_path else None
            init_causal = ""
            init_att = ""
            init_comp = ""
            if isinstance(initial_test_json, dict):
                v = initial_test_json.get("validation_scores", {})
                if isinstance(v, dict):
                    init_causal = v.get("causal_f1", "")
                    init_att = v.get("attention_f1", v.get("direct_attention_f1", ""))
                    c = float(init_causal or 0.0)
                    a = float(init_att or 0.0)
                    init_comp = math.sqrt(c * a) if c > 0 and a > 0 else 0.0

            rows.append(
                {
                    "head_type": family_label,
                    "standard_explanation": reference_text,
                    "head": head,
                    "type": "initial",
                    "hypothesis": initial_hypothesis,
                    "causal_f1": init_causal,
                    "att_f1": init_att,
                    "composite_score": init_comp,
                    "llm_winner_mode_vs_all": "",
                    "llm_confidence": "",
                    "llm_mode_score_a": "",
                    "llm_all_score_b": "",
                    "llm_comments": "",
                    "run_dir": str(all_iter1_path.parent.parent) if all_iter1_path else "",
                }
            )

            for mode in MODES:
                test_path = _latest_test_result(results_root, family_dir, head, mode)
                test_json = _load_json(test_path) if test_path else None

                hypothesis = ""
                causal_f1 = 0.0
                att_f1 = 0.0
                composite_score = 0.0
                run_dir = ""
                if test_json:
                    hypothesis = str(test_json.get("hypothesis", "") or "")
                    causal_f1, att_f1, composite_score = _extract_scores(test_json)
                    run_dir = str(test_path.parent)

                llm_winner = ""
                llm_confidence = ""
                llm_a_score = ""
                llm_b_score = ""
                llm_comments = ""

                cmp_json = _load_compare_json_for_mode(all_test_path, mode)
                if cmp_json:
                    ev = cmp_json.get("llm_evaluation", {})
                    llm_winner = str(ev.get("winner", "") or "")
                    llm_confidence = str(ev.get("confidence", "") or "")
                    llm_a_score = str(ev.get("a_score", "") or "")
                    llm_b_score = str(ev.get("b_score", "") or "")
                    llm_comments = str(ev.get("comments", "") or "")

                rows.append(
                    {
                        "head_type": family_label,
                        "standard_explanation": reference_text,
                        "head": head,
                        "type": mode,
                        "hypothesis": hypothesis,
                        "causal_f1": causal_f1,
                        "att_f1": att_f1,
                        "composite_score": composite_score,
                        "llm_winner_mode_vs_all": llm_winner,
                        "llm_confidence": llm_confidence,
                        "llm_mode_score_a": llm_a_score,
                        "llm_all_score_b": llm_b_score,
                        "llm_comments": llm_comments,
                        "run_dir": run_dir,
                    }
                )
    return rows


def write_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "head_type",
        "standard_explanation",
        "head",
        "type",
        "hypothesis",
        "causal_f1",
        "att_f1",
        "composite_score",
        "llm_winner_mode_vs_all",
        "llm_confidence",
        "llm_mode_score_a",
        "llm_all_score_b",
        "llm_comments",
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


def _xlsx_cell_xml(cell_ref: str, value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{cell_ref}"><v>{value}</v></c>'
    text = "" if value is None else str(value)
    text = escape(text)
    return f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">{text}</t></is></c>'


def write_xlsx_if_available(rows: List[Dict[str, object]], out_xlsx: Path) -> bool:
    # Pure-Python minimal XLSX writer (no openpyxl dependency).
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "head_type",
        "standard_explanation",
        "head",
        "type",
        "hypothesis",
        "causal_f1",
        "att_f1",
        "composite_score",
        "llm_winner_mode_vs_all",
        "llm_confidence",
        "llm_mode_score_a",
        "llm_all_score_b",
        "llm_comments",
        "run_dir",
    ]

    all_rows: List[List[object]] = [headers]
    for r in rows:
        all_rows.append([r.get(h, "") for h in headers])

    # Build merge ranges:
    # - standard_explanation + head_type merged by family block
    # - head merged by each 4-row block (initial/all/causal/att)
    merge_ranges: List[str] = []
    col_idx = {h: i + 1 for i, h in enumerate(headers)}
    c_head_type = _col_name(col_idx["head_type"])
    c_std = _col_name(col_idx["standard_explanation"])
    c_head = _col_name(col_idx["head"])

    # rows in sheet: header=1, data starts at 2
    data_start = 2
    n = len(rows)
    i = 0
    while i < n:
        fam = str(rows[i].get("head_type", ""))
        fam_start = i
        while i < n and str(rows[i].get("head_type", "")) == fam:
            i += 1
        fam_end = i - 1
        r1 = data_start + fam_start
        r2 = data_start + fam_end
        if r2 > r1:
            merge_ranges.append(f"{c_head_type}{r1}:{c_head_type}{r2}")
            merge_ranges.append(f"{c_std}{r1}:{c_std}{r2}")

    i = 0
    while i < n:
        fam = str(rows[i].get("head_type", ""))
        head = str(rows[i].get("head", ""))
        block_start = i
        while i < n and str(rows[i].get("head_type", "")) == fam and str(rows[i].get("head", "")) == head:
            i += 1
        block_end = i - 1
        r1 = data_start + block_start
        r2 = data_start + block_end
        if r2 > r1:
            merge_ranges.append(f"{c_head}{r1}:{c_head}{r2}")

    sheet_rows_xml: List[str] = []
    for r_idx, row in enumerate(all_rows, start=1):
        cells = []
        for c_idx, val in enumerate(row, start=1):
            ref = f"{_col_name(c_idx)}{r_idx}"
            cells.append(_xlsx_cell_xml(ref, val))
        sheet_rows_xml.append(f'<row r="{r_idx}">{"".join(cells)}</row>')

    merge_xml = ""
    if merge_ranges:
        merge_xml = '<mergeCells count="{}">{}</mergeCells>'.format(
            len(merge_ranges),
            "".join([f'<mergeCell ref="{escape(ref)}"/>' for ref in merge_ranges]),
        )

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        '<sheetData>'
        + "".join(sheet_rows_xml) +
        '</sheetData>'
        + merge_xml +
        '</worksheet>'
    )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="summary" sheetId="1" r:id="rId1"/></sheets>'
        '</workbook>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '</Relationships>'
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )

    with zipfile.ZipFile(out_xlsx, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml)
        zf.writestr("_rels/.rels", rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)

    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export IOI hypothesis+LLM-compare summary table.")
    p.add_argument(
        "--results-root",
        default="/home/wangziran/eap_auto/results/ioi_0301",
        help="Root with hypothesis/ subdir.",
    )
    p.add_argument(
        "--reference-root",
        default="/home/wangziran/eap_auto/results/ioi_0126/answer",
        help="Directory containing NMH.txt/NNMH.txt/SIH.txt/DTH.txt.",
    )
    p.add_argument(
        "--out-csv",
        default="",
        help="Output CSV path. Default: <results-root>/summary/ioi_compare_summary.csv",
    )
    p.add_argument(
        "--out-xlsx",
        default="",
        help="Output XLSX path (optional). Default: <results-root>/summary/ioi_compare_summary.xlsx",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    reference_root = Path(args.reference_root)

    out_csv = Path(args.out_csv) if args.out_csv else results_root / "summary" / "ioi_compare_summary.csv"
    out_xlsx = Path(args.out_xlsx) if args.out_xlsx else results_root / "summary" / "ioi_compare_summary.xlsx"

    rows = build_rows(results_root, reference_root)
    write_csv(rows, out_csv)
    print(f"✅ CSV written: {out_csv}")

    ok_xlsx = write_xlsx_if_available(rows, out_xlsx)
    if ok_xlsx:
        print(f"✅ XLSX written: {out_xlsx}")


if __name__ == "__main__":
    main()
