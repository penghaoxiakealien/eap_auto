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


RESULTS_ROOT = Path("/home/wangziran/eap_auto/results/ioi_0312")
SUMMARY_DIR = RESULTS_ROOT / "summary"
OLD_SUMMARY_CSV = SUMMARY_DIR / "ioi_0312_initial_final_summary.csv"

EXPECTED_RUNS = [
    ("NMH", "9.6", "causal"),
    ("NMH", "9.6", "attention"),
    ("NMH", "9.9", "causal"),
    ("NMH", "9.9", "attention"),
    ("NMH", "10.0", "causal"),
    ("NMH", "10.0", "attention"),
    ("NNMH", "10.7", "causal"),
    ("NNMH", "10.7", "attention"),
    ("NNMH", "11.10", "causal"),
    ("NNMH", "11.10", "attention"),
    ("SIH", "7.9", "causal"),
    ("SIH", "7.9", "attention"),
    ("SIH", "8.6", "causal"),
    ("SIH", "8.6", "attention"),
    ("SIH", "8.10", "causal"),
    ("SIH", "8.10", "attention"),
    ("DTH", "0.1", "causal"),
    ("DTH", "0.1", "attention"),
    ("DTH", "0.10", "causal"),
    ("DTH", "0.10", "attention"),
    ("DTH", "3.0", "causal"),
    ("DTH", "3.0", "attention"),
]

FIELDS = [
    "source_table",
    "head_type",
    "standard_explanation",
    "head",
    "run_label",
    "optimize_only",
    "causal_direction",
    "status",
    "run_dir",
    "best_iteration",
    "initial_hypothesis",
    "initial_causal_f1",
    "initial_att_f1",
    "initial_composite_f1",
    "final_hypothesis",
    "final_causal_f1",
    "final_att_f1",
    "final_composite_f1",
    "best_hypothesis",
    "best_val_causal_f1",
    "best_val_att_f1",
    "best_val_composite_f1",
    "delta_causal_final_minus_initial",
    "delta_att_final_minus_initial",
    "delta_composite_final_minus_initial",
    "accept_count",
    "rollback_count",
    "candidate_count",
]


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _score_triplet(d: Optional[dict]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not isinstance(d, dict):
        return None, None, None
    scores = d.get("validation_scores", d)
    if not isinstance(scores, dict):
        return None, None, None
    causal = scores.get("causal_f1")
    att = scores.get("attention_f1", scores.get("direct_attention_f1"))
    comp = scores.get("composite_score", scores.get("composite_f1"))
    if comp is None and causal is not None and att is not None:
        try:
            c = float(causal)
            a = float(att)
            comp = math.sqrt(c * a) if c >= 0 and a >= 0 else None
        except Exception:
            comp = None
    return causal, att, comp


def _to_float_or_blank(v):
    if v is None or v == "":
        return ""
    try:
        return float(v)
    except Exception:
        return v


def _delta(a, b):
    if a in ("", None) or b in ("", None):
        return ""
    try:
        return float(a) - float(b)
    except Exception:
        return ""


def _get_standard_explanations() -> Dict[str, str]:
    out: Dict[str, str] = {}
    with OLD_SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            head_type = (row.get("head_type") or "").strip()
            text = (row.get("standard_explanation") or "").strip()
            if head_type and text and head_type not in out:
                out[head_type] = text
    return out


def _latest_matching_dir(head_type: str, head: str, optimize_only: str) -> Optional[Path]:
    if head_type == "NMH":
        family_dir = RESULTS_ROOT / "hypothesis" / "Name_Mover_Head"
        pattern = f"{head}_{optimize_only}*"
    elif head_type == "NNMH":
        family_dir = RESULTS_ROOT / "hypothesis" / "Negative_Name_Mover_Head"
        pattern = f"{head}_{optimize_only}*"
    elif head_type == "SIH":
        family_dir = RESULTS_ROOT / "hypothesis" / "Middle_Head"
        pattern = f"{head}_{optimize_only}_decrease_*"
    else:
        family_dir = RESULTS_ROOT / "hypothesis" / "Middle_Head_Plus"
        pattern = f"{head}_to_9_6__9_9__10_0_{optimize_only}_decrease_*"
    matches = sorted(
        [p for p in family_dir.glob(pattern) if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _build_single_task_rows() -> List[Dict[str, object]]:
    explanations = _get_standard_explanations()
    rows: List[Dict[str, object]] = []

    for head_type, head, optimize_only in EXPECTED_RUNS:
        run_dir = _latest_matching_dir(head_type, head, optimize_only)
        row: Dict[str, object] = {
            "source_table": "single_task_runs",
            "head_type": head_type,
            "standard_explanation": explanations.get(head_type, ""),
            "head": head,
            "run_label": "",
            "optimize_only": optimize_only,
            "causal_direction": "decrease" if head_type in {"SIH", "DTH"} else "",
            "status": "missing_run",
            "run_dir": "",
            "best_iteration": "",
            "initial_hypothesis": "",
            "initial_causal_f1": "",
            "initial_att_f1": "",
            "initial_composite_f1": "",
            "final_hypothesis": "",
            "final_causal_f1": "",
            "final_att_f1": "",
            "final_composite_f1": "",
            "best_hypothesis": "",
            "best_val_causal_f1": "",
            "best_val_att_f1": "",
            "best_val_composite_f1": "",
            "delta_causal_final_minus_initial": "",
            "delta_att_final_minus_initial": "",
            "delta_composite_final_minus_initial": "",
            "accept_count": "",
            "rollback_count": "",
            "candidate_count": "",
        }
        if run_dir is None:
            rows.append(row)
            continue

        row["run_label"] = run_dir.name
        row["run_dir"] = str(run_dir)
        final_result = _load_json(run_dir / "final_result_all_rounds.json") or {}
        test_result = _load_json(run_dir / "test_results.json")
        initial_test = _load_json(run_dir / "initial_test_results.json")
        best_hyp = _load_json(run_dir / "best_hypothesis.json")
        validation_results = _load_json(run_dir / "validation_results.json")

        # NMH/NNMH single-task runs do not currently write final_result_all_rounds.json.
        # Treat the run as complete as long as the core evaluation artifacts exist.
        complete = bool(test_result and initial_test and best_hyp)
        row["status"] = "complete" if complete else "incomplete"
        row["best_iteration"] = final_result.get("best_validation_iteration", "")

        init_c, init_a, init_comp = _score_triplet(initial_test)
        fin_c, fin_a, fin_comp = _score_triplet(test_result)
        best_scores = (best_hyp or {}).get("validation_scores", {}) if isinstance(best_hyp, dict) else {}
        best_c = best_scores.get("causal_f1", "")
        best_a = best_scores.get("attention_f1", best_scores.get("direct_attention_f1", ""))
        best_comp = best_scores.get("composite_score", best_scores.get("composite_f1", ""))

        row["initial_hypothesis"] = (initial_test or {}).get("hypothesis", "")
        row["initial_causal_f1"] = _to_float_or_blank(init_c)
        row["initial_att_f1"] = _to_float_or_blank(init_a)
        row["initial_composite_f1"] = _to_float_or_blank(init_comp)
        row["final_hypothesis"] = (test_result or {}).get("hypothesis", "")
        row["final_causal_f1"] = _to_float_or_blank(fin_c)
        row["final_att_f1"] = _to_float_or_blank(fin_a)
        row["final_composite_f1"] = _to_float_or_blank(fin_comp)
        row["best_hypothesis"] = (best_hyp or {}).get("best_hypothesis", "")
        row["best_val_causal_f1"] = _to_float_or_blank(best_c)
        row["best_val_att_f1"] = _to_float_or_blank(best_a)
        row["best_val_composite_f1"] = _to_float_or_blank(best_comp)
        row["delta_causal_final_minus_initial"] = _delta(row["final_causal_f1"], row["initial_causal_f1"])
        row["delta_att_final_minus_initial"] = _delta(row["final_att_f1"], row["initial_att_f1"])
        row["delta_composite_final_minus_initial"] = _delta(row["final_composite_f1"], row["initial_composite_f1"])

        if isinstance(validation_results, list):
            row["accept_count"] = sum(1 for x in validation_results if x.get("decision") == "accept_current")
            row["rollback_count"] = sum(1 for x in validation_results if x.get("decision") == "rollback_to_previous_validation")
            row["candidate_count"] = sum(1 for x in validation_results if x.get("candidate_index") is not None)

        if isinstance(final_result, dict):
            row["causal_direction"] = final_result.get("causal_direction", row["causal_direction"])

        rows.append(row)

    return rows


def _build_combined_rows(single_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    combined: List[Dict[str, object]] = []
    with OLD_SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row = {k: "" for k in FIELDS}
            row.update(
                {
                    "source_table": "ioi_0312_initial_final_summary",
                    "head_type": r.get("head_type", ""),
                    "standard_explanation": r.get("standard_explanation", ""),
                    "head": r.get("head", ""),
                    "run_label": Path(r.get("run_dir", "")).name if r.get("run_dir") else "",
                    "optimize_only": "",
                    "causal_direction": "",
                    "status": "complete",
                    "run_dir": r.get("run_dir", ""),
                    "best_iteration": r.get("best_iteration", ""),
                    "initial_hypothesis": r.get("initial_hypothesis", ""),
                    "initial_causal_f1": r.get("initial_causal_f1", ""),
                    "initial_att_f1": r.get("initial_att_f1", ""),
                    "initial_composite_f1": r.get("initial_composite_f1", ""),
                    "final_hypothesis": r.get("final_hypothesis", ""),
                    "final_causal_f1": r.get("final_causal_f1", ""),
                    "final_att_f1": r.get("final_att_f1", ""),
                    "final_composite_f1": r.get("final_composite_f1", ""),
                    "best_hypothesis": r.get("best_hypothesis", ""),
                    "best_val_causal_f1": r.get("best_val_causal_f1", ""),
                    "best_val_att_f1": r.get("best_val_att_f1", ""),
                    "best_val_composite_f1": r.get("best_val_composite_f1", ""),
                    "delta_causal_final_minus_initial": r.get("delta_causal_final_minus_initial", ""),
                    "delta_att_final_minus_initial": r.get("delta_att_final_minus_initial", ""),
                    "delta_composite_final_minus_initial": r.get("delta_composite_final_minus_initial", ""),
                    "accept_count": r.get("accept_count", ""),
                    "rollback_count": r.get("rollback_count", ""),
                    "candidate_count": r.get("candidate_count", ""),
                }
            )
            combined.append(row)
    combined.extend(single_rows)
    return combined


def write_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
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


def write_xlsx(rows: List[Dict[str, object]], out_xlsx: Path) -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    all_rows: List[List[object]] = [FIELDS]
    for r in rows:
        all_rows.append([r.get(h, "") for h in FIELDS])

    sheet_rows_xml: List[str] = []
    for r_idx, row in enumerate(all_rows, start=1):
        cells = []
        for c_idx, val in enumerate(row, start=1):
            ref = f"{_col_name(c_idx)}{r_idx}"
            cells.append(_xlsx_cell_xml(ref, val))
        sheet_rows_xml.append(f'<row r="{r_idx}">{"".join(cells)}</row>')

    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
        '<sheetData>' + "".join(sheet_rows_xml) + '</sheetData>'
        '</worksheet>'
    )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="summary" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        '</Relationships>'
    )
    root_rels_xml = (
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
        zf.writestr("_rels/.rels", root_rels_xml)
        zf.writestr("xl/workbook.xml", workbook_xml)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-dir", default=str(SUMMARY_DIR))
    args = parser.parse_args()

    summary_dir = Path(args.summary_dir)
    single_rows = _build_single_task_rows()
    combined_rows = _build_combined_rows(single_rows)

    single_csv = summary_dir / "ioi_0312_single_task_decrease_summary.csv"
    single_xlsx = summary_dir / "ioi_0312_single_task_decrease_summary.xlsx"
    combined_csv = summary_dir / "ioi_0312_initial_final_summary_with_single_task.csv"
    combined_xlsx = summary_dir / "ioi_0312_initial_final_summary_with_single_task.xlsx"

    write_csv(single_rows, single_csv)
    write_xlsx(single_rows, single_xlsx)
    write_csv(combined_rows, combined_csv)
    write_xlsx(combined_rows, combined_xlsx)

    print(f"Wrote: {single_csv}")
    print(f"Wrote: {single_xlsx}")
    print(f"Wrote: {combined_csv}")
    print(f"Wrote: {combined_xlsx}")


if __name__ == "__main__":
    main()
