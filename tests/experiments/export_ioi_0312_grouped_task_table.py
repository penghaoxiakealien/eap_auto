#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from xml.sax.saxutils import escape


SUMMARY_DIR = Path("/home/wangziran/eap_auto/results/ioi_0312/summary")
SOURCE_CSV = SUMMARY_DIR / "ioi_0312_initial_final_summary_with_single_task.csv"


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


def write_xlsx(rows: List[Dict[str, object]], headers: List[str], out_xlsx: Path) -> None:
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    all_rows: List[List[object]] = [headers]
    for r in rows:
        all_rows.append([r.get(h, "") for h in headers])

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
        '<sheets><sheet name="grouped" sheetId="1" r:id="rId1"/></sheets></workbook>'
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


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def blank_group_row(head_type: str, head: str, standard_explanation: str) -> Dict[str, object]:
    row: Dict[str, object] = {
        "head_type": head_type,
        "head": head,
        "standard_explanation": standard_explanation,
    }
    for prefix in ["all", "causal", "attention"]:
        for suffix in [
            "status",
            "run_label",
            "run_dir",
            "best_iteration",
            "initial_causal_f1",
            "initial_att_f1",
            "initial_composite_f1",
            "final_causal_f1",
            "final_att_f1",
            "final_composite_f1",
            "best_val_causal_f1",
            "best_val_att_f1",
            "best_val_composite_f1",
            "delta_composite_final_minus_initial",
            "candidate_count",
            "accept_count",
            "rollback_count",
            "causal_direction",
            "initial_hypothesis",
            "final_hypothesis",
            "best_hypothesis",
        ]:
            row[f"{prefix}_{suffix}"] = ""
    return row


def build_grouped_rows(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], Dict[str, object]] = {}
    for r in rows:
        key = (r.get("head_type", ""), r.get("head", ""))
        if key not in grouped:
            grouped[key] = blank_group_row(key[0], key[1], r.get("standard_explanation", ""))

        if r.get("source_table") == "ioi_0312_initial_final_summary":
            prefix = "all"
        elif r.get("optimize_only") == "causal":
            prefix = "causal"
        elif r.get("optimize_only") == "attention":
            prefix = "attention"
        else:
            continue

        target = grouped[key]
        for suffix in [
            "status",
            "run_label",
            "run_dir",
            "best_iteration",
            "initial_causal_f1",
            "initial_att_f1",
            "initial_composite_f1",
            "final_causal_f1",
            "final_att_f1",
            "final_composite_f1",
            "best_val_causal_f1",
            "best_val_att_f1",
            "best_val_composite_f1",
            "delta_composite_final_minus_initial",
            "candidate_count",
            "accept_count",
            "rollback_count",
            "causal_direction",
            "initial_hypothesis",
            "final_hypothesis",
            "best_hypothesis",
        ]:
            source_key = suffix
            target[f"{prefix}_{suffix}"] = r.get(source_key, "")
    return sorted(grouped.values(), key=lambda x: (x["head_type"], x["head"]))


def write_csv(rows: List[Dict[str, object]], headers: List[str], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", default=str(SOURCE_CSV))
    parser.add_argument("--output-prefix", default=str(SUMMARY_DIR / "ioi_0312_grouped_by_head"))
    args = parser.parse_args()

    rows = load_rows(Path(args.source_csv))
    grouped_rows = build_grouped_rows(rows)
    headers = list(grouped_rows[0].keys()) if grouped_rows else ["head_type", "head", "standard_explanation"]

    out_prefix = Path(args.output_prefix)
    out_csv = out_prefix.with_suffix(".csv")
    out_xlsx = out_prefix.with_suffix(".xlsx")
    write_csv(grouped_rows, headers, out_csv)
    write_xlsx(grouped_rows, headers, out_xlsx)
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_xlsx}")


if __name__ == "__main__":
    main()
