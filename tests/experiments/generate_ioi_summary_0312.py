#!/usr/bin/env python3
import csv
import json
import os
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from xml.sax.saxutils import escape

ROOT = "/home/wangziran/eap_auto/results/ioi_0312/hypothesis"
OUT_DIR = "/home/wangziran/eap_auto/results/ioi_0312/summary"
os.makedirs(OUT_DIR, exist_ok=True)

FAMILY_MAP = {
    "Name_Mover_Head": "NMH",
    "Negative_Name_Mover_Head": "NNMH",
    "Middle_Head": "SIH",
    "Middle_Head_Plus": "DTH",
}

STANDARD_EXPLANATIONS = {
    "NMH": "This attention head outputs the remaining name. It is active at END, attends to previous names in the sentence, and copies the names it attends to.",
    "NNMH": "This attention head is active at END, attends to previous names in the sentence, but decreases the confidence of the predictions, which means it writes in the opposite direction of the Name Mover Heads that attend to copy the indirect object to END.",
    "SIH": "S-Inhibition Heads mediate clause-structure or participant-role signals that influence downstream IO-relevant heads. They are explained through how they causally change downstream attention and what direct tokens they attend to.",
    "DTH": "Duplicate Token Heads identify tokens that have already appeared in the sentence. They are active at the S2 token, attend primarily to the S1 token, and signal that token duplication has occurred by writing the position of the duplicate token.",
}


def col_letter(idx: int) -> str:
    s = ""
    n = idx + 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def excel_safe(value):
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@"):
        return "'" + value
    return value


def write_xlsx(path, sheets):
    # sheets: list[(name, rows)] rows is list[list[str/num]]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", """<?xml version='1.0' encoding='UTF-8'?>
<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>
  <Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/>
  <Default Extension='xml' ContentType='application/xml'/>
  <Override PartName='/xl/workbook.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml'/>
  <Override PartName='/docProps/core.xml' ContentType='application/vnd.openxmlformats-package.core-properties+xml'/>
  <Override PartName='/docProps/app.xml' ContentType='application/vnd.openxmlformats-officedocument.extended-properties+xml'/>
  %s
</Types>""" % "\n  ".join(
            f"<Override PartName='/xl/worksheets/sheet{i+1}.xml' ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml'/>"
            for i in range(len(sheets))
        ))
        zf.writestr("_rels/.rels", """<?xml version='1.0' encoding='UTF-8'?>
<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>
  <Relationship Id='rId1' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument' Target='xl/workbook.xml'/>
  <Relationship Id='rId2' Type='http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties' Target='docProps/core.xml'/>
  <Relationship Id='rId3' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties' Target='docProps/app.xml'/>
</Relationships>""")
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        zf.writestr("docProps/core.xml", f"""<?xml version='1.0' encoding='UTF-8'?>
<cp:coreProperties xmlns:cp='http://schemas.openxmlformats.org/package/2006/metadata/core-properties' xmlns:dc='http://purl.org/dc/elements/1.1/' xmlns:dcterms='http://purl.org/dc/terms/' xmlns:dcmitype='http://purl.org/dc/dcmitype/' xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance'>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type='dcterms:W3CDTF'>{now}</dcterms:created>
  <dcterms:modified xsi:type='dcterms:W3CDTF'>{now}</dcterms:modified>
</cp:coreProperties>""")
        zf.writestr("docProps/app.xml", f"""<?xml version='1.0' encoding='UTF-8'?>
<Properties xmlns='http://schemas.openxmlformats.org/officeDocument/2006/extended-properties' xmlns:vt='http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes'>
  <Application>Codex</Application>
  <TitlesOfParts><vt:vector size='{len(sheets)}' baseType='lpstr'>%s</vt:vector></TitlesOfParts>
</Properties>""" % "".join(f"<vt:lpstr>{escape(name)}</vt:lpstr>" for name, _ in sheets))
        workbook_sheets = []
        workbook_rels = []
        for i, (name, rows) in enumerate(sheets, start=1):
            workbook_sheets.append(f"<sheet name='{escape(name)}' sheetId='{i}' r:id='rId{i}'/>")
            workbook_rels.append(f"<Relationship Id='rId{i}' Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet' Target='worksheets/sheet{i}.xml'/>")
            sheet_rows = []
            for r_idx, row in enumerate(rows, start=1):
                cells = []
                for c_idx, value in enumerate(row):
                    ref = f"{col_letter(c_idx)}{r_idx}"
                    if value is None:
                        continue
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        cells.append(f"<c r='{ref}'><v>{value}</v></c>")
                    else:
                        text = escape(str(excel_safe(value)))
                        cells.append(f"<c r='{ref}' t='inlineStr'><is><t xml:space='preserve'>{text}</t></is></c>")
                sheet_rows.append(f"<row r='{r_idx}'>" + "".join(cells) + "</row>")
            zf.writestr(f"xl/worksheets/sheet{i}.xml", """<?xml version='1.0' encoding='UTF-8'?>
<worksheet xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'>
  <sheetData>%s</sheetData>
</worksheet>""" % "".join(sheet_rows))
        zf.writestr("xl/workbook.xml", """<?xml version='1.0' encoding='UTF-8'?>
<workbook xmlns='http://schemas.openxmlformats.org/spreadsheetml/2006/main' xmlns:r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'>
  <sheets>%s</sheets>
</workbook>""" % "".join(workbook_sheets))
        zf.writestr("xl/_rels/workbook.xml.rels", """<?xml version='1.0' encoding='UTF-8'?>
<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>
  %s
</Relationships>""" % "\n  ".join(workbook_rels))


def parse_head(family_dir, run_name):
    if family_dir == "Middle_Head_Plus":
        return run_name.split("_to_", 1)[0]
    return run_name.split("_", 1)[0]


def parse_ts(run_name):
    m = re.search(r"(20\d{6}_\d{4})", run_name)
    return m.group(1) if m else ""


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_latest_complete_runs():
    latest = {}
    for family_dir in sorted(os.listdir(ROOT)):
        family_path = os.path.join(ROOT, family_dir)
        if not os.path.isdir(family_path):
            continue
        for run_name in sorted(os.listdir(family_path)):
            run_path = os.path.join(family_path, run_name)
            if not os.path.isdir(run_path):
                continue
            needed = [
                os.path.join(run_path, "initial_test_results.json"),
                os.path.join(run_path, "test_results.json"),
                os.path.join(run_path, "validation_results.json"),
                os.path.join(run_path, "best_hypothesis.json"),
            ]
            if not all(os.path.exists(p) for p in needed):
                continue
            head = parse_head(family_dir, run_name)
            key = (family_dir, head)
            ts = parse_ts(run_name)
            if key not in latest or ts > latest[key][0]:
                latest[key] = (ts, run_path)
    return {k: v[1] for k, v in latest.items()}


def get_scores(obj):
    return obj.get("validation_scores", {}) if isinstance(obj, dict) else {}


def build_initial_final_rows(latest_runs):
    headers = [
        "head_type", "standard_explanation", "head", "run_dir", "best_iteration",
        "initial_hypothesis", "initial_causal_f1", "initial_att_f1", "initial_composite_f1",
        "final_hypothesis", "final_causal_f1", "final_att_f1", "final_composite_f1",
        "best_hypothesis", "best_val_causal_f1", "best_val_att_f1", "best_val_composite_f1",
        "delta_causal_final_minus_initial", "delta_att_final_minus_initial", "delta_composite_final_minus_initial",
        "accept_count", "rollback_count", "candidate_count"
    ]
    rows = [headers]
    data_rows = []
    for (family_dir, head), run_path in sorted(latest_runs.items(), key=lambda x: (FAMILY_MAP.get(x[0][0], x[0][0]), x[0][1])):
        head_type = FAMILY_MAP.get(family_dir, family_dir)
        init = load_json(os.path.join(run_path, "initial_test_results.json"))
        final = load_json(os.path.join(run_path, "test_results.json"))
        best = load_json(os.path.join(run_path, "best_hypothesis.json"))
        val = load_json(os.path.join(run_path, "validation_results.json"))
        init_sc = get_scores(init)
        final_sc = get_scores(final)
        best_sc = best.get("validation_scores", {})
        accept_count = sum(1 for x in val if x.get("decision") == "accept_current")
        rollback_count = sum(1 for x in val if x.get("decision") == "rollback_to_previous_validation")
        candidate_count = sum(1 for x in val if str(x.get("label", "")).startswith("validation_epoch_0_candidate_"))
        row = {
            "head_type": head_type,
            "standard_explanation": STANDARD_EXPLANATIONS.get(head_type, ""),
            "head": head,
            "run_dir": run_path,
            "best_iteration": best.get("iteration", ""),
            "initial_hypothesis": init.get("hypothesis", ""),
            "initial_causal_f1": init_sc.get("causal_f1", ""),
            "initial_att_f1": init_sc.get("direct_attention_f1", init_sc.get("attention_f1", "")),
            "initial_composite_f1": init_sc.get("composite_score", init_sc.get("composite_f1", "")),
            "final_hypothesis": final.get("hypothesis", ""),
            "final_causal_f1": final_sc.get("causal_f1", ""),
            "final_att_f1": final_sc.get("direct_attention_f1", final_sc.get("attention_f1", "")),
            "final_composite_f1": final_sc.get("composite_score", final_sc.get("composite_f1", "")),
            "best_hypothesis": best.get("best_hypothesis", ""),
            "best_val_causal_f1": best_sc.get("causal_f1", ""),
            "best_val_att_f1": best_sc.get("direct_attention_f1", best_sc.get("attention_f1", "")),
            "best_val_composite_f1": best_sc.get("composite_score", best_sc.get("composite_f1", "")),
            "accept_count": accept_count,
            "rollback_count": rollback_count,
            "candidate_count": candidate_count,
        }
        def to_num(v):
            try:
                return float(v)
            except Exception:
                return None
        ic, ia, ip = map(to_num, [row['initial_causal_f1'], row['initial_att_f1'], row['initial_composite_f1']])
        fc, fa, fp = map(to_num, [row['final_causal_f1'], row['final_att_f1'], row['final_composite_f1']])
        row['delta_causal_final_minus_initial'] = "" if ic is None or fc is None else fc - ic
        row['delta_att_final_minus_initial'] = "" if ia is None or fa is None else fa - ia
        row['delta_composite_final_minus_initial'] = "" if ip is None or fp is None else fp - ip
        data_rows.append(row)
        rows.append([row[h] for h in headers])
    return headers, data_rows, rows


def build_review_rows(latest_runs):
    headers = [
        "head_type", "head", "run_dir", "row_type", "label_or_epoch", "decision",
        "causal_f1", "attention_f1", "composite_f1",
        "causal_increase_f1", "causal_decrease_f1", "attention_ndcg",
        "hypothesis", "hypothesis_before", "hypothesis_after",
        "sampled_sentence_ids", "sentence_ids", "candidate_index",
        "refinement_reasoning"
    ]
    rows = [headers]
    for (family_dir, head), run_path in sorted(latest_runs.items(), key=lambda x: (FAMILY_MAP.get(x[0][0], x[0][0]), x[0][1])):
        head_type = FAMILY_MAP.get(family_dir, family_dir)
        val = load_json(os.path.join(run_path, "validation_results.json"))
        for item in val:
            sc = get_scores(item)
            rows.append([
                head_type, head, run_path, "validation", item.get("label", ""), item.get("decision", ""),
                sc.get("causal_f1", ""),
                sc.get("direct_attention_f1", sc.get("attention_f1", "")),
                sc.get("composite_score", sc.get("composite_f1", "")),
                sc.get("causal_increase_f1", ""),
                sc.get("causal_decrease_f1", ""),
                sc.get("direct_attention_ndcg", sc.get("attention_ndcg", "")),
                item.get("hypothesis", ""),
                "", "",
                json.dumps(item.get("sampled_sentence_ids", []), ensure_ascii=False),
                json.dumps(item.get("sentence_ids", []), ensure_ascii=False),
                item.get("candidate_index", ""),
                item.get("hypothesis_analysis", ""),
            ])
        iter_dir = os.path.join(run_path, "iteration_results")
        if os.path.isdir(iter_dir):
            files = sorted(
                [f for f in os.listdir(iter_dir) if f.startswith("iteration_") and f.endswith(".json")],
                key=lambda x: int(re.search(r"(\d+)", x).group(1))
            )
            for fname in files:
                item = load_json(os.path.join(iter_dir, fname))
                scores = item.get("scores", {})
                rows.append([
                    head_type, head, run_path, "iteration", item.get("epoch", ""), "",
                    scores.get("causal_f1", item.get("causal_f1", "")),
                    scores.get("attention_f1", item.get("attention_f1", "")),
                    "",
                    item.get("causal_increase_f1", ""),
                    item.get("causal_decrease_f1", ""),
                    item.get("attention_ndcg", ""),
                    item.get("hypothesis", ""),
                    item.get("hypothesis_before", ""),
                    item.get("hypothesis_after", ""),
                    "", "", "",
                    item.get("refinement_reasoning", item.get("hypothesis_analysis", "")),
                ])
    return rows


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow([excel_safe(v) for v in row])


def main():
    latest_runs = choose_latest_complete_runs()
    initial_headers, initial_data_rows, initial_rows = build_initial_final_rows(latest_runs)
    review_rows = build_review_rows(latest_runs)

    initial_csv = os.path.join(OUT_DIR, 'ioi_0312_initial_final_summary.csv')
    initial_xlsx = os.path.join(OUT_DIR, 'ioi_0312_initial_final_summary.xlsx')
    review_csv = os.path.join(OUT_DIR, 'ioi_0312_epoch_hypothesis_refine_review.csv')
    review_xlsx = os.path.join(OUT_DIR, 'ioi_0312_epoch_hypothesis_refine_review.xlsx')

    write_csv(initial_csv, initial_rows)
    write_csv(review_csv, review_rows)
    write_xlsx(initial_xlsx, [('initial_final_summary', initial_rows)])
    write_xlsx(review_xlsx, [('epoch_refine_review', review_rows)])

    manifest = {
        'latest_runs_used': {f'{FAMILY_MAP.get(k[0], k[0])}:{k[1]}': v for k, v in sorted(latest_runs.items())},
        'initial_final_csv': initial_csv,
        'initial_final_xlsx': initial_xlsx,
        'review_csv': review_csv,
        'review_xlsx': review_xlsx,
        'row_count_initial_final': len(initial_rows) - 1,
        'row_count_review': len(review_rows) - 1,
    }
    with open(os.path.join(OUT_DIR, 'ioi_0312_summary_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
