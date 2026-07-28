"""
Table-lookup evaluation on the 2024 ground truth (NO LLM).

Scores hazard_label and hazard_category_label against the GT file, split into
has_hazards / no_hazards / overall, for both label and category.

Both branches are pure table lookups:
  - has_hazards -> hazard_eval.rule_predict_has_hazards (table-lookup engine + canon937)
  - no_hazards  -> hazard_eval.infer_no_hazards (n-gram keyword table-lookup over
                   no_hazards_mapping1.json / no_hazards_mapping2.json)
The shared post-pipeline (canon_filter -> post_process -> canon937 -> derive_categories)
and the Metric are identical across branches, so numbers are directly comparable.

GT file: <RASFF_ROOT>/rasff_2024_ground_truth_labels.json
(carries hazards/subject as input, hazard_label/hazard_category_label as gold labels)

Metrics: exact-match accuracy (full set equality), mean Jaccard, micro precision/recall/F1,
split into has_hazards / no_hazards / overall, computed separately for label and category.
"""
import os
import json

import hazard_eval as E   # has 段 table-lookup + no 段 n-gram 规则，均在此模块内

RASFF_ROOT = os.environ.get("RASFF_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_PATH = os.environ.get(
    "GROUND_TRUTH_PATH",
    os.path.join(RASFF_ROOT, "rasff_2024_ground_truth_labels.json"),
)
OUTPUT_DIR = os.path.join(E.CODE_DIR, "Outputs_hazard_eval_gt2024")


def norm_set(labels):
    return {x.strip().lower() for x in labels if isinstance(x, str) and x.strip()}


class Metric:
    def __init__(self):
        self.n = self.exact = 0
        self.jac = 0.0
        self.tp = self.fp = self.fn = 0

    def add(self, pred, truth):
        self.n += 1
        if pred == truth:
            self.exact += 1
        inter, union = pred & truth, pred | truth
        self.jac += (len(inter) / len(union)) if union else 1.0
        self.tp += len(inter)
        self.fp += len(pred - truth)
        self.fn += len(truth - pred)

    def report(self):
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"n": self.n, "exact_accuracy": round(self.exact / self.n, 4) if self.n else 0.0,
                "mean_jaccard": round(self.jac / self.n, 4) if self.n else 0.0,
                "tp": self.tp, "fp": self.fp, "fn": self.fn, "micro_precision": round(p, 4), "micro_recall": round(r, 4), "micro_f1": round(f1, 4)}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building has_hazards table-lookup engine + 937 label space ...")
    ctx = E.build_context()
    label_list, label_lookup, label_to_cat = E.build_label_space()
    E.build_canon(label_list)

    print("Building no_hazards n-gram table-lookup context ...")
    has_data = E.load_json(E.HAS_HAZARDS_LABELLED_PATH)
    no_data = E.load_json(E.NO_HAZARDS_LABELLED_PATH)
    has_hazard_label_inventory = E.load_json(E.HAS_HAZARDS_MAPPING_HAZARD_LABEL_PATH)
    has_hazard_category_inventory = E.load_json(E.HAS_HAZARDS_MAPPING_HAZARD_CATEGORY_LABEL_PATH)
    no_mapping1_raw = E.load_json(E.NO_HAZARDS_MAPPING1_PATH)
    no_mapping2_raw = E.load_json(E.NO_HAZARDS_MAPPING2_PATH)
    label_to_category_raw = E.load_json(E.MAPPING_LABEL_TO_CATEGORY_PATH)

    allowed_hazard_labels, allowed_hazard_category_labels = E.build_allowed_label_sets(
        has_labelled=has_data, no_labelled=no_data,
        has_hazard_label_inventory=has_hazard_label_inventory,
        has_hazard_category_inventory=has_hazard_category_inventory,
        no_mapping1=no_mapping1_raw, no_mapping2=no_mapping2_raw,
        label_to_category=label_to_category_raw,
    )
    preferred_hazard_lookup = E.build_preferred_lookup_from_gold(has_data + no_data, "hazard_label")
    preferred_hazard_category_lookup = E.build_preferred_lookup_from_gold(has_data + no_data, "hazard_category_label")
    allowed_hazard_lookup = E.merge_preferred_and_allowed(preferred_hazard_lookup, allowed_hazard_labels)
    allowed_hazard_category_lookup = E.merge_preferred_and_allowed(preferred_hazard_category_lookup, allowed_hazard_category_labels)
    label_to_category_map = E.compile_label_to_category_map(
        label_to_category_raw, allowed_hazard_lookup=allowed_hazard_lookup,
        allowed_hazard_category_lookup=allowed_hazard_category_lookup,
    )
    mapping1 = E.compile_ngram_mapping(no_mapping1_raw)
    mapping2 = E.compile_ngram_mapping(no_mapping2_raw)
    pesticide_labels = sorted(
        [lbl for lbl, cat in label_to_category_map.items() if cat == "Pesticide residues"],
        key=lambda x: len(E.normalize_text_for_match(x)), reverse=True,
    )

    def lookup_no_hazards_labels(r):
        labels = E.infer_no_hazards(
            item=r, mapping1=mapping1, mapping2=mapping2,
            label_to_category_map=label_to_category_map, pesticide_labels=pesticide_labels,
            allowed_hazard_lookup=allowed_hazard_lookup, allowed_hazard_labels=allowed_hazard_labels,
            allowed_hazard_category_lookup=allowed_hazard_category_lookup,
            allowed_hazard_category_labels=allowed_hazard_category_labels,
        )[0]
        return labels

    gt = E.load_json(GT_PATH)
    has_n = sum(1 for r in gt if isinstance(r.get("hazards"), list) and len(r.get("hazards")) > 0)
    print(f"GT total={len(gt)} | has_hazards={has_n} | no_hazards={len(gt) - has_n}")

    M = {("label", s): Metric() for s in ("has", "no", "all")}
    M.update({("cat", s): Metric() for s in ("has", "no", "all")})
    mismatches = []

    for r in gt:
        hz = r.get("hazards", [])
        is_has = isinstance(hz, list) and len(hz) > 0

        if is_has:
            labels = E.canon_list(E.rule_predict_has_hazards(r, ctx))
        else:
            labels = E.canon_list(E.post_process_labels(
                E.canon_filter(lookup_no_hazards_labels(r), label_lookup)))
        cats = E.derive_categories(labels, label_to_cat)

        pred_l, truth_l = norm_set(labels), norm_set(r.get("hazard_label", []))
        pred_c, truth_c = norm_set(cats), norm_set(r.get("hazard_category_label", []))

        seg = "has" if is_has else "no"
        M[("label", seg)].add(pred_l, truth_l); M[("label", "all")].add(pred_l, truth_l)
        M[("cat", seg)].add(pred_c, truth_c); M[("cat", "all")].add(pred_c, truth_c)

        if pred_l != truth_l or pred_c != truth_c:
            mismatches.append({"reference": str(r.get("reference", "")), "segment": seg, "subject": r.get("subject", ""),
                               "gold_label": sorted(truth_l), "pred_label": sorted(pred_l),
                               "gold_category": sorted(truth_c), "pred_category": sorted(pred_c)})

    report = {"gt_path": GT_PATH, "total": len(gt), "mode": "TABLE-LOOKUP ONLY (no LLM)",
              "hazard_label": {s: M[("label", s)].report() for s in ("has", "no", "all")},
              "hazard_category_label": {s: M[("cat", s)].report() for s in ("has", "no", "all")}}
    json.dump(report, open(os.path.join(OUTPUT_DIR, "metrics.json"), "w"), ensure_ascii=False, indent=2)
    json.dump(mismatches, open(os.path.join(OUTPUT_DIR, "mismatches.json"), "w"), ensure_ascii=False, indent=2)

    for field in ("hazard_label", "hazard_category_label"):
        print(f"\n==== {field} (TABLE-LOOKUP ONLY) ====")
        hdr = f"{'segment':8} {'n':>6} {'exact_acc':>10} {'jaccard':>8} {'P':>7} {'R':>7} {'F1':>7}"
        print(hdr); print("-" * len(hdr))
        for s, name in [("has", "has_haz"), ("no", "no_haz"), ("all", "OVERALL")]:
            m = report[field][s]
            print(f"{name:8} {m['n']:>6} {m['exact_accuracy']:>10} {m['mean_jaccard']:>8} "
                  f"{m['micro_precision']:>7} {m['micro_recall']:>7} {m['micro_f1']:>7}")
    print(f"\nSaved: {OUTPUT_DIR}/metrics.json + mismatches.json")


if __name__ == "__main__":
    main()
