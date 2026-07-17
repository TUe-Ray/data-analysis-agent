"""Deterministic fully-static cross-study biomarker harmonization task."""
# ruff: noqa: E501, E701, E702
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import shutil
import statistics
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from data_analysis_agent.benchmark_grading import grade_candidate
from data_analysis_agent.benchmark_types import PrivateGradingSpec

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TARGET_ROOT = PROJECT_ROOT / "benchmark_tasks/cross_study_biomarker_harmonization"
TASK_ID = "cross_study_biomarker_harmonization"
SEED = 20260717
PUBLIC_FILES = [
    "protocol/study_alpha_protocol.md", "protocol/study_alpha_amendment_01.md", "protocol/study_beta_protocol.md", "protocol/harmonization_plan.md",
    "documentation/study_alpha_data_dictionary.csv", "documentation/study_beta_data_dictionary.csv", "documentation/value_codebook.csv", "documentation/identifier_crosswalk_dictionary.csv",
    "data/study_alpha_subjects.csv", "data/study_alpha_visits.csv", "data/study_alpha_assays.csv", "data/study_alpha_exclusions.csv",
    "data/study_beta_subjects.csv", "data/study_beta_visits.csv", "data/study_beta_assays.csv", "data/study_beta_exclusions.csv", "data/study_beta_subject_crosswalk.csv",
]

def _csv(fields: list[str], rows: list[dict[str, Any]]) -> str:
    out = io.StringIO(newline=""); writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows); return out.getvalue()

def _finite(value: str) -> float | None:
    try: result = float(value)
    except (ValueError, TypeError): return None
    return result if math.isfinite(result) else None

def _unique(rows: list[dict[str, str]], fields: list[str]) -> tuple[list[dict[str, str]], int]:
    seen: set[tuple[str, ...]] = set(); output = []
    for row in rows:
        key = tuple(row[x] for x in fields)
        if key not in seen: seen.add(key); output.append(row)
    return output, len(rows) - len(output)

def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: output[row[key]].append(row)
    return output

ALPHA = """# Study Alpha base protocol

Canonical analysis identifiers are in study_alpha_subjects.csv; visits use source_subject_id and must map through that table. Eligible subjects are aged 18 through 75 inclusive, consent code C_Y, and physical arms PX17 (A) or PX29 (B). Baseline accepts V only in days -14 through -1 inclusive, target -7. Legacy follow-up accepts V only in days 30 through 45 inclusive, target 38. Deduplicate exact logical visit rows before validation; select nearest target, then higher quality, earlier timestamp, then lexical record ID. Alpha assays: deduplicate exact logical rows, reject nonfinite and QC other than AQ_OK, select highest assay quality then earliest timestamp then lexical assay record ID; never average. Alpha reported values are pg/mL, exactly equal dimensionally to ng/L; Platform X harmonized_value = reported_value * 1.08. Pre-start event_date <= start; legacy post-start upper bound is strict.
"""
AMENDMENT = """# Study Alpha amendment 01

Where this amendment conflicts with the Alpha base protocol, it governs; all unaffected Alpha baseline, eligibility, assay, calibration, identifier, duplicate, and tie-breaking rules remain in force. Follow-up accepts V and R, window day 28 through day 42 inclusive, target day 35. Post-start exclusion is treatment_start < exclusion_date <= selected_followup_date. It does not apply to Study Beta.
"""
BETA = """# Study Beta protocol

Visits and assays use visit_source_id; exclusions use a different event_source_id. Map both using study_beta_subject_crosswalk.csv to the canonical analysis ID; similar-looking identifiers must not be directly joined. Eligible subjects are age 18 through 75, consent B_CONSENT, arms B01 (A) and B02 (B). Only visit status OK is accepted. Baseline is days -21 through -3 inclusive, target -7; follow-up is days 56 through 70 inclusive, target 63. Select nearest target, then higher quality, earlier timestamp, then lexical record ID. Beta assays are ng/L without calibration: deduplicate exact scientific rows, retain finite B_ACCEPT records, require two accepted replicates per specimen, and take their median. Pre-start event <= start; post-start is strict, start < event < selected follow-up.
"""
HARMONIZATION = """# Harmonization plan

Apply study-specific eligibility, visit, assay, and exclusion rules first, convert to ng/L, select one baseline and follow-up value per complete subject, and compute change = followup_harmonized_value - baseline_harmonized_value. Report study/arm n, means, sample SD using n-1, and SE = SD / sqrt(n). Study contrast is mean_change_B - mean_change_A. Variance = sample_variance_A/n_A + sample_variance_B/n_B; weight = 1/variance; pooled contrast = sum(weight * contrast)/sum(weight). Round to six decimals. No p-values, confidence intervals, hypothesis tests, or causal claims.

A non-binding workflow is reconcile rules; normalize eligible cohorts; harmonize assays; select pairs and apply exclusions; calculate study and pooled summaries; assemble exact JSON. Artifacts may be effective_rules.json, eligible_subjects.csv, harmonized_assays.csv, and selected_pairs.csv, but filenames are not required.
"""

def _documents() -> dict[str, str]:
    return {
        "protocol/study_alpha_protocol.md": ALPHA, "protocol/study_alpha_amendment_01.md": AMENDMENT, "protocol/study_beta_protocol.md": BETA, "protocol/harmonization_plan.md": HARMONIZATION,
        "documentation/study_alpha_data_dictionary.csv": "table,column,meaning\nstudy_alpha_subjects,source_subject_id,source participant key\nstudy_alpha_visits,technical_row_id,technical row excluded from logical identity\nstudy_alpha_assays,assay_record_id,scientific replicate ID\n",
        "documentation/study_beta_data_dictionary.csv": "table,column,meaning\nstudy_beta_subject_crosswalk,visit_source_id,visit mapping key\nstudy_beta_exclusions,event_source_id,different exclusion mapping key\nstudy_beta_assays,specimen_id,replicate aggregation key\n",
        "documentation/value_codebook.csv": "field,physical_value,meaning\nalpha_consent,C_Y,consented\nalpha_consent,C_N,not consented\nalpha_status,V,valid\nalpha_status,R,reviewed accepted only by amendment\nalpha_status,RX,similar-looking rejected\nalpha_qc,AQ_OK,accepted\nalpha_qc,AQ_FAIL,rejected\nbeta_consent,B_CONSENT,consented\nbeta_status,OK,accepted\nbeta_status,O_K,similar-looking rejected\nbeta_qc,B_ACCEPT,accepted\nbeta_qc,B_REJECT,rejected\n",
        "documentation/identifier_crosswalk_dictionary.csv": "study,identifier,use\nAlpha,source_subject_id,map against alpha subject table\nBeta,visit_source_id,map visits through crosswalk\nBeta,event_source_id,map exclusions through crosswalk\n",
    }

def _data() -> dict[str, str]:
    start = date(2025, 1, 1); a_sub=[]; a_vis=[]; a_ass=[]; a_exc=[]; b_sub=[]; b_vis=[]; b_ass=[]; b_exc=[]; cross=[]
    def visit(rows, source, rec, day, status, quality, tech):
        rows.append({"source_subject_id":source,"visit_record_id":rec,"day_relative":day,"visit_status_code":status,"visit_quality":quality,"visit_timestamp":f"2025-02-{max(1,min(28,day+15)):02d}T{8+len(rows)%4:02d}:00:00","technical_row_id":tech})
    def alpha_assay(rec, val, invalid=False):
        base={"visit_record_id":rec,"assay_record_id":rec+"-R1","specimen_id":"S-"+rec,"reported_value":"nan" if invalid else f"{val/1.08:.6f}","unit":"pg/mL","platform":"X","qc_code":"AQ_FAIL" if invalid else "AQ_OK","assay_quality":.99 if invalid else .8,"assay_timestamp":"2025-03-01T08:00:00","technical_row_id":"T1"}
        a_ass.extend([base,{**base,"technical_row_id":"T1_DUP"},{**base,"assay_record_id":rec+"-R2","reported_value":f"{(val+9)/1.08:.6f}","qc_code":"AQ_FAIL","assay_quality":1.0,"technical_row_id":"T2"}])
    def beta_assay(rec, val, insufficient=False):
        for n, off in enumerate([0.0] if insufficient else [-.4,.4],1):
            base={"visit_record_id":rec,"specimen_id":"BS-"+rec,"assay_record_id":f"{rec}-R{n}","reported_value":f"{val+off:.6f}","unit":"ng/L","qc_code":"B_ACCEPT","assay_timestamp":f"2025-03-01T0{n}:00:00","technical_row_id":"T1"}; b_ass.extend([base,{**base,"technical_row_id":"T1_DUP"}])
        b_ass.append({"visit_record_id":rec,"specimen_id":"BS-"+rec,"assay_record_id":rec+"-BAD","reported_value":"inf","unit":"ng/L","qc_code":"B_REJECT","assay_timestamp":"2025-03-01T09:00:00","technical_row_id":"TB"})
    for i in range(1,33):
        aid=f"ALPHA-{i:03d}"; src=f"ALPHA-S-{i:03d}"; acode="PX17" if i%2 else "PX29"; a_sub.append({"analysis_subject_id":aid,"source_subject_id":src,"age_years":30+i,"consent_code":"C_N" if i==31 else "C_Y","physical_arm_code":acode,"treatment_start":start.isoformat()})
        base=10+i%7+(acode=="PX29"); follow=base+(1+i%4 if acode=="PX17" else 3+i%5)
        visit(a_vis,src,f"A{i:03d}-B",-7,"V",.8,"T1"); alpha_assay(f"A{i:03d}-B",base,i==28); visit(a_vis,src,f"A{i:03d}-B-BOUND",-14 if i%2 else -1,"V",.5,"T2"); alpha_assay(f"A{i:03d}-B-BOUND",base+8)
        if i!=27: visit(a_vis,src,f"A{i:03d}-F",35,"R" if i==1 else "V",.9,"T3"); alpha_assay(f"A{i:03d}-F",follow,i==25)
        visit(a_vis,src,f"A{i:03d}-F-BOUND",42,"V",.4,"T4"); alpha_assay(f"A{i:03d}-F-BOUND",follow+8)
        if i==2:
            for rec,day in [("A002-F34-A",34),("A002-F36-B",36)]: visit(a_vis,src,rec,day,"V",.7,rec); alpha_assay(rec,follow)
            visit(a_vis,src,"A002-F-BAD",35,"RX",.99,"BAD"); alpha_assay("A002-F-BAD",follow+99)
        if i==29: a_exc.append({"analysis_subject_id":aid,"event_date":start.isoformat(),"event_code":"PRE"})
        if i==26: a_exc.append({"analysis_subject_id":aid,"event_date":(start+timedelta(days=35)).isoformat(),"event_code":"ON_FOLLOWUP"})
        bid=f"BETA-{i:03d}"; vsrc=f"BETA{i:03d}"; esrc=f"BETA-E-{i:03d}"; bcode="B01" if i%3 else "B02"; b_sub.append({"analysis_subject_id":bid,"age_years":25+i,"consent_code":"B_NO" if i==31 else "B_CONSENT","physical_arm_code":bcode,"treatment_start":start.isoformat()}); cross.append({"visit_source_id":vsrc,"event_source_id":esrc,"analysis_subject_id":bid})
        bbase=12+i%6+(bcode=="B02"); bfollow=bbase+(1.5+i%4 if bcode=="B01" else 4+i%5)
        visit(b_vis,vsrc,f"B{i:03d}-B",-7,"OK",.8,"T1"); beta_assay(f"B{i:03d}-B",bbase,i==28); visit(b_vis,vsrc,f"B{i:03d}-B-BOUND",-21 if i%2 else -3,"OK",.5,"T2"); beta_assay(f"B{i:03d}-B-BOUND",bbase+8)
        if i!=27: visit(b_vis,vsrc,f"B{i:03d}-F",63,"OK",.9,"T3"); beta_assay(f"B{i:03d}-F",bfollow,i==25)
        visit(b_vis,vsrc,f"B{i:03d}-F-BOUND",70,"OK",.4,"T4"); beta_assay(f"B{i:03d}-F-BOUND",bfollow+7)
        if i==1:
            for rec,day in [("B001-F62-A",62),("B001-F64-B",64)]: visit(b_vis,vsrc,rec,day,"OK",.7,rec); beta_assay(rec,bfollow)
            visit(b_vis,vsrc,"B001-F-BAD",63,"O_K",.99,"BAD"); beta_assay("B001-F-BAD",bfollow+99)
        if i==29: b_exc.append({"event_source_id":esrc,"event_date":start.isoformat(),"event_code":"PRE"})
        if i==26: b_exc.append({"event_source_id":esrc,"event_date":(start+timedelta(days=63)).isoformat(),"event_code":"ON_FOLLOWUP_STRICT"})
    return {
        "data/study_alpha_subjects.csv":_csv(["analysis_subject_id","source_subject_id","age_years","consent_code","physical_arm_code","treatment_start"],a_sub), "data/study_alpha_visits.csv":_csv(["source_subject_id","visit_record_id","day_relative","visit_status_code","visit_quality","visit_timestamp","technical_row_id"],a_vis), "data/study_alpha_assays.csv":_csv(["visit_record_id","assay_record_id","specimen_id","reported_value","unit","platform","qc_code","assay_quality","assay_timestamp","technical_row_id"],a_ass), "data/study_alpha_exclusions.csv":_csv(["analysis_subject_id","event_date","event_code"],a_exc),
        "data/study_beta_subjects.csv":_csv(["analysis_subject_id","age_years","consent_code","physical_arm_code","treatment_start"],b_sub), "data/study_beta_visits.csv":_csv(["source_subject_id","visit_record_id","day_relative","visit_status_code","visit_quality","visit_timestamp","technical_row_id"],b_vis), "data/study_beta_assays.csv":_csv(["visit_record_id","specimen_id","assay_record_id","reported_value","unit","qc_code","assay_timestamp","technical_row_id"],b_ass), "data/study_beta_exclusions.csv":_csv(["event_source_id","event_date","event_code"],b_exc), "data/study_beta_subject_crosswalk.csv":_csv(["visit_source_id","event_source_id","analysis_subject_id"],cross)}

def _read(root: Path, name: str) -> list[dict[str,str]]:
    with (root/"public"/name).open(encoding="utf-8",newline="") as handle: return list(csv.DictReader(handle))

def compute_oracle(root: Path = TARGET_ROOT) -> dict[str, Any]:
    """Apply public rules to public tables; no private reference is read."""
    all_pairs=[]; attr={}; quality={}; study_stats={}; comparisons={}
    for study in ("alpha","beta"):
        sub=_read(root,f"data/study_{study}_subjects.csv"); vis=_read(root,f"data/study_{study}_visits.csv"); assays=_read(root,f"data/study_{study}_assays.csv"); events=_read(root,f"data/study_{study}_exclusions.csv")
        if study=="alpha": vmap={r["source_subject_id"]:r["analysis_subject_id"] for r in sub}; emap=vmap; arms={"PX17":"A","PX29":"B"}; consent="C_Y"; accepted={"V","R"}; bw,bt,fw,ft=(-14,-1),-7,(28,42),35
        else:
            cw=_read(root,"data/study_beta_subject_crosswalk.csv"); vmap={r["visit_source_id"]:r["analysis_subject_id"] for r in cw}; emap={r["event_source_id"]:r["analysis_subject_id"] for r in cw}; arms={"B01":"A","B02":"B"}; consent="B_CONSENT"; accepted={"OK"}; bw,bt,fw,ft=(-21,-3),-7,(56,70),63
        eligible={r["analysis_subject_id"]:{**r,"arm":arms.get(r["physical_arm_code"],"")} for r in sub if _finite(r["age_years"]) is not None and 18<=float(r["age_years"])<=75 and r["consent_code"]==consent and r["physical_arm_code"] in arms}
        cvis,dv=_unique(vis,["source_subject_id","visit_record_id","day_relative","visit_status_code","visit_quality","visit_timestamp"]); valid_vis=[]
        for r in cvis:
            d=_finite(r["day_relative"]); q=_finite(r["visit_quality"])
            if r["source_subject_id"] in vmap and r["visit_status_code"] in accepted and d is not None and q is not None: valid_vis.append({**r,"sid":vmap[r["source_subject_id"]],"day":d,"q":q})
        cass,da=_unique(assays,[x for x in assays[0] if x!="technical_row_id"]); valid_ass=[]
        for r in cass:
            val=_finite(r["reported_value"]); good=r["qc_code"]==("AQ_OK" if study=="alpha" else "B_ACCEPT") and val is not None
            if good: valid_ass.append({**r,"value":val})
        assay_values={}; insufficient=0
        if study=="alpha":
            for vid,rows in _group(valid_ass,"visit_record_id").items():
                rows=[r for r in rows if _finite(r["assay_quality"]) is not None]
                if rows:
                    chosen=min(rows,key=lambda r:(-float(r["assay_quality"]),r["assay_timestamp"],r["assay_record_id"])); assay_values[vid]=([chosen["assay_record_id"]],chosen["value"]*1.08)
        else:
            for vid,rows in _group(valid_ass,"visit_record_id").items():
                for _,rep in _group(rows,"specimen_id").items():
                    if len(rep)<2: insufficient+=1
                    else: assay_values[vid]=(sorted(r["assay_record_id"] for r in rep),statistics.median(r["value"] for r in rep))
        eby=defaultdict(list)
        for r in events:
            sid=r.get("analysis_subject_id") if study=="alpha" else emap.get(r["event_source_id"])
            if sid: eby[sid].append(date.fromisoformat(r["event_date"]))
        vby=_group([r for r in valid_vis if r["sid"] in eligible],"sid"); pre=nob=nof=post=0; final=[]
        for sid,s in eligible.items():
            start=date.fromisoformat(s["treatment_start"])
            if any(x<=start for x in eby[sid]): pre+=1; continue
            def choose(window,target):
                rows=[r for r in vby.get(sid,[]) if window[0]<=r["day"]<=window[1] and r["visit_record_id"] in assay_values]
                return min(rows,key=lambda r:(abs(r["day"]-target),-r["q"],r["visit_timestamp"],r["visit_record_id"])) if rows else None
            base=choose(bw,bt)
            if not base: nob+=1; continue
            follow=choose(fw,ft)
            if not follow: nof+=1; continue
            fdate=start+timedelta(days=int(follow["day"])); excluded=any(start<x<=fdate for x in eby[sid]) if study=="alpha" else any(start<x<fdate for x in eby[sid])
            if excluded: post+=1; continue
            bi,bv=assay_values[base["visit_record_id"]]; fi,fv=assay_values[follow["visit_record_id"]]; final.append({"study_id":study,"analysis_subject_id":sid,"arm":s["arm"],"baseline_visit_record_id":base["visit_record_id"],"followup_visit_record_id":follow["visit_record_id"],"baseline_assay_record_ids":bi,"followup_assay_record_ids":fi,"baseline_harmonized_value":round(bv,6),"followup_harmonized_value":round(fv,6),"change":round(fv-bv,6)})
        final.sort(key=lambda r:r["analysis_subject_id"]); attr[study]={"total_subjects":len(sub),"basic_ineligible":len(sub)-len(eligible),"eligible_after_basic_checks":len(eligible),"excluded_pre_start":pre,"no_valid_baseline":nob,"no_valid_followup":nof,"excluded_post_start_before_followup":post,"complete_pairs":len(final),"complete_pairs_arm_a":sum(r["arm"]=="A" for r in final),"complete_pairs_arm_b":sum(r["arm"]=="B" for r in final)}; quality[study]={"scientific_duplicate_rows_removed":dv+da,"invalid_visit_rows_excluded":len(cvis)-len(valid_vis),"invalid_assay_rows_excluded":len(cass)-len(valid_ass),"insufficient_replicate_specimens":insufficient}; study_stats[study]={}
        for arm in ("A","B"):
            rows=[r for r in final if r["arm"]==arm]; changes=[r["change"] for r in rows]; sd=statistics.stdev(changes); study_stats[study][arm]={"n":len(rows),"mean_baseline":round(statistics.mean(r["baseline_harmonized_value"] for r in rows),6),"mean_followup":round(statistics.mean(r["followup_harmonized_value"] for r in rows),6),"mean_change":round(statistics.mean(changes),6),"sample_sd_change":round(sd,6),"sample_se_change":round(sd/math.sqrt(len(rows)),6)}
        a,b=study_stats[study]["A"],study_stats[study]["B"]; var=a["sample_sd_change"]**2/a["n"]+b["sample_sd_change"]**2/b["n"]; comparisons[study]={"difference_in_mean_change_b_minus_a":round(b["mean_change"]-a["mean_change"],6),"variance_of_difference":round(var,6),"inverse_variance_weight":round(1/var,6)}; all_pairs.extend(final)
    ws=[comparisons[x]["inverse_variance_weight"] for x in ("alpha","beta")]; pooled=sum(comparisons[x]["difference_in_mean_change_b_minus_a"]*comparisons[x]["inverse_variance_weight"] for x in ("alpha","beta"))/sum(ws)
    return {"status":"completed","answer":"Descriptive cross-study biomarker harmonization completed from all public evidence.","key_results":{"study_attrition":attr,"study_statistics":study_stats,"study_between_arm_comparisons":comparisons,"pooled_comparison":{"pooled_difference_in_mean_change_b_minus_a":round(pooled,6),"sum_of_inverse_variance_weights":round(sum(ws),6)},"selected_pairs":sorted(all_pairs,key=lambda r:(r["study_id"],r["analysis_subject_id"])),"data_quality_audit":quality},"limitations":["Descriptive harmonization only; no causal claim or hypothesis test."]}

_number = {"type": "number"}
_attrition = {"type": "object", "additionalProperties": False, "required": ["total_subjects", "basic_ineligible", "eligible_after_basic_checks", "excluded_pre_start", "no_valid_baseline", "no_valid_followup", "excluded_post_start_before_followup", "complete_pairs", "complete_pairs_arm_a", "complete_pairs_arm_b"], "properties": {name: {"type": "integer"} for name in ["total_subjects", "basic_ineligible", "eligible_after_basic_checks", "excluded_pre_start", "no_valid_baseline", "no_valid_followup", "excluded_post_start_before_followup", "complete_pairs", "complete_pairs_arm_a", "complete_pairs_arm_b"]}}
_arm_stats = {"type": "object", "additionalProperties": False, "required": ["n", "mean_baseline", "mean_followup", "mean_change", "sample_sd_change", "sample_se_change"], "properties": {"n": {"type": "integer"}, "mean_baseline": _number, "mean_followup": _number, "mean_change": _number, "sample_sd_change": _number, "sample_se_change": _number}}
_study_stats = {"type": "object", "additionalProperties": False, "required": ["A", "B"], "properties": {"A": _arm_stats, "B": _arm_stats}}
_comparison = {"type": "object", "additionalProperties": False, "required": ["difference_in_mean_change_b_minus_a", "variance_of_difference", "inverse_variance_weight"], "properties": {"difference_in_mean_change_b_minus_a": _number, "variance_of_difference": _number, "inverse_variance_weight": _number}}
_quality = {"type": "object", "additionalProperties": False, "required": ["scientific_duplicate_rows_removed", "invalid_visit_rows_excluded", "invalid_assay_rows_excluded", "insufficient_replicate_specimens"], "properties": {name: {"type": "integer"} for name in ["scientific_duplicate_rows_removed", "invalid_visit_rows_excluded", "invalid_assay_rows_excluded", "insufficient_replicate_specimens"]}}
_pair = {"type": "object", "additionalProperties": False, "required": ["study_id", "analysis_subject_id", "arm", "baseline_visit_record_id", "followup_visit_record_id", "baseline_assay_record_ids", "followup_assay_record_ids", "baseline_harmonized_value", "followup_harmonized_value", "change"], "properties": {"study_id": {"enum": ["alpha", "beta"]}, "analysis_subject_id": {"type": "string"}, "arm": {"enum": ["A", "B"]}, "baseline_visit_record_id": {"type": "string"}, "followup_visit_record_id": {"type": "string"}, "baseline_assay_record_ids": {"type": "array", "items": {"type": "string"}}, "followup_assay_record_ids": {"type": "array", "items": {"type": "string"}}, "baseline_harmonized_value": _number, "followup_harmonized_value": _number, "change": _number}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["status", "answer", "key_results", "limitations"], "properties": {"status": {"enum": ["completed"]}, "answer": {"type": "string"}, "limitations": {"type": "array", "items": {"type": "string"}}, "key_results": {"type": "object", "additionalProperties": False, "required": ["study_attrition", "study_statistics", "study_between_arm_comparisons", "pooled_comparison", "selected_pairs", "data_quality_audit"], "properties": {"study_attrition": {"type": "object", "additionalProperties": False, "required": ["alpha", "beta"], "properties": {"alpha": _attrition, "beta": _attrition}}, "study_statistics": {"type": "object", "additionalProperties": False, "required": ["alpha", "beta"], "properties": {"alpha": _study_stats, "beta": _study_stats}}, "study_between_arm_comparisons": {"type": "object", "additionalProperties": False, "required": ["alpha", "beta"], "properties": {"alpha": _comparison, "beta": _comparison}}, "pooled_comparison": {"type": "object", "additionalProperties": False, "required": ["pooled_difference_in_mean_change_b_minus_a", "sum_of_inverse_variance_weights"], "properties": {"pooled_difference_in_mean_change_b_minus_a": _number, "sum_of_inverse_variance_weights": _number}}, "selected_pairs": {"type": "array", "items": _pair}, "data_quality_audit": {"type": "object", "additionalProperties": False, "required": ["alpha", "beta"], "properties": {"alpha": _quality, "beta": _quality}}}}}}
GRADER='''import math\nfrom data_analysis_agent.benchmark_types import GradeResult\ndef leaves(x): return sum(leaves(v) for v in x.values()) if isinstance(x,dict) else sum(leaves(v) for v in x) if isinstance(x,list) else 1\ndef match(a,e,p,errs,t):\n if isinstance(e,dict):\n  if not isinstance(a,dict): errs.append("type mismatch: "+p); return 0\n  [errs.append("unexpected field: "+p+"."+k) for k in set(a)-set(e)]\n  return sum(match(a[k],v,p+"."+k,errs,t) if k in a else (errs.append("missing field: "+p+"."+k) or 0) for k,v in e.items())\n if isinstance(e,list):\n  if not isinstance(a,list): errs.append("type mismatch: "+p); return 0\n  if len(a)!=len(e): errs.append("length mismatch: "+p)\n  return sum(match(a[i],v,p+"["+str(i)+"]",errs,t) if i<len(a) else (errs.append("missing item: "+p) or 0) for i,v in enumerate(e))\n if isinstance(e,(int,float)) and not isinstance(e,bool):\n  if not isinstance(a,(int,float)) or isinstance(a,bool) or not math.isfinite(a) or abs(a-e)>t: errs.append("numerical mismatch: "+p); return 0\n  return 1\n if a!=e: errs.append("value mismatch: "+p); return 0\n return 1\ndef grade(candidate,reference):\n errs=[]\n if not isinstance(candidate,dict) or set(candidate)!={"status","answer","key_results","limitations"}: errs.append("top-level schema mismatch")\n if not isinstance(candidate,dict) or candidate.get("status")!="completed" or not isinstance(candidate.get("answer"),str) or not isinstance(candidate.get("limitations"),list): errs.append("invalid top-level values")\n actual=candidate.get("key_results") if isinstance(candidate,dict) else None; expected=reference["key_results"]; n=match(actual,expected,"key_results",errs,reference["absolute_tolerance"]) if isinstance(actual,dict) else 0\n return GradeResult(passed=not errs,score=n/leaves(expected),errors=errs[:80],details={"error_category":"none" if not errs else "strict_scientific_mismatch"})\n'''

def generated_files() -> dict[str,str]:
    task={"public_files":PUBLIC_FILES,"answer_schema":SCHEMA,"metadata":{"description":"Fully static cross-study biomarker harmonization","document_files":[x for x in PUBLIC_FILES if x.startswith(("protocol/","documentation/"))],"document_precedence":["Alpha amendment supersedes only conflicting Alpha rules","Beta does not inherit Alpha amendment","Harmonization follows study-specific rules"],"static_fairness":{"all_public_files_staged_initially":True,"deferred_files":[],"release_gates":[]}}}
    files={"public/task.json":json.dumps(task,indent=2)+"\n","public/prompt.txt":"Use all public protocol, amendment, dictionary, codebook, crosswalk, and raw-data files. Respect precedence, determine the harmonization plan, return the exact JSON with complete audit records, and avoid causal claims and hypothesis tests.\n","private/grader.py":GRADER}; files.update({"public/"+k:v for k,v in {**_documents(),**_data()}.items()}); return files

def write_task() -> None:
    shutil.rmtree(TARGET_ROOT,ignore_errors=True)
    for name,text in generated_files().items():
        path=TARGET_ROOT/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text,encoding="utf-8")
    (TARGET_ROOT/"private/reference.json").write_text(json.dumps({"absolute_tolerance":1e-6,"key_results":compute_oracle()["key_results"]},indent=2)+"\n",encoding="utf-8")

def validate_task() -> None:
    stale=[name for name,text in generated_files().items() if not (TARGET_ROOT/name).is_file() or (TARGET_ROOT/name).read_text(encoding="utf-8")!=text]
    if stale: raise RuntimeError("generated task files are stale: "+", ".join(stale))
    grade=grade_candidate(compute_oracle(),PrivateGradingSpec(grader_path=str(TARGET_ROOT/"private/grader.py"),reference_path=str(TARGET_ROOT/"private/reference.json")))
    if not grade.passed: raise RuntimeError("cross-study oracle failed private grading: "+repr(grade.errors))

def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); mode=parser.add_mutually_exclusive_group(required=True); mode.add_argument("--write",action="store_true"); mode.add_argument("--check",action="store_true"); args=parser.parse_args()
    if args.write: write_task()
    validate_task(); print("Cross-study biomarker task is deterministic and oracle-valid."); return 0
