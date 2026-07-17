from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

TASK_ID = 'cross_study_biomarker_harmonization_small'
ROUND_DIGITS = 3


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + '\n', encoding='utf-8')


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def iso_day(start: str, day: int) -> str:
    return (date.fromisoformat(start) + timedelta(days=day)).isoformat()


def iso_time(start: str, day: int, hour: int = 9, minute: int = 0) -> str:
    return f"{iso_day(start, day)}T{hour:02d}:{minute:02d}:00"


def r3(value: float) -> float:
    return round(float(value), ROUND_DIGITS)


def build_rows() -> dict[str, list[dict[str, Any]]]:
    alpha_start = '2026-01-15'
    beta_start = '2026-02-01'

    alpha_subjects: list[dict[str, Any]] = []
    for i in range(1, 12):
        alpha_subjects.append({
            'source_subject_id': f'AS{i:02d}',
            'analysis_subject_id': f'ALPHA-{i:03d}',
            'arm_code': 'TA' if i in {1, 2, 3, 7, 8, 9} else 'TB',
            'age_years': 17 if i == 7 else 30 + i,
            'consent_code': 'Y',
            'treatment_start_date': alpha_start,
        })

    beta_subjects: list[dict[str, Any]] = []
    for i in range(1, 12):
        beta_subjects.append({
            'analysis_subject_id': f'BETA-{i:03d}',
            'blind_arm_code': 'BA' if i in {1, 2, 3, 7, 8, 9} else 'BB',
            'age_years': 40 + i,
            'consent_flag': '0' if i == 7 else '1',
            'treatment_start_date': beta_start,
        })

    beta_crosswalk: list[dict[str, Any]] = []
    # Cyclic identifiers make a wrong direct source->canonical join look plausible.
    for i in range(1, 12):
        source_num = (i % 11) + 1
        beta_crosswalk.append({
            'source_subject_id': f'BETA-{source_num:03d}',
            'analysis_subject_id': f'BETA-{i:03d}',
        })
    beta_source_for = {row['analysis_subject_id']: row['source_subject_id'] for row in beta_crosswalk}

    alpha_visits: list[dict[str, Any]] = []
    alpha_assays: list[dict[str, Any]] = []
    beta_visits: list[dict[str, Any]] = []
    beta_assays: list[dict[str, Any]] = []

    visit_row_counter = 1
    assay_row_counter = 1

    def add_alpha_visit(subject_n: int, visit_id: str, day: int, status: str = 'V', quality: float = 0.9,
                        hour: int = 9, minute: int = 0, duplicate: bool = False) -> None:
        nonlocal visit_row_counter
        base = {
            'technical_row_id': f'AVROW-{visit_row_counter:04d}',
            'visit_record_id': visit_id,
            'source_subject_id': f'AS{subject_n:02d}',
            'study_day': day,
            'visit_status_code': status,
            'visit_quality_score': quality,
            'collected_at': iso_time(alpha_start, day, hour, minute),
        }
        alpha_visits.append(base)
        visit_row_counter += 1
        if duplicate:
            dup = dict(base)
            dup['technical_row_id'] = f'AVROW-{visit_row_counter:04d}'
            alpha_visits.append(dup)
            visit_row_counter += 1

    def add_alpha_assay(subject_n: int, visit_id: str, assay_id: str, harmonized_value: float,
                        status: str = 'Q', quality: float = 0.9, hour: int = 11,
                        minute: int = 0, duplicate: bool = False, raw_override: str | None = None,
                        unit: str = 'pg/mL', platform: str = 'PX') -> None:
        nonlocal assay_row_counter
        raw_value = raw_override if raw_override is not None else f'{harmonized_value / 1.08:.9f}'
        base = {
            'technical_row_id': f'AAROW-{assay_row_counter:04d}',
            'assay_record_id': assay_id,
            'source_subject_id': f'AS{subject_n:02d}',
            'visit_record_id': visit_id,
            'specimen_id': f'ASP-{subject_n:02d}-{visit_id}',
            'assay_timestamp': f'{iso_day(alpha_start, 0)}T{hour:02d}:{minute:02d}:00',
            'reported_value': raw_value,
            'reported_unit': unit,
            'assay_status_code': status,
            'assay_quality_score': quality,
            'platform_code': platform,
        }
        alpha_assays.append(base)
        assay_row_counter += 1
        if duplicate:
            dup = dict(base)
            dup['technical_row_id'] = f'AAROW-{assay_row_counter:04d}'
            alpha_assays.append(dup)
            assay_row_counter += 1

    # Alpha complete pairs: A01-A06.
    alpha_targets = {
        1: (100, 92),
        2: (110, 100),
        3: (120, 111),
        4: (130, 116),
        5: (140, 125),
        6: (150, 137),
    }
    for n, (baseline, followup) in alpha_targets.items():
        if n == 3:
            add_alpha_visit(n, 'A03-BL-M8', -8, quality=0.95, hour=9)
            add_alpha_visit(n, 'A03-BL-M6', -6, quality=0.80, hour=8)
            add_alpha_visit(n, 'A03-BL-INVALID', -7, status='X', quality=1.0, hour=7)
            add_alpha_assay(n, 'A03-BL-M8', 'A03-ASSAY-BL', baseline)
            add_alpha_assay(n, 'A03-BL-M6', 'A03-ASSAY-BL-DECOY', 70)
            add_alpha_assay(n, 'A03-BL-INVALID', 'A03-ASSAY-BL-INVALID', 999)
        else:
            bl_id = f'A{n:02d}-BL'
            add_alpha_visit(n, bl_id, -7, duplicate=(n == 5))
            if n == 4:
                add_alpha_assay(n, bl_id, 'A04-ASSAY-BL-VALID', baseline, quality=0.95, hour=9)
                add_alpha_assay(n, bl_id, 'A04-ASSAY-BL-LOWQ', 160, quality=0.70, hour=8)
                add_alpha_assay(n, bl_id, 'A04-ASSAY-BL-INVALID', 999, status='F', quality=1.0, hour=7)
            else:
                add_alpha_assay(n, bl_id, f'A{n:02d}-ASSAY-BL', baseline, duplicate=(n == 5))

        if n == 2:
            add_alpha_visit(n, 'A02-FU-AMENDED', 35, status='R', quality=0.90)
            add_alpha_visit(n, 'A02-FU-LEGACY', 38, status='V', quality=0.99)
            add_alpha_assay(n, 'A02-FU-AMENDED', 'A02-ASSAY-FU-AMENDED', followup)
            add_alpha_assay(n, 'A02-FU-LEGACY', 'A02-ASSAY-FU-LEGACY', 80)
        elif n == 6:
            add_alpha_visit(n, 'A06-FU-D34', 34, quality=0.90, hour=10)
            add_alpha_visit(n, 'A06-FU-D36', 36, quality=0.90, hour=8)
            add_alpha_assay(n, 'A06-FU-D34', 'A06-ASSAY-FU-D34', followup)
            add_alpha_assay(n, 'A06-FU-D36', 'A06-ASSAY-FU-D36', 160)
        else:
            fu_id = f'A{n:02d}-FU'
            add_alpha_visit(n, fu_id, 35, status='V')
            add_alpha_assay(n, fu_id, f'A{n:02d}-ASSAY-FU', followup)

    # Alpha attrition subjects.
    add_alpha_visit(9, 'A09-BL-OUTSIDE', -20)
    add_alpha_assay(9, 'A09-BL-OUTSIDE', 'A09-ASSAY-BL-OUTSIDE', 100)
    add_alpha_visit(9, 'A09-FU', 35)
    add_alpha_assay(9, 'A09-FU', 'A09-ASSAY-FU', 90)

    add_alpha_visit(10, 'A10-BL', -7)
    add_alpha_assay(10, 'A10-BL', 'A10-ASSAY-BL', 100)
    add_alpha_visit(10, 'A10-FU-OUTSIDE', 50)
    add_alpha_assay(10, 'A10-FU-OUTSIDE', 'A10-ASSAY-FU-OUTSIDE', 90)

    add_alpha_visit(11, 'A11-BL', -7)
    add_alpha_assay(11, 'A11-BL', 'A11-ASSAY-BL', 100)
    add_alpha_visit(11, 'A11-FU', 35)
    add_alpha_assay(11, 'A11-FU', 'A11-ASSAY-FU', 90)

    # Additional invalid assay rows for audit.
    add_alpha_assay(1, 'A01-BL', 'A01-ASSAY-NAN', 0, raw_override='NaN', quality=1.0)
    add_alpha_assay(1, 'A01-FU', 'A01-ASSAY-WRONGUNIT', 50, unit='ng/mL', quality=1.0)

    alpha_exclusions = [
        {
            'event_id': 'AX-08-PRE',
            'source_subject_id': 'AS08',
            'event_effective_date': alpha_start,
            'event_type_code': 'EXC',
        },
        {
            'event_id': 'AX-11-POST-BOUNDARY',
            'source_subject_id': 'AS11',
            'event_effective_date': iso_day(alpha_start, 35),
            'event_type_code': 'EXC',
        },
    ]

    def add_beta_visit(canonical_n: int, visit_id: str, day: int, status: str = 'OK', quality: float = 0.9,
                       hour: int = 9, minute: int = 0, duplicate: bool = False) -> None:
        nonlocal visit_row_counter
        canonical = f'BETA-{canonical_n:03d}'
        base = {
            'technical_row_id': f'BVROW-{visit_row_counter:04d}',
            'visit_record_id': visit_id,
            'source_subject_id': beta_source_for[canonical],
            'study_day': day,
            'visit_status_code': status,
            'visit_quality_score': quality,
            'collected_at': iso_time(beta_start, day, hour, minute),
        }
        beta_visits.append(base)
        visit_row_counter += 1
        if duplicate:
            dup = dict(base)
            dup['technical_row_id'] = f'BVROW-{visit_row_counter:04d}'
            beta_visits.append(dup)
            visit_row_counter += 1

    def add_beta_assay(canonical_n: int, visit_id: str, assay_id: str, value: str | float,
                       status: str = 'PASS', replicate: int = 1, duplicate: bool = False,
                       unit: str = 'ng/L', platform: str = 'PY') -> None:
        nonlocal assay_row_counter
        canonical = f'BETA-{canonical_n:03d}'
        base = {
            'technical_row_id': f'BAROW-{assay_row_counter:04d}',
            'assay_record_id': assay_id,
            'source_subject_id': beta_source_for[canonical],
            'visit_record_id': visit_id,
            'specimen_id': f'BSP-{canonical_n:02d}-{visit_id}',
            'replicate_number': replicate,
            'reported_value': value,
            'reported_unit': unit,
            'assay_status_code': status,
            'platform_code': platform,
        }
        beta_assays.append(base)
        assay_row_counter += 1
        if duplicate:
            dup = dict(base)
            dup['technical_row_id'] = f'BAROW-{assay_row_counter:04d}'
            beta_assays.append(dup)
            assay_row_counter += 1

    beta_targets = {
        1: (95, 91),
        2: (105, 99),
        3: (115, 107),
        4: (125, 118),
        5: (135, 126),
        6: (145, 134),
    }

    def add_beta_replicates(n: int, visit_id: str, target: float, prefix: str, invalid_high: bool = False,
                            duplicate_middle: bool = False) -> None:
        add_beta_assay(n, visit_id, f'{prefix}-R1', target - 1, replicate=1)
        add_beta_assay(n, visit_id, f'{prefix}-R2', target, replicate=2, duplicate=duplicate_middle)
        add_beta_assay(n, visit_id, f'{prefix}-R3', target + 1, replicate=3)
        if invalid_high:
            add_beta_assay(n, visit_id, f'{prefix}-INVALID', 999, status='FAIL', replicate=4)

    for n, (baseline, followup) in beta_targets.items():
        if n == 3:
            add_beta_visit(n, 'B03-BL-LOWER-BOUND', -21)
            add_beta_replicates(n, 'B03-BL-LOWER-BOUND', baseline, 'B03-ASSAY-BL')
            add_beta_visit(n, 'B03-FU-UPPER-BOUND', 70)
            add_beta_replicates(n, 'B03-FU-UPPER-BOUND', followup, 'B03-ASSAY-FU')
        elif n == 6:
            add_beta_visit(n, 'B06-BL-M6', -6, quality=0.90, hour=10)
            add_beta_visit(n, 'B06-BL-M8', -8, quality=0.90, hour=8)
            add_beta_replicates(n, 'B06-BL-M6', 170, 'B06-ASSAY-BL-M6')
            add_beta_replicates(n, 'B06-BL-M8', baseline, 'B06-ASSAY-BL-M8')
            add_beta_visit(n, 'B06-FU', 63)
            add_beta_replicates(n, 'B06-FU', followup, 'B06-ASSAY-FU')
        else:
            bl_id = f'B{n:02d}-BL'
            fu_id = f'B{n:02d}-FU'
            add_beta_visit(n, bl_id, -7, duplicate=(n == 2))
            add_beta_visit(n, fu_id, 63)
            add_beta_replicates(n, bl_id, baseline, f'B{n:02d}-ASSAY-BL', invalid_high=(n == 4))
            add_beta_replicates(n, fu_id, followup, f'B{n:02d}-ASSAY-FU', duplicate_middle=(n == 5))

    # Invalid visit row that would win if status filtering were omitted.
    add_beta_visit(1, 'B01-FU-INVALID', 63, status='BAD', quality=1.0, hour=7)
    add_beta_replicates(1, 'B01-FU-INVALID', 20, 'B01-ASSAY-FU-INVALID')

    # Beta attrition subjects.
    add_beta_visit(9, 'B09-BL-ONE-REP', -7)
    add_beta_assay(9, 'B09-BL-ONE-REP', 'B09-ASSAY-BL-R1', 100, replicate=1)
    add_beta_visit(9, 'B09-FU', 63)
    add_beta_replicates(9, 'B09-FU', 90, 'B09-ASSAY-FU')

    add_beta_visit(10, 'B10-BL', -7)
    add_beta_replicates(10, 'B10-BL', 100, 'B10-ASSAY-BL')
    add_beta_visit(10, 'B10-FU-ONE-REP', 63)
    add_beta_assay(10, 'B10-FU-ONE-REP', 'B10-ASSAY-FU-R1', 90, replicate=1)

    add_beta_visit(11, 'B11-BL', -7)
    add_beta_replicates(11, 'B11-BL', 100, 'B11-ASSAY-BL')
    add_beta_visit(11, 'B11-FU', 63)
    add_beta_replicates(11, 'B11-FU', 90, 'B11-ASSAY-FU')

    # Extra invalid assay rows.
    add_beta_assay(2, 'B02-BL', 'B02-ASSAY-NAN', 'NaN', replicate=9)
    add_beta_assay(2, 'B02-FU', 'B02-ASSAY-WRONGUNIT', 99, replicate=9, unit='pg/mL')

    beta_exclusions = [
        {
            'event_id': 'BX-08-PRE',
            'source_subject_id': beta_source_for['BETA-008'],
            'event_effective_date': beta_start,
            'event_type_code': 'EXCLUDE',
        },
        {
            'event_id': 'BX-05-ON-FOLLOWUP',
            'source_subject_id': beta_source_for['BETA-005'],
            'event_effective_date': iso_day(beta_start, 63),
            'event_type_code': 'EXCLUDE',
        },
        {
            'event_id': 'BX-11-BEFORE-FOLLOWUP',
            'source_subject_id': beta_source_for['BETA-011'],
            'event_effective_date': iso_day(beta_start, 60),
            'event_type_code': 'EXCLUDE',
        },
    ]

    return {
        'study_alpha_subjects.csv': alpha_subjects,
        'study_alpha_visits.csv': alpha_visits,
        'study_alpha_assays.csv': alpha_assays,
        'study_alpha_exclusions.csv': alpha_exclusions,
        'study_beta_subjects.csv': beta_subjects,
        'study_beta_visits.csv': beta_visits,
        'study_beta_assays.csv': beta_assays,
        'study_beta_exclusions.csv': beta_exclusions,
        'study_beta_subject_crosswalk.csv': beta_crosswalk,
    }


def _dedupe(rows: list[dict[str, str]], keys: list[str]) -> tuple[list[dict[str, str]], int]:
    seen: set[tuple[str, ...]] = set()
    kept: list[dict[str, str]] = []
    removed = 0
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in seen:
            removed += 1
        else:
            seen.add(key)
            kept.append(row)
    return kept, removed


def _finite_positive(value: str) -> bool:
    try:
        parsed = float(value)
        return math.isfinite(parsed) and parsed > 0
    except ValueError:
        return False


def _choose_visit(candidates: list[dict[str, Any]], target: int) -> dict[str, Any] | None:
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda row: (
            abs(int(row['study_day']) - target),
            -float(row['visit_quality_score']),
            row['collected_at'],
            row['visit_record_id'],
        ),
    )[0]


def _study_oracle(public_dir: Path, study: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    data_dir = public_dir / 'data'
    subjects = read_csv(data_dir / f'study_{study}_subjects.csv')
    visits_raw = read_csv(data_dir / f'study_{study}_visits.csv')
    assays_raw = read_csv(data_dir / f'study_{study}_assays.csv')
    exclusions = read_csv(data_dir / f'study_{study}_exclusions.csv')

    if study == 'alpha':
        canonical_for_source = {row['source_subject_id']: row['analysis_subject_id'] for row in subjects}
        arm_map = {'TA': 'A', 'TB': 'B'}
        consent_key, consent_ok = 'consent_code', 'Y'
        arm_key = 'arm_code'
        accepted_visit = {'V', 'R'}
        baseline_window, baseline_target = (-14, -1), -7
        followup_window, followup_target = (28, 42), 35
        post_inclusive = True
    else:
        crosswalk = read_csv(data_dir / 'study_beta_subject_crosswalk.csv')
        canonical_for_source = {row['source_subject_id']: row['analysis_subject_id'] for row in crosswalk}
        arm_map = {'BA': 'A', 'BB': 'B'}
        consent_key, consent_ok = 'consent_flag', '1'
        arm_key = 'blind_arm_code'
        accepted_visit = {'OK'}
        baseline_window, baseline_target = (-21, -3), -7
        followup_window, followup_target = (56, 70), 63
        post_inclusive = False

    visit_identity = [
        'visit_record_id', 'source_subject_id', 'study_day', 'visit_status_code',
        'visit_quality_score', 'collected_at'
    ]
    visits_deduped, visit_dups = _dedupe(visits_raw, visit_identity)
    valid_visits: list[dict[str, Any]] = []
    invalid_visit_count = 0
    for row in visits_deduped:
        try:
            day = int(row['study_day'])
            quality = float(row['visit_quality_score'])
            good = row['visit_status_code'] in accepted_visit and math.isfinite(quality)
        except ValueError:
            good = False
        if not good:
            invalid_visit_count += 1
            continue
        if row['source_subject_id'] not in canonical_for_source:
            invalid_visit_count += 1
            continue
        item = dict(row)
        item['analysis_subject_id'] = canonical_for_source[row['source_subject_id']]
        item['study_day'] = day
        item['visit_quality_score'] = quality
        valid_visits.append(item)

    if study == 'alpha':
        assay_identity = [
            'assay_record_id', 'source_subject_id', 'visit_record_id', 'specimen_id',
            'assay_timestamp', 'reported_value', 'reported_unit', 'assay_status_code',
            'assay_quality_score', 'platform_code'
        ]
    else:
        assay_identity = [
            'assay_record_id', 'source_subject_id', 'visit_record_id', 'specimen_id',
            'replicate_number', 'reported_value', 'reported_unit', 'assay_status_code',
            'platform_code'
        ]
    assays_deduped, assay_dups = _dedupe(assays_raw, assay_identity)
    valid_assays: list[dict[str, Any]] = []
    invalid_assay_count = 0
    for row in assays_deduped:
        if row['source_subject_id'] not in canonical_for_source:
            invalid_assay_count += 1
            continue
        if study == 'alpha':
            good = (
                row['assay_status_code'] == 'Q'
                and row['reported_unit'] == 'pg/mL'
                and row['platform_code'] == 'PX'
                and _finite_positive(row['reported_value'])
            )
        else:
            good = (
                row['assay_status_code'] == 'PASS'
                and row['reported_unit'] == 'ng/L'
                and row['platform_code'] == 'PY'
                and _finite_positive(row['reported_value'])
            )
        if not good:
            invalid_assay_count += 1
            continue
        item = dict(row)
        item['analysis_subject_id'] = canonical_for_source[row['source_subject_id']]
        item['reported_value'] = float(row['reported_value'])
        valid_assays.append(item)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in valid_assays:
        grouped.setdefault(row['visit_record_id'], []).append(row)

    visit_assay: dict[str, dict[str, Any]] = {}
    insufficient = 0
    for visit_id, rows in grouped.items():
        if study == 'alpha':
            selected = sorted(
                rows,
                key=lambda row: (
                    -float(row['assay_quality_score']),
                    row['assay_timestamp'],
                    row['assay_record_id'],
                ),
            )[0]
            visit_assay[visit_id] = {
                'value': selected['reported_value'] * 1.08,
                'assay_record_ids': [selected['assay_record_id']],
            }
        else:
            if len(rows) < 2:
                insufficient += 1
                continue
            ordered = sorted(rows, key=lambda row: row['assay_record_id'])
            visit_assay[visit_id] = {
                'value': float(median([row['reported_value'] for row in ordered])),
                'assay_record_ids': [row['assay_record_id'] for row in ordered],
            }

    subject_by_id = {row['analysis_subject_id']: row for row in subjects}
    exclusions_by_subject: dict[str, list[dict[str, str]]] = {}
    for row in exclusions:
        canonical = canonical_for_source.get(row['source_subject_id'])
        if canonical:
            exclusions_by_subject.setdefault(canonical, []).append(row)

    total = len(subjects)
    basic_eligible: list[str] = []
    for row in subjects:
        if int(row['age_years']) >= 18 and row[consent_key] == consent_ok and row[arm_key] in arm_map:
            basic_eligible.append(row['analysis_subject_id'])
    basic_ineligible = total - len(basic_eligible)

    after_pre: list[str] = []
    excluded_pre = 0
    for sid in basic_eligible:
        start = date.fromisoformat(subject_by_id[sid]['treatment_start_date'])
        has_pre = any(
            date.fromisoformat(event['event_effective_date']) <= start
            for event in exclusions_by_subject.get(sid, [])
        )
        if has_pre:
            excluded_pre += 1
        else:
            after_pre.append(sid)

    visits_by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in valid_visits:
        if row['visit_record_id'] in visit_assay:
            visits_by_subject.setdefault(row['analysis_subject_id'], []).append(row)

    no_baseline = 0
    no_followup = 0
    excluded_post = 0
    pairs: list[dict[str, Any]] = []

    for sid in after_pre:
        candidates = visits_by_subject.get(sid, [])
        baseline = _choose_visit(
            [row for row in candidates if baseline_window[0] <= row['study_day'] <= baseline_window[1]],
            baseline_target,
        )
        if baseline is None:
            no_baseline += 1
            continue
        followup = _choose_visit(
            [row for row in candidates if followup_window[0] <= row['study_day'] <= followup_window[1]],
            followup_target,
        )
        if followup is None:
            no_followup += 1
            continue
        start = date.fromisoformat(subject_by_id[sid]['treatment_start_date'])
        followup_date = datetime.fromisoformat(followup['collected_at']).date()
        if post_inclusive:
            has_post = any(
                start < date.fromisoformat(event['event_effective_date']) <= followup_date
                for event in exclusions_by_subject.get(sid, [])
            )
        else:
            has_post = any(
                start < date.fromisoformat(event['event_effective_date']) < followup_date
                for event in exclusions_by_subject.get(sid, [])
            )
        if has_post:
            excluded_post += 1
            continue
        baseline_info = visit_assay[baseline['visit_record_id']]
        followup_info = visit_assay[followup['visit_record_id']]
        baseline_value = float(baseline_info['value'])
        followup_value = float(followup_info['value'])
        pair = {
            'study_id': study,
            'analysis_subject_id': sid,
            'arm': arm_map[subject_by_id[sid][arm_key]],
            'baseline_visit_record_id': baseline['visit_record_id'],
            'followup_visit_record_id': followup['visit_record_id'],
            'baseline_assay_record_ids': baseline_info['assay_record_ids'],
            'followup_assay_record_ids': followup_info['assay_record_ids'],
            'baseline_harmonized_value': r3(baseline_value),
            'followup_harmonized_value': r3(followup_value),
            'change': r3(followup_value - baseline_value),
        }
        pairs.append(pair)

    pairs.sort(key=lambda row: row['analysis_subject_id'])
    attrition = {
        'total_subjects': total,
        'basic_ineligible': basic_ineligible,
        'eligible_after_basic_checks': len(basic_eligible),
        'excluded_pre_start': excluded_pre,
        'no_valid_baseline': no_baseline,
        'no_valid_followup': no_followup,
        'excluded_post_start_before_followup': excluded_post,
        'complete_pairs': len(pairs),
        'complete_pairs_arm_a': sum(row['arm'] == 'A' for row in pairs),
        'complete_pairs_arm_b': sum(row['arm'] == 'B' for row in pairs),
    }
    audit = {
        'scientific_duplicate_visit_rows_removed': visit_dups,
        'invalid_visit_rows_excluded': invalid_visit_count,
        'scientific_duplicate_assay_rows_removed': assay_dups,
        'invalid_assay_rows_excluded': invalid_assay_count,
        'insufficient_replicate_visit_summaries': insufficient,
    }
    return attrition, audit, pairs


def compute_reference(public_dir: Path) -> dict[str, Any]:
    study_attrition: dict[str, Any] = {}
    data_quality: dict[str, Any] = {}
    all_pairs: list[dict[str, Any]] = []
    study_stats: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}

    for study in ('alpha', 'beta'):
        attrition, audit, pairs = _study_oracle(public_dir, study)
        study_attrition[study] = attrition
        data_quality[study] = audit
        all_pairs.extend(pairs)
        arm_stats: dict[str, Any] = {}
        for arm in ('A', 'B'):
            arm_pairs = [row for row in pairs if row['arm'] == arm]
            changes = [float(row['change']) for row in arm_pairs]
            baselines = [float(row['baseline_harmonized_value']) for row in arm_pairs]
            followups = [float(row['followup_harmonized_value']) for row in arm_pairs]
            sd = stdev(changes)
            arm_stats[arm] = {
                'n': len(arm_pairs),
                'mean_baseline': r3(mean(baselines)),
                'mean_followup': r3(mean(followups)),
                'mean_change': r3(mean(changes)),
                'sample_sd_change': r3(sd),
                'sample_se_change': r3(sd / math.sqrt(len(changes))),
            }
        study_stats[study] = arm_stats
        diff = arm_stats['B']['mean_change'] - arm_stats['A']['mean_change']
        var_a = arm_stats['A']['sample_sd_change'] ** 2 / arm_stats['A']['n']
        var_b = arm_stats['B']['sample_sd_change'] ** 2 / arm_stats['B']['n']
        variance = var_a + var_b
        weight = 1.0 / variance
        comparisons[study] = {
            'difference_in_mean_change_b_minus_a': r3(diff),
            'variance_of_difference': r3(variance),
            'inverse_variance_weight': r3(weight),
        }

    all_pairs.sort(key=lambda row: (row['study_id'], row['analysis_subject_id']))
    sum_weights = sum(item['inverse_variance_weight'] for item in comparisons.values())
    pooled = sum(
        item['inverse_variance_weight'] * item['difference_in_mean_change_b_minus_a']
        for item in comparisons.values()
    ) / sum_weights
    return {
        'status': 'completed',
        'answer': (
            'Study-specific biomarker responses were harmonized and summarized using '
            'the governing protocols and fixed-effect inverse-variance pooling.'
        ),
        'key_results': {
            'study_attrition': study_attrition,
            'study_statistics': study_stats,
            'study_between_arm_comparisons': comparisons,
            'pooled_comparison': {
                'pooled_difference_in_mean_change_b_minus_a': r3(pooled),
                'sum_of_inverse_variance_weights': r3(sum_weights),
            },
            'selected_pairs': all_pairs,
            'data_quality_audit': data_quality,
        },
        'limitations': [
            'Descriptive analysis only; no causal or hypothesis-testing claims are made.'
        ],
    }


def answer_schema() -> dict[str, Any]:
    number = {'type': 'number'}
    integer = {'type': 'integer'}
    attrition_props = {
        key: integer for key in (
            'total_subjects', 'basic_ineligible', 'eligible_after_basic_checks',
            'excluded_pre_start', 'no_valid_baseline', 'no_valid_followup',
            'excluded_post_start_before_followup', 'complete_pairs',
            'complete_pairs_arm_a', 'complete_pairs_arm_b'
        )
    }
    audit_props = {
        key: integer for key in (
            'scientific_duplicate_visit_rows_removed', 'invalid_visit_rows_excluded',
            'scientific_duplicate_assay_rows_removed', 'invalid_assay_rows_excluded',
            'insufficient_replicate_visit_summaries'
        )
    }
    arm_props = {
        'n': integer,
        'mean_baseline': number,
        'mean_followup': number,
        'mean_change': number,
        'sample_sd_change': number,
        'sample_se_change': number,
    }
    pair_props = {
        'study_id': {'enum': ['alpha', 'beta']},
        'analysis_subject_id': {'type': 'string'},
        'arm': {'enum': ['A', 'B']},
        'baseline_visit_record_id': {'type': 'string'},
        'followup_visit_record_id': {'type': 'string'},
        'baseline_assay_record_ids': {'type': 'array', 'items': {'type': 'string'}},
        'followup_assay_record_ids': {'type': 'array', 'items': {'type': 'string'}},
        'baseline_harmonized_value': number,
        'followup_harmonized_value': number,
        'change': number,
    }
    return {
        'type': 'object',
        'required': ['status', 'answer', 'key_results', 'limitations'],
        'properties': {
            'status': {'enum': ['completed', 'completed_with_limitations']},
            'answer': {'type': 'string'},
            'key_results': {
                'type': 'object',
                'required': [
                    'study_attrition', 'study_statistics',
                    'study_between_arm_comparisons', 'pooled_comparison',
                    'selected_pairs', 'data_quality_audit'
                ],
                'properties': {
                    'study_attrition': {
                        'type': 'object',
                        'required': ['alpha', 'beta'],
                        'properties': {
                            s: {'type': 'object', 'required': list(attrition_props),
                                'properties': attrition_props, 'additionalProperties': False}
                            for s in ('alpha', 'beta')
                        },
                        'additionalProperties': False,
                    },
                    'study_statistics': {
                        'type': 'object',
                        'required': ['alpha', 'beta'],
                        'properties': {
                            s: {
                                'type': 'object', 'required': ['A', 'B'],
                                'properties': {
                                    arm: {'type': 'object', 'required': list(arm_props),
                                          'properties': arm_props, 'additionalProperties': False}
                                    for arm in ('A', 'B')
                                },
                                'additionalProperties': False,
                            } for s in ('alpha', 'beta')
                        },
                        'additionalProperties': False,
                    },
                    'study_between_arm_comparisons': {
                        'type': 'object', 'required': ['alpha', 'beta'],
                        'properties': {
                            s: {
                                'type': 'object',
                                'required': [
                                    'difference_in_mean_change_b_minus_a',
                                    'variance_of_difference', 'inverse_variance_weight'
                                ],
                                'properties': {
                                    'difference_in_mean_change_b_minus_a': number,
                                    'variance_of_difference': number,
                                    'inverse_variance_weight': number,
                                },
                                'additionalProperties': False,
                            } for s in ('alpha', 'beta')
                        },
                        'additionalProperties': False,
                    },
                    'pooled_comparison': {
                        'type': 'object',
                        'required': [
                            'pooled_difference_in_mean_change_b_minus_a',
                            'sum_of_inverse_variance_weights'
                        ],
                        'properties': {
                            'pooled_difference_in_mean_change_b_minus_a': number,
                            'sum_of_inverse_variance_weights': number,
                        },
                        'additionalProperties': False,
                    },
                    'selected_pairs': {
                        'type': 'array',
                        'items': {
                            'type': 'object', 'required': list(pair_props),
                            'properties': pair_props, 'additionalProperties': False
                        },
                    },
                    'data_quality_audit': {
                        'type': 'object', 'required': ['alpha', 'beta'],
                        'properties': {
                            s: {'type': 'object', 'required': list(audit_props),
                                'properties': audit_props, 'additionalProperties': False}
                            for s in ('alpha', 'beta')
                        },
                        'additionalProperties': False,
                    },
                },
                'additionalProperties': False,
            },
            'limitations': {'type': 'array', 'items': {'type': 'string'}},
        },
        'additionalProperties': False,
    }


def create_public_docs(root: Path) -> None:
    public = root / 'public'
    write_text(public / 'prompt.txt', '''
Analyze and harmonize the two longitudinal biomarker studies using every public
protocol, amendment, harmonization, dictionary, codebook, crosswalk, and raw-data
file. Respect document precedence and study-specific rules. Construct the eligible
paired cohort, produce complete audit records, compute study-specific summaries and
the protocol-defined pooled comparison, and return exactly one JSON object matching
the public answer schema. Do not make causal claims or perform hypothesis tests.
''')

    write_text(public / 'protocol/study_alpha_protocol.md', '''
# Study Alpha Base Protocol

## Eligibility
Subjects are eligible when age is at least 18 years, consent code is `Y`, and arm
code is recognized. `TA` maps to analysis arm A and `TB` maps to analysis arm B.
An exclusion event effective on or before treatment start is a pre-start exclusion.

## Visit rules
Accepted visit status is `V` only. Baseline is study day -14 through -1 inclusive,
target day -7. Follow-up is study day 30 through 45 inclusive, target day 38.
Tie-break order is: minimum absolute distance to target, higher visit quality,
earlier collection timestamp, lexical visit record ID.

## Assay rules
Alpha Platform `PX` reports `pg/mL`. One pg/mL equals one ng/L. After conversion,
apply the Platform PX calibration multiplier 1.08. Exact scientific assay duplicates
exclude only the technical row ID. Keep status `Q`, unit `pg/mL`, platform `PX`, and
finite positive values. For one visit, select the valid assay with highest quality,
then earliest assay timestamp, then lexical assay record ID. Do not average Alpha
technical replicates.

## Exclusions and statistics
The legacy post-start exclusion rule is treatment_start < event_date < selected
follow-up date. Change is follow-up minus baseline. Use sample SD with denominator
n-1 and sample SE = sample SD / sqrt(n). Study contrast is B minus A.
''')

    write_text(public / 'protocol/study_alpha_amendment_01.md', '''
# Study Alpha Amendment 01

This amendment overrides only conflicting sections of the Study Alpha Base Protocol.
All unrelated baseline, assay, eligibility, tie-break, and statistical rules remain
in force.

Effective amended rules:

- Accepted visit statuses are `V` and `R`.
- Follow-up is study day 28 through 42 inclusive.
- Follow-up target day is 35.
- Post-start exclusion is treatment_start < event_date <= selected follow-up date.
''')

    write_text(public / 'protocol/study_beta_protocol.md', '''
# Study Beta Protocol

## Eligibility and identifiers
Subjects are eligible when age is at least 18 years, consent flag is `1`, and blind
arm code is recognized. `BA` maps to analysis arm A and `BB` maps to analysis arm B.
Visit, assay, and exclusion files use source-system subject IDs. They must be mapped
through the subject crosswalk to canonical `analysis_subject_id`; never join source
IDs directly to canonical IDs. An exclusion event effective on or before treatment
start is a pre-start exclusion.

## Visit rules
Accepted visit status is `OK` only. Baseline is study day -21 through -3 inclusive,
target day -7. Follow-up is study day 56 through 70 inclusive, target day 63.
Tie-break order is: minimum absolute distance to target, higher visit quality,
earlier collection timestamp, lexical visit record ID.

## Assay rules
Platform `PY` reports the harmonized unit `ng/L`; no calibration multiplier applies.
Exact scientific assay duplicates exclude only the technical row ID. Keep status
`PASS`, unit `ng/L`, platform `PY`, and finite positive values. A visit summary
requires at least two accepted replicates after deduplication. Use the median of all
accepted replicates and retain every contributing assay record ID.

## Exclusions and statistics
Post-start exclusion is treatment_start < event_date < selected follow-up date; an
event exactly on the selected follow-up date does not exclude the subject. Change is
follow-up minus baseline. Use sample SD with denominator n-1 and sample SE = sample
SD / sqrt(n). Study contrast is B minus A.
''')

    write_text(public / 'protocol/harmonization_plan.md', '''
# Cross-Study Harmonization Plan

Apply each study's own eligibility, identifier, visit, assay, and exclusion rules
before harmonization. Express both studies in ng/L. Produce one selected baseline and
follow-up pair per complete subject and preserve canonical analysis IDs plus selected
visit and assay record IDs.

For each study and arm, report n, mean baseline, mean follow-up, mean change, sample
SD of change, and sample SE of change. Round reported numeric results to three decimal
places.

For each study:

variance_of_study_contrast = sample_variance_A / n_A + sample_variance_B / n_B
weight = 1 / variance_of_study_contrast
contrast = mean_change_B - mean_change_A

The pooled contrast is sum(weight * contrast) / sum(weight). Report each study's
variance and weight and the sum of weights. Do not replace this with a simple average
or a subject-level pooled mean. No p-values, confidence intervals, or causal claims.

Subject-level attrition is sequential. Data-quality row counts are separate and must
not be inserted into the subject conservation identity.
''')

    write_csv(public / 'documentation/value_codebook.csv', [
        {'study': 'alpha', 'field': 'arm_code', 'physical_value': 'TA', 'semantic_meaning': 'Analysis arm A', 'analysis_use': 'accepted'},
        {'study': 'alpha', 'field': 'arm_code', 'physical_value': 'TB', 'semantic_meaning': 'Analysis arm B', 'analysis_use': 'accepted'},
        {'study': 'alpha', 'field': 'consent_code', 'physical_value': 'Y', 'semantic_meaning': 'Consented', 'analysis_use': 'accepted'},
        {'study': 'alpha', 'field': 'visit_status_code', 'physical_value': 'V', 'semantic_meaning': 'Valid', 'analysis_use': 'accepted'},
        {'study': 'alpha', 'field': 'visit_status_code', 'physical_value': 'R', 'semantic_meaning': 'Reviewed', 'analysis_use': 'accepted by amendment'},
        {'study': 'alpha', 'field': 'visit_status_code', 'physical_value': 'X', 'semantic_meaning': 'Rejected', 'analysis_use': 'not accepted'},
        {'study': 'alpha', 'field': 'assay_status_code', 'physical_value': 'Q', 'semantic_meaning': 'QC accepted', 'analysis_use': 'accepted'},
        {'study': 'alpha', 'field': 'assay_status_code', 'physical_value': 'F', 'semantic_meaning': 'QC failed', 'analysis_use': 'not accepted'},
        {'study': 'beta', 'field': 'blind_arm_code', 'physical_value': 'BA', 'semantic_meaning': 'Analysis arm A', 'analysis_use': 'accepted'},
        {'study': 'beta', 'field': 'blind_arm_code', 'physical_value': 'BB', 'semantic_meaning': 'Analysis arm B', 'analysis_use': 'accepted'},
        {'study': 'beta', 'field': 'consent_flag', 'physical_value': '1', 'semantic_meaning': 'Consented', 'analysis_use': 'accepted'},
        {'study': 'beta', 'field': 'visit_status_code', 'physical_value': 'OK', 'semantic_meaning': 'Accepted visit', 'analysis_use': 'accepted'},
        {'study': 'beta', 'field': 'visit_status_code', 'physical_value': 'BAD', 'semantic_meaning': 'Rejected visit', 'analysis_use': 'not accepted'},
        {'study': 'beta', 'field': 'assay_status_code', 'physical_value': 'PASS', 'semantic_meaning': 'Accepted replicate', 'analysis_use': 'accepted'},
        {'study': 'beta', 'field': 'assay_status_code', 'physical_value': 'FAIL', 'semantic_meaning': 'Rejected replicate', 'analysis_use': 'not accepted'},
    ])

    alpha_dictionary = [
        {'file': 'study_alpha_subjects.csv', 'physical_column': 'source_subject_id', 'semantic_meaning': 'Alpha source subject ID', 'join_key_or_relation': 'joins Alpha raw files'},
        {'file': 'study_alpha_subjects.csv', 'physical_column': 'analysis_subject_id', 'semantic_meaning': 'canonical identifier', 'join_key_or_relation': 'final output ID'},
        {'file': 'study_alpha_visits.csv', 'physical_column': 'visit_record_id', 'semantic_meaning': 'scientific visit record ID', 'join_key_or_relation': 'joins assays'},
        {'file': 'study_alpha_visits.csv', 'physical_column': 'technical_row_id', 'semantic_meaning': 'technical row key', 'join_key_or_relation': 'exclude from logical duplicate identity'},
        {'file': 'study_alpha_assays.csv', 'physical_column': 'visit_record_id', 'semantic_meaning': 'visit relation', 'join_key_or_relation': 'joins visits'},
        {'file': 'study_alpha_assays.csv', 'physical_column': 'technical_row_id', 'semantic_meaning': 'technical row key', 'join_key_or_relation': 'exclude from logical duplicate identity'},
    ]
    beta_dictionary = [
        {'file': 'study_beta_subjects.csv', 'physical_column': 'analysis_subject_id', 'semantic_meaning': 'canonical identifier', 'join_key_or_relation': 'crosswalk target and final output ID'},
        {'file': 'study_beta_subject_crosswalk.csv', 'physical_column': 'source_subject_id', 'semantic_meaning': 'source identifier', 'join_key_or_relation': 'joins Beta raw files'},
        {'file': 'study_beta_subject_crosswalk.csv', 'physical_column': 'analysis_subject_id', 'semantic_meaning': 'canonical identifier', 'join_key_or_relation': 'joins subjects'},
        {'file': 'study_beta_visits.csv', 'physical_column': 'visit_record_id', 'semantic_meaning': 'scientific visit record ID', 'join_key_or_relation': 'joins assays'},
        {'file': 'study_beta_visits.csv', 'physical_column': 'technical_row_id', 'semantic_meaning': 'technical row key', 'join_key_or_relation': 'exclude from logical duplicate identity'},
        {'file': 'study_beta_assays.csv', 'physical_column': 'technical_row_id', 'semantic_meaning': 'technical row key', 'join_key_or_relation': 'exclude from logical duplicate identity'},
    ]
    write_csv(public / 'documentation/study_alpha_data_dictionary.csv', alpha_dictionary)
    write_csv(public / 'documentation/study_beta_data_dictionary.csv', beta_dictionary)
    write_csv(public / 'documentation/identifier_crosswalk_dictionary.csv', [
        {'file': 'study_beta_subject_crosswalk.csv', 'source_column': 'source_subject_id', 'target_column': 'analysis_subject_id', 'rule': 'Many raw source IDs resemble another subject canonical ID; always use this explicit mapping.'}
    ])


def grader_source() -> str:
    return r'''from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

TOLERANCE = 0.0011
REFERENCE_PATH = Path(__file__).with_name("reference.json")


def _compare(expected: Any, actual: Any, path: str, errors: list[str]) -> None:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if expected != actual:
            errors.append(f"{path}: expected {expected!r}, got {actual!r}")
        return
    if isinstance(expected, int):
        if type(actual) is not int or expected != actual:
            errors.append(f"{path}: expected integer {expected}, got {actual!r}")
        return
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            errors.append(f"{path}: expected number {expected}, got {actual!r}")
        elif not math.isfinite(float(actual)) or abs(float(actual) - expected) > TOLERANCE:
            errors.append(f"{path}: expected {expected}, got {actual}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list):
            errors.append(f"{path}: expected list, got {type(actual).__name__}")
            return
        if path.endswith("selected_pairs"):
            key = lambda row: (row.get("study_id"), row.get("analysis_subject_id"))
            expected = sorted(expected, key=key)
            actual = sorted(actual, key=key)
        if len(expected) != len(actual):
            errors.append(f"{path}: expected length {len(expected)}, got {len(actual)}")
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, f"{path}[{index}]", errors)
        return
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            errors.append(f"{path}: expected object, got {type(actual).__name__}")
            return
        if set(expected) != set(actual):
            errors.append(f"{path}: expected keys {sorted(expected)}, got {sorted(actual)}")
        for key in expected.keys() & actual.keys():
            _compare(expected[key], actual[key], f"{path}.{key}", errors)
        return
    errors.append(f"{path}: unsupported expected type {type(expected).__name__}")


def grade(candidate: dict[str, Any]) -> dict[str, Any]:
    reference = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    errors: list[str] = []
    if not isinstance(candidate, dict):
        return {"score": 0.0, "passed": False, "errors": ["candidate is not an object"]}
    if candidate.get("status") not in {"completed", "completed_with_limitations"}:
        errors.append("status must be completed or completed_with_limitations")
    if not isinstance(candidate.get("answer"), str) or not candidate.get("answer", "").strip():
        errors.append("answer must be a non-empty string")
    if not isinstance(candidate.get("limitations"), list):
        errors.append("limitations must be a list")
    _compare(reference["key_results"], candidate.get("key_results"), "$.key_results", errors)
    return {"score": 1.0 if not errors else 0.0, "passed": not errors, "errors": errors}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate")
    args = parser.parse_args()
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    print(json.dumps(grade(candidate), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def create_task(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    create_public_docs(root)
    rows = build_rows()
    for filename, data in rows.items():
        write_csv(root / 'public/data' / filename, data)

    public_files = [
        'protocol/study_alpha_protocol.md',
        'protocol/study_alpha_amendment_01.md',
        'protocol/study_beta_protocol.md',
        'protocol/harmonization_plan.md',
        'documentation/study_alpha_data_dictionary.csv',
        'documentation/study_beta_data_dictionary.csv',
        'documentation/value_codebook.csv',
        'documentation/identifier_crosswalk_dictionary.csv',
        *[f'data/{name}' for name in rows],
    ]
    task = {
        'public_files': public_files,
        'answer_schema': answer_schema(),
        'metadata': {
            'description': 'Small adversarial static cross-study biomarker harmonization task',
            'domain': 'clinical biomarker data analysis',
            'difficulty': 'distributed multi-step static workflow',
            'document_files': public_files[:8],
            'document_precedence': [
                'protocol/study_alpha_amendment_01.md overrides only conflicting sections of protocol/study_alpha_protocol.md'
            ],
            'all_public_files_available_initially': True,
            'deferred_public_files': [],
            'rounding_decimal_places': 3,
        },
    }
    write_json(root / 'public/task.json', task)
    reference = compute_reference(root / 'public')
    write_json(root / 'private/reference.json', reference)
    write_text(root / 'private/grader.py', grader_source())

    write_text(root / 'README.md', f'''
# {TASK_ID}

A small but adversarial fully static benchmark for comparing `single_agent`,
`single_agent_checker`, and Planner-Executor-Verifier workflows under identical
public evidence.

## Contents

- `public/`: prompt, schema, protocols, amendment, dictionaries, codebook, crosswalk,
  and all raw CSV files. Every public file is available from the first model call.
- `private/reference.json`: deterministic oracle answer.
- `private/grader.py`: strict grader for scientific outputs and audit record IDs.
- `build_task.py`: reproducible builder/oracle/check command.
- `tests/test_mutations.py`: reference-pass and adversarial mutation checks.

## Validate

```bash
python build_task.py --check
python tests/test_mutations.py
python private/grader.py private/reference.json
```

## Repository integration

Copy this directory under your repository's `benchmark_tasks/` directory. If the
repository requires task builders inside an importable package, move the reusable
functions from `build_task.py` into that package and keep this script as a thin CLI.

## Intended six-goal workflow

1. Reconcile Alpha amendment precedence, Beta rules, codes, mappings, and pooling.
2. Normalize study-specific eligible cohorts and pre-start exclusions.
3. Validate, deduplicate, convert/calibrate, and aggregate assay records.
4. Validate visits, select baseline/follow-up records, and apply post-start exclusions.
5. Compute attrition, study summaries, contrasts, weights, and pooled comparison.
6. Assemble the exact final JSON and selected-record audit.

This task does not make single-agent approaches structurally incapable of solving it.
It tests whether intermediate scientific verification and bounded local correction
improve empirical reliability under the same evidence and execution environment.
''')

    write_text(root / 'tests/test_mutations.py', r'''from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "private/reference.json").read_text(encoding="utf-8"))
spec = importlib.util.spec_from_file_location("task_grader", ROOT / "private/grader.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def require_fail(name: str, candidate: dict) -> None:
    result = module.grade(candidate)
    assert not result["passed"], f"mutation unexpectedly passed: {name}"


def main() -> int:
    assert module.grade(REFERENCE)["passed"]

    legacy = deepcopy(REFERENCE)
    pair = next(row for row in legacy["key_results"]["selected_pairs"] if row["analysis_subject_id"] == "ALPHA-002")
    pair["followup_visit_record_id"] = "A02-FU-LEGACY"
    require_fail("legacy_alpha_followup", legacy)

    calibration = deepcopy(REFERENCE)
    pair = next(row for row in calibration["key_results"]["selected_pairs"] if row["analysis_subject_id"] == "ALPHA-004")
    pair["baseline_harmonized_value"] = round(pair["baseline_harmonized_value"] / 1.08, 3)
    require_fail("omitted_alpha_calibration", calibration)

    crosswalk = deepcopy(REFERENCE)
    pair = next(row for row in crosswalk["key_results"]["selected_pairs"] if row["analysis_subject_id"] == "BETA-001")
    pair["analysis_subject_id"] = "BETA-002"
    require_fail("beta_direct_join", crosswalk)

    replicate = deepcopy(REFERENCE)
    pair = next(row for row in replicate["key_results"]["selected_pairs"] if row["analysis_subject_id"] == "BETA-004")
    pair["baseline_assay_record_ids"] = pair["baseline_assay_record_ids"][:1]
    require_fail("beta_selected_one_replicate", replicate)

    pooling = deepcopy(REFERENCE)
    comps = pooling["key_results"]["study_between_arm_comparisons"]
    pooling["key_results"]["pooled_comparison"]["pooled_difference_in_mean_change_b_minus_a"] = round(
        (comps["alpha"]["difference_in_mean_change_b_minus_a"] + comps["beta"]["difference_in_mean_change_b_minus_a"]) / 2,
        3,
    )
    require_fail("simple_average_pooling", pooling)

    se = deepcopy(REFERENCE)
    se["key_results"]["study_statistics"]["alpha"]["A"]["sample_se_change"] = se["key_results"]["study_statistics"]["alpha"]["A"]["sample_sd_change"]
    require_fail("sd_se_confusion", se)

    direction = deepcopy(REFERENCE)
    direction["key_results"]["study_between_arm_comparisons"]["alpha"]["difference_in_mean_change_b_minus_a"] *= -1
    require_fail("reversed_direction", direction)

    attrition = deepcopy(REFERENCE)
    attrition["key_results"]["study_attrition"]["beta"]["complete_pairs"] += 1
    require_fail("inconsistent_attrition", attrition)

    print("reference passed; 8 adversarial mutations rejected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''')

    # Bundle a standalone copy of this builder.
    source = Path(__file__).read_text(encoding='utf-8')
    write_text(root / 'build_task.py', source)


def manifest(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name != '.DS_Store':
            result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def check_task(root: Path) -> None:
    if not root.exists():
        raise SystemExit(f'task directory not found: {root}')
    tmp = root.parent / f'.{root.name}.check'
    create_task(tmp)
    left = manifest(root)
    right = manifest(tmp)
    shutil.rmtree(tmp)
    if left != right:
        missing = sorted(set(left) ^ set(right))
        changed = sorted(key for key in set(left) & set(right) if left[key] != right[key])
        raise SystemExit(f'generated task differs; missing={missing}; changed={changed}')
    reference = json.loads((root / 'private/reference.json').read_text(encoding='utf-8'))
    assert reference == compute_reference(root / 'public')
    alpha = reference['key_results']['study_between_arm_comparisons']['alpha']
    beta = reference['key_results']['study_between_arm_comparisons']['beta']
    pooled = reference['key_results']['pooled_comparison']['pooled_difference_in_mean_change_b_minus_a']
    simple = r3((alpha['difference_in_mean_change_b_minus_a'] + beta['difference_in_mean_change_b_minus_a']) / 2)
    assert pooled != simple
    assert alpha['variance_of_difference'] > 0 and beta['variance_of_difference'] > 0
    print(json.dumps({
        'task_id': TASK_ID,
        'files': len(left),
        'selected_pairs': len(reference['key_results']['selected_pairs']),
        'alpha_complete_pairs': reference['key_results']['study_attrition']['alpha']['complete_pairs'],
        'beta_complete_pairs': reference['key_results']['study_attrition']['beta']['complete_pairs'],
        'alpha_contrast': alpha['difference_in_mean_change_b_minus_a'],
        'beta_contrast': beta['difference_in_mean_change_b_minus_a'],
        'pooled_contrast': pooled,
        'simple_average_contrast': simple,
    }, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=(Path(__file__).resolve().parent if Path(__file__).name == 'build_task.py' else Path(__file__).resolve().parent / TASK_ID))
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        check_task(args.output)
    else:
        create_task(args.output)
        print(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
