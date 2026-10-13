"""Offline evidence validation and pre-submit gates. No network or desktop calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SHANGHAI = timezone(timedelta(hours=8), "Asia/Shanghai")
STAMP = r"\d{4}[/-]\d{2}[/-]\d{2} \d{2}:\d{2}:\d{2}"
TERMINAL = {"Pass", "Wrong Answer", "Runtime Error", "Compile Error", "Time Limit Exceeded"}


class EvidenceError(ValueError):
    pass


def now_beijing():
    return datetime.now(SHANGHAI)


def stamp(value):
    if not isinstance(value, str) or not value:
        raise EvidenceError("Missing timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("/", "-").replace("Z", "+00:00"))
    except ValueError as error:
        raise EvidenceError("Malformed timestamp") from error
    return parsed.replace(tzinfo=SHANGHAI) if parsed.tzinfo is None else parsed.astimezone(SHANGHAI)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".new")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def route_id(url):
    # The result route uses a Mongo token; the displayed submission ID is a different number.
    match = re.search(r"^https://cannjudge\.cn/(?:.*/)?submission/([0-9a-f]{24}|\d+)(?:[/?#]|$)", str(url), re.I)
    return match[1] if match else None


def normalized_source(data):
    return data.decode("utf-8-sig").replace("\r\n", "\n").encode("utf-8")


def source_review(folder, config):
    folder = Path(folder)
    if folder.name in config.get("known_quarantine", []):
        raise EvidenceError("Candidate is quarantined")
    source = (folder / "kernel.asc").read_bytes()
    if source != normalized_source(source):
        raise EvidenceError("Candidate must use UTF-8 without BOM and LF, before review")
    review = load(folder / "review_checks.json")
    source_hash = digest(source)
    reviewed_hash = review.get("sha256", review.get("source_sha256"))
    if source_hash in config.get("quarantine_sha256", []):
        raise EvidenceError("Candidate source digest is quarantined even if its folder was renamed")
    if review.get("review_pass") is not True or reviewed_hash != source_hash:
        raise EvidenceError("Passing review does not bind to current source bytes")
    if review.get("post_submission_finding") or review.get("quarantined"):
        raise EvidenceError("Candidate has an unresolved correctness finding")
    checks = review.get("checks", {})
    for name in config.get("required_review_checks", []):
        if checks.get(name) is not True:
            raise EvidenceError("Missing passing framework-specific review: " + name)
    return source_hash


def parse_history_page(page):
    """Prefer actual row IDs. A timestamp is a fallback, never a unique ID by assertion."""
    text = page.get("text", "")
    rows = page.get("entries")
    if rows is None:
        rows = [{"submitted_at": value} for value in re.findall(STAMP, text)]
    parsed = []
    for row in rows:
        date = stamp(row.get("submitted_at", row.get("time")))
        ident = str(row.get("id", row.get("submission_id", "")))
        route = row.get("route_id") or route_id(row.get("url", ""))
        if ident and not ident.isdecimal():
            raise EvidenceError("Malformed history submission ID")
        parsed.append({"id": ident or None, "route_id": route,
                       "submitted_at": date.isoformat()})
    total = page.get("total_count")
    if total is None:
        found = re.search(r"共\s*(\d+)\s*条|(?:Total|total)\s*[:：]?\s*(\d+)\s*(?:items|records)", text)
        total = int(next(group for group in found.groups() if group is not None)) if found else None
    if total is not None and (not isinstance(total, int) or total < 0):
        raise EvidenceError("Malformed total history count")
    return parsed, total


def reconcile_history(pages, current=None, page_size=20, expected_scan=None):
    current = current or now_beijing()
    today = current.date()
    if not pages:
        raise EvidenceError("No current-day account history")
    today_rows = []
    all_ids = {}
    fallback_times = set()
    observed = []
    total_values = set()
    raw_rows = 0
    previous = None
    complete = False
    for number, page in enumerate(pages, 1):
        if page.get("page", number) != number:
            raise EvidenceError("History pages are not contiguous from page 1")
        if expected_scan is not None and page.get("scan_id") != expected_scan:
            raise EvidenceError("Mixed or stale history scan pages")
        observed_at = stamp(page.get("observed_at", page.get("observed_at_beijing")))
        if observed_at.date() != today or observed_at > current + timedelta(seconds=5):
            raise EvidenceError("History was not observed on current Shanghai day")
        observed.append(observed_at)
        rows, total = parse_history_page(page)
        if total is not None:
            total_values.add(total)
            if len(total_values) > 1:
                raise EvidenceError("History changed during pagination; repeat a complete scan")
        raw_rows += len(rows)
        for row in rows:
            date = stamp(row["submitted_at"])
            if date > current + timedelta(seconds=5):
                raise EvidenceError("History contains future submissions")
            if previous is not None and date > previous:
                raise EvidenceError("History is not in descending timestamp order")
            previous = date
            ident = ("route:" + row["route_id"]) if row.get("route_id") else (("display:" + row["id"]) if row["id"] else None)
            if ident:
                if ident in all_ids:
                    if all_ids[ident] != row["submitted_at"]:
                        raise EvidenceError("One history ID has conflicting timestamps")
                    # Repeated known IDs can be deduplicated; completeness still needs boundary evidence.
                    continue
                all_ids[ident] = row["submitted_at"]
            else:
                if row["submitted_at"] in fallback_times:
                    raise EvidenceError("Repeated timestamp without IDs is ambiguous; capture actual row IDs")
                fallback_times.add(row["submitted_at"])
            if date.date() == today:
                today_rows.append(row)
            elif date.date() < today:
                complete = True
        if complete:
            break
        known_total = next(iter(total_values)) if total_values else None
        if page.get("end_of_history") is True:
            complete = True
        elif known_total is not None and raw_rows == known_total:
            if len(all_ids) + len(fallback_times) != known_total:
                raise EvidenceError("Duplicate known IDs prevent proving a full account-history end")
            complete = True
        elif known_total is not None and raw_rows > known_total:
            raise EvidenceError("Pagination row count exceeds the observed account total")
        if complete:
            break
        if len(rows) != page_size:
            raise EvidenceError("Short/empty page without verified end or previous-day boundary")
    if not complete:
        raise EvidenceError("History did not reach previous day or a verified account-history end")
    # Mixed identified/fallback rows at one timestamp cannot prove whether they are duplicates.
    by_time = {}
    for row in today_rows:
        by_time.setdefault(row["submitted_at"], []).append(row)
    if any(len(group) > 1 and any(not row["id"] and not row.get("route_id") for row in group) for group in by_time.values()):
        raise EvidenceError("Mixed ID/timestamp identities are ambiguous")
    return {"date": str(today), "observed_at": min(observed).isoformat(),
            "completed_at": max(observed).isoformat(), "complete": True,
            "account_used": len(today_rows), "entries": today_rows,
            "pages_read": len(observed), "identity_mode": "id" if all(row["id"] or row.get("route_id") for row in today_rows) else "unique_timestamp_fallback",
            "scan_id": expected_scan}


def candidate_folders(config, config_path):
    base = Path(config_path).resolve().parent
    folders = set()
    for root_value in config.get("candidate_roots", [".."]):
        root = (base / root_value).resolve()
        if not root.exists():
            continue
        for pattern in ("submission_attempt.json", "submission_info.json", "platform_result.json"):
            folders.update(file.parent for file in root.rglob(pattern))
    return sorted(folders)


def own_records(folders):
    confirmed = {}
    unresolved = []
    latest_capture = None
    for folder in folders:
        info_path = folder / "submission_info.json"
        attempt_path = folder / "submission_attempt.json"
        info = load(info_path) if info_path.exists() else None
        attempt = load(attempt_path) if attempt_path.exists() else None
        if attempt:
            state = attempt.get("state")
            if state == "not_submitted_verified":
                resolution = attempt.get("resolution", {})
                if resolution.get("reviewed") is not True or not resolution.get("evidence"):
                    unresolved.append(folder.name)
            elif not info or not route_id(info.get("url", "")):
                # An unknown clicked attempt remains unknown on later calendar days.
                unresolved.append(folder.name)
            elif attempt.get("url") and attempt["url"] != info["url"]:
                unresolved.append(folder.name)
        if not info:
            continue
        ident = route_id(info.get("url", ""))
        if not ident:
            if folder.name not in unresolved:
                unresolved.append(folder.name)
            continue
        captured_at = stamp(info["captured_at"])
        if latest_capture is None or captured_at > latest_capture:
            latest_capture = captured_at
        submitted_at = None
        result_file = folder / "platform_result.json"
        if result_file.exists():
            raw = load(result_file)
            if raw.get("url") != info["url"]:
                raise EvidenceError("Result evidence URL differs from stored submission URL")
            match = re.search(r"(?:提交时间|Submitted at|Submission time)\s*[:：]\s*(" + STAMP + r")", raw.get("text", ""), re.I)
            if match:
                submitted_at = stamp(match[1])
        display_id = info.get("submission_id")
        if result_file.exists():
            numeric = re.search(r"(?:提交\s*ID|Submission ID)\s*[:：]\s*(\d+)", raw.get("text", ""), re.I)
            if numeric:
                if display_id is not None and str(display_id) != numeric[1]:
                    raise EvidenceError("Displayed submission ID differs from captured metadata")
                display_id = numeric[1]
        record = {"id": ident, "display_id": str(display_id) if display_id is not None else None,
                  "captured_at": captured_at, "submitted_at": submitted_at, "folder": folder.name}
        if ident in confirmed and confirmed[ident].get("submitted_at") != submitted_at:
            raise EvidenceError("Local evidence conflicts for one submission ID")
        confirmed[ident] = record
    return confirmed, sorted(set(unresolved)), latest_capture


def budget_gate(config, history, folders, current=None):
    current = current or now_beijing()
    if history.get("complete") is not True or history.get("date") != str(current.date()):
        raise EvidenceError("Fresh complete current-day history is required; new day never implies zero external submissions")
    age = (current - stamp(history["observed_at"])).total_seconds()
    if age < -5 or age > config.get("history_max_age_seconds", 60):
        raise EvidenceError("Account history is stale; refresh immediately before submit")
    entries = history["entries"]
    history_ids = {row["id"] for row in entries if row.get("id")}
    history_routes = {row["route_id"] for row in entries if row.get("route_id")}
    fallback_times = {row["submitted_at"] for row in entries if not row.get("id") and not row.get("route_id")}
    own, unresolved, latest = own_records(folders)
    if unresolved:
        raise EvidenceError("Unresolved submit attempts (including earlier days): " + ", ".join(unresolved))
    additional = set()
    own_today = set()
    for ident, record in own.items():
        # Actual submitted_at wins over delayed capture; absent metadata is counted conservatively.
        actual = record["submitted_at"] or record["captured_at"]
        if actual.date() != current.date():
            continue
        own_today.add(ident)
        represented = ident in history_routes or record.get("display_id") in history_ids
        if record["submitted_at"] is not None:
            represented = represented or record["submitted_at"].isoformat() in fallback_times
        if not represented:
            additional.add(ident)
    used = len(entries) + len(additional)
    limit, reserve = config.get("daily_limit", 50), config.get("reserve", 10)
    if not isinstance(limit, int) or not isinstance(reserve, int) or not 0 <= reserve < limit:
        raise EvidenceError("Invalid daily limit/reserve policy")
    spacing = max(0, math.ceil((latest + timedelta(seconds=config.get("spacing_seconds", 180)) - current).total_seconds())) if latest else 0
    report = {"date": str(current.date()), "account_history_used": len(entries),
              "own_today_confirmed": len(own_today), "own_not_yet_represented": len(additional),
              "conservative_account_used": used, "daily_limit": limit, "reserve": reserve,
              "remaining_before_reserve": limit - reserve - used, "minimum_wait_seconds": spacing,
              "history_observed_at": history["observed_at"], "history_identity_mode": history["identity_mode"]}
    if used >= limit - reserve:
        raise EvidenceError("Daily reserve reached: conservative account used=" + str(used))
    assumption = config.get("limit_assumption", {})
    if config.get("platform_limit_verified") is not True:
        if not assumption.get("source") or not assumption.get("date") or not isinstance(assumption.get("pilot_cap"), int):
            raise EvidenceError("Unverified current limit requires a documented conservative pilot cap")
        if assumption["date"] != str(current.date()):
            raise EvidenceError("Pilot policy must be renewed against current-day history")
        pilot_start = stamp(assumption.get("pilot_started_at"))
        pilot_used = sum((record["submitted_at"] or record["captured_at"]) >= pilot_start for record in own.values())
        if not 0 < assumption["pilot_cap"] <= 4:
            raise EvidenceError("Unverified-limit pilot must be at most four own submissions")
        if pilot_used >= assumption["pilot_cap"]:
            raise EvidenceError("Conservative unverified-limit pilot cap reached")
        report.update(limit_assumed=True, pilot_used=pilot_used, pilot_cap=assumption["pilot_cap"])
    return report


def gate(config_path, folder, history_path, current=None):
    config = load(config_path)
    if config.get("automatic_submission_enabled") is not True:
        raise EvidenceError("Automatic submission is disabled in this configuration")
    folder = Path(folder).resolve()
    if (folder / "submission_info.json").exists() or (folder / "submission_attempt.json").exists():
        raise EvidenceError("Candidate already has submission evidence; use a fresh immutable repeat folder")
    source_hash = source_review(folder, config)
    report = budget_gate(config, load(history_path), candidate_folders(config, config_path), current)
    report["source_sha256"] = source_hash
    return report


def parse_result(raw, expected_points=15):
    text = raw.get("text", "")
    row = {"status": "Unknown", "points": [], "all_pass": False, "terminal": False}
    field_patterns = {
        "submission_id": r"(?:提交\s*ID|Submission ID)\s*[:：]\s*(\d+)",
        "status": r"(?:提交状态|Submission status|Status)\s*[:：]\s*([^\r\n]+)",
        "submitted_at": r"(?:提交时间|Submission time|Submitted at)\s*[:：]\s*(" + STAMP + r")",
    }
    for name, expression in field_patterns.items():
        match = re.search(expression, text, re.I)
        if match:
            row[name] = match[1].strip()
    status = row["status"]
    row["terminal"] = status in TERMINAL
    # Each point occupies one copied table row; whitespace supports tabs, CRLF and narrow spaces.
    pattern = r"(?m)^\s*(\d+)\s+(Pass|Wrong Answer|Runtime Error|Compile Error|Time Limit Exceeded)\s+([\d.]+)\s*%\s+([\d.]+)\s*(μs|µs|us|ms)\s+([\d.]+)\s*(μs|µs|us|ms)\s*$"
    for match in re.finditer(pattern, text):
        ident, status, error, elapsed, unit, reference, ref_unit = match.groups()
        try:
            row["points"].append({"id": int(ident), "status": status, "error_pct": float(error),
                                  "us": float(elapsed) * (1000 if unit == "ms" else 1),
                                  "ref_us": float(reference) * (1000 if ref_unit == "ms" else 1)})
        except ValueError:
            raise EvidenceError("Malformed result number")
    row["all_pass"] = row["status"] == "Pass" and [point["id"] for point in row["points"]] == list(range(1, expected_points + 1)) and all(
        point["status"] == "Pass" and point["error_pct"] == 0 and math.isfinite(point["us"]) and point["us"] > 0 and math.isfinite(point["ref_us"]) and point["ref_us"] > 0 for point in row["points"])
    row["times_us"] = [point["us"] for point in row["points"]]
    row["reference_us"] = [point["ref_us"] for point in row["points"]]
    row["result_route_valid"] = bool(route_id(raw.get("url", "")) and row.get("submission_id"))
    if not row["result_route_valid"]:
        row["all_pass"] = False
    if row["status"] == "Pass" and not row["all_pass"]:
        row["terminal"] = False
    return row


def evidence(folder, config):
    folder = Path(folder)
    row = {"version": folder.name, "status": "Not submitted", "all_pass": False,
           "eligible_for_recommendation": False}
    row["sha256"] = digest((folder / "kernel.asc").read_bytes())
    try:
        row["source_review_pass"] = source_review(folder, config) == row["sha256"]
    except (EvidenceError, FileNotFoundError, UnicodeError, json.JSONDecodeError):
        row["source_review_pass"] = False
    web_path = folder / "web_verified.asc"
    row["submitted_source_hash_match"] = web_path.exists() and digest(normalized_source(web_path.read_bytes())) == row["sha256"]
    result_path = folder / "platform_result.json"
    info_path = folder / "submission_info.json"
    if result_path.exists():
        raw = load(result_path)
        row.update(parse_result(raw, config.get("expected_points", 15)))
        info = load(info_path) if info_path.exists() else {}
        row["result_url_matches_info"] = info.get("url") == raw.get("url") and bool(route_id(raw.get("url", "")))
        row["result_submission_id_matches_info"] = info.get("submission_id") is None or str(info["submission_id"]) == row.get("submission_id")
        if info.get("source_sha256") and info["source_sha256"] != row["sha256"]:
            row["result_url_matches_info"] = False
        row["eligible_for_recommendation"] = bool(row["all_pass"] and row["source_review_pass"] and row["submitted_source_hash_match"] and row["result_url_matches_info"] and row["result_submission_id_matches_info"])
    return row


def estimated_score(times, reference):
    if len(times) != len(reference) or not times:
        raise EvidenceError("Score requires equal nonempty vectors")
    if not all(math.isfinite(x) and x > 0 for x in times + reference):
        raise EvidenceError("Score requires finite positive durations")
    return sum(100 if elapsed <= ref else 100 / (1 + math.log(elapsed / ref, 1.5)) for elapsed, ref in zip(times, reference)) / len(times)


def analyze(folders, config, baseline=None):
    records = [evidence(folder, config) for folder in folders]
    base = next((row for row in records if baseline and row["version"] == Path(baseline).name and row["eligible_for_recommendation"]), None)
    if baseline and not base:
        raise EvidenceError("Selected baseline lacks complete source-bound device evidence")
    if base:
        for row in records:
            if row["all_pass"]:
                row["estimated_score_common_T"] = estimated_score(row["times_us"], base["reference_us"])
                row["estimated_score_delta_vs_baseline"] = row["estimated_score_common_T"] - estimated_score(base["times_us"], base["reference_us"])
                row["delta_pct_vs_baseline"] = [100 * (value / control - 1) for value, control in zip(row["times_us"], base["times_us"])]
    return {"updated_at_beijing": now_beijing().isoformat(), "reference_version": base["version"] if base else None,
            "common_T": base["reference_us"] if base else [], "records": records,
            "limitation": "Timing-derived scores are estimates; page rounding and device noise apply. Repeated paired controls are required to establish improvement; these values are not the live team scoreboard."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    history_cli = sub.add_parser("history")
    history_cli.add_argument("--pages", required=True)
    history_cli.add_argument("--output", required=True)
    history_cli.add_argument("--scan-id")
    history_cli.add_argument("--page-size", type=int, default=20)
    gate_cli = sub.add_parser("gate")
    gate_cli.add_argument("--config", required=True)
    gate_cli.add_argument("--candidate", required=True)
    gate_cli.add_argument("--history", required=True)
    gate_cli.add_argument("--output")
    result_cli = sub.add_parser("result")
    result_cli.add_argument("--config", required=True)
    result_cli.add_argument("--candidate", required=True)
    analysis_cli = sub.add_parser("analyze")
    analysis_cli.add_argument("--config", required=True)
    analysis_cli.add_argument("--candidate", action="append", required=True)
    analysis_cli.add_argument("--baseline")
    analysis_cli.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "history":
            paths = sorted(Path(args.pages).glob("page*.json"), key=lambda path: int(re.search(r"page(\d+)", path.name)[1]))
            result = reconcile_history([load(path) for path in paths], page_size=args.page_size, expected_scan=args.scan_id)
            save(args.output, result)
            # Account row text/IDs/URLs stay in local evidence, never printed.
            print(json.dumps({key: result[key] for key in ("date", "account_used", "complete", "pages_read", "identity_mode")}, ensure_ascii=False))
        elif args.command == "gate":
            result = gate(args.config, args.candidate, args.history)
            if args.output:
                save(args.output, result)
            print(json.dumps(result, ensure_ascii=False))
            return 3 if result["minimum_wait_seconds"] else 0
        elif args.command == "result":
            result = evidence(args.candidate, load(args.config))
            save(Path(args.candidate) / "hardware_validation.json", result)
            print(json.dumps({key: result.get(key) for key in ("version", "status", "all_pass", "eligible_for_recommendation")}, ensure_ascii=False))
            return 0 if result.get("terminal") else 4
        else:
            result = analyze(args.candidate, load(args.config), args.baseline)
            save(args.output, result)
            csv_path = Path(args.output).with_suffix(".csv")
            with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["version", "point", "status", "error_pct", "us", "ref_us", "delta_pct_vs_baseline"])
                for row in result["records"]:
                    for index, point in enumerate(row.get("points", [])):
                        writer.writerow([row["version"], point["id"], point["status"], point["error_pct"], point["us"], point["ref_us"], row.get("delta_pct_vs_baseline", [None] * len(row["points"]))[index]])
            print(json.dumps({"records": len(result["records"]), "eligible": sum(row["eligible_for_recommendation"] for row in result["records"]), "reference_version": result["reference_version"]}, ensure_ascii=False))
    except (EvidenceError, OSError, ValueError, KeyError) as error:
        print("BLOCKED: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
