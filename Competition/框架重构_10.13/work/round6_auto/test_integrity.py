"""Synthetic, isolated tests; no browser, network, or account records are read."""
import copy
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import cli


TIME = datetime(2026, 10, 13, 11, 0, 0, tzinfo=cli.SHANGHAI)
URL = "https://cannjudge.cn/public/test/quant/submission/000000000000000000000123"


def raw_result(status="Pass", ids=range(1, 16)):
    return {"url": URL, "text": "提交 ID：111111\n提交状态：" + status + "\n提交时间：2026/10/13 10:59:00\n" + "\n".join(
        f"{i}\tPass\t0%\t10μs\t2µs" for i in ids)}


def history(rows=None, observed=TIME):
    return {"date": "2026-10-13", "complete": True, "observed_at": observed.isoformat(),
            "entries": rows or [], "identity_mode": "id"}


class Integrity(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="r6_test_", dir=Path(__file__).parent)
        self.root = Path(self.temporary.name)
        self.folder = self.root / "v134_test"
        self.folder.mkdir()
        source = b"// synthetic source\nvoid kernel() {}\n"
        (self.folder / "kernel.asc").write_bytes(source)
        cli.save(self.folder / "review_checks.json", {"review_pass": True, "sha256": cli.digest(source)})
        self.config = {"daily_limit": 50, "reserve": 10, "platform_limit_verified": True,
                       "history_max_age_seconds": 60, "spacing_seconds": 180, "expected_points": 15}

    def tearDown(self):
        self.temporary.cleanup()

    def bind_source(self):
        # Native Windows clipboard BOM/CRLF is normalized; reviewed kernel bytes remain LF/no BOM.
        (self.folder / "web_verified.asc").write_bytes(b"\xef\xbb\xbf" + (self.folder / "kernel.asc").read_bytes().replace(b"\n", b"\r\n"))
        cli.save(self.folder / "submission_info.json", {"url": URL, "captured_at": "2026-10-13T10:59:03+08:00", "submission_id": "111111"})
        cli.save(self.folder / "platform_result.json", raw_result())

    def test_mongo_route_and_numeric_display_id_are_distinct(self):
        self.assertEqual(cli.route_id(URL), "000000000000000000000123")
        self.assertTrue(cli.parse_result(raw_result())["all_pass"])
        self.bind_source()
        self.assertTrue(cli.evidence(self.folder, self.config)["eligible_for_recommendation"])

    def test_numeric_display_id_conflict_blocks(self):
        self.bind_source()
        info = cli.load(self.folder / "submission_info.json")
        info["submission_id"] = "111"
        cli.save(self.folder / "submission_info.json", info)
        self.assertFalse(cli.evidence(self.folder, self.config)["eligible_for_recommendation"])

    def test_mutated_source_blocks_even_with_pass(self):
        self.bind_source()
        (self.folder / "kernel.asc").write_bytes(b"// different source\n")
        row = cli.evidence(self.folder, self.config)
        self.assertTrue(row["all_pass"])
        self.assertFalse(row["eligible_for_recommendation"])

    def test_missing_web_copy_blocks(self):
        self.bind_source()
        (self.folder / "web_verified.asc").unlink()
        self.assertFalse(cli.evidence(self.folder, self.config)["eligible_for_recommendation"])

    def test_duplicate_or_missing_point_ids_are_incomplete(self):
        for ids in ([1] * 15, range(1, 15), range(1, 17)):
            row = cli.parse_result(raw_result(ids=ids))
            self.assertFalse(row["all_pass"])
            self.assertFalse(row["terminal"])

    def test_running_zero_points_not_complete(self):
        row = cli.parse_result(raw_result(status="Running", ids=[]))
        self.assertFalse(row["all_pass"])
        self.assertFalse(row["terminal"])

    def test_compile_error_is_terminal_but_ineligible(self):
        row = cli.parse_result(raw_result(status="Compile Error", ids=[]))
        self.assertTrue(row["terminal"])
        self.assertFalse(row["all_pass"])

    def test_unit_normalization(self):
        raw = raw_result()
        raw["text"] = raw["text"].replace("10μs", "0.01ms")
        self.assertEqual(cli.parse_result(raw)["times_us"], [10.0] * 15)

    def page(self, entries, number=1, **extra):
        return {"observed_at": TIME.isoformat(), "page": number, "scan_id": "current", "entries": entries, **extra}

    def test_previous_day_boundary_completes_current_day(self):
        row = {"submitted_at": "2026/10/13 10:58:00"}
        older = {"submitted_at": "2026/10/12 23:58:00"}
        reconciled = cli.reconcile_history([self.page([row, older])], TIME)
        self.assertEqual(reconciled["account_used"], 1)

    def test_explicit_zero_total_proves_empty_day(self):
        reconciled = cli.reconcile_history([self.page([], total_count=0)], TIME)
        self.assertEqual(reconciled["account_used"], 0)

    def test_short_page_without_boundary_is_rejected(self):
        with self.assertRaises(cli.EvidenceError):
            cli.reconcile_history([self.page([{ "submitted_at": "2026/10/13 10:58:00"}])], TIME)

    def test_same_second_without_ids_cannot_be_deduplicated(self):
        row = {"submitted_at": "2026/10/13 10:58:00"}
        with self.assertRaises(cli.EvidenceError):
            cli.reconcile_history([self.page([row, row], end_of_history=True)], TIME)

    def test_same_second_distinct_numeric_ids_count_both(self):
        rows = [{"submitted_at": "2026/10/13 10:58:00", "id": str(i)} for i in [1, 2]]
        result = cli.reconcile_history([self.page(rows, end_of_history=True)], TIME)
        self.assertEqual(result["account_used"], 2)

    def test_one_id_is_deduplicated(self):
        row = {"submitted_at": "2026/10/13 10:58:00", "id": "1"}
        result = cli.reconcile_history([self.page([row, row], end_of_history=True)], TIME)
        self.assertEqual(result["account_used"], 1)

    def test_history_total_changing_during_scan_blocks(self):
        first = self.page([{ "submitted_at": "2026/10/13 10:59:00", "id": "1"}], total_count=2)
        second = self.page([{ "submitted_at": "2026/10/13 10:58:00", "id": "2"}], number=2, total_count=3)
        with self.assertRaises(cli.EvidenceError):
            cli.reconcile_history([first, second], TIME, page_size=1)

    def test_mixed_stale_scan_blocks(self):
        with self.assertRaises(cli.EvidenceError):
            cli.reconcile_history([self.page([], total_count=0)], TIME, expected_scan="another")

    def test_new_day_never_defaults_external_count_to_zero(self):
        old = history()
        old["date"] = "2026-10-12"
        with self.assertRaises(cli.EvidenceError):
            cli.budget_gate(self.config, old, [], TIME)

    def test_stale_history_blocks(self):
        with self.assertRaises(cli.EvidenceError):
            cli.budget_gate(self.config, history(observed=TIME - timedelta(seconds=61)), [], TIME)

    def test_old_unknown_clicked_attempt_still_blocks(self):
        cli.save(self.folder / "submission_attempt.json", {"state": "submit_invocation_started", "started_at": "2026-10-12T20:00:00+08:00"})
        with self.assertRaises(cli.EvidenceError):
            cli.budget_gate(self.config, history(), [self.folder], TIME)

    def test_old_valid_mongo_record_is_not_unresolved(self):
        self.bind_source()
        cli.save(self.folder / "submission_attempt.json", {"state": "url_captured", "url": URL, "started_at": "2026-10-13T10:59:00+08:00"})
        row = {"id": "111111", "route_id": None, "submitted_at": "2026-10-13T10:59:00+08:00"}
        report = cli.budget_gate(self.config, history([row]), [self.folder], TIME)
        self.assertEqual(report["conservative_account_used"], 1)
        self.assertEqual(report["minimum_wait_seconds"], 123)

    def test_other_account_activity_uses_full_budget(self):
        rows = [{"id": str(i), "submitted_at": "2026-10-13T10:00:00+08:00"} for i in range(40)]
        with self.assertRaises(cli.EvidenceError):
            cli.budget_gate(self.config, history(rows), [], TIME)

    def test_duplicate_ids_cannot_falsely_prove_total_end(self):
        row = {"submitted_at": "2026/10/13 10:58:00", "id": "1"}
        with self.assertRaises(cli.EvidenceError):
            cli.reconcile_history([self.page([row, row], total_count=2)], TIME)

    def test_daily_bucket_uses_actual_submission_not_delayed_capture(self):
        self.bind_source()
        info = cli.load(self.folder / "submission_info.json")
        info["captured_at"] = "2026-10-13T00:00:02+08:00"
        cli.save(self.folder / "submission_info.json", info)
        raw = raw_result()
        raw["text"] = raw["text"].replace("2026/10/13 10:59:00", "2026/10/12 23:59:59")
        cli.save(self.folder / "platform_result.json", raw)
        result = cli.budget_gate(self.config, history(), [self.folder], TIME)
        self.assertEqual(result["conservative_account_used"], 0)

    def test_pilot_cap_is_enforced_after_four_own_submissions(self):
        folders = []
        rows = []
        for index in range(4):
            folder = self.root / ("run_" + str(index))
            folder.mkdir()
            url = "https://cannjudge.cn/public/test/submission/" + ("%024x" % (index + 1))
            cli.save(folder / "submission_info.json", {"url": url, "captured_at": "2026-10-13T10:30:00+08:00"})
            folders.append(folder)
            rows.append({"id": None, "route_id": cli.route_id(url), "submitted_at": "2026-10-13T10:30:00+08:00"})
        config = dict(self.config, platform_limit_verified=False, limit_assumption={"source": "historical policy", "date": "2026-10-13", "pilot_started_at": "2026-10-13T10:00:00+08:00", "pilot_cap": 4})
        with self.assertRaises(cli.EvidenceError):
            cli.budget_gate(config, history(rows), folders, TIME)

    def test_unverified_limit_requires_documented_small_pilot(self):
        config = dict(self.config, platform_limit_verified=False)
        with self.assertRaises(cli.EvidenceError):
            cli.budget_gate(config, history(), [], TIME)
        config["limit_assumption"] = {"source": "historical working policy", "date": "2026-10-13", "pilot_started_at": "2026-10-13T10:00:00+08:00", "pilot_cap": 4}
        report = cli.budget_gate(config, history(), [], TIME)
        self.assertEqual(report["pilot_cap"], 4)
        self.assertTrue(report["limit_assumed"])

    def test_quarantine_is_enforced_independent_of_pass(self):
        config = dict(self.config, known_quarantine=[self.folder.name])
        with self.assertRaises(cli.EvidenceError):
            cli.source_review(self.folder, config)

    def test_quarantine_digest_survives_folder_rename(self):
        config = dict(self.config, quarantine_sha256=[cli.digest((self.folder / "kernel.asc").read_bytes())])
        with self.assertRaises(cli.EvidenceError):
            cli.source_review(self.folder, config)

    def test_framework_specific_required_review_fields(self):
        config = dict(self.config, required_review_checks=["precision", "memory"])
        with self.assertRaises(cli.EvidenceError):
            cli.source_review(self.folder, config)
        review = cli.load(self.folder / "review_checks.json")
        review["checks"] = {"precision": True, "memory": True}
        cli.save(self.folder / "review_checks.json", review)
        self.assertEqual(cli.source_review(self.folder, config), review["sha256"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
