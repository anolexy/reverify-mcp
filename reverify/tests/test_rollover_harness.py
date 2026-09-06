"""Rollover across harnesses: Codex, Gemini and OpenCode adapters share one state machine.

What must hold: each adapter measures the live context from its own transcript, extracts the
user's verbatim anchors, and speaks its harness's hook dialect (block / deny / clearContext /
plugin actions); install and uninstall are idempotent, reversible and confined to the
harness's own config files; a request without a session id is keyed by the directory.
"""

import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

tools_root = Path(__file__).resolve().parent.parent
if str(tools_root) not in sys.path:
    sys.path.insert(0, str(tools_root))

import rollover_harness as rh  # noqa: E402

HANDOFF_OK = textwrap.dedent("""\
    # Rollover hand-off
    written: now
    ## Task and goal
    map the loader
    ## Decisions the user made
    none
    ## Next step
    exports next
    """)


def codex_line(kind, payload, when="2026-09-01T14:12:08.871Z", ordinal=0):
    return json.dumps({"timestamp": when, "ordinal": ordinal, "type": kind, "payload": payload})


def codex_user(text, when="2026-09-01T14:11:57.051Z"):
    return codex_line("response_item", {"type": "message", "role": "user",
                                        "content": [{"type": "input_text", "text": text}]}, when)


def codex_tokens(total, when="2026-09-01T14:12:08.871Z"):
    usage = {"input_tokens": total - 200, "cached_input_tokens": 100, "cache_write_input_tokens": 0,
             "output_tokens": 200, "reasoning_output_tokens": 50, "total_tokens": total}
    return codex_line("event_msg", {"type": "token_count", "info": {"total_token_usage": usage, "last_token_usage": usage,
                                                                      "model_context_window": 258400}}, when)


def gemini_user(text, when="2026-09-04T09:00:00.000Z", parts=False):
    content = [{"text": text}] if parts else text
    return json.dumps({"id": "u", "timestamp": when, "type": "user", "content": content})


def gemini_reply(total, when="2026-09-04T09:00:10.000Z"):
    return json.dumps({"id": "g", "timestamp": when, "type": "gemini", "content": [{"text": "ok"}],
                       "tokens": {"input": total - 30, "output": 20, "cached": 5, "thoughts": 10, "tool": 0, "total": total}})


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.home = self.root / "home"
        self.home.mkdir()
        self.cwd = self.root / "proj"
        self.cwd.mkdir()
        self._env = dict(os.environ)
        os.environ[rh.ENV_HOME] = str(self.home)
        os.environ[rh.ENV_STATE_DIR] = str(self.root / "state")
        # the successor opt-in reaches every shell started inside a Claude Code session (settings.json `env`);
        # with it set, a receipt in these tests would start a real `claude --bg` session
        for key in (rh.ENV_TOKENS, rh.ENV_STEP, rh.ENV_LAUNCH_ID, rh.ENV_SETTINGS, "OPENCODE_CONFIG_DIR", "OPENCODE_DB",
                    rh.ClaudeHarness.SUCCESSOR_ENV) + rh.SESSION_ENV_VARS:
            os.environ.pop(key, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)
        self.tmp.cleanup()

    def hook(self, harness, event, payload, env=None):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(rh.run_hook(harness, event, payload, env), 0)
        text = out.getvalue().strip()
        return json.loads(text) if text.startswith("{") else (text or None)

    @property
    def handoff(self):
        return self.cwd / ".reverify" / rh.HANDOFF_NAME


class CodexTranscript(Base):
    def rollout(self, name="rollout-2026-09-01T22-11-53-sess-codex-1.jsonl"):
        day = self.home / ".codex" / "sessions" / "2026" / "09" / "01"
        day.mkdir(parents=True, exist_ok=True)
        path = day / name
        path.write_text("\n".join([
            codex_line("session_meta", {"session_id": "sess-codex-1", "cwd": str(self.cwd)}, "2026-09-01T14:11:53.817Z"),
            codex_user("<user_instructions>\nignore me\n</user_instructions>", "2026-09-01T14:11:57.001Z"),
            codex_user("write the AOB scanner", "2026-09-01T14:11:57.051Z"),
            codex_tokens(21307, "2026-09-01T14:12:08.871Z"),
            codex_user("now anti-debug", "2026-09-01T14:20:00.000Z"),
            codex_tokens(47617, "2026-09-01T14:21:00.000Z"),
        ]) + "\n", encoding="utf-8")
        return path

    def test_tokens_anchors_and_lookup(self):
        path = self.rollout()
        self.assertEqual(rh.codex_context_tokens(path), 47617)
        anchors = rh.codex_anchors(path)
        self.assertEqual(anchors["first_user_message"], "write the AOB scanner")
        self.assertEqual(anchors["last_user_message"], "now anti-debug")
        self.assertEqual(len(anchors["transcript_sha256"]), 64)
        self.assertEqual(rh.codex_find_rollout("sess-codex-1"), path)
        self.assertIsNone(rh.codex_find_rollout("nope"))
        self.assertTrue(rh.codex_user_message_after(path, rh.parse_iso("2026-09-01T14:15:00Z")))
        self.assertFalse(rh.codex_user_message_after(path, rh.parse_iso("2026-09-01T14:25:00Z")))

    def test_stop_hook_blocks_then_receipt_with_null_transcript_path(self):
        self.rollout()
        harness = rh.CodexHarness()
        os.environ[rh.ENV_TOKENS] = "40k"
        payload = {"session_id": "sess-codex-1", "transcript_path": None, "cwd": str(self.cwd), "hook_event_name": "Stop"}
        first = self.hook(harness, "stop", payload)
        self.assertEqual(first["decision"], "block")
        self.assertIn(str(self.handoff), first["reason"])
        self.assertIn("48k", first["reason"])
        self.handoff.parent.mkdir(parents=True)
        self.handoff.write_text(HANDOFF_OK, encoding="utf-8")
        self.assertIsNone(self.hook(harness, "stop", dict(payload, stop_hook_active=True)))
        receipt = json.loads(rh.receipt_path("sess-codex-1").read_text(encoding="utf-8"))
        self.assertEqual(receipt["harness"], "codex")
        self.assertEqual(receipt["first_user_message"], "write the AOB scanner")
        self.assertTrue(receipt["transcript_path"].endswith(".jsonl"))
        pointer = self.hook(harness, "session-start", payload)
        self.assertIn(str(self.handoff), pointer)


class CodexInstall(Base):
    def test_config_toml_transform(self):
        empty = rh.CodexHarness.apply_config_toml("", True)
        self.assertIn(f"model_auto_compact_token_limit = {rh.CODEX_NO_COMPACT_LIMIT}", empty)
        self.assertIn("[features]\nhooks = true", empty)
        existing = 'model = "gpt-5"\n\n[features]\nhooks = false\nother = 1\n\n[tui]\ntheme = "x"\n'
        once = rh.CodexHarness.apply_config_toml(existing, True)
        twice = rh.CodexHarness.apply_config_toml(once, True)
        self.assertEqual(once, twice)
        lines = once.splitlines()
        self.assertTrue(lines[1].startswith("model_auto_compact_token_limit ="), lines)
        self.assertEqual(lines.index("[features]") + 1, lines.index("hooks = true"))
        self.assertNotIn("hooks = false", once)
        self.assertIn("other = 1", once)
        self.assertIn('theme = "x"', once)
        kept = rh.CodexHarness.apply_config_toml(existing, False)
        self.assertNotIn("model_auto_compact_token_limit", kept)
        stripped = rh.CodexHarness.strip_config_toml(once)
        self.assertNotIn("model_auto_compact_token_limit", stripped)
        self.assertIn("hooks = true", stripped)

    def test_install_and_uninstall_on_disk(self):
        codex = self.home / ".codex"
        codex.mkdir()
        (codex / "config.toml").write_text('model = "gpt-5"\n[features]\nhooks = false\n', encoding="utf-8")
        (codex / "hooks.json").write_text(json.dumps({"hooks": {"PostCompact": [{"hooks": [{"type": "command", "command": "node x.mjs"}]}]}}), encoding="utf-8")
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(rh.run_install(["--harness", "codex"]), 0)
        hooks = json.loads((codex / "hooks.json").read_text(encoding="utf-8"))["hooks"]
        self.assertEqual(len(hooks["PostCompact"]), 1)
        self.assertIn("hook codex stop", hooks["Stop"][0]["hooks"][0]["command"])
        self.assertIn("hook codex session-start", hooks["SessionStart"][0]["hooks"][0]["command"])
        config = (codex / "config.toml").read_text(encoding="utf-8")
        self.assertIn("hooks = true", config)
        self.assertIn("model_auto_compact_token_limit", config)
        self.assertTrue(any(p.name.startswith("config.toml.bak-reverify-") for p in codex.iterdir()))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(rh.run_install(["--harness", "codex"]), 0)
        self.assertEqual(len(json.loads((codex / "hooks.json").read_text(encoding="utf-8"))["hooks"]["Stop"]), 1)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(rh.run_uninstall(["--harness", "codex"]), 0)
        hooks = json.loads((codex / "hooks.json").read_text(encoding="utf-8"))["hooks"]
        self.assertEqual(list(hooks), ["PostCompact"])
        self.assertNotIn("model_auto_compact_token_limit", (codex / "config.toml").read_text(encoding="utf-8"))


class GeminiTranscript(Base):
    def chat(self):
        chats = self.home / ".gemini" / "tmp" / "proj" / "chats"
        chats.mkdir(parents=True)
        path = chats / "session-2026-09-04T09-00-abcd1234.jsonl"
        path.write_text("\n".join([
            json.dumps({"sessionId": "sess-gem-1", "projectHash": "proj", "startTime": "2026-09-04T09:00:00.000Z"}),
            gemini_user("map the loader"),
            gemini_reply(12000),
            gemini_user("now the exports", "2026-09-04T09:05:00.000Z", parts=True),
            gemini_reply(250000, "2026-09-04T09:05:10.000Z"),
        ]) + "\n", encoding="utf-8")
        return path

    def test_tokens_and_anchors(self):
        path = self.chat()
        self.assertEqual(rh.gemini_context_tokens(path), 250000)
        anchors = rh.gemini_anchors(path)
        self.assertEqual(anchors["first_user_message"], "map the loader")
        self.assertEqual(anchors["last_user_message"], "now the exports")
        self.assertTrue(rh.gemini_user_message_after(path, rh.parse_iso("2026-09-04T09:04:00Z")))
        self.assertFalse(rh.gemini_user_message_after(path, rh.parse_iso("2026-09-04T09:06:00Z")))

    def test_after_agent_deny_then_clear_context_then_before_agent_injection(self):
        path = self.chat()
        harness = rh.GeminiHarness()
        payload = {"session_id": "sess-gem-1", "transcript_path": str(path), "cwd": str(self.cwd),
                   "hook_event_name": "AfterAgent", "prompt": "x", "prompt_response": "y", "stop_hook_active": False}
        first = self.hook(harness, "stop", payload)
        self.assertEqual(first["decision"], "deny")
        self.assertIn(str(self.handoff), first["reason"])
        # nothing written -> fail closed, no clearContext
        self.assertIsNone(self.hook(harness, "stop", dict(payload, stop_hook_active=True)))
        state = rh.load_state("sess-gem-1", rh.DEFAULT_THRESHOLD)
        self.assertIn("hand-off file missing", state["last_outcome"])
        self.assertIsNone(state["opening_pending"])
        # re-armed further up: grow the transcript and block again, this time with a real hand-off
        with path.open("a", encoding="utf-8") as handle:
            handle.write(gemini_reply(360000, "2026-09-04T09:30:00.000Z") + "\n")
        self.assertEqual(self.hook(harness, "stop", payload)["decision"], "deny")
        self.handoff.parent.mkdir(parents=True)
        self.handoff.write_text(HANDOFF_OK, encoding="utf-8")
        cleared = self.hook(harness, "stop", dict(payload, stop_hook_active=True))
        self.assertEqual(cleared, {"hookSpecificOutput": {"hookEventName": "AfterAgent", "clearContext": True}})
        receipt = json.loads(rh.receipt_path("sess-gem-1").read_text(encoding="utf-8"))
        self.assertEqual(receipt["harness"], "gemini")
        self.assertEqual(receipt["first_user_message"], "map the loader")
        injected = self.hook(harness, "before-agent", dict(payload, hook_event_name="BeforeAgent"))
        self.assertEqual(injected["hookSpecificOutput"]["hookEventName"], "BeforeAgent")
        context = injected["hookSpecificOutput"]["additionalContext"]
        self.assertIn(str(self.handoff), context)
        self.assertIn("«map the loader»", context)
        self.assertIsNone(self.hook(harness, "before-agent", dict(payload, hook_event_name="BeforeAgent")))
        start = self.hook(harness, "session-start", dict(payload, hook_event_name="SessionStart", source="startup"))
        self.assertEqual(start["hookSpecificOutput"]["hookEventName"], "SessionStart")
        self.assertIn(str(self.handoff), start["hookSpecificOutput"]["additionalContext"])

    def test_launcher_mode_leaves_the_reset_to_the_launcher(self):
        path = self.chat()
        harness = rh.GeminiHarness()
        env = dict(os.environ)
        env[rh.ENV_LAUNCH_ID] = "launch-g"
        payload = {"session_id": "sess-gem-2", "transcript_path": str(path), "cwd": str(self.cwd)}
        self.hook(harness, "stop", payload, env=env)
        self.handoff.parent.mkdir(parents=True)
        self.handoff.write_text(HANDOFF_OK, encoding="utf-8")
        self.assertIsNone(self.hook(harness, "stop", dict(payload, stop_hook_active=True), env=env))
        self.assertTrue(rh.receipt_path("launch-g").is_file())
        self.assertIsNone(rh.load_state("sess-gem-2", rh.DEFAULT_THRESHOLD)["opening_pending"])
        self.assertEqual(harness.launch_args(["--model", "x"], "OPEN"), ["--model", "x", "-i", "OPEN"])

    def test_install_and_uninstall(self):
        gem = self.home / ".gemini"
        gem.mkdir()
        (gem / "settings.json").write_text(json.dumps({"security": {"auth": {"selectedType": "oauth-personal"}}}), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(rh.run_install(["--harness", "gemini"]), 0)
            self.assertEqual(rh.run_install(["--harness", "gemini"]), 0)
        data = json.loads((gem / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(set(data["hooks"]), {"AfterAgent", "BeforeAgent", "SessionStart"})
        self.assertEqual(len(data["hooks"]["AfterAgent"]), 1)
        entry = data["hooks"]["AfterAgent"][0]["hooks"][0]
        self.assertEqual(entry["name"], "reverify-rollover")
        self.assertIn("hook gemini stop", entry["command"])
        self.assertEqual(entry["timeout"], 20000)
        self.assertEqual(data["model"]["compressionThreshold"], 2)
        self.assertEqual(data["security"]["auth"]["selectedType"], "oauth-personal")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(rh.run_uninstall(["--harness", "gemini"]), 0)
        data = json.loads((gem / "settings.json").read_text(encoding="utf-8"))
        self.assertEqual(data, {"security": {"auth": {"selectedType": "oauth-personal"}}})


class OpenCodeAdapter(Base):
    def test_idle_protocol_prompt_then_rollover(self):
        harness = rh.OpenCodeHarness()
        payload = {"session_id": "ses_1", "cwd": str(self.cwd), "tokens": 250000,
                   "first_user_message": "map the loader", "last_user_message": "now exports", "stop_hook_active": False}
        first = self.hook(harness, "idle", payload)
        self.assertEqual(first["action"], "prompt")
        self.assertIn(str(self.handoff), first["text"])
        missing = self.hook(harness, "idle", dict(payload, stop_hook_active=True))
        self.assertEqual(missing["action"], "none")
        with_room = self.hook(harness, "idle", dict(payload, tokens=300000))
        self.assertEqual(with_room["action"], "none")
        again = self.hook(harness, "idle", dict(payload, tokens=360000))
        self.assertEqual(again["action"], "prompt")
        self.handoff.parent.mkdir(parents=True)
        self.handoff.write_text(HANDOFF_OK, encoding="utf-8")
        done = self.hook(harness, "idle", dict(payload, tokens=361000, stop_hook_active=True))
        self.assertEqual(done["action"], "rollover")
        self.assertIn("«map the loader»", done["opening"])
        self.assertIn("«now exports»", done["opening"])
        self.assertEqual(done["handoff"], str(self.handoff))
        receipt = json.loads(rh.receipt_path("ses_1").read_text(encoding="utf-8"))
        self.assertEqual(receipt["harness"], "opencode")
        self.assertEqual(receipt["context_tokens"], 361000)
        self.assertEqual(self.hook(harness, "idle", dict(payload, tokens=10))["action"], "none")

    def test_session_stats_from_sqlite(self):
        db = self.root / "opencode.db"
        con = sqlite3.connect(db)
        con.executescript("""
            create table message (id text primary key, session_id text, time_created integer, time_updated integer, data text);
            create table part (id text primary key, message_id text, session_id text, time_created integer, time_updated integer, data text);
        """)
        rows = [
            ("m1", "ses_1", 1, json.dumps({"role": "user"})),
            ("m2", "ses_1", 2, json.dumps({"role": "assistant", "tokens": {"total": 11185, "input": 9249, "output": 16, "cache": {"read": 1920}}})),
            ("m3", "ses_1", 3, json.dumps({"role": "user"})),
            ("m4", "ses_1", 4, json.dumps({"role": "assistant", "tokens": {"total": 0, "input": 20000, "output": 100, "cache": {"read": 500}}})),
            ("m9", "ses_2", 5, json.dumps({"role": "assistant", "tokens": {"total": 999999}})),
        ]
        for mid, sid, t, data in rows:
            con.execute("insert into message values (?,?,?,?,?)", (mid, sid, t, t, data))
        parts = [("p1", "m1", json.dumps({"type": "text", "text": "map the loader"})),
                 ("p2", "m3", json.dumps({"type": "text", "text": "<system>ignored"})),
                 ("p3", "m3", json.dumps({"type": "file", "url": "x"}))]
        for pid, mid, data in parts:
            con.execute("insert into part values (?,?,?,?,?,?)", (pid, mid, "ses_1", 1, 1, data))
        con.commit()
        con.close()
        stats = rh.opencode_session_stats("ses_1", db)
        self.assertEqual(stats["tokens"], 20600)
        self.assertEqual(stats["first_user_message"], "map the loader")
        self.assertEqual(stats["last_user_message"], "map the loader")
        self.assertEqual(rh.opencode_session_stats("nope", db)["tokens"], None)
        os.environ["OPENCODE_DB"] = str(db)
        self.assertEqual(rh.OpenCodeHarness().context_tokens(None, "ses_2"), 999999)

    def test_install_writes_plugin_and_config(self):
        cfg = self.home / ".config" / "opencode"
        cfg.mkdir(parents=True)
        (cfg / "opencode.json").write_text(json.dumps({"model": "x/y", "compaction": {"prune": True}}), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(rh.run_install(["--harness", "opencode"]), 0)
        plugin = cfg / "plugins" / rh.OPENCODE_PLUGIN_NAME
        source = plugin.read_text(encoding="utf-8")
        self.assertNotIn("__REVERIFY_COMMAND__", source)
        self.assertIn("rollover_harness.py", source)
        self.assertIn('"hook", "opencode", "idle"', source)
        data = json.loads((cfg / "opencode.json").read_text(encoding="utf-8"))
        self.assertEqual(data["compaction"], {"prune": True, "auto": False})
        self.assertEqual(data["model"], "x/y")
        node = shutil.which("node")
        if node:
            check = subprocess.run([node, "--check", str(plugin)], capture_output=True, text=True)
            self.assertEqual(check.returncode, 0, check.stderr)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(rh.run_uninstall(["--harness", "opencode"]), 0)
        self.assertFalse(plugin.exists())
        self.assertEqual(json.loads((cfg / "opencode.json").read_text(encoding="utf-8"))["compaction"], {"prune": True})
        self.assertEqual(rh.OpenCodeHarness().launch_args([], "OPEN"), ["--prompt", "OPEN"])


class WindowSafeguard(Base):
    def test_threshold_is_capped_by_the_model_window(self):
        self.assertEqual(rh.effective_threshold(200_000, None), 200_000)
        self.assertEqual(rh.effective_threshold(200_000, 258_400), 193_800)
        self.assertEqual(rh.effective_threshold(200_000, 1_000_000), 200_000)
        self.assertEqual(rh.effective_threshold(50_000, 128_000), 50_000)

    def test_codex_small_window_triggers_below_the_configured_threshold(self):
        day = self.home / ".codex" / "sessions" / "2026" / "09" / "01"
        day.mkdir(parents=True)
        path = day / "rollout-2026-09-01T22-11-53-sess-small.jsonl"

        def tokens(total):
            usage = {"input_tokens": total - 100, "cached_input_tokens": 0, "cache_write_input_tokens": 0,
                     "output_tokens": 100, "reasoning_output_tokens": 0, "total_tokens": total}
            return codex_line("event_msg", {"type": "token_count", "info": {"total_token_usage": usage, "last_token_usage": usage,
                                                                              "model_context_window": 128000}})

        path.write_text("\n".join([codex_user("go"), tokens(97_000)]) + "\n", encoding="utf-8")
        self.assertEqual(rh.codex_context_window(path), 128000)
        harness = rh.CodexHarness()
        payload = {"session_id": "sess-small", "transcript_path": str(path), "cwd": str(self.cwd)}
        blocked = self.hook(harness, "stop", payload)          # 97k >= 75% of 128k, though < 200k
        self.assertEqual(blocked["decision"], "block")
        self.assertIn("threshold 96k", blocked["reason"])
        self.assertIsNone(self.hook(harness, "stop", dict(payload, stop_hook_active=True)))
        state = rh.load_state("sess-small", rh.DEFAULT_THRESHOLD)
        self.assertEqual(state["context_window"], 128000)
        self.assertEqual(state["next_trigger"], 97_000 + 12_800)   # re-arm step shrinks with the window


class RunAlias(Base):
    def test_cli_name_can_lead_the_arguments(self):
        harness, rest = rh.resolve_run_harness([], ["codex", "--full-auto"])
        self.assertEqual((harness.name, rest), ("codex", ["--full-auto"]))
        harness, rest = rh.resolve_run_harness(["--harness", "gemini"], ["codex", "x"])
        self.assertEqual((harness.name, rest), ("gemini", ["codex", "x"]))
        original = shutil.which
        try:
            shutil.which = lambda name: "/bin/" + name if name == "opencode" else None
            harness, rest = rh.resolve_run_harness([], ["--model", "m"])
            self.assertEqual((harness.name, rest), ("opencode", ["--model", "m"]))
            shutil.which = lambda name: None
            with self.assertRaises(FileNotFoundError):
                rh.resolve_run_harness([], [])
            shutil.which = lambda name: "/bin/" + name if name in ("codex", "gemini") else None
            with self.assertRaises(ValueError):
                rh.resolve_run_harness([], [])
            shutil.which = lambda name: "/bin/" + name
            harness, _ = rh.resolve_run_harness([], [])
            self.assertEqual(harness.name, "claude")
        finally:
            shutil.which = original


class DoctorAndInstructions(Base):
    def test_doctor_reports_install_state_and_broken_commands(self):
        os.environ[rh.ENV_SETTINGS] = str(self.root / "settings.json")
        with redirect_stdout(io.StringIO()):
            rh.run_install(["--harness", "claude,gemini"])
        rows = {r["harness"]: r for r in rh.doctor_report()}
        self.assertTrue(rows["claude"]["compaction_off"])
        self.assertTrue(all(rows["claude"]["hooks"].values()))
        self.assertEqual(rows["claude"]["problems"], [])
        self.assertTrue(rows["gemini"]["compaction_off"])
        self.assertIn("Stop: not installed", rows["codex"]["problems"])
        self.assertFalse(rows["opencode"]["compaction_off"])
        # break the interpreter path -> doctor notices
        data = json.loads((self.root / "settings.json").read_text(encoding="utf-8"))
        data["hooks"]["Stop"][0]["hooks"][0]["command"] = '"/nonexistent/python" "/nonexistent/rollover_harness.py" hook claude stop'
        (self.root / "settings.json").write_text(json.dumps(data), encoding="utf-8")
        rows = {r["harness"]: r for r in rh.doctor_report()}
        self.assertTrue(any("missing" in p for p in rows["claude"]["problems"]))
        out = io.StringIO()
        with redirect_stdout(out):
            code = rh.run_doctor([])
        self.assertIn("threshold  :", out.getvalue())
        self.assertIn("claude", out.getvalue())

    def test_doctor_flags_receipts_no_launcher_consumed(self):
        os.environ[rh.ENV_SETTINGS] = str(self.root / "settings.json")
        with redirect_stdout(io.StringIO()):
            rh.run_install(["--harness", "claude"])
        events = rh.state_dir() / "events.jsonl"
        events.parent.mkdir(parents=True, exist_ok=True)
        events.write_text("\n".join([
            json.dumps({"at": "2026-09-04T17:53:28Z", "event": "receipt", "harness": "claude", "session": "s1",
                        "tokens": 371615, "launch_id": None, "inline": False}),
            json.dumps({"at": "2026-09-05T07:50:14Z", "event": "receipt", "harness": "claude", "session": "s1",
                        "tokens": 815959, "launch_id": None, "inline": False}),
            json.dumps({"at": "2026-09-05T08:00:00Z", "event": "receipt", "harness": "claude", "session": "s2",
                        "tokens": 205000, "launch_id": "abc", "inline": False}),          # consumed by a launcher
            json.dumps({"at": "2026-09-05T08:10:00Z", "event": "receipt", "harness": "gemini", "session": "g1",
                        "tokens": 210000, "launch_id": None, "inline": True}),            # inline reset: fine
            "not json",
        ]) + "\n", encoding="utf-8")
        rows = {r["harness"]: r for r in rh.doctor_report()}
        self.assertEqual(rows["claude"]["unconsumed_receipts"]["count"], 2)
        self.assertEqual(rows["claude"]["unconsumed_receipts"]["peak"], 815959)
        problem = " ".join(rows["claude"]["problems"])
        self.assertIn("launcher did not start", problem)
        self.assertIn("816k", problem)
        self.assertIn("reverify rollover claude", problem)
        self.assertEqual(rows["gemini"]["unconsumed_receipts"]["count"], 0)
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(rh.run_doctor([]), 1)
        self.assertIn("launcher did not start", out.getvalue())

    def test_claude_successor_uses_job_respawn_flags_and_strips_session_identity(self):
        harness = rh.ClaudeHarness()
        job = self.root / "job"
        job.mkdir()
        (job / "state.json").write_text(json.dumps({"respawnFlags": ["--reply-on-resume", "--effort", "low", "--model", "m"]}),
                                        encoding="utf-8")
        original_resolve = rh.ClaudeHarness.resolve_exe
        rh.ClaudeHarness.resolve_exe = lambda self: "/bin/claude"
        calls = []

        class Done:
            returncode = 0
            stdout = "backgrounded \u00b7 \x1b[36mdeadbeef\x1b[39m\n  claude attach deadbeef".encode("utf-8")
            stderr = b""

        def fake_run(cmd, **kwargs):
            calls.append((cmd, kwargs))
            return Done()

        original_run = rh.subprocess.run
        rh.subprocess.run = fake_run
        try:
            env = {"CLAUDE_JOB_DIR": str(job), "CLAUDE_CODE_SESSION_ID": "old", "CLAUDECODE": "1", "PATH": "x",
                   rh.ClaudeHarness.SUCCESSOR_ENV: "bg"}
            self.assertEqual(harness.successor_command("OPEN", env),
                             ["/bin/claude", "--bg", "--effort", "low", "--model", "m", "OPEN"])
            self.assertEqual(harness.spawn_successor("OPEN", str(self.cwd), env), "deadbeef")
            cmd, kwargs = calls[0]
            self.assertEqual(cmd[-1], "OPEN")
            self.assertNotIn("CLAUDE_CODE_SESSION_ID", kwargs["env"])
            self.assertNotIn("CLAUDECODE", kwargs["env"])
            self.assertEqual(kwargs["env"]["PATH"], "x")
            self.assertEqual(kwargs["cwd"], str(self.cwd))
            # not opted in -> nothing spawned
            self.assertIsNone(harness.spawn_successor("OPEN", None, {"CLAUDE_JOB_DIR": str(job)}))
            self.assertEqual(len(calls), 1)
        finally:
            rh.subprocess.run = original_run
            rh.ClaudeHarness.resolve_exe = original_resolve

    def test_only_one_successor_per_session(self):
        calls = []

        class Fake(rh.Harness):
            name = "claude"

            def context_tokens(self, transcript, session_id):
                return 250_000

            def anchors(self, transcript, session_id):
                return {}

            def spawn_successor(self, opening, cwd, env):
                calls.append(opening)
                return f"succ{len(calls)}"

        harness = Fake()
        transcript = self.cwd / "t.jsonl"
        transcript.write_text("", encoding="utf-8")
        env = {}
        for _ in range(2):                                   # two full block -> receipt cycles
            result = rh.run_guard(harness, "sess-1", str(transcript), str(self.cwd), 250_000, lambda: {}, env)
            self.assertEqual(result["action"], "block")
            time.sleep(1.1)
            self.handoff.parent.mkdir(parents=True, exist_ok=True)
            self.handoff.write_text(HANDOFF_OK, encoding="utf-8")
            result = rh.run_guard(harness, "sess-1", str(transcript), str(self.cwd), 250_000, lambda: {}, env)
            self.assertEqual(result["action"], "receipt")
            state = rh.load_state("sess-1", 200_000)
            state["next_trigger"] = 0                         # force the next cycle to fire again
            rh.save_state(state)
        self.assertEqual(calls, [calls[0]])                  # spawned once, not twice
        self.assertEqual(rh.load_state("sess-1", 200_000)["successor"], "succ1")

    def test_doctor_downgrades_old_receipts_to_a_note_once_the_successor_is_on(self):
        os.environ[rh.ENV_SETTINGS] = str(self.root / "settings.json")
        with redirect_stdout(io.StringIO()):
            rh.run_install(["--harness", "claude"])
        settings_file = self.root / "settings.json"
        settings = json.loads(settings_file.read_text(encoding="utf-8"))
        settings.setdefault("env", {})[rh.ClaudeHarness.SUCCESSOR_ENV] = "bg"
        settings_file.write_text(json.dumps(settings), encoding="utf-8")
        events = rh.state_dir() / "events.jsonl"
        events.parent.mkdir(parents=True, exist_ok=True)
        events.write_text(json.dumps({"at": "2026-09-05T07:50:14Z", "event": "receipt", "harness": "claude", "session": "s1",
                                      "tokens": 815959, "launch_id": None, "inline": False}) + "\n", encoding="utf-8")
        rows = {r["harness"]: r for r in rh.doctor_report()}
        self.assertTrue(rows["claude"]["successor"])
        self.assertEqual(rows["claude"]["problems"], [])
        self.assertIn("1 earlier hand-off receipt(s) had no successor (peak 816k", " ".join(rows["claude"]["notes"]))
        self.assertNotIn("successor", rows["codex"])
        out = io.StringIO()
        with redirect_stdout(out):
            rh.run_doctor([])
        self.assertIn("successor: on", out.getvalue())
        self.assertIn("     . 1 earlier hand-off receipt", out.getvalue())

    def test_doctor_counts_receipt_with_successor_as_consumed(self):
        events = rh.state_dir() / "events.jsonl"
        events.parent.mkdir(parents=True, exist_ok=True)
        events.write_text(json.dumps({"at": "2026-09-06T00:00:00Z", "event": "receipt", "harness": "claude",
                                      "session": "s", "tokens": 205000, "launch_id": None, "inline": False,
                                      "successor": "deadbeef"}) + "\n", encoding="utf-8")
        self.assertEqual(rh.unconsumed_receipts("claude")["count"], 0)

    def test_claude_successor_starts_in_the_old_sessions_project(self):
        harness = rh.ClaudeHarness()
        project = self.cwd
        transcript = self.home / ".claude" / "projects" / rh.claude_project_slug(str(project)) / "sess.jsonl"
        deep = project / "worktree" / "sub"                  # where a `cd` inside the old session left the hook
        deep.mkdir(parents=True)
        self.assertEqual(rh.claude_project_slug(r"C:\Users\u\AppData\Local\Temp\reverify-reset-sandbox"),
                         "C--Users-u-AppData-Local-Temp-reverify-reset-sandbox")
        # the hook environment names the project: that wins
        self.assertEqual(harness.successor_cwd(str(deep), str(transcript), {"CLAUDE_PROJECT_DIR": str(project)}),
                         str(project))
        # no env: the ancestor whose slug names the transcript's project folder
        self.assertEqual(harness.successor_cwd(str(deep), str(transcript), {}), str(project))
        # a transcript of some other project: fall back to the hook's cwd
        other = self.home / ".claude" / "projects" / "elsewhere" / "sess.jsonl"
        self.assertEqual(harness.successor_cwd(str(deep), str(other), {}), str(deep))
        self.assertIsNone(harness.successor_cwd(None, str(transcript), {}))
        # the base harness keeps the hook's cwd
        self.assertEqual(rh.Harness().successor_cwd(str(deep), str(transcript), {"CLAUDE_PROJECT_DIR": str(project)}),
                         str(deep))

    def test_doctor_flags_successor_job_that_never_started(self):
        jobs = self.home / ".claude" / "jobs"
        marker = "[rollover] This is a fresh session that continues the previous one"
        for job_id, state in (("8cbdefe0", {"state": "blocked", "detail": "2 new MCP servers need approval",
                                            "cwd": str(self.cwd / "worktree"), "intent": marker}),
                              ("24056d0d", {"state": "working", "detail": "reading the hand-off",
                                            "cwd": str(self.cwd), "intent": marker}),
                              ("feabbd57", {"state": "blocked", "detail": "unrelated", "intent": "fix the tests"})):
            (jobs / job_id).mkdir(parents=True)
            (jobs / job_id / "state.json").write_text(json.dumps(state), encoding="utf-8")
        info = rh.successor_jobs()
        self.assertEqual([j["id"] for j in info["blocked"]], ["8cbdefe0"])
        self.assertEqual(info["other"], 1)
        os.environ[rh.ENV_SETTINGS] = str(self.root / "settings.json")
        with redirect_stdout(io.StringIO()):
            rh.run_install(["--harness", "claude"])
        rows = {r["harness"]: r for r in rh.doctor_report()}
        problem = " ".join(rows["claude"]["problems"])
        self.assertIn("successor session 8cbdefe0 never started: 2 new MCP servers need approval", problem)
        self.assertIn("claude respawn 8cbdefe0", problem)
        self.assertNotIn("24056d0d", problem)
        self.assertNotIn("successor_jobs", rows["gemini"])

    def test_claude_launch_args_keep_opening_prompt_out_of_remote_control_name(self):
        harness = rh.ClaudeHarness()
        bare = harness.launch_args(["--remote-control"], "OPENING")
        self.assertEqual(bare[0], "--remote-control")
        self.assertNotEqual(bare[1], "OPENING")          # a generated name sits between flag and prompt
        self.assertEqual(bare[-1], "OPENING")
        named = harness.launch_args(["--rc", "My box"], "OPENING")
        self.assertEqual(named, ["--rc", "My box", "OPENING"])
        before_flag = harness.launch_args(["--remote-control", "--verbose"], None)
        self.assertEqual(before_flag[0], "--remote-control")
        self.assertEqual(before_flag[2], "--verbose")
        self.assertEqual(harness.launch_args(["--verbose"], "OPENING"), ["--verbose", "OPENING"])

    def test_instructions_snippet_prints_and_appends_once(self):
        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(rh.run_instructions([]), 0)
        self.assertIn("reverify rollover request", out.getvalue())
        target = self.cwd / "AGENTS.md"
        target.write_text("# Agents\nexisting", encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(rh.run_instructions(["--write", str(target)]), 0)
            self.assertEqual(rh.run_instructions(["--write", str(target)]), 0)
        text = target.read_text(encoding="utf-8")
        self.assertEqual(text.count("## Context rollover (reverify)"), 1)
        self.assertTrue(text.startswith("# Agents\nexisting\n\n"))

    def test_main_routes_cli_name_and_doctor(self):
        original = shutil.which
        try:
            shutil.which = lambda name: "/bin/claude" if name == "claude" else None   # CI runners have no CLI on PATH
            out = io.StringIO()
            with redirect_stdout(out):
                self.assertEqual(rh.main(["doctor"]), 1)      # claude on PATH but nothing installed -> a problem
                self.assertEqual(rh.main(["instructions"]), 0)
            text = out.getvalue()
            self.assertIn("hooks: no", text)
            self.assertIn("to install : reverify rollover install --harness claude", text)
            shutil.which = lambda name: None
            with redirect_stdout(io.StringIO()):
                self.assertEqual(rh.main(["doctor"]), 0)      # no CLI at all: nothing to fix
        finally:
            shutil.which = original


class Selection(Base):
    def test_detect_and_select(self):
        original = shutil.which
        try:
            shutil.which = lambda name: "/bin/" + name if name in ("claude", "gemini") else None
            self.assertEqual(rh.detect_harnesses(), ["claude", "gemini"])
            self.assertEqual([h.name for h in rh._selected_harnesses([])], ["claude", "gemini"])
            self.assertEqual([h.name for h in rh._selected_harnesses(["--harness", "codex,opencode"])], ["codex", "opencode"])
        finally:
            shutil.which = original
        with self.assertRaises(ValueError):
            rh.get_harness("cursor")

    def test_launch_args_and_block_dialects(self):
        self.assertEqual(rh.ClaudeHarness().launch_args(["-p"], "O"), ["-p", "O"])
        self.assertEqual(rh.CodexHarness().launch_args(["--full-auto"], "O"), ["--full-auto", "O"])
        self.assertEqual(rh.ClaudeHarness().format_block("t"), {"decision": "block", "reason": "t"})
        self.assertEqual(rh.CodexHarness().format_block("t"), {"decision": "block", "reason": "t"})
        self.assertEqual(rh.GeminiHarness().format_block("t"), {"decision": "deny", "reason": "t"})

    def test_main_hook_with_missing_args_fails_open(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(rh.main(["hook"]), 0)
            self.assertEqual(rh.main(["hook", "claude"]), 0)

    def test_one_harness_failure_does_not_stop_the_others(self):
        # claude installs into a real settings file; opencode cannot write its config dir
        os.environ[rh.ENV_SETTINGS] = str(self.root / "settings.json")
        os.environ["OPENCODE_CONFIG_DIR"] = str(self.root / "denied" / "opencode")
        original = rh.OpenCodeHarness.install

        def boom(self, *a, **k):
            raise PermissionError("[WinError 5] access denied")

        rh.OpenCodeHarness.install = boom
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                code = rh.run_install(["--harness", "claude,opencode"])
        finally:
            rh.OpenCodeHarness.install = original
        text = out.getvalue()
        self.assertEqual(code, 0)                        # claude succeeded -> overall success
        self.assertIn("claude: hooks", text)
        self.assertIn("opencode: FAILED", text)
        self.assertIn("OPENCODE_CONFIG_DIR", text)
        self.assertIn("failed: opencode", text)
        self.assertTrue((self.root / "settings.json").is_file())


if __name__ == "__main__":
    unittest.main()
