"""Tests for exp2.board.root_helper (the process) and exp2.board.sysfs.RootHelper (the client)."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from exp2.board import sysfs
from exp2.board.dryroot import make_dry_root
from exp2.board.sysfs import RootHelper
from exp2.contracts import SUDO_PW_ENV, ClockError

FAKE_SUDO = str(Path(__file__).with_name("fake_sudo.py"))
GPU = "/sys/class/devfreq/17000000.gpu/"
MIN_F, MAX_F, CUR_F = GPU + "min_freq", GPU + "max_freq", GPU + "cur_freq"
PW_TEXT = "pw-for-fake-sudo-1"


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "root"
    make_dry_root(path)
    return path


def fread(root, path):
    return (root / path.lstrip("/")).read_text().strip()


@pytest.fixture
def dry(root):
    helper = RootHelper.start(mode="dry", fake_root=root)
    yield helper
    helper.close()


def fake_sudo_argv(root, log, *flags):
    helper_cmd = [sys.executable, "-B", "-m", "exp2.board.root_helper", "--fake-root", str(root)]
    return [sys.executable, FAKE_SUDO, "--log", str(log), *flags, "--", *helper_cmd]


def log_lines(log):
    return [ln.split(" ", 2) for ln in Path(log).read_text().splitlines()]


# ---- dry helper ----------------------------------------------------------------------------------
def test_dry_hello_says_fake_root(dry, root):
    assert dry.hello["op"] == "hello"
    assert dry.hello["fake_root"] == str(root)


def test_dry_write_returns_readback_and_changes_file(dry, root):
    assert dry.write(MAX_F, 816000000) == "816000000"
    assert fread(root, MAX_F) == "816000000"


def test_dry_read_returns_file_content_stripped(dry, root):
    assert dry.read(GPU + "available_frequencies").split()[0] == "306000000"
    assert dry.read("/proc/sys/kernel/random/boot_id") == "dry-boot"


def test_dry_write_creates_missing_file_under_fake_root(dry, root):
    assert dry.write("/tmp/exp2_probe/x", "abc") == "abc"
    assert (root / "tmp" / "exp2_probe" / "x").read_text() == "abc"


def test_dry_devfreq_clamps_cur_freq_into_new_range(dry, root):
    dry.write(MAX_F, 612000000)
    assert fread(root, CUR_F) == "612000000"
    dry.write(MIN_F, 510000000)
    assert fread(root, CUR_F) == "612000000"
    dry.write(MAX_F, 918000000)
    dry.write(MIN_F, 816000000)
    assert fread(root, CUR_F) == "816000000"


def test_dry_devfreq_rejects_min_above_max_like_the_kernel(dry, root):
    dry.write(MAX_F, 612000000)
    with pytest.raises(ClockError, match="Invalid argument"):
        dry.write(MIN_F, 714000000)
    assert fread(root, MIN_F) == "306000000"
    with pytest.raises(ClockError, match="Invalid argument"):
        dry.write(MAX_F, 204000000)
    assert fread(root, MAX_F) == "612000000"


def test_dry_read_missing_path_raises_clockerror_and_helper_survives(dry):
    with pytest.raises(ClockError, match="no_such_file"):
        dry.read("/sys/no_such_file")
    assert dry.read("/proc/sys/kernel/random/boot_id") == "dry-boot"


def test_dry_nvpmodel_query_and_set(dry):
    assert dry.run(["nvpmodel", "-m", "0"]) == (0, "")
    rc, out = dry.run(["nvpmodel", "-q"])
    assert rc == 0
    assert out.splitlines() == ["NV Power Mode: 15W", "0"]
    assert dry.run(["nvpmodel", "-m", "2"]) == (0, "")
    assert dry.run(["nvpmodel", "-q"])[1].splitlines() == ["NV Power Mode: MAXN_SUPER", "2"]
    rc, out = dry.run(["nvpmodel", "-m", "9"])
    assert rc != 0 and "Invalid" in out


def test_dry_jetson_clocks_pins_gpu_to_top_level(dry, root):
    assert dry.run(["jetson_clocks"]) == (0, "")
    assert fread(root, MIN_F) == fread(root, MAX_F) == fread(root, CUR_F) == "1020000000"


def test_dry_jetson_clocks_store_then_restore(dry, root):
    dry.write(MAX_F, 816000000)
    dry.write(MIN_F, 612000000)
    assert dry.run(["jetson_clocks", "--store", "/tmp/jc_store"]) == (0, "")
    dry.run(["jetson_clocks"])
    assert fread(root, MIN_F) == "1020000000"
    assert dry.run(["jetson_clocks", "--restore", "/tmp/jc_store"]) == (0, "")
    assert (fread(root, MIN_F), fread(root, MAX_F)) == ("612000000", "816000000")
    assert fread(root, CUR_F) == "816000000"


def test_dry_run_unknown_command_is_an_error(dry):
    with pytest.raises(ClockError, match="does not emulate"):
        dry.run(["reboot"])


def test_oplog_has_writes_and_runs_in_order_and_no_reads(dry):
    dry.read(MAX_F)
    dry.write(MAX_F, 816000000)
    dry.run(["nvpmodel", "-q"])
    dry.write(MIN_F, 612000000)
    assert dry.oplog == [("write", MAX_F, "816000000"), ("run", ("nvpmodel", "-q"), None),
                         ("write", MIN_F, "612000000")]


def test_close_stops_the_process_and_later_calls_fail(root):
    helper = RootHelper.start(mode="dry", fake_root=root)
    proc = helper._proc
    helper.close()
    assert proc.poll() is not None
    helper.close()  # idempotent
    with pytest.raises(ClockError, match="closed"):
        helper.read(MAX_F)


def test_helper_killed_mid_session_raises_clockerror(root):
    helper = RootHelper.start(mode="dry", fake_root=root)
    try:
        helper._proc.kill()
        helper._proc.wait()
        with pytest.raises(ClockError):
            helper.read(MAX_F)
    finally:
        helper.close()


SCRIPTED_HELPER = """
import json, sys, time
sys.stdin.readline()
print(json.dumps({'op': 'hello', 'pid': 1, 'uid': 0, 'fake_root': None}), flush=True)
sys.stdin.readline()
%s
"""


def scripted_helper(monkeypatch, tail, **kw):
    monkeypatch.setenv(SUDO_PW_ENV, PW_TEXT)
    return RootHelper.start(mode="sudo", sudo_argv=[sys.executable, "-c", SCRIPTED_HELPER % tail], **kw)


def test_helper_exiting_while_handling_a_request_raises_clockerror(monkeypatch):
    helper = scripted_helper(monkeypatch, "sys.exit(3)")
    try:
        with pytest.raises(ClockError, match="exited while handling read"):
            helper.read(MAX_F)
        with pytest.raises(ClockError, match="closed"):
            helper.read(MAX_F)
    finally:
        helper.close()


def test_helper_that_never_answers_times_out_and_is_killed(monkeypatch):
    helper = scripted_helper(monkeypatch, "time.sleep(60)")
    try:
        start = time.monotonic()
        with pytest.raises(ClockError, match="did not answer read within 0.5 s"):
            helper._request({"op": "read", "path": MAX_F}, timeout_s=0.5)
        assert time.monotonic() - start < 10
        assert helper._proc.poll() is not None
    finally:
        helper.close()


def test_start_rejects_bad_mode_and_dry_without_root():
    with pytest.raises(ValueError):
        RootHelper.start(mode="root")
    with pytest.raises(ValueError):
        RootHelper.start(mode="dry")


def test_non_json_lines_get_no_reply_and_valid_lines_still_work(root):
    cmd = [sys.executable, "-B", "-m", "exp2.board.root_helper", "--fake-root", str(root)]
    repo = Path(sysfs.__file__).resolve().parents[2]
    junk = ["not json at all\n", "\n", "[1, 2, 3]\n", '"a string"\n', "42\n", "{broken\n"]
    msgs = [json.dumps({"op": "read", "path": "/proc/sys/kernel/random/boot_id"}) + "\n",
            json.dumps({"op": "stop"}) + "\n"]
    res = subprocess.run(cmd, input="".join(junk + msgs), capture_output=True, text=True, cwd=repo, timeout=30)
    replies = [json.loads(ln) for ln in res.stdout.splitlines()]
    assert [r.get("op") for r in replies[:1]] == ["hello"]
    assert replies[1:] == [{"ok": True, "value": "dry-boot"}, {"ok": True}]


def test_unknown_op_gets_error_reply_not_silence(root):
    cmd = [sys.executable, "-B", "-m", "exp2.board.root_helper", "--fake-root", str(root)]
    repo = Path(sysfs.__file__).resolve().parents[2]
    res = subprocess.run(cmd, input=json.dumps({"op": "reboot"}) + "\n", capture_output=True, text=True,
                         cwd=repo, timeout=30)
    replies = [json.loads(ln) for ln in res.stdout.splitlines()]
    assert replies[1]["ok"] is False and "reboot" in replies[1]["err"]


# ---- password and sudo start ---------------------------------------------------------------------
def test_default_sudo_argv_is_the_k17_command():
    assert sysfs.default_sudo_argv() == ["sudo", "-S", "-k", "-p", "", sys.executable, "-B", "-m",
                                         "exp2.board.root_helper"]


def test_read_password_prefers_env(monkeypatch):
    monkeypatch.setenv(SUDO_PW_ENV, PW_TEXT)
    monkeypatch.setattr(sysfs.getpass, "getpass", lambda *a, **k: pytest.fail("getpass must not be called"))
    assert sysfs.read_password() == PW_TEXT


def test_read_password_falls_back_to_getpass(monkeypatch):
    monkeypatch.delenv(SUDO_PW_ENV, raising=False)
    monkeypatch.setattr(sysfs.getpass, "getpass", lambda *a, **k: "typed-in")
    assert sysfs.read_password() == "typed-in"


def test_sudo_mode_hello_write_read_via_fake_sudo(root, tmp_path, monkeypatch):
    monkeypatch.setenv(SUDO_PW_ENV, PW_TEXT)
    monkeypatch.setenv("FAKE_SUDO_EXPECT", PW_TEXT)
    log = tmp_path / "sudo.log"
    helper = RootHelper.start(mode="sudo", sudo_argv=fake_sudo_argv(root, log))
    try:
        assert helper.hello["op"] == "hello"
        assert helper.write(MAX_F, 714000000) == "714000000"
        assert fread(root, MAX_F) == "714000000"
        assert helper.read(MAX_F) == "714000000"
    finally:
        helper.close()
    kinds = [k for _, k, _ in log_lines(log)]
    assert kinds[0] == "PW"
    assert log_lines(log)[0][2] == f"len={len(PW_TEXT)}"


def test_request_is_sent_only_after_hello(root, tmp_path, monkeypatch):
    """The fake holds the hello for 0.6 s; a client that sends at once logs IN before OUT hello."""
    monkeypatch.setenv(SUDO_PW_ENV, PW_TEXT)
    log = tmp_path / "sudo.log"
    helper = RootHelper.start(mode="sudo", sudo_argv=fake_sudo_argv(root, log, "--delay-hello", "0.6"))
    try:
        helper.read(MAX_F)
    finally:
        helper.close()
    rows = log_lines(log)
    kinds = [(k, text) for _, k, text in rows]
    hello_at = next(i for i, (k, t) in enumerate(kinds) if k == "OUT" and '"hello"' in t)
    first_in = next(i for i, (k, _) in enumerate(kinds) if k == "IN")
    assert kinds[0][0] == "PW"
    assert hello_at < first_in


def test_wrong_password_is_refused_with_clockerror(root, tmp_path, monkeypatch):
    monkeypatch.setenv(SUDO_PW_ENV, "the-wrong-one")
    monkeypatch.setenv("FAKE_SUDO_EXPECT", PW_TEXT)
    start = time.monotonic()
    with pytest.raises(ClockError, match="exited before hello"):
        RootHelper.start(mode="sudo", sudo_argv=fake_sudo_argv(root, tmp_path / "sudo.log"))
    assert time.monotonic() - start < 15


def test_refusing_sudo_raises_clockerror_within_15_s(root, tmp_path, monkeypatch):
    monkeypatch.setenv(SUDO_PW_ENV, PW_TEXT)
    start = time.monotonic()
    with pytest.raises(ClockError):
        RootHelper.start(mode="sudo", sudo_argv=fake_sudo_argv(root, tmp_path / "sudo.log", "--refuse"))
    assert time.monotonic() - start < 15
    assert "REFUSE" in (tmp_path / "sudo.log").read_text()


def test_sudo_that_never_says_hello_times_out_and_is_killed(root, tmp_path, monkeypatch):
    monkeypatch.setenv(SUDO_PW_ENV, PW_TEXT)
    started = []
    real_popen = subprocess.Popen

    def spy(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        started.append(proc)
        return proc

    monkeypatch.setattr(sysfs.subprocess, "Popen", spy)
    start = time.monotonic()
    with pytest.raises(ClockError, match="no hello within"):
        RootHelper.start(mode="sudo", sudo_argv=fake_sudo_argv(root, tmp_path / "sudo.log", "--delay-hello", "60"),
                         hello_timeout_s=0.8)
    assert 0.7 < time.monotonic() - start < 10
    assert len(started) == 1 and started[0].poll() is not None
