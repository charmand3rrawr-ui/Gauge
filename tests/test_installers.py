"""The install kit: the scripts cannot be run on every platform in CI, so pin the contract they rely on."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from gauge import settings as st
from gauge.cli import main
from gauge.config import GaugeConfig

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "installer" / "bootstrap.ps1"
GET_BAT = ROOT / "installer" / "Get Gauge.bat"
UPDATE_BAT = ROOT / "installer" / "Update Gauge.bat"
INSTALL_BAT = ROOT / "Install Gauge.bat"
INSTALL_SH = ROOT / "install.sh"
BASH = shutil.which("bash")


def _powershell_balanced(src: str) -> bool:
    """Brackets and quotes balance, ignoring comments and string contents. A stand-in for a parser."""
    stack, i, n = [], 0, len(src)
    pairs = {")": "(", "}": "{", "]": "["}
    while i < n:
        c = src[i]
        if src[i:i + 2] == "<#":
            i = src.find("#>", i) + 2
            continue
        if c == "#":
            j = src.find("\n", i)
            i = n if j == -1 else j
            continue
        if c in "\"'":
            j = i + 1
            while j < n:
                if src[j] == "`" and c == '"':
                    j += 2
                    continue
                if src[j] == c:
                    if src[j + 1:j + 2] == c:
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                return False
            i = j + 1
            continue
        if c in "({[":
            stack.append(c)
        elif c in ")}]":
            if not stack or stack.pop() != pairs[c]:
                return False
        i += 1
    return not stack


# ------------------------------------------------------------------ the bug the old installer hit
def test_config_with_a_byte_order_mark_still_loads(tmp_path):
    # Windows PowerShell 5.1 and Notepad save UTF-8 with a BOM. A config.json like that must not break Gauge.
    GaugeConfig(data_dir=str(tmp_path)).save()
    p = tmp_path / "config.json"
    p.write_bytes(b"\xef\xbb\xbf" + p.read_bytes())
    assert GaugeConfig.load(str(tmp_path)).llm.model == "llama3.1:8b"
    r = CliRunner(env={"COLUMNS": "200"}).invoke(main, ["--data-dir", str(tmp_path), "doctor"])
    assert r.exit_code == 0, r.output


def test_installers_let_gauge_write_its_own_settings():
    ps = BOOTSTRAP.read_text(encoding="ascii")
    sh = INSTALL_SH.read_text(encoding="utf-8")
    for bad in ("Set-Content", "Out-File", "ConvertTo-Json -Depth 10"):
        assert bad not in ps, f"bootstrap.ps1 must not rewrite config.json itself ({bad})"
    for script in (ps, sh):
        assert "settings set llm.fast_model" in script and "init --provider ollama --model" in script
        assert "init --provider native" in script
    assert "llm.fast_model" in st.BY_KEY                     # the switch the installers set really exists


def test_the_cli_calls_the_installers_make_work(tmp_path):
    data = str(tmp_path / "data")
    run = CliRunner(env={"COLUMNS": "200"})
    assert run.invoke(main, ["--data-dir", data, "init", "--provider", "ollama", "--model", "qwen2.5:14b"]).exit_code == 0
    assert run.invoke(main, ["--data-dir", data, "settings", "set", "llm.fast_model", "llama3.2:3b"]).exit_code == 0
    cfg = GaugeConfig.load(data)
    assert (cfg.llm.provider, cfg.llm.model, cfg.llm.fast_model) == ("ollama", "qwen2.5:14b", "llama3.2:3b")
    assert run.invoke(main, ["--data-dir", data, "init", "--provider", "native"]).exit_code == 0
    assert GaugeConfig.load(data).llm.provider == "native"
    assert run.invoke(main, ["--data-dir", data, "--version"]).exit_code == 0


# ------------------------------------------------------------------ Windows scripts (cannot be executed here)
@pytest.mark.parametrize("path", [GET_BAT, UPDATE_BAT, INSTALL_BAT, BOOTSTRAP])
def test_windows_scripts_are_ascii_with_crlf(path):
    raw = path.read_bytes()
    raw.decode("ascii")                                       # no smart quotes or other surprises for cmd.exe
    assert b"\r\n" in raw and b"\n" not in raw.replace(b"\r\n", b"")


def test_gitattributes_keeps_crlf_in_a_downloaded_zip():
    attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.bat text eol=crlf" in attrs and "*.ps1 text eol=crlf" in attrs and "*.sh text eol=lf" in attrs


def test_powershell_scripts_balance_and_avoid_powershell_7_syntax():
    ps = BOOTSTRAP.read_text(encoding="ascii").replace("\r\n", "\n")
    bat = GET_BAT.read_text(encoding="ascii").replace("\r\n", "\n")
    assert bat.count("#---POWERSHELL---") == 2                # once in the command that finds it, once as the marker
    embedded = bat[bat.rindex("#---POWERSHELL---"):]
    upd = UPDATE_BAT.read_text(encoding="ascii").replace("\r\n", "\n")
    assert upd.count("#---POWERSHELL---") == 2
    for name, src in (("bootstrap.ps1", ps), ("Get Gauge.bat", embedded),
                      ("Update Gauge.bat", upd[upd.rindex("#---POWERSHELL---"):])):
        assert _powershell_balanced(src), name
        assert not re.search(r"\?\?|&&|\?\.", src), f"{name} uses PowerShell 7 syntax; Windows ships 5.1"
    assert _powershell_balanced('if ($a) { "x)" }') and not _powershell_balanced("if ($a) { ")   # the checker itself works


def test_get_gauge_downloads_then_runs_the_installer():
    bat = GET_BAT.read_text(encoding="ascii")
    assert "archive/refs/heads/$b.zip" in bat and "installer\\bootstrap.ps1" in bat
    assert "Tls12" in bat and "exit /b %RC%" in bat
    assert '"main"' in bat                                    # falls back to main once the work is merged
    assert "Tls12" in BOOTSTRAP.read_text(encoding="ascii")


def test_bootstrap_keeps_existing_settings_and_asks_a_real_question():
    ps = BOOTSTRAP.read_text(encoding="ascii")
    assert "$PSBoundParameters.ContainsKey" in ps and "Keeping your existing settings" in ps
    assert "/api/generate" in ps and "The AI answers" in ps
    assert "exit 1" in ps and ps.rstrip().endswith("exit 0")


# ------------------------------------------------------------------ install.sh (runs here)
@pytest.mark.skipif(not BASH, reason="bash not available")
def test_install_sh_is_valid_bash_and_documents_itself():
    assert subprocess.run([BASH, "-n", str(INSTALL_SH)], capture_output=True).returncode == 0
    out = subprocess.run([BASH, str(INSTALL_SH), "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "--skip-engine" in out.stdout and "Safe to run again" in out.stdout
    bad = subprocess.run([BASH, str(INSTALL_SH), "--no-such-flag"], capture_output=True, text=True)
    assert bad.returncode == 2


@pytest.mark.skipif(not BASH, reason="bash not available")
@pytest.mark.parametrize("ram,vram,kind,want", [
    (8, 0, "", "llama3.2:3b"),        # small machine, no GPU
    (16, 0, "", "llama3.1:8b"),       # enough memory to run the default on the processor
    (16, 8, "", "llama3.1:8b"),       # mid-range graphics card
    (32, 12, "", "qwen2.5:14b"),      # big graphics card
    (8, 24, "", "qwen2.5:14b"),       # the graphics card decides, not the system memory
    (32, 0, "apple", "qwen2.5:14b"),  # Apple Silicon shares memory with the graphics chip
    (16, 0, "apple", "llama3.1:8b"),
])
def test_model_choice_fits_the_hardware(ram, vram, kind, want):
    src = INSTALL_SH.read_text(encoding="utf-8")
    fn = src[src.index("pick_model() {"):]
    fn = fn[:fn.index("\n}\n") + 3]
    out = subprocess.run([BASH, "-c", f'{fn}\npick_model "$1" "$2" "$3"', "_", str(ram), str(vram), kind],
                         capture_output=True, text=True)
    assert out.stdout.strip() == want, out.stderr


def test_both_installers_pick_the_same_models():
    ps = BOOTSTRAP.read_text(encoding="ascii")
    sh = INSTALL_SH.read_text(encoding="utf-8")
    for model in ("llama3.2:3b", "llama3.1:8b", "qwen2.5:14b", "nomic-embed-text"):
        assert model in ps and model in sh
    # Same thresholds on both sides: 12 GB and 6 GB of graphics memory, 16 GB of system memory.
    assert "$vram -ge 12" in ps and "$vram -ge 6" in ps and "$ram -ge 16" in ps
    assert '"$vram" -ge 12' in sh and '"$vram" -ge 6' in sh and '"$ram" -ge 16' in sh


# ------------------------------------------------------------------ the updater
def test_windows_updater_only_updates_the_program():
    bat = UPDATE_BAT.read_text(encoding="ascii")
    code = bat[bat.rindex("#---POWERSHELL---"):]
    # It must not be able to reinstall anything or fetch a model: none of that machinery is in it.
    for forbidden in ("winget", "ollama pull", "bootstrap.ps1", "python.org", "OllamaSetup", "gauge.cli init", "-m venv"):
        assert forbidden not in code, f"the updater must not contain {forbidden!r}"
    assert "already up to date" in code and "Nothing was downloaded" in code     # no download when current
    assert "api.github.com/repos/$repo/commits/$b" in code                       # one small question first
    assert "installed-version.txt" in code
    assert "$oldDeps -ne $newDeps" in code and "pip install -e" in code          # packages only when they changed
    assert 'Join-Path $src "gauge"' in code and '-Filter "*.py"' in code         # stale files: program folder only
    assert "git checkout" in code                                                # never overwrites a developer's clone
    assert "Run 'Get Gauge.bat' first" in code                                   # clear message when not installed


def test_windows_updater_can_replace_itself_while_running():
    # cmd reads a parenthesised block completely before running it; the tail of the file must be in one,
    # or overwriting the running .bat would send cmd into the middle of the new file.
    text = UPDATE_BAT.read_bytes().decode("ascii").replace("\r\n", "\n")
    start = text.index("\n(\n")                               # the block opens on a line of its own
    end = text.rindex("#---POWERSHELL---")                    # and everything up to the PowerShell part is inside it
    tail = text[start:end]
    assert "powershell -NoProfile" in tail and "if errorlevel 1 (" in tail
    assert tail.count("exit /b") == 2 and tail.rstrip().endswith(")")
    assert "%ERRORLEVEL%" not in tail and "%RC%" not in tail  # nothing that cmd would expand when it parses the block
    commands_before = [ln for ln in text[:start].splitlines() if ln.strip() and not ln.lstrip().upper().startswith("REM")]
    assert not any("powershell" in ln.lower() for ln in commands_before)   # the download never starts outside the block


def test_installers_record_the_version_and_skip_models_already_present():
    get = GET_BAT.read_text(encoding="ascii")
    ps = BOOTSTRAP.read_text(encoding="ascii")
    sh = INSTALL_SH.read_text(encoding="utf-8")
    assert "installed-version.txt" in get and "installed-version.txt" in sh
    assert "is already on this computer" in ps and "is already on this computer" in sh
    assert '($m + ":latest")' in ps and "(:latest)?" in sh


@pytest.mark.skipif(not BASH, reason="bash not available")
def test_model_presence_check_understands_ollama_names():
    src = INSTALL_SH.read_text(encoding="utf-8")
    fn = src[src.index("have_model() {"):]
    fn = fn[:fn.index("\n}\n") + 3]
    listing = '{"models":[{"name":"nomic-embed-text:latest"},{"name": "llama3.1:8b"}]}'
    script = (f"curl() {{ printf '%s' '{listing}'; }}\n{fn}\n"
              "have_model nomic-embed-text && have_model llama3.1:8b && ! have_model llama3.1 && ! have_model qwen2.5:14b")
    assert subprocess.run([BASH, "-c", script], capture_output=True).returncode == 0


@pytest.mark.skipif(not BASH, reason="bash not available")
def test_latest_version_lookup_rejects_anything_that_is_not_a_version():
    src = INSTALL_SH.read_text(encoding="utf-8")
    fn = src[src.index("latest_sha() {"):]
    fn = fn[:fn.index("\n}\n") + 3]
    good = "d2ae973e3e615c36a76f5291cc271ae31b788694"
    for answer, want in ((good, good), ("Not Found", ""), ("", ""), (good[:39], ""), ("<html>rate limit</html>", "")):
        script = f"REPO=x/y\ncurl() {{ printf '%s' '{answer}'; }}\n{fn}\nlatest_sha main"
        out = subprocess.run([BASH, "-c", script], capture_output=True, text=True)
        assert out.stdout.strip() == want, (answer, out.stdout)


@pytest.mark.skipif(not BASH, reason="bash not available")
def test_updater_refuses_when_not_installed_and_never_touches_a_git_checkout(tmp_path):
    # The script is copied out of the repository first: run in place it would find this checkout.
    script = tmp_path / "bin" / "install.sh"
    script.parent.mkdir()
    script.write_text(INSTALL_SH.read_text(encoding="utf-8"), encoding="utf-8")
    env = {"HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin", "GAUGE_DIR": str(tmp_path / "nowhere")}
    (tmp_path / "home").mkdir()
    missing = subprocess.run([BASH, str(script), "--update"], capture_output=True, text=True, env=env)
    assert missing.returncode == 1 and "not installed" in missing.stderr
    clone = tmp_path / "clone"
    (clone / "gauge").mkdir(parents=True)
    (clone / ".git").mkdir()
    (clone / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (clone / "gauge" / "mine.py").write_text("# my uncommitted work\n", encoding="utf-8")
    env["GAUGE_DIR"] = str(clone)
    refused = subprocess.run([BASH, str(script), "--update"], capture_output=True, text=True, env=env)
    assert refused.returncode == 1 and "git checkout" in refused.stderr
    assert (clone / "gauge" / "mine.py").read_text(encoding="utf-8") == "# my uncommitted work\n"   # untouched


def test_update_mode_in_install_sh_cannot_reach_the_engine():
    src = INSTALL_SH.read_text(encoding="utf-8")
    body = src[src.index("update_only() {"):src.index('if [ "$UPDATE_ONLY" -eq 1 ]; then update_only; fi')]
    for forbidden in ("ollama", "apt-get", "brew", "-m venv", "gauge.cli init"):
        assert forbidden not in body, f"update mode must not contain {forbidden!r}"
    assert body.count("exit 0") == 2 and "Nothing was downloaded" in body
    # It runs before the installer proper, so nothing below it can execute in update mode.
    assert src.index("update_only; fi") < src.index("1. This computer")
