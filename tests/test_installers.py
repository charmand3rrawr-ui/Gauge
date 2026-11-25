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
@pytest.mark.parametrize("path", [GET_BAT, INSTALL_BAT, BOOTSTRAP])
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
    for name, src in (("bootstrap.ps1", ps), ("Get Gauge.bat", embedded)):
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
