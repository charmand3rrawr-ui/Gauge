#!/usr/bin/env bash
# Gauge installer for macOS and Linux. Installs everything Gauge needs and sets it up.
#
#   Inside a copy of Gauge:   ./install.sh
#   From nothing at all:      curl -fsSL https://raw.githubusercontent.com/charmand3rrawr-ui/Gauge/ccr-24dc418f-hsneyx/install.sh | bash
#
# What it does, skipping anything already done:
#   1. looks at this computer (memory, graphics card, free disk) and picks a writer model that fits
#   2. downloads Gauge into ~/Gauge when it is not being run from a copy of Gauge
#   3. Python 3.10+ (asks the system package manager if it is missing)
#   4. Ollama, the local AI engine
#   5. a private Python environment with Gauge installed into it
#   6. the AI models: a writer, a small helper, and an embedding model
#   7. Gauge's settings, a `gauge` command, a real test question to the AI, and a checklist of what worked
#
# Safe to run again: it updates Gauge and keeps your stories and settings.
#
# Options:
#   --model NAME        choose the writer model yourself (default: picked for this computer)
#   --fast-model NAME   the small helper model (default: llama3.2:3b)
#   --engine native     no Ollama: use GaugeLM, the built-in model (you train it yourself)
#   --skip-engine       do not install Ollama or download models (set Gauge up only)
#   --dir PATH          where to put Gauge when downloading it (default: ~/Gauge)
#   --yes               do not ask before using sudo to install system packages
set -euo pipefail

REPO="charmand3rrawr-ui/Gauge"
BRANCH="${GAUGE_BRANCH:-ccr-24dc418f-hsneyx}"
MODEL="" FAST_MODEL="llama3.2:3b" EMBED_MODEL="nomic-embed-text"
ENGINE="ollama" SKIP_ENGINE=0 ASSUME_YES=0 EXPLICIT=0
INSTALL_DIR="${GAUGE_DIR:-$HOME/Gauge}"

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL="$2"; EXPLICIT=1; shift 2 ;;
    --fast-model) FAST_MODEL="$2"; EXPLICIT=1; shift 2 ;;
    --engine) ENGINE="$2"; EXPLICIT=1; shift 2 ;;
    --skip-engine) SKIP_ENGINE=1; shift ;;
    --dir) INSTALL_DIR="$2"; shift 2 ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done
case "$ENGINE" in ollama|native) ;; *) echo "--engine must be ollama or native" >&2; exit 2 ;; esac

if [ -t 1 ]; then C_STEP=$'\033[36m' C_OK=$'\033[32m' C_WARN=$'\033[33m' C_ERR=$'\033[31m' C_OFF=$'\033[0m'
else C_STEP="" C_OK="" C_WARN="" C_ERR="" C_OFF=""; fi
step() { printf '\n%s==> %s%s\n' "$C_STEP" "$1" "$C_OFF"; }
ok()   { printf '    %s%s%s\n' "$C_OK" "$1" "$C_OFF"; }
warn() { printf '    %s%s%s\n' "$C_WARN" "$1" "$C_OFF"; }
die()  { printf '\n%sERROR: %s%s\n' "$C_ERR" "$1" "$C_OFF" >&2; exit 1; }

CHECK_NAMES=() CHECK_OK=() CHECK_NOTES=() HARD_FAILED=0
check() { CHECK_NAMES+=("$1"); CHECK_OK+=("$2"); CHECK_NOTES+=("${3:-}"); [ "$2" = "1" ] || HARD_FAILED=$((HARD_FAILED + 1)); }
check_optional() {  # nice to have: reported, but never fails the install
  if [ "$2" = "1" ]; then CHECK_NAMES+=("$1"); CHECK_OK+=("1"); else CHECK_NAMES+=("$1"); CHECK_OK+=("optional"); fi
  CHECK_NOTES+=("${3:-}")
}

OS="$(uname -s)"
have() { command -v "$1" >/dev/null 2>&1; }

as_root() {
  # Run a command as root, asking first unless --yes was given.
  if [ "$(id -u)" -eq 0 ]; then "$@"; return; fi
  have sudo || die "This step needs administrator rights and 'sudo' is not available: $*"
  if [ "$ASSUME_YES" -ne 1 ] && [ -t 0 ]; then
    printf '    This needs administrator rights:  %s\n    Continue? [Y/n] ' "$*"
    read -r reply
    case "$reply" in n|N|no|No) die "Stopped at your request." ;; esac
  fi
  sudo "$@"
}

# ---------------------------------------------------------------- 1. This computer
ram_gb() {
  if [ "$OS" = "Darwin" ]; then echo $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
  elif [ -r /proc/meminfo ]; then awk '/MemTotal/ {printf "%d", ($2 + 524288) / 1048576}' /proc/meminfo
  else echo 0; fi
}
vram_gb() {
  if have nvidia-smi; then
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{printf "%d", ($1 + 512) / 1024}'
  else echo 0; fi
}
free_gb() {  # free space, in GB, on the disk that holds $1 (or its nearest existing parent)
  local p="$1"
  while [ ! -e "$p" ] && [ "$p" != "/" ]; do p="$(dirname "$p")"; done
  df -Pk "$p" 2>/dev/null | awk 'NR==2 {printf "%d", $4 / 1048576}'
}
model_size_gb() {
  case "$1" in llama3.2:3b) echo 2 ;; llama3.1:8b|qwen2.5:7b) echo 5 ;; qwen2.5:14b) echo 9 ;; nomic-embed-text) echo 1 ;; *) echo 6 ;; esac
}
pick_model() {  # $1 = RAM GB, $2 = NVIDIA VRAM GB, $3 = "apple" on Apple Silicon (memory is shared with the GPU)
  local ram="$1" vram="$2" kind="${3:-}"
  if [ "$vram" -ge 12 ]; then echo "qwen2.5:14b"
  elif [ "$vram" -ge 6 ]; then echo "llama3.1:8b"
  elif [ "$kind" = "apple" ] && [ "$ram" -ge 32 ]; then echo "qwen2.5:14b"
  elif [ "$ram" -ge 16 ]; then echo "llama3.1:8b"
  else echo "llama3.2:3b"; fi
}

step "Looking at this computer"
RAM="$(ram_gb)"; VRAM="$(vram_gb)"; RAM="${RAM:-0}"; VRAM="${VRAM:-0}"
KIND=""; [ "$OS" = "Darwin" ] && [ "$(uname -m)" = "arm64" ] && KIND="apple"
FREE="$(free_gb "${OLLAMA_MODELS:-$HOME}")"; FREE="${FREE:-0}"
if [ "$VRAM" -gt 0 ]; then GPU_TEXT="NVIDIA graphics card with ${VRAM} GB"
elif [ "$KIND" = "apple" ]; then GPU_TEXT="Apple Silicon (memory shared with the graphics chip)"
else GPU_TEXT="no NVIDIA graphics card found (the AI will run on the processor: slower, still works)"; fi
ok "Memory: ${RAM} GB.  Graphics: ${GPU_TEXT}.  Free disk: ${FREE} GB."


# ---------------------------------------------------------------- 2. Where is Gauge?
SELF_DIR=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; fi
if [ -n "$SELF_DIR" ] && [ -f "$SELF_DIR/pyproject.toml" ] && [ -d "$SELF_DIR/gauge" ]; then
  ROOT="$SELF_DIR"
  step "Using the copy of Gauge in $ROOT"
else
  step "Downloading Gauge into $INSTALL_DIR"
  have curl || die "'curl' is needed to download Gauge. Install curl and run this again."
  have tar || die "'tar' is needed to unpack Gauge. Install tar and run this again."
  TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
  GOT=""
  for b in "$BRANCH" main; do
    if curl -fsSL "https://github.com/$REPO/archive/refs/heads/$b.tar.gz" -o "$TMP/gauge.tar.gz" 2>/dev/null; then
      rm -rf "$TMP/x"; mkdir -p "$TMP/x"; tar -xzf "$TMP/gauge.tar.gz" -C "$TMP/x"
      TOP="$(find "$TMP/x" -mindepth 1 -maxdepth 1 -type d | head -1)"
      if [ -n "$TOP" ] && [ -f "$TOP/pyproject.toml" ]; then GOT="$TOP"; ok "downloaded ($b)"; break; fi
      warn "the '$b' copy has no program in it; trying the next one"
    fi
  done
  [ -n "$GOT" ] || die "Could not download Gauge. Check your internet connection and run this again."
  mkdir -p "$INSTALL_DIR"
  # Copy the program over the install folder. Data, stories and the Python environment are in folders
  # the download does not contain, so they are left exactly as they were.
  cp -R "$GOT"/. "$INSTALL_DIR"/
  ROOT="$INSTALL_DIR"
fi
cd "$ROOT"

# ---------------------------------------------------------------- 3. Python
find_python() {
  local c v
  for c in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if have "$c"; then
      v="$("$c" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
      case "$v" in 3.1[0-9]|3.[2-9][0-9]) command -v "$c"; return 0 ;; esac
    fi
  done
  return 1
}
step "Checking for Python"
PY="$(find_python || true)"
if [ -z "$PY" ]; then
  warn "Python 3.10 or newer not found. Installing it..."
  if [ "$OS" = "Darwin" ]; then
    have brew || die "Install Python 3.12 from https://www.python.org/downloads/ (or install Homebrew), then run this again."
    brew install python@3.12 python-tk@3.12
  elif have apt-get; then as_root apt-get update -y && as_root apt-get install -y python3 python3-venv python3-pip python3-tk
  elif have dnf; then as_root dnf install -y python3 python3-pip python3-tkinter
  elif have pacman; then as_root pacman -S --noconfirm python python-pip tk
  elif have zypper; then as_root zypper install -y python3 python3-pip python3-tk
  else die "Could not find a package manager to install Python. Install Python 3.10 or newer, then run this again."; fi
  PY="$(find_python || true)"
  [ -n "$PY" ] || die "Python was installed but is still not found. Open a new terminal and run this again."
fi
ok "Python: $PY ($("$PY" --version 2>&1))"
check "Python 3.10 or newer" 1 "$PY"

# ---------------------------------------------------------------- Which model?
# An existing config.json is yours: unless you named a model or engine on the command line, its choices
# are kept, so running this again never resets them (and it re-downloads only what is missing).
CONFIG="$ROOT/data/config.json"
KEEP=0
if [ -f "$CONFIG" ] && [ "$EXPLICIT" -eq 0 ]; then
  SAVED="$("$PY" - "$CONFIG" <<'READCFG' 2>/dev/null || true
import json, sys
llm = json.load(open(sys.argv[1], encoding="utf-8-sig")).get("llm", {})
print(llm.get("provider", ""), llm.get("model", ""), llm.get("fast_model", "") or llm.get("model", ""))
READCFG
)"
  set -- $SAVED
  if [ $# -eq 3 ]; then
    KEEP=1
    case "$1" in ollama) ENGINE="ollama"; MODEL="$2"; FAST_MODEL="$3" ;; native) ENGINE="native" ;; *) ENGINE="other" ;; esac
    ok "Keeping your existing settings (engine: $1, writer model: $2)"
  fi
fi
if [ "$ENGINE" = "ollama" ] && [ "$SKIP_ENGINE" -eq 0 ]; then
  if [ "$KEEP" -eq 0 ]; then
    if [ -z "$MODEL" ]; then MODEL="$(pick_model "$RAM" "$VRAM" "$KIND")"; ok "Chosen writer model for this computer: $MODEL"
    else ok "Writer model (your choice): $MODEL"; fi
    NEED=$(( $(model_size_gb "$MODEL") + 4 ))
    [ "$FAST_MODEL" != "$MODEL" ] && NEED=$(( NEED + $(model_size_gb "$FAST_MODEL") ))
    if [ "$FREE" -lt "$NEED" ] && [ "$MODEL" != "llama3.2:3b" ]; then
      warn "Only ${FREE} GB free; $MODEL needs about ${NEED} GB. Using the small model (llama3.2:3b) instead."
      MODEL="llama3.2:3b"
    fi
    [ "$MODEL" = "llama3.2:3b" ] && FAST_MODEL="$MODEL"   # one small model does both jobs: one download
    [ "$FREE" -lt 6 ] && die "Not enough free disk space: ${FREE} GB free, at least 6 GB needed. Free some space and run this again."
  fi
elif [ -z "$MODEL" ]; then
  MODEL="$(pick_model "$RAM" "$VRAM" "$KIND")"
fi
check "Disk space" 1 "${FREE} GB free"

# ---------------------------------------------------------------- 4. Ollama
ENGINE_UP=0
ollama_up() { curl -fsS -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; }
if [ "$ENGINE" = "ollama" ] && [ "$SKIP_ENGINE" -eq 0 ]; then
  step "Checking for Ollama (the local AI engine)"
  if ! have ollama; then
    warn "Ollama not found. Installing..."
    if [ "$OS" = "Darwin" ]; then
      have brew || die "Download Ollama from https://ollama.com/download, open it once, then run this again."
      brew install ollama
    else
      have curl || die "'curl' is needed to install Ollama."
      curl -fsSL https://ollama.com/install.sh | sh
    fi
    have ollama || die "Ollama did not install. Get it from https://ollama.com/download and run this again."
  fi
  ok "Ollama: $(command -v ollama)"
  if ! ollama_up; then
    warn "Starting the Ollama engine..."
    (OLLAMA_NUM_PARALLEL=4 nohup ollama serve >/dev/null 2>&1 &)
    for _ in $(seq 1 30); do sleep 2; ollama_up && break; done
  fi
  ollama_up && ENGINE_UP=1
  check "Ollama engine running" "$ENGINE_UP" "start it with: ollama serve"
fi

# ---------------------------------------------------------------- 5. Gauge
step "Installing Gauge into its own Python environment"
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv 2>/dev/null || {
    warn "Python's venv module is missing. Installing it..."
    if have apt-get; then as_root apt-get install -y python3-venv; else die "Install the Python 'venv' module, then run this again."; fi
    "$PY" -m venv .venv
  }
fi
VPY="$ROOT/.venv/bin/python"
PIPLOG="$ROOT/install.log"
if ! { "$VPY" -m pip install --upgrade pip --disable-pip-version-check \
       && "$VPY" -m pip install -e ".[ml]" --disable-pip-version-check; } >"$PIPLOG" 2>&1; then
  tail -15 "$PIPLOG" >&2
  die "Installing Gauge failed (full log: $PIPLOG). Check your internet connection and run this again."
fi
"$VPY" -m gauge.cli --version >/dev/null && check "Gauge installed" 1 "$ROOT/.venv" || check "Gauge installed" 0 "$ROOT/.venv"
ok "Gauge installed"
if ! "$VPY" -c "import tkinter" 2>/dev/null && [ "$OS" != "Darwin" ] && { [ "$(id -u)" -eq 0 ] || [ -t 0 ]; }; then
  warn "The window toolkit (tkinter) is missing. Trying to install it..."
  if have apt-get; then as_root apt-get install -y python3-tk >>"$PIPLOG" 2>&1 || true
  elif have dnf; then as_root dnf install -y python3-tkinter >>"$PIPLOG" 2>&1 || true
  elif have pacman; then as_root pacman -S --noconfirm tk >>"$PIPLOG" 2>&1 || true
  fi
fi
if "$VPY" -c "import tkinter" 2>/dev/null; then check_optional "Gauge Studio window (tkinter)" 1
else check_optional "Gauge Studio window (tkinter)" 0 "install your system's python3-tk package; the command line works without it"; fi

# A `gauge` command that works from any folder: it uses Gauge's own Python environment and data folder,
# and leaves you where you are, so a path like ./mybook still means what you expect.
mkdir -p "$ROOT/bin"
LAUNCHER="$ROOT/bin/gauge"
cat > "$LAUNCHER" <<LAUNCH
#!/usr/bin/env bash
# Gauge launcher (written by install.sh).
export GAUGE_DATA_DIR="\${GAUGE_DATA_DIR:-$ROOT/data}"
exec "$VPY" -m gauge.cli "\$@"
LAUNCH
chmod +x "$LAUNCHER"
mkdir -p "$HOME/.local/bin" 2>/dev/null && ln -sf "$LAUNCHER" "$HOME/.local/bin/gauge" 2>/dev/null || true

# ---------------------------------------------------------------- 6. Models
if [ "$ENGINE" = "ollama" ] && [ "$SKIP_ENGINE" -eq 0 ]; then
  step "Downloading the AI models (the slow part; a few GB)"
  DONE=""
  for m in "$MODEL" "$FAST_MODEL" "$EMBED_MODEL"; do
    case " $DONE " in *" $m "*) continue ;; esac
    DONE="$DONE $m"
    echo "    pulling $m ..."
    if ollama pull "$m"; then check "Model $m" 1; else warn "could not pull $m; later, run: ollama pull $m"; check "Model $m" 0 "ollama pull $m"; fi
  done
fi

# ---------------------------------------------------------------- 7. Settings and a real test
step "Setting up Gauge"
# Settings are written by Gauge itself, never edited by this script.
if [ "$KEEP" -eq 1 ]; then
  ok "Your existing settings were kept ($CONFIG)"
  check "Settings kept" 1
elif [ "$ENGINE" = "native" ]; then
  "$VPY" -m gauge.cli init --provider native >/dev/null && check "Settings written" 1 || check "Settings written" 0 "$CONFIG"
else
  if "$VPY" -m gauge.cli init --provider ollama --model "$MODEL" >/dev/null \
     && "$VPY" -m gauge.cli settings set llm.fast_model "$FAST_MODEL" >/dev/null; then check "Settings written" 1
  else check "Settings written" 0 "$CONFIG"; fi
fi
"$VPY" -m gauge.cli doctor || true

# The check that matters: does the AI actually answer? (The first answer loads the model, so it can be slow.)
if [ "$ENGINE" = "ollama" ] && [ "$SKIP_ENGINE" -eq 0 ]; then
  step "Asking the AI a test question"
  USE_MODEL="$MODEL"
  BODY="{\"model\":\"$USE_MODEL\",\"prompt\":\"Reply with the single word: ready\",\"stream\":false,\"options\":{\"num_predict\":8}}"
  if curl -fsS -m 600 http://localhost:11434/api/generate -H 'Content-Type: application/json' -d "$BODY" 2>/dev/null | grep -q '"response"'; then
    ok "The AI answered."
    check "The AI answers ($USE_MODEL)" 1
  else
    warn "No answer from $USE_MODEL."
    check "The AI answers ($USE_MODEL)" 0 "make sure Ollama is running, then: ollama run $USE_MODEL"
  fi
fi

printf '\n%sWhat was installed%s\n' "$C_STEP" "$C_OFF"
FAILED=0
for i in "${!CHECK_NAMES[@]}"; do
  if [ "${CHECK_OK[$i]}" = "1" ]; then printf '  %s[ OK ]%s %s\n' "$C_OK" "$C_OFF" "${CHECK_NAMES[$i]}"
  elif [ "${CHECK_OK[$i]}" = "optional" ]; then printf '  %s[ -- ]%s %s (optional)   -> %s\n' "$C_WARN" "$C_OFF" "${CHECK_NAMES[$i]}" "${CHECK_NOTES[$i]}"
  else FAILED=$((FAILED + 1)); printf '  %s[FAIL]%s %s   -> %s\n' "$C_WARN" "$C_OFF" "${CHECK_NAMES[$i]}" "${CHECK_NOTES[$i]}"; fi
done
printf '\nGauge is in: %s\n' "$ROOT"
case ":$PATH:" in *":$HOME/.local/bin:"*) RUN="gauge" ;; *) RUN="$LAUNCHER" ;; esac
if [ "$ENGINE" = "native" ]; then
  printf '%sYou chose GaugeLM, the built-in model. It has to be trained before it can write:%s\n' "$C_WARN" "$C_OFF"
  printf '  %s corpus fetch --source gutenberg --query adventure\n  %s native train --size tiny --minutes 240\n' "$RUN" "$RUN"
elif [ "$SKIP_ENGINE" -eq 1 ]; then
  printf '%sThe AI engine was skipped. Install Ollama and pull a model before writing:%s\n  ollama pull %s\n' "$C_WARN" "$C_OFF" "$MODEL"
fi
if [ "$FAILED" -gt 0 ]; then
  printf '%s%d step(s) need attention (see above). Everything else is installed.%s\n' "$C_WARN" "$FAILED" "$C_OFF"
else
  printf '%sAll done.%s Try:  %s start "a courier who can rewind ten seconds" --chapters 1\n' "$C_OK" "$C_OFF" "$RUN"
  printf '           %s gui      (opens Gauge Studio, the big-button window)\n' "$RUN"
fi
[ "$HARD_FAILED" -eq 0 ] || exit 1
