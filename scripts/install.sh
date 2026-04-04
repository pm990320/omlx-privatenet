#!/usr/bin/env bash
set -euo pipefail

OMLX_REPO="https://github.com/pm990320/omlx.git"
OMLX_REF="v0.3.2"
MLX_LM_FORK="git+https://github.com/pm990320/mlx-lm@feat/gemma4-tool-calling"
MODEL_1="gemma-4-26b-a4b-it-4bit"
MODEL_2="gemma-4-31b-it-4bit"
STATE_DIR="$HOME/.omlx-privatenet"
OMLX_BASE="$HOME/.omlx"
INSTALL_ROOT="$HOME/omlx-privatenet"
OMLX_SRC="$INSTALL_ROOT/omlx"
VENV_DIR="$STATE_DIR/venv"
NODE_ENV="$STATE_DIR/node.env"
NODE_JSON="$STATE_DIR/node.json"
START_SCRIPT="$STATE_DIR/start-edge.sh"
MODEL_DIR="$OMLX_BASE/models"
LAUNCH_AGENT="$HOME/Library/LaunchAgents/com.omlx-privatenet.edge.plist"
LAUNCH_LABEL="com.omlx-privatenet.edge"
TAILSCALE_IP=""
BREW_BIN=""
PYTHON_BIN=""
PIP_BIN=""
HF_BIN=""
STEP=0
TOTAL_STEPS=9

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

BOLD='\033[1m'
DIM='\033[2m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
RESET='\033[0m'

step() {
  STEP=$((STEP + 1))
  printf '\n\033[1m\033[0;36m[%d/%d]\033[0m \033[1m%s\033[0m\n' "$STEP" "$TOTAL_STEPS" "$1"
  if [ -n "${2:-}" ]; then
    printf '\033[2m      %s\033[0m\n' "$2"
  fi
}

info() {
  printf '\033[2m      %s\033[0m\n' "$*"
}

success() {
  printf '\033[0;32m  ✓   %s\033[0m\n' "$*"
}

warn() {
  printf '\033[0;33m  ⚠   %s\033[0m\n' "$*" >&2
}

die() {
  printf '\n\033[0;31m  ✗   %s\033[0m\n' "$*" >&2
  exit 1
}

write_text_file() {
  local path="$1"
  local mode="$2"
  local content="$3"
  python3 -c 'from pathlib import Path; import sys; p = Path(sys.argv[1]).expanduser(); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(sys.argv[3], encoding="utf-8"); p.chmod(int(sys.argv[2], 8))' "$path" "$mode" "$content"
}

require_supported_host() {
  [ "$(uname -s)" = "Darwin" ] || die "This installer only works on macOS. Linux and Windows are not supported."
  [ "$(uname -m)" = "arm64" ] || die "This requires an Apple Silicon Mac (M1, M2, M3, or M4 chip). Older Intel Macs can't run the AI models we need."
}

ensure_homebrew() {
  if command -v brew >/dev/null 2>&1; then
    BREW_BIN="$(command -v brew)"
    success "Homebrew is already installed."
  elif [ -x /opt/homebrew/bin/brew ]; then
    BREW_BIN="/opt/homebrew/bin/brew"
    success "Homebrew is already installed."
  else
    info "Homebrew is a package manager for macOS — it lets us install the software we need."
    info "Installing now (you may be asked for your Mac password)..."
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    BREW_BIN="/opt/homebrew/bin/brew"
    success "Homebrew installed."
  fi

  [ -x "$BREW_BIN" ] || die "Homebrew installation failed. Try running the installer again, or visit https://brew.sh for help."
  eval "$($BREW_BIN shellenv)"
}

ensure_brew_package() {
  local formula="$1"
  local label="${2:-$1}"
  if "$BREW_BIN" list "$formula" >/dev/null 2>&1; then
    success "$label is already installed."
  else
    info "Installing $label..."
    "$BREW_BIN" install "$formula"
    success "$label installed."
  fi
}

ensure_brew_cask() {
  local cask="$1"
  local label="${2:-$1}"
  if "$BREW_BIN" list --cask "$cask" >/dev/null 2>&1; then
    success "$label is already installed."
  else
    info "Installing $label..."
    "$BREW_BIN" install --cask "$cask"
    success "$label installed."
  fi
}

ensure_dependencies() {
  ensure_brew_package python@3.13 "Python 3.13 (the programming language that powers the AI server)"
  ensure_brew_package git "Git (for downloading source code)"
  ensure_brew_cask tailscale "Tailscale (the private network that connects all the Macs together)"

  PYTHON_BIN="$($BREW_BIN --prefix python@3.13)/bin/python3.13"
  [ -x "$PYTHON_BIN" ] || die "Python 3.13 didn't install correctly. Try running the installer again."
}

ensure_tailscale_ip() {
  if ! command -v tailscale >/dev/null 2>&1 && [ -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ]; then
    export PATH="/Applications/Tailscale.app/Contents/MacOS:$PATH"
  fi

  command -v tailscale >/dev/null 2>&1 || die "Tailscale CLI not found. Try re-running this installer."
  open -ga Tailscale >/dev/null 2>&1 || true

  TAILSCALE_IP="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
  if [ -z "$TAILSCALE_IP" ]; then
    printf '\n'
    info "Tailscale needs to be connected before we can continue."
    info "A sign-in window may have opened in your browser."
    printf '\n'
    printf '      1. Sign in to Tailscale (or create a free account) in the browser window.\n'
    printf '      2. Wait until the Tailscale menu bar icon shows \"Connected\".\n'
    printf '      3. Come back here and press Enter.\n\n'
    read -r
    TAILSCALE_IP="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
  fi

  [ -n "$TAILSCALE_IP" ] || die "Couldn't get your Tailscale address. Make sure Tailscale is connected, then re-run this installer."
  success "Tailscale connected — your private network address is $TAILSCALE_IP"
}

ensure_omlx_source() {
  mkdir -p "$INSTALL_ROOT"
  if [ -d "$OMLX_SRC/.git" ]; then
    info "oMLX source already exists — updating to $OMLX_REF..."
    git -C "$OMLX_SRC" remote set-url origin "$OMLX_REPO"
    git -C "$OMLX_SRC" fetch --tags origin
    git -C "$OMLX_SRC" checkout "$OMLX_REF"
    success "oMLX updated."
  elif [ -e "$OMLX_SRC" ]; then
    die "$OMLX_SRC already exists but isn't a proper install. Please delete or move that folder, then re-run."
  else
    info "Downloading oMLX $OMLX_REF (the AI inference server)..."
    info "This is a small download — should only take a few seconds."
    git clone --branch "$OMLX_REF" --depth 1 "$OMLX_REPO" "$OMLX_SRC"
    success "oMLX downloaded."
  fi
}

ensure_venv() {
  mkdir -p "$STATE_DIR"
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    info "Creating an isolated Python environment (keeps everything tidy and separate from your system)..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    success "Python environment created."
  else
    success "Python environment already exists."
  fi

  PIP_BIN="$VENV_DIR/bin/pip"
  HF_BIN="$VENV_DIR/bin/huggingface-cli"

  info "Now installing the AI software. This involves several packages and may"
  info "take 2-5 minutes depending on your internet speed. You'll see progress below."
  printf '\n'

  info "Upgrading core Python tools..."
  "$PIP_BIN" install --upgrade pip setuptools wheel
  success "Core tools ready."

  info "Installing Hugging Face tools (for downloading AI models)..."
  "$PIP_BIN" install --upgrade huggingface-hub
  success "Hugging Face tools installed."

  info "Installing oMLX (the AI inference server that runs on your Mac)..."
  "$PIP_BIN" install -e "$OMLX_SRC"
  success "oMLX installed."

  info "Installing xgrammar (helps the AI follow structured output formats)..."
  "$PIP_BIN" install --upgrade xgrammar
  success "xgrammar installed."

  info "Installing our custom AI language model library (adds Gemma 4 support)..."
  "$PIP_BIN" install --upgrade --force-reinstall "$MLX_LM_FORK"
  success "Custom mlx-lm installed."
}

ensure_models() {
  mkdir -p "$MODEL_DIR"
  [ -x "$HF_BIN" ] || die "Hugging Face CLI wasn't installed properly. Try re-running the installer."

  download_model() {
    local model="$1"
    local size_hint="$2"
    local repo="mlx-community/$model"
    local target="$MODEL_DIR/$model"
    if [ -f "$target/config.json" ]; then
      success "$model is already downloaded."
      return
    fi

    info "Downloading $model (~${size_hint})..."
    info "This is a large download and may take 10-30 minutes depending on your internet speed."
    info "You'll see a progress bar below. Feel free to grab a coffee! ☕"
    printf '\n'
    if ! "$HF_BIN" download "$repo" --local-dir "$target"; then
      printf '\n'
      warn "Download failed. This usually means one of:"
      warn "  • Your internet connection dropped"
      warn "  • Hugging Face needs you to accept a license agreement"
      warn ""
      warn "To fix: run '$HF_BIN login', accept the model terms at huggingface.co,"
      warn "then re-run this installer. It will pick up where it left off."
      die "Could not download $model."
    fi
    printf '\n'
    success "$model downloaded."
  }

  download_model "$MODEL_1" "15 GB"
  download_model "$MODEL_2" "18 GB"
}

ensure_api_key() {
  if [ -f "$NODE_ENV" ]; then
    # shellcheck disable=SC1090
    source "$NODE_ENV"
  fi

  if [ -z "${OMLX_API_KEY:-}" ]; then
    info "Generating a unique secret key for this node (like a password for the AI server)..."
    OMLX_API_KEY="$(python3 -c 'import secrets, string; alphabet = string.ascii_letters + string.digits; print("pn-" + "".join(secrets.choice(alphabet) for _ in range(40)))')"
    success "API key generated."
  else
    success "Using existing API key."
  fi
}

write_node_env() {
  local content
  content="$(printf '%s\n' \
    "export OMLX_API_KEY=$OMLX_API_KEY" \
    "export OMLX_HOST=0.0.0.0" \
    "export OMLX_PORT=5741" \
    "export OMLX_MODEL_DIR=$MODEL_DIR" \
    "export OMLX_LOG_LEVEL=info")"
  write_text_file "$NODE_ENV" 0600 "$content"
}

write_settings_json() {
  python3 -c 'from pathlib import Path; import json, sys; path = Path(sys.argv[1]).expanduser(); path.parent.mkdir(parents=True, exist_ok=True); payload = {"version": "1.0", "server": {"host": "0.0.0.0", "port": 5741, "log_level": "info", "cors_origins": ["*"]}, "model": {"model_dirs": [sys.argv[2]], "model_dir": sys.argv[2], "max_model_memory": "auto", "model_fallback": False}, "memory": {"max_process_memory": "auto", "prefill_memory_guard": True}, "scheduler": {"max_num_seqs": 8, "completion_batch_size": 8}, "cache": {"enabled": True, "ssd_cache_dir": None, "ssd_cache_max_size": "auto", "hot_cache_max_size": "0", "initial_cache_blocks": 256}, "auth": {"api_key": sys.argv[3], "secret_key": None, "skip_api_key_verification": False, "sub_keys": []}, "mcp": {"config_path": None}, "huggingface": {"endpoint": ""}, "modelscope": {"endpoint": ""}, "sampling": {"max_context_window": 128000, "max_tokens": 32768, "temperature": 1.0, "top_p": 0.95, "top_k": 0, "repetition_penalty": 1.0}, "logging": {"log_dir": None, "retention_days": 7}, "claude_code": {"context_scaling_enabled": False, "target_context_size": 200000, "mode": "cloud", "opus_model": None, "sonnet_model": None, "haiku_model": None}, "integrations": {"codex_model": None, "opencode_model": None, "openclaw_model": None, "openclaw_tools_profile": "coding"}, "ui": {"language": "en"}}; path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")' "$OMLX_BASE/settings.json" "$MODEL_DIR" "$OMLX_API_KEY"
}

write_start_script() {
  local content
  content="$(printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    "source \"$NODE_ENV\"" \
    "source \"$VENV_DIR/bin/activate\"" \
    "exec omlx serve --base-path \"$OMLX_BASE\" --host \"0.0.0.0\" --port \"5741\" --model-dir \"$MODEL_DIR\" --api-key \"$OMLX_API_KEY\""
  )"
  write_text_file "$START_SCRIPT" 0755 "$content"
}

write_launchagent() {
  python3 -c 'from pathlib import Path; import plistlib, sys; plist = Path(sys.argv[1]).expanduser(); script = Path(sys.argv[2]).expanduser(); logs = Path(sys.argv[3]).expanduser(); label = sys.argv[4]; plist.parent.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True); payload = {"Label": label, "ProgramArguments": [str(script)], "RunAtLoad": True, "KeepAlive": True, "WorkingDirectory": str(script.parent), "StandardOutPath": str(logs / "edge.stdout.log"), "StandardErrorPath": str(logs / "edge.stderr.log")}; plist.write_bytes(plistlib.dumps(payload))' "$LAUNCH_AGENT" "$START_SCRIPT" "$STATE_DIR/logs" "$LAUNCH_LABEL"
}

load_launchagent() {
  local uid
  uid="$(id -u)"
  launchctl bootout "gui/$uid" "$LAUNCH_AGENT" >/dev/null 2>&1 || true
  if ! launchctl bootstrap "gui/$uid" "$LAUNCH_AGENT" >/dev/null 2>&1; then
    launchctl unload "$LAUNCH_AGENT" >/dev/null 2>&1 || true
    launchctl load -w "$LAUNCH_AGENT"
  fi
  launchctl kickstart -k "gui/$uid/$LAUNCH_LABEL" >/dev/null 2>&1 || true
}

write_node_json() {
  python3 -c 'from pathlib import Path; import json, sys; path = Path(sys.argv[1]).expanduser(); path.parent.mkdir(parents=True, exist_ok=True); payload = {"tailscale_ip": sys.argv[2], "port": 5741, "api_key": sys.argv[3], "models": [sys.argv[4], sys.argv[5]], "endpoint": f"http://{sys.argv[2]}:5741/v1"}; path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")' "$NODE_JSON" "$TAILSCALE_IP" "$OMLX_API_KEY" "$MODEL_1" "$MODEL_2"
}

print_summary() {
  printf '\n'
  printf '\033[1m\033[0;32m════════════════════════════════════════════════════════════\033[0m\n'
  printf '\033[1m\033[0;32m  🎉  All done! Your Mac is now an AI edge node.\033[0m\n'
  printf '\033[1m\033[0;32m════════════════════════════════════════════════════════════\033[0m\n'
  printf '\n'
  printf '  \033[1mWhat just happened:\033[0m\n'
  printf '  • The AI server (oMLX) is running in the background\n'
  printf '  • It will start automatically when you reboot your Mac\n'
  printf '  • Two AI models are loaded and ready to go\n'
  printf '\n'
  printf '  \033[1mYour node details:\033[0m\n'
  printf '  • Tailscale IP:  %s\n' "$TAILSCALE_IP"
  printf '  • Models:        %s\n' "$MODEL_1"
  printf '                   %s\n' "$MODEL_2"
  printf '  • Config file:   %s\n' "$NODE_JSON"
  printf '\n'
  printf '  \033[1m\033[0;36mNext step:\033[0m Send your node.json file to the network admin.\n'
  printf '  They will add your Mac to the cluster so it can start\n'
  printf '  processing AI requests from the network.\n'
  printf '\n'
  printf '  \033[2mTo send it: open Finder, press Cmd+Shift+G, paste this path:\033[0m\n'
  printf '  \033[2m%s\033[0m\n' "$STATE_DIR"
  printf '\n'
}

main() {
  printf '\n'
  printf '\033[1m\033[0;36m╔════════════════════════════════════════════════════════════\033[0m\n'
  printf '\033[1m\033[0;36m║           oMLX PrivateNet — Edge Node Installer            ║\033[0m\n'
  printf '\033[1m\033[0;36m╚════════════════════════════════════════════════════════════\033[0m\n'
  printf '\n'
  printf '  This script will turn your Mac into an AI processing node\n'
  printf '  on a private network. Here is what it will do:\n'
  printf '\n'
  printf '  1. Check that your Mac is compatible (Apple Silicon required)\n'
  printf '  2. Install developer tools (Homebrew, Python, Git)\n'
  printf '  3. Set up Tailscale (the private network)\n'
  printf '  4. Download the AI server software (oMLX)\n'
  printf '  5. Install Python libraries for AI inference\n'
  printf '  6. Download two AI models (~33 GB total — this takes a while!)\n'
  printf '  7. Configure everything and start the server\n'
  printf '\n'
  printf '  \033[2mThe whole process takes about 30-60 minutes, mostly waiting\n'
  printf '  for the AI models to download. You can use your Mac normally\n'
  printf '  while it runs.\033[0m\n'
  printf '\n'
  printf '  \033[2mSafe to re-run — it will skip anything already installed.\033[0m\n'
  printf '\n'

  # ── Step 1: Compatibility check ──
  step "Checking your Mac" "Making sure this is an Apple Silicon Mac (M1/M2/M3/M4)..."
  require_supported_host
  success "Apple Silicon Mac confirmed — you're good to go!"

  # ── Step 2: Homebrew + dev tools ──
  step "Installing developer tools" "These are standard Mac tools used by millions of developers."
  ensure_homebrew

  # ── Step 3: Dependencies ──
  step "Installing required software" "Python (runs the AI), Git (downloads code), Tailscale (private network)."
  ensure_dependencies

  # ── Step 4: Tailscale ──
  step "Connecting to Tailscale" "Tailscale creates a secure private network between all the Macs."
  ensure_tailscale_ip

  # ── Step 5: oMLX source ──
  step "Downloading the AI server" "oMLX is the server that runs AI models on Apple Silicon."
  ensure_omlx_source

  # ── Step 6: Python environment + packages ──
  step "Setting up the AI software" "Installing all the Python libraries needed to run AI models."
  ensure_venv

  # ── Step 7: AI models ──
  step "Downloading AI models" "These are the actual AI brains — two versions of Google's Gemma 4."
  info "⏱  This is the longest step. Total download: ~33 GB."
  info "   On a 100 Mbps connection: ~45 minutes. On gigabit: ~5 minutes."
  printf '\n'
  ensure_models

  # ── Step 8: Configuration ──
  step "Configuring your node" "Setting up security keys and server settings."
  ensure_api_key
  write_node_env
  success "Environment config written."
  write_settings_json
  success "Server settings written."

  # ── Step 9: Launch ──
  step "Starting the AI server" "Setting it up to run automatically, even after reboots."
  write_start_script
  success "Startup script created."
  write_launchagent
  success "Automatic startup configured (LaunchAgent)."
  load_launchagent
  success "AI server is now running!"
  write_node_json
  success "Node registration file created."

  print_summary
}

main "$@"
