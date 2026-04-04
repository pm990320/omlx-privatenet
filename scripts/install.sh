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

log() {
  printf '\n==> %s\n' "$*"
}

warn() {
  printf 'WARNING: %s\n' "$*" >&2
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

write_text_file() {
  local path="$1"
  local mode="$2"
  local content="$3"
  python3 -c 'from pathlib import Path; import sys; p = Path(sys.argv[1]).expanduser(); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(sys.argv[3], encoding="utf-8"); p.chmod(int(sys.argv[2], 8))' "$path" "$mode" "$content"
}

require_supported_host() {
  [ "$(uname -s)" = "Darwin" ] || die "omlx-privatenet edge nodes only support macOS."
  [ "$(uname -m)" = "arm64" ] || die "omlx-privatenet requires Apple Silicon (arm64). Intel Macs are not supported."
}

ensure_homebrew() {
  if command -v brew >/dev/null 2>&1; then
    BREW_BIN="$(command -v brew)"
  elif [ -x /opt/homebrew/bin/brew ]; then
    BREW_BIN="/opt/homebrew/bin/brew"
  else
    log "Installing Homebrew"
    NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    BREW_BIN="/opt/homebrew/bin/brew"
  fi

  [ -x "$BREW_BIN" ] || die "Homebrew installation failed."
  eval "$($BREW_BIN shellenv)"
}

ensure_brew_package() {
  local formula="$1"
  if ! "$BREW_BIN" list "$formula" >/dev/null 2>&1; then
    log "Installing $formula"
    "$BREW_BIN" install "$formula"
  fi
}

ensure_brew_cask() {
  local cask="$1"
  if ! "$BREW_BIN" list --cask "$cask" >/dev/null 2>&1; then
    log "Installing $cask"
    "$BREW_BIN" install --cask "$cask"
  fi
}

ensure_dependencies() {
  ensure_brew_package python@3.13
  ensure_brew_package git
  ensure_brew_cask tailscale

  PYTHON_BIN="$($BREW_BIN --prefix python@3.13)/bin/python3.13"
  [ -x "$PYTHON_BIN" ] || die "python3.13 was not installed correctly."
}

ensure_tailscale_ip() {
  if ! command -v tailscale >/dev/null 2>&1 && [ -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ]; then
    export PATH="/Applications/Tailscale.app/Contents/MacOS:$PATH"
  fi

  command -v tailscale >/dev/null 2>&1 || die "tailscale CLI not found after installation."
  open -ga Tailscale >/dev/null 2>&1 || true

  TAILSCALE_IP="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
  if [ -z "$TAILSCALE_IP" ]; then
    printf '\nTailscale needs to be connected before this node can register.\n'
    printf '1. Complete Tailscale sign-in if a browser or app prompt appears.\n'
    printf '2. Wait until the app shows Connected.\n'
    printf '3. Press Enter here to continue.\n\n'
    read -r
    TAILSCALE_IP="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
  fi

  [ -n "$TAILSCALE_IP" ] || die "Could not determine a Tailscale IPv4 address. Connect Tailscale, then re-run the installer."
}

ensure_omlx_source() {
  mkdir -p "$INSTALL_ROOT"
  if [ -d "$OMLX_SRC/.git" ]; then
    log "Updating oMLX source"
    git -C "$OMLX_SRC" remote set-url origin "$OMLX_REPO"
    git -C "$OMLX_SRC" fetch --tags origin
    git -C "$OMLX_SRC" checkout "$OMLX_REF"
  elif [ -e "$OMLX_SRC" ]; then
    die "$OMLX_SRC exists but is not a git checkout. Move it away and re-run."
  else
    log "Cloning oMLX $OMLX_REF"
    git clone --branch "$OMLX_REF" --depth 1 "$OMLX_REPO" "$OMLX_SRC"
  fi
}

ensure_venv() {
  mkdir -p "$STATE_DIR"
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    log "Creating Python virtual environment"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi

  PIP_BIN="$VENV_DIR/bin/pip"
  HF_BIN="$VENV_DIR/bin/huggingface-cli"

  log "Installing oMLX, Hugging Face CLI, xgrammar, and the Gemma 4 mlx-lm fork"
  "$PIP_BIN" install --upgrade pip setuptools wheel
  "$PIP_BIN" install --upgrade huggingface-hub
  "$PIP_BIN" install -e "$OMLX_SRC"
  "$PIP_BIN" install --upgrade xgrammar
  "$PIP_BIN" install --upgrade --force-reinstall "$MLX_LM_FORK"
}

ensure_models() {
  mkdir -p "$MODEL_DIR"
  [ -x "$HF_BIN" ] || die "huggingface-cli was not installed into $VENV_DIR."

  download_model() {
    local model="$1"
    local repo="mlx-community/$model"
    local target="$MODEL_DIR/$model"
    if [ -f "$target/config.json" ]; then
      log "Model already present: $model"
      return
    fi

    log "Downloading $model"
    if ! "$HF_BIN" download "$repo" --local-dir "$target"; then
      die "Failed to download $repo. If Hugging Face prompts for auth or license acceptance, run '$HF_BIN login', accept the model terms, then re-run this installer."
    fi
  }

  download_model "$MODEL_1"
  download_model "$MODEL_2"
}

ensure_api_key() {
  if [ -f "$NODE_ENV" ]; then
    # shellcheck disable=SC1090
    source "$NODE_ENV"
  fi

  if [ -z "${OMLX_API_KEY:-}" ]; then
    OMLX_API_KEY="$(python3 -c 'import secrets, string; alphabet = string.ascii_letters + string.digits; print("pn-" + "".join(secrets.choice(alphabet) for _ in range(40)))')"
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
  printf '\nNode ready! Share your node.json with the network admin to join the cluster\n'
  printf 'node.json: %s\n' "$NODE_JSON"
  printf 'Tailscale IP: %s\n' "$TAILSCALE_IP"
  printf 'Models: %s, %s\n' "$MODEL_1" "$MODEL_2"
}

main() {
  require_supported_host
  ensure_homebrew
  ensure_dependencies
  ensure_tailscale_ip
  ensure_omlx_source
  ensure_venv
  ensure_models
  ensure_api_key
  write_node_env
  write_settings_json
  write_start_script
  write_launchagent
  load_launchagent
  write_node_json
  print_summary
}

main "$@"
