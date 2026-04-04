#!/usr/bin/env bash
set -euo pipefail

# ── Configurable defaults ────────────────────────────────────────────────────
OMLX_REPO="https://github.com/pm990320/omlx.git"
OMLX_REF="v0.3.2"
PRIVATENET_REPO="https://github.com/pm990320/omlx-privatenet.git"
PRIVATENET_REF="main"
MLX_LM_FORK="git+https://github.com/pm990320/mlx-lm@feat/gemma4-tool-calling"
MODEL_1="gemma-4-26b-a4b-it-4bit"
MODEL_2="gemma-4-31b-it-4bit"
TAILSCALE_TAG="tag:omlx-node"

# ── Paths ────────────────────────────────────────────────────────────────────
STATE_DIR="$HOME/.omlx-privatenet"
OMLX_BASE="$HOME/.omlx"
INSTALL_ROOT="$HOME/omlx-privatenet"
OMLX_SRC="$INSTALL_ROOT/omlx"
PRIVATENET_SRC="$INSTALL_ROOT/omlx-privatenet"
VENV_DIR="$STATE_DIR/venv"
NODE_ENV="$STATE_DIR/node.env"
ROUTER_CONFIG="$STATE_DIR/router.json"
OMLX_START_SCRIPT="$STATE_DIR/start-omlx.sh"
ROUTER_START_SCRIPT="$STATE_DIR/start-router.sh"
MODEL_DIR="$OMLX_BASE/models"
OMLX_AGENT="$HOME/Library/LaunchAgents/com.omlx-privatenet.omlx.plist"
ROUTER_AGENT="$HOME/Library/LaunchAgents/com.omlx-privatenet.router.plist"
OMLX_LABEL="com.omlx-privatenet.omlx"
ROUTER_LABEL="com.omlx-privatenet.router"

# ── Runtime state ────────────────────────────────────────────────────────────
TAILSCALE_IP=""
TAILSCALE_TAG_STATUS=""
BREW_BIN=""
PYTHON_BIN=""
PIP_BIN=""
HF_BIN=""
NODE_ID=""
OMLX_API_KEY=""
EXISTING_OMLX="false"      # true when oMLX is already running and managed externally
OPENCLAW_BIN=""             # path to openclaw CLI, empty if not found
INSTALL_OPENCLAW="false"    # true when user opts in to plugin install
STEP=0
TOTAL_STEPS=0               # computed dynamically

# ── Non-interactive / CI mode ────────────────────────────────────────────────
# Set NONINTERACTIVE=1 to skip all prompts (defaults to no for optional steps).
# Override specific choices with environment variables:
#   OMLX_PRIVATENET_INSTALL_OPENCLAW=1   Install the OpenClaw plugin without asking
#   OMLX_PRIVATENET_INSTALL_OPENCLAW=0   Skip the OpenClaw plugin without asking
NONINTERACTIVE="${NONINTERACTIVE:-0}"

# ── Colours (used inline via printf escape codes) ────────────────────────────

# ── Helpers ──────────────────────────────────────────────────────────────────
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
  python3 -c '
import os, sys
from pathlib import Path
p = Path(sys.argv[1]).expanduser()
p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
# Write with restrictive umask so file is never world-readable, even briefly
old_umask = os.umask(0o177)  # creates files as 0600
try:
    p.write_text(sys.argv[3], encoding="utf-8")
finally:
    os.umask(old_umask)
p.chmod(int(sys.argv[2], 8))
' "$path" "$mode" "$content"
}

# ── Detection: is oMLX already running? ──────────────────────────────────────
detect_existing_omlx() {
  # Check if oMLX is responding on its default port
  if curl -sf http://127.0.0.1:5741/health >/dev/null 2>&1; then
    # Verify we have a settings file to read the API key from
    if [ -f "$OMLX_BASE/settings.json" ]; then
      EXISTING_OMLX="true"
      return
    fi
  fi

  # Also check for any oMLX LaunchAgent (ours or third-party)
  for plist in "$HOME"/Library/LaunchAgents/*omlx*.plist; do
    if [ -f "$plist" ] && [ "$plist" != "$OMLX_AGENT" ]; then
      if [ -f "$OMLX_BASE/settings.json" ]; then
        EXISTING_OMLX="true"
        return
      fi
    fi
  done
}

# Read the API key from an existing oMLX settings.json
read_existing_api_key() {
  local key
  key="$(python3 -c '
import json, sys
try:
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    print(cfg.get("auth", {}).get("api_key", ""))
except Exception:
    pass
' "$OMLX_BASE/settings.json" 2>/dev/null || true)"
  if [ -n "$key" ]; then
    OMLX_API_KEY="$key"
  fi
}

# Read models from existing oMLX (live endpoint preferred, fall back to disk)
read_existing_models() {
  local models_json
  models_json="$(curl -sf -H "Authorization: Bearer $OMLX_API_KEY" \
    http://127.0.0.1:5741/v1/models 2>/dev/null || true)"

  if [ -n "$models_json" ]; then
    local detected
    detected="$(python3 -c '
import json, sys
try:
    data = json.loads(sys.argv[1])
    models = data if isinstance(data, list) else data.get("data", data.get("models", []))
    for m in models:
        name = m.get("id", "") if isinstance(m, dict) else str(m)
        if name:
            print(name)
except Exception:
    pass
' "$models_json" 2>/dev/null || true)"
    if [ -n "$detected" ]; then
      info "Detected models from running oMLX:"
      while IFS= read -r m; do
        info "  $m"
      done <<< "$detected"
      return
    fi
  fi

  # Fall back to model directories on disk
  if [ -d "$MODEL_DIR" ]; then
    for d in "$MODEL_DIR"/*/; do
      [ -f "$d/config.json" ] && info "  $(basename "$d")"
    done
  fi
}

# ── Compute step count ───────────────────────────────────────────────────────
# ── OpenClaw integration ──────────────────────────────────────────────────────
detect_openclaw() {
  # Check common locations for the openclaw CLI
  for candidate in \
    "$(command -v openclaw 2>/dev/null || true)" \
    "$HOME/.local/bin/openclaw" \
    "/usr/local/bin/openclaw" \
    "$HOME/openclaw/openclaw/openclaw" \
    ; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      OPENCLAW_BIN="$candidate"
      return
    fi
  done
}

ask_openclaw_plugin() {
  if [ -z "$OPENCLAW_BIN" ]; then
    return
  fi

  # Check for env var override (works in both interactive and non-interactive modes)
  local env_override="${OMLX_PRIVATENET_INSTALL_OPENCLAW:-}"
  if [ "$env_override" = "1" ] || [ "$env_override" = "true" ] || [ "$env_override" = "yes" ]; then
    INSTALL_OPENCLAW="true"
    return
  fi
  if [ "$env_override" = "0" ] || [ "$env_override" = "false" ] || [ "$env_override" = "no" ]; then
    return
  fi

  # In non-interactive mode with no explicit override, skip
  if [ "$NONINTERACTIVE" = "1" ]; then
    info "OpenClaw detected but skipping plugin install (non-interactive mode)."
    info "Set OMLX_PRIVATENET_INSTALL_OPENCLAW=1 to install automatically."
    return
  fi

  printf '\n'
  printf '  \033[1mOpenClaw detected!\033[0m\n'
  printf '  The openclaw-omlx plugin lets OpenClaw use your PrivateNet models.\n'
  printf '  It will configure OpenClaw to connect to the router on this Mac.\n'
  printf '\n'
  printf '  \033[1mInstall the openclaw-omlx plugin? [Y/n]\033[0m '
  local answer
  read -r answer < /dev/tty
  case "$answer" in
    [nN]|[nN][oO])
      info "Skipping OpenClaw plugin."
      ;;
    *)
      INSTALL_OPENCLAW="true"
      ;;
  esac
}

install_openclaw_plugin() {
  # Check if already installed
  local already_installed="false"
  if "$OPENCLAW_BIN" plugins list 2>/dev/null | grep -q "omlx"; then
    already_installed="true"
  fi

  if [ "$already_installed" = "true" ]; then
    info "openclaw-omlx plugin is already installed — updating configuration..."
  else
    info "Installing the openclaw-omlx plugin..."
    if ! "$OPENCLAW_BIN" plugins install openclaw-omlx 2>&1; then
      warn "Plugin install failed. You can install it manually later with:"
      warn "  openclaw plugins install openclaw-omlx"
      return
    fi
    success "openclaw-omlx plugin installed."
  fi

  # Configure the plugin to point at the router
  local router_url="http://${TAILSCALE_IP}:8741/v1"
  local openclaw_config="$HOME/.openclaw/openclaw.json"

  if [ -f "$openclaw_config" ]; then
    info "Configuring plugin to use router at $router_url..."
    python3 -c '
import json, shutil, sys
from pathlib import Path

config_path = Path(sys.argv[1])
router_url = sys.argv[2]
api_key = sys.argv[3]

# Back up before modifying
backup_path = config_path.with_suffix(".json.pre-privatenet-bak")
shutil.copy2(config_path, backup_path)

with open(config_path, "r") as f:
    config = json.load(f)

plugins = config.setdefault("plugins", {})
entries = plugins.setdefault("entries", {})
omlx = entries.setdefault("omlx", {})
omlx["enabled"] = True
omlx_config = omlx.setdefault("config", {})

# Only set baseUrl if missing or still pointing at a local oMLX default
# (do not overwrite intentional user customizations)
DEFAULT_OMLX_URLS = {
    "http://127.0.0.1:5741/v1",
    "http://127.0.0.1:8000/v1",
    "http://localhost:5741/v1",
    "http://localhost:8000/v1",
}
current_url = omlx_config.get("baseUrl", "")
if not current_url or current_url in DEFAULT_OMLX_URLS:
    omlx_config["baseUrl"] = router_url
    print(f"UPDATED baseUrl -> {router_url}")
else:
    print(f"KEPT existing baseUrl: {current_url}")

# Only set apiKey if not already configured
if not omlx_config.get("apiKey") and api_key:
    omlx_config["apiKey"] = api_key
    print("UPDATED apiKey")
else:
    print("KEPT existing apiKey")

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
' "$openclaw_config" "$router_url" "$OMLX_API_KEY" 2>&1 | while IFS= read -r line; do
      info "$line"
    done
    success "OpenClaw plugin configured (backup at openclaw.json.pre-privatenet-bak)"
  else
    warn "Could not find openclaw.json at $openclaw_config"
    warn "After installing OpenClaw, configure the plugin manually:"
    warn "  baseUrl: $router_url"
  fi
}

compute_steps() {
  # Steps that always run:
  #  1. Check Mac
  #  2. Install developer tools
  #  3. Install required software
  #  4. Connect Tailscale
  #  5. Advertise tag
  #  6. Writing configuration
  #  7. Installing automatic startup
  #  8. Starting services
  TOTAL_STEPS=8

  if [ "$EXISTING_OMLX" = "false" ]; then
    # Extra steps for fresh oMLX install:
    #  +1 Download software (repos)
    #  +1 Set up Python environment
    #  +1 Download AI models
    TOTAL_STEPS=$((TOTAL_STEPS + 3))
  else
    # Still need to download the router code
    #  +1 Download router code
    #  +1 Set up Python environment (router deps only)
    TOTAL_STEPS=$((TOTAL_STEPS + 2))
  fi

  if [ "$INSTALL_OPENCLAW" = "true" ]; then
    TOTAL_STEPS=$((TOTAL_STEPS + 1))
  fi
}

# ── System checks ────────────────────────────────────────────────────────────
require_supported_host() {
  [ "$(uname -s)" = "Darwin" ] || die "This installer only works on macOS. Linux and Windows are not supported."
  [ "$(uname -m)" = "arm64" ] || die "This requires an Apple Silicon Mac (M1, M2, M3, or M4 chip). Older Intel Macs can't run the AI models we need."
}

# ── Homebrew ─────────────────────────────────────────────────────────────────
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

ensure_tailscale_installed() {
  # Tailscale can be installed many ways: Homebrew cask, Mac App Store, .pkg,
  # or the standalone CLI. Check all of them before trying brew install.
  if "$BREW_BIN" list --cask tailscale >/dev/null 2>&1; then
    success "Tailscale (the private network that connects all the Macs together) is already installed."
    return
  fi
  if [ -d "/Applications/Tailscale.app" ]; then
    success "Tailscale is already installed (App Store / standalone)."
    return
  fi
  if command -v tailscale >/dev/null 2>&1; then
    success "Tailscale CLI is already available."
    return
  fi
  if [ -x /usr/local/bin/tailscale ]; then
    success "Tailscale is already installed (system package)."
    return
  fi

  info "Installing Tailscale (the private network that connects all the Macs together)..."
  "$BREW_BIN" install --cask tailscale
  success "Tailscale installed."
}

ensure_git_installed() {
  # Git can come from Xcode CLI tools, Homebrew, or standalone installers
  if command -v git >/dev/null 2>&1; then
    success "Git (for downloading source code) is already installed."
    return
  fi
  info "Installing Git (for downloading source code)..."
  "$BREW_BIN" install git
  success "Git installed."
}

ensure_python_installed() {
  # Check Homebrew Python 3.13 first (preferred), then fall back to any 3.13+
  if [ -x "$($BREW_BIN --prefix python@3.13 2>/dev/null)/bin/python3.13" ] 2>/dev/null; then
    PYTHON_BIN="$($BREW_BIN --prefix python@3.13)/bin/python3.13"
    success "Python 3.13 (the programming language that powers the AI server) is already installed."
    return
  fi

  # Check for any system Python >= 3.13
  local sys_python
  for candidate in python3.13 python3; do
    sys_python="$(command -v "$candidate" 2>/dev/null || true)"
    if [ -n "$sys_python" ]; then
      local ver
      ver="$("$sys_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
      if [ -n "$ver" ] && python3 -c "exit(0 if tuple(map(int, '$ver'.split('.'))) >= (3, 13) else 1)" 2>/dev/null; then
        PYTHON_BIN="$sys_python"
        success "Python $ver is already installed."
        return
      fi
    fi
  done

  info "Installing Python 3.13 (the programming language that powers the AI server)..."
  "$BREW_BIN" install python@3.13
  PYTHON_BIN="$($BREW_BIN --prefix python@3.13)/bin/python3.13"
  success "Python 3.13 installed."
}

ensure_dependencies() {
  ensure_python_installed
  ensure_git_installed
  ensure_tailscale_installed

  [ -x "$PYTHON_BIN" ] || die "Python 3.13 didn't install correctly. Try running the installer again."
}

# ── Tailscale ────────────────────────────────────────────────────────────────
ensure_tailscale_cli() {
  if ! command -v tailscale >/dev/null 2>&1 && [ -x /Applications/Tailscale.app/Contents/MacOS/Tailscale ]; then
    export PATH="/Applications/Tailscale.app/Contents/MacOS:$PATH"
  fi
  command -v tailscale >/dev/null 2>&1 || die "Tailscale CLI not found. Try re-running this installer."
}

ensure_tailscale_ip() {
  ensure_tailscale_cli
  open -ga Tailscale >/dev/null 2>&1 || true

  TAILSCALE_IP="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
  if [ -z "$TAILSCALE_IP" ]; then
    printf '\n'
    info "Tailscale needs to be connected before we can continue."
    info "Tailscale is a free private network that securely connects your Macs."
    printf '\n'
    if [ "$NONINTERACTIVE" = "1" ]; then
      die "Tailscale is not connected and installer is running in non-interactive mode. Connect Tailscale first."
    fi
    printf '      \033[1mIf this is your first time using Tailscale:\033[0m\n'
    printf '      1. macOS may ask you to approve a system extension.\n'
    printf '         Go to System Settings > Privacy & Security and click Allow.\n'
    printf '      2. A sign-in window should open in your browser.\n'
    printf '         Sign in or create a free Tailscale account.\n'
    printf '      3. Wait until the Tailscale menu bar icon shows "Connected".\n'
    printf '\n'
    printf '      \033[1mImportant:\033[0m All the Macs in your PrivateNet cluster must be\n'
    printf '      on the \033[1msame Tailscale network\033[0m (tailnet). If someone else set this\n'
    printf '      up, ask them to invite you from the Tailscale admin console.\n'
    printf '\n'
    printf '      Press Enter when Tailscale shows "Connected".\n\n'
    read -r < /dev/tty
    TAILSCALE_IP="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
  fi

  [ -n "$TAILSCALE_IP" ] || die "Couldn't get your Tailscale address. Make sure Tailscale is connected, then re-run this installer."
  success "Tailscale connected — your private network address is $TAILSCALE_IP"
}

ensure_tailscale_tag() {
  # Check if we already have the tag
  local current_tags
  current_tags="$(tailscale status --json 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    tags = d.get("Self", {}).get("Tags", [])
    print(",".join(tags))
except Exception:
    pass
' 2>/dev/null || true)"

  if echo "$current_tags" | grep -qF "$TAILSCALE_TAG"; then
    TAILSCALE_TAG_STATUS="ok"
    success "This Mac is already advertising the tag $TAILSCALE_TAG"
    return
  fi

  # Build the new tag list by appending ours to any existing tags
  local new_tags="$TAILSCALE_TAG"
  if [ -n "$current_tags" ]; then
    new_tags="$current_tags,$TAILSCALE_TAG"
  fi

  # Show the user what we're about to do and ask permission
  printf '\n'
  info "PrivateNet peers discover each other using the Tailscale tag $TAILSCALE_TAG."
  info "To join the network, this Mac needs to advertise that tag."
  if [ -n "$current_tags" ]; then
    info "Your current tags: $current_tags"
    info "New tags will be:  $new_tags"
  fi
  printf '\n'

  local do_tag="false"
  if [ "$NONINTERACTIVE" = "1" ]; then
    do_tag="true"
  else
    printf '  \033[1mAdd the %s tag to this Mac? [Y/n]\033[0m ' "$TAILSCALE_TAG"
    local answer
    read -r answer < /dev/tty
    case "$answer" in
      [nN]|[nN][oO]) do_tag="false" ;;
      *) do_tag="true" ;;
    esac
  fi

  if [ "$do_tag" = "false" ]; then
    TAILSCALE_TAG_STATUS="skipped"
    warn "Skipped. Other PrivateNet nodes won't discover this Mac automatically."
    warn "You can add the tag later with: tailscale set --advertise-tags=$new_tags"
    return
  fi

  # Use 'tailscale set' (modifies only the specified field, unlike 'tailscale up')
  if tailscale set --advertise-tags="$new_tags" >/dev/null 2>&1; then
    TAILSCALE_TAG_STATUS="ok"
    success "This Mac is now advertising the tag $TAILSCALE_TAG"
    return
  fi

  # Fallback: try 'tailscale up' with --reset for older Tailscale versions
  if tailscale up --advertise-tags="$new_tags" --reset >/dev/null 2>&1; then
    TAILSCALE_TAG_STATUS="ok"
    success "This Mac is now advertising the tag $TAILSCALE_TAG"
    return
  fi

  TAILSCALE_TAG_STATUS="needs-admin"
  warn "Couldn't set the tag automatically. This usually means your Tailscale admin"
  warn "needs to allow the tag $TAILSCALE_TAG in the tailnet ACL policy."
  warn "After the policy is updated, run: tailscale set --advertise-tags=$new_tags"
}

# ── Source code ──────────────────────────────────────────────────────────────
ensure_repos() {
  mkdir -p "$INSTALL_ROOT"

  if [ "$EXISTING_OMLX" = "false" ]; then
    if [ -d "$OMLX_SRC/.git" ]; then
      info "oMLX source already exists — updating to $OMLX_REF..."
      git -C "$OMLX_SRC" remote set-url origin "$OMLX_REPO"
      git -C "$OMLX_SRC" fetch --tags origin
      git -C "$OMLX_SRC" checkout -f "$OMLX_REF"
      success "oMLX updated."
    elif [ -e "$OMLX_SRC" ]; then
      die "$OMLX_SRC already exists but isn't a proper install. Please delete or move that folder, then re-run."
    else
      info "Downloading oMLX $OMLX_REF (the AI inference server)..."
      git clone --branch "$OMLX_REF" --depth 1 "$OMLX_REPO" "$OMLX_SRC"
      success "oMLX downloaded."
    fi
  fi

  if [ -d "$PRIVATENET_SRC/.git" ]; then
    info "PrivateNet router source already exists — updating to $PRIVATENET_REF..."
    git -C "$PRIVATENET_SRC" remote set-url origin "$PRIVATENET_REPO"
    git -C "$PRIVATENET_SRC" fetch origin
    # This is a managed checkout — discard any local changes and force-update
    git -C "$PRIVATENET_SRC" checkout -f "$PRIVATENET_REF"
    git -C "$PRIVATENET_SRC" reset --hard "origin/$PRIVATENET_REF"
    success "PrivateNet router updated."
  elif [ -e "$PRIVATENET_SRC" ]; then
    die "$PRIVATENET_SRC already exists but isn't a proper git checkout. Please delete or move that folder, then re-run."
  else
    info "Downloading the PrivateNet router code..."
    git clone --branch "$PRIVATENET_REF" --depth 1 "$PRIVATENET_REPO" "$PRIVATENET_SRC"
    success "PrivateNet router downloaded."
  fi
}

# ── Python environment ───────────────────────────────────────────────────────
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

  info "Installing required software. This may take a few minutes depending on your internet speed."
  printf '\n'

  info "Upgrading core Python tools..."
  "$PIP_BIN" install --upgrade pip setuptools wheel >/dev/null 2>&1
  success "Core tools ready."

  if [ "$EXISTING_OMLX" = "false" ]; then
    info "Installing Hugging Face tools (for downloading AI models)..."
    "$PIP_BIN" install --upgrade huggingface-hub >/dev/null 2>&1
    success "Hugging Face tools installed."

    info "Installing oMLX (the local inference server)..."
    "$PIP_BIN" install -e "$OMLX_SRC" >/dev/null 2>&1
    success "oMLX installed."

    info "Installing xgrammar (helps the AI follow structured output formats)..."
    "$PIP_BIN" install --upgrade xgrammar >/dev/null 2>&1
    success "xgrammar installed."

    info "Installing our custom AI language model library (adds Gemma 4 support)..."
    "$PIP_BIN" install --upgrade --force-reinstall "$MLX_LM_FORK" >/dev/null 2>&1
    success "Custom mlx-lm installed."
  fi

  info "Installing the PrivateNet router dependencies..."
  "$PIP_BIN" install -r "$PRIVATENET_SRC/router/requirements.txt" >/dev/null 2>&1
  success "Router dependencies installed."
}

# ── Models ───────────────────────────────────────────────────────────────────
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
    info "You'll see a progress bar below. Feel free to grab a coffee!"
    printf '\n'
    if ! "$HF_BIN" download "$repo" --local-dir "$target"; then
      printf '\n'
      warn "Download failed. This usually means one of:"
      warn "  - Your internet connection dropped"
      warn "  - Hugging Face needs you to accept a license agreement"
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

# ── Configuration ────────────────────────────────────────────────────────────
ensure_api_key() {
  # If we already read it from an existing oMLX install, keep it
  if [ -n "$OMLX_API_KEY" ]; then
    success "Using API key from existing oMLX installation."
    return
  fi

  # Check our own node.env
  if [ -f "$NODE_ENV" ]; then
    # shellcheck disable=SC1090
    source "$NODE_ENV"
  fi

  if [ -z "${OMLX_API_KEY:-}" ]; then
    info "Generating a unique secret key for the local oMLX server..."
    OMLX_API_KEY="$(python3 -c 'import secrets, string; alphabet = string.ascii_letters + string.digits; print("pn-" + "".join(secrets.choice(alphabet) for _ in range(40)))')"
    success "API key generated."
  else
    success "Using existing oMLX API key."
  fi
}

ensure_node_id() {
  NODE_ID="$(scutil --get ComputerName 2>/dev/null || hostname)"
  NODE_ID="$(printf '%s' "$NODE_ID" | tr '[:space:]/' '--' | tr -cd '[:alnum:]._-')"
  [ -n "$NODE_ID" ] || NODE_ID="$(hostname)"
}

write_node_env() {
  local content
  content="$(printf '%s\n' \
    "export OMLX_API_KEY=$OMLX_API_KEY" \
    "export OMLX_HOST=127.0.0.1" \
    "export OMLX_PORT=5741" \
    "export OMLX_MODEL_DIR=$MODEL_DIR" \
    "export OMLX_LOG_LEVEL=info")"
  write_text_file "$NODE_ENV" 0600 "$content"
}

write_settings_json() {
  if [ -f "$OMLX_BASE/settings.json" ]; then
    success "oMLX settings.json already exists — not overwriting."
    return
  fi

  python3 -c '
from pathlib import Path; import json, sys
path = Path(sys.argv[1]).expanduser()
path.parent.mkdir(parents=True, exist_ok=True)
payload = {
  "version": "1.0",
  "server": {"host": "127.0.0.1", "port": 5741, "log_level": "info", "cors_origins": ["*"]},
  "model": {"model_dirs": [sys.argv[2]], "model_dir": sys.argv[2], "max_model_memory": "auto", "model_fallback": False},
  "memory": {"max_process_memory": "auto", "prefill_memory_guard": True},
  "scheduler": {"max_num_seqs": 8, "completion_batch_size": 8},
  "cache": {"enabled": True, "ssd_cache_dir": None, "ssd_cache_max_size": "auto", "hot_cache_max_size": "0", "initial_cache_blocks": 256},
  "auth": {"api_key": sys.argv[3], "secret_key": None, "skip_api_key_verification": False, "sub_keys": []},
  "mcp": {"config_path": None},
  "huggingface": {"endpoint": ""},
  "modelscope": {"endpoint": ""},
  "sampling": {"max_context_window": 128000, "max_tokens": 32768, "temperature": 1.0, "top_p": 0.95, "top_k": 0, "repetition_penalty": 1.0},
  "logging": {"log_dir": None, "retention_days": 7},
  "claude_code": {"context_scaling_enabled": False, "target_context_size": 200000, "mode": "cloud", "opus_model": None, "sonnet_model": None, "haiku_model": None},
  "integrations": {"codex_model": None, "opencode_model": None, "openclaw_model": None, "openclaw_tools_profile": "coding"},
  "ui": {"language": "en"}
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
' "$OMLX_BASE/settings.json" "$MODEL_DIR" "$OMLX_API_KEY"
  success "oMLX settings written."
}

build_local_models_list() {
  # Build the models list for router config.
  # Prefer what's actually available on disk.
  local found_models=()
  if [ -d "$MODEL_DIR" ]; then
    for d in "$MODEL_DIR"/*/; do
      if [ -f "$d/config.json" ]; then
        found_models+=("$(basename "$d")")
      fi
    done
  fi

  if [ ${#found_models[@]} -gt 0 ]; then
    printf '%s\n' "${found_models[@]}"
  else
    # Fallback to defaults
    printf '%s\n%s\n' "$MODEL_1" "$MODEL_2"
  fi
}

write_router_config() {
  local models_json
  models_json="$(python3 -c '
import json, sys
models = [line for line in sys.stdin.read().strip().split("\n") if line]
print(json.dumps(models))
' <<< "$(build_local_models_list)")"

  python3 -c '
from pathlib import Path; import json, sys
path = Path(sys.argv[1]).expanduser()
path.parent.mkdir(parents=True, exist_ok=True)
models = json.loads(sys.argv[7])
payload = {
  "host": "0.0.0.0",
  "port": 8741,
  "api_key": None,
  "connect_timeout_seconds": 10,
  "request_timeout_seconds": 600,
  "discovery_interval_seconds": 30,
  "health_check_timeout_seconds": 5,
  "failure_threshold": 3,
  "prefix_message_count": 3,
  "overload_threshold": None,
  "consistent_hash_replicas": 128,
  "tailscale_tag": sys.argv[2],
  "local_node_id": sys.argv[3],
  "local_tailscale_ip": sys.argv[4],
  "local_omlx_url": "http://127.0.0.1:5741",
  "local_omlx_api_key": sys.argv[5],
  "local_models": models,
  "local_max_concurrent": 8
}
# If config already exists, preserve user edits to fields we do not need to change
existing = {}
if path.exists():
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
# Always update these discovery-critical fields
for key in ("local_node_id", "local_tailscale_ip", "local_omlx_api_key", "local_models"):
    existing[key] = payload[key]
# Set defaults for any missing keys
for key, val in payload.items():
    existing.setdefault(key, val)
path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
' "$ROUTER_CONFIG" "$TAILSCALE_TAG" "$NODE_ID" "$TAILSCALE_IP" "$OMLX_API_KEY" "unused" "$models_json"
}

write_start_scripts() {
  if [ "$EXISTING_OMLX" = "false" ]; then
    local omlx_content
    omlx_content="$(printf '%s\n' \
      '#!/usr/bin/env bash' \
      'set -euo pipefail' \
      "source \"$NODE_ENV\"" \
      "source \"$VENV_DIR/bin/activate\"" \
      "exec omlx serve --base-path \"$OMLX_BASE\" --host \"127.0.0.1\" --port \"5741\" --model-dir \"$MODEL_DIR\" --api-key \"$OMLX_API_KEY\"" \
    )"
    write_text_file "$OMLX_START_SCRIPT" 0700 "$omlx_content"
  fi

  local router_content
  router_content="$(printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -euo pipefail' \
    "source \"$VENV_DIR/bin/activate\"" \
    "cd \"$PRIVATENET_SRC\"" \
    "exec python -m router.server --config \"$ROUTER_CONFIG\" --host \"0.0.0.0\" --port \"8741\"" \
  )"
  write_text_file "$ROUTER_START_SCRIPT" 0700 "$router_content"

  # CLI tool for enable/disable/status
  local cli_content
  cli_content="$(printf '%s\n' \
    '#!/usr/bin/env bash' \
    "export OMLX_PRIVATENET_STATE_DIR=\"$STATE_DIR\"" \
    "source \"$VENV_DIR/bin/activate\"" \
    "cd \"$PRIVATENET_SRC\"" \
    'exec python -m router.cli "$@"' \
  )"
  write_text_file "$STATE_DIR/bin/privatenet" 0755 "$cli_content"

  # Add to PATH via shell profile if not already there
  local bin_dir="$STATE_DIR/bin"
  local path_line="export PATH=\"$bin_dir:\$PATH\""
  local profile=""

  # Detect the right shell profile
  if [ -n "${ZSH_VERSION:-}" ] || [ "$(basename "${SHELL:-}")" = "zsh" ]; then
    profile="$HOME/.zprofile"
  elif [ -f "$HOME/.bash_profile" ]; then
    profile="$HOME/.bash_profile"
  else
    profile="$HOME/.profile"
  fi

  if ! grep -qF "$bin_dir" "$profile" 2>/dev/null; then
    printf '\n# oMLX PrivateNet CLI\n%s\n' "$path_line" >> "$profile"
    success "Added $bin_dir to PATH in $profile"
    info "Run 'source $profile' or open a new terminal to use the 'privatenet' command."
  else
    success "privatenet command is already on PATH."
  fi

  # Also try /usr/local/bin symlink as a convenience
  if [ -d /usr/local/bin ] && [ -w /usr/local/bin ]; then
    ln -sf "$bin_dir/privatenet" /usr/local/bin/privatenet
  fi
}

# ── LaunchAgents ─────────────────────────────────────────────────────────────
write_launchagent() {
  local plist_path="$1"
  local script_path="$2"
  local label="$3"
  local stdout_name="$4"
  local stderr_name="$5"
  python3 -c '
from pathlib import Path; import plistlib, sys
plist = Path(sys.argv[1]).expanduser()
script = Path(sys.argv[2]).expanduser()
logs = Path(sys.argv[3]).expanduser()
label = sys.argv[4]
stdout_name = sys.argv[5]
stderr_name = sys.argv[6]
plist.parent.mkdir(parents=True, exist_ok=True)
logs.mkdir(parents=True, exist_ok=True)
payload = {
    "Label": label,
    "ProgramArguments": [str(script)],
    "RunAtLoad": True,
    "KeepAlive": True,
    "WorkingDirectory": str(script.parent),
    "StandardOutPath": str(logs / stdout_name),
    "StandardErrorPath": str(logs / stderr_name),
}
plist.write_bytes(plistlib.dumps(payload))
' "$plist_path" "$script_path" "$STATE_DIR/logs" "$label" "$stdout_name" "$stderr_name"
}

load_launchagent() {
  local plist_path="$1"
  local label="$2"
  local uid
  uid="$(id -u)"
  launchctl bootout "gui/$uid/$label" >/dev/null 2>&1 || true
  if ! launchctl bootstrap "gui/$uid" "$plist_path" >/dev/null 2>&1; then
    launchctl unload "$plist_path" >/dev/null 2>&1 || true
    launchctl load -w "$plist_path"
  fi
  launchctl kickstart -k "gui/$uid/$label" >/dev/null 2>&1 || true
}

# ── Summary ──────────────────────────────────────────────────────────────────
print_summary() {
  printf '\n'
  printf '\033[1m\033[0;32m════════════════════════════════════════════════════════════\033[0m\n'
  if [ "$EXISTING_OMLX" = "true" ]; then
    printf '\033[1m\033[0;32m  All done! The PrivateNet router is now running.\033[0m\n'
  else
    printf '\033[1m\033[0;32m  All done! This Mac now runs both oMLX and the router.\033[0m\n'
  fi
  printf '\033[1m\033[0;32m════════════════════════════════════════════════════════════\033[0m\n'
  printf '\n'
  printf '  \033[1mWhat just happened:\033[0m\n'
  if [ "$EXISTING_OMLX" = "true" ]; then
    printf '  - Found your existing oMLX on 127.0.0.1:5741 (untouched)\n'
    printf '  - The PrivateNet router is running on 0.0.0.0:8741\n'
    printf '  - The router will start automatically after reboots\n'
  else
    printf '  - oMLX is running locally on 127.0.0.1:5741\n'
    printf '  - The router is running on 0.0.0.0:8741\n'
    printf '  - Both services will start automatically after reboots\n'
  fi
  printf '  - This Mac will discover peer nodes automatically through Tailscale\n'
  printf '\n'
  printf '  \033[1mThis node:\033[0m\n'
  printf '  - Node ID:       %s\n' "$NODE_ID"
  printf '  - Tailscale IP:  %s\n' "$TAILSCALE_IP"
  printf '  - Router URL:    http://%s:8741/v1\n' "$TAILSCALE_IP"
  printf '  - Router config: %s\n' "$ROUTER_CONFIG"
  printf '\n'
  printf '  \033[1mModels available:\033[0m\n'
  build_local_models_list | while IFS= read -r m; do
    printf '  - %s\n' "$m"
  done
  printf '\n'
  if [ "$TAILSCALE_TAG_STATUS" = "needs-admin" ]; then
    printf '  \033[1m\033[0;33mAction still needed:\033[0m Ask your Tailscale admin to allow the tag\n'
    printf '  \033[0;33m  %s\033[0m in the tailnet ACL policy, then run:\n' "$TAILSCALE_TAG"
    printf '  \033[2m  tailscale set --advertise-tags=%s\033[0m\n' "$TAILSCALE_TAG"
    printf '\n'
  elif [ "$TAILSCALE_TAG_STATUS" = "skipped" ]; then
    printf '  \033[0;33mTailscale tag was skipped.\033[0m Add it later with:\n'
    printf '  \033[2m  tailscale set --advertise-tags=%s\033[0m\n' "$TAILSCALE_TAG"
    printf '\n'
  else
    printf '  \033[0;32mThis Mac is advertising the Tailscale tag %s.\033[0m\n' "$TAILSCALE_TAG"
    printf '\n'
  fi
  if [ "$INSTALL_OPENCLAW" = "true" ]; then
    printf '  \033[0;32mOpenClaw is configured to use PrivateNet.\033[0m\n'
    printf '  Restart OpenClaw to pick up the new plugin.\n'
    printf '\n'
  fi
  printf '  \033[1mNext step:\033[0m Point OpenClaw (or any OpenAI-compatible client) at:\n'
  printf '  \033[2mhttp://%s:8741/v1\033[0m\n' "$TAILSCALE_IP"
  printf '\n'
  printf '  \033[1mQuick test:\033[0m\n'
  printf '  \033[2mcurl http://%s:8741/health\033[0m\n' "$TAILSCALE_IP"
  printf '\n'
  printf '  \033[1mManage this node:\033[0m\n'
  printf '  \033[2mprivatenet status\033[0m     Check node status\n'
  printf '  \033[2mprivatenet disable\033[0m    Take this node out of service\n'
  printf '  \033[2mprivatenet enable\033[0m     Bring this node back into service\n'
  printf '\n'
}

# ── Main ─────────────────────────────────────────────────────────────────────
main() {
  printf '\n'
  printf '\033[1m\033[0;36m╔════════════════════════════════════════════════════════════╗\033[0m\n'
  printf '\033[1m\033[0;36m║         oMLX PrivateNet — Peer Node Installer              ║\033[0m\n'
  printf '\033[1m\033[0;36m╚════════════════════════════════════════════════════════════╝\033[0m\n'
  printf '\n'

  # Early detection — changes messaging and step count
  detect_existing_omlx

  if [ "$EXISTING_OMLX" = "true" ]; then
    printf '  \033[0;32mFound an existing oMLX installation on this Mac.\033[0m\n'
    printf '  The installer will add the PrivateNet router alongside it\n'
    printf '  without changing your oMLX configuration or services.\n'
    printf '\n'
    printf '  The installer will:\n'
    printf '  1. Check that your Mac is compatible (Apple Silicon required)\n'
    printf '  2. Ensure developer tools are installed (Homebrew, Python, Git)\n'
    printf '  3. Set up Tailscale and try to advertise %s\n' "$TAILSCALE_TAG"
    printf '  4. Download the PrivateNet router code\n'
    printf '  5. Install router dependencies in an isolated Python environment\n'
    printf '  6. Configure the router to work with your existing oMLX\n'
    printf '  7. Start the router as a background service\n'
  else
    printf '  This script will turn your Mac into a full PrivateNet peer.\n'
    printf '  Every peer runs two things:\n'
    printf '\n'
    printf '  - oMLX on 127.0.0.1:5741  (local inference server)\n'
    printf '  - Router on 0.0.0.0:8741 (OpenAI-compatible API + peer discovery)\n'
    printf '\n'
    printf '  The installer will:\n'
    printf '  1. Check that your Mac is compatible (Apple Silicon required)\n'
    printf '  2. Install developer tools (Homebrew, Python, Git)\n'
    printf '  3. Set up Tailscale and try to advertise %s\n' "$TAILSCALE_TAG"
    printf '  4. Download oMLX and the PrivateNet router code\n'
    printf '  5. Install Python libraries for both services\n'
    printf '  6. Download two AI models (~33 GB total — this takes a while!)\n'
    printf '  7. Configure everything and start both background services\n'
  fi
  printf '\n'
  printf '  \033[2mSafe to re-run — it will skip anything already installed.\033[0m\n'
  printf '\n'

  # Check for OpenClaw and ask about plugin before computing steps
  detect_openclaw
  ask_openclaw_plugin

  compute_steps

  # ── Step 1: Check Mac ──────────────────────────────────────────────────
  step "Checking your Mac" "Making sure this is an Apple Silicon Mac (M1/M2/M3/M4)..."
  require_supported_host
  success "Apple Silicon Mac confirmed — you're good to go!"

  # ── Step 2: Developer tools ────────────────────────────────────────────
  step "Installing developer tools" "These are standard Mac tools used by millions of developers."
  ensure_homebrew

  # ── Step 3: Required software ──────────────────────────────────────────
  step "Installing required software" "Python (runs the AI), Git (downloads code), Tailscale (private network)."
  ensure_dependencies

  # ── Step 4: Tailscale ──────────────────────────────────────────────────
  step "Connecting to Tailscale" "Tailscale creates a secure private network between all the Macs."
  ensure_tailscale_ip

  # ── Step 5: Tag ────────────────────────────────────────────────────────
  step "Advertising the PrivateNet tag" "Peers find each other automatically by the Tailscale tag $TAILSCALE_TAG."
  ensure_tailscale_tag

  # ── Existing oMLX: read its API key now ────────────────────────────────
  if [ "$EXISTING_OMLX" = "true" ]; then
    read_existing_api_key
    if [ -z "$OMLX_API_KEY" ]; then
      die "Found oMLX settings.json but couldn't read the API key from it. Check $OMLX_BASE/settings.json"
    fi
  fi

  # ── Steps 6+: Source code ──────────────────────────────────────────────
  if [ "$EXISTING_OMLX" = "true" ]; then
    step "Downloading the router" "Fetching the PrivateNet router code."
  else
    step "Downloading the software" "Fetching both oMLX and the PrivateNet router code."
  fi
  ensure_repos

  # ── Python environment ─────────────────────────────────────────────────
  if [ "$EXISTING_OMLX" = "true" ]; then
    step "Setting up the Python environment" "Installing the router dependencies in an isolated environment."
  else
    step "Setting up the Python environment" "Installing the libraries needed for both oMLX and the router."
  fi
  ensure_venv

  # ── Models (fresh install only) ────────────────────────────────────────
  if [ "$EXISTING_OMLX" = "false" ]; then
    step "Downloading AI models" "These are the actual AI brains — two versions of Google's Gemma 4."
    info "This is the longest step. Total download: ~33 GB."
    printf '\n'
    ensure_models
  fi

  # ── Configuration ──────────────────────────────────────────────────────
  step "Writing configuration" "Creating the local config files for the router."
  ensure_api_key
  ensure_node_id
  write_node_env
  success "Environment config written."
  if [ "$EXISTING_OMLX" = "false" ]; then
    write_settings_json
  else
    success "oMLX settings.json already exists — not overwriting."
  fi
  write_router_config
  success "Router config written."
  write_start_scripts
  success "Startup scripts created."

  # ── LaunchAgents ───────────────────────────────────────────────────────
  step "Installing automatic startup" "Setting up the router to start automatically."

  if [ "$EXISTING_OMLX" = "false" ]; then
    write_launchagent "$OMLX_AGENT" "$OMLX_START_SCRIPT" "$OMLX_LABEL" "omlx.stdout.log" "omlx.stderr.log"
    success "oMLX LaunchAgent written."
  else
    success "oMLX is already managed by an existing LaunchAgent — skipping."
  fi
  write_launchagent "$ROUTER_AGENT" "$ROUTER_START_SCRIPT" "$ROUTER_LABEL" "router.stdout.log" "router.stderr.log"
  success "Router LaunchAgent written."

  # ── Start services ─────────────────────────────────────────────────────
  step "Starting services" "Bringing up the router."

  if [ "$EXISTING_OMLX" = "false" ]; then
    load_launchagent "$OMLX_AGENT" "$OMLX_LABEL"
    success "oMLX is running."
  else
    success "oMLX is already running (managed externally)."
  fi
  load_launchagent "$ROUTER_AGENT" "$ROUTER_LABEL"
  success "Router is running."

  # Wait briefly for the router to start, then verify
  info "Waiting a few seconds for the router to come up..."
  sleep 3
  if curl -sf http://127.0.0.1:8741/health >/dev/null 2>&1; then
    success "Router health check passed!"
  else
    warn "Router hasn't responded yet — it may need a few more seconds."
    warn "Check the logs at: $STATE_DIR/logs/router.stderr.log"
  fi

  # ── OpenClaw plugin ───────────────────────────────────────────────────
  if [ "$INSTALL_OPENCLAW" = "true" ]; then
    step "Setting up OpenClaw" "Installing the openclaw-omlx plugin so OpenClaw can use your PrivateNet models."
    install_openclaw_plugin
  fi

  print_summary
}

main "$@"
