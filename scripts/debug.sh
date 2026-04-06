#!/usr/bin/env bash
# oMLX PrivateNet — Diagnostic Report
# Run: curl -fsSL https://raw.githubusercontent.com/pm990320/omlx-privatenet/main/scripts/debug.sh | bash

set +e  # don't exit on errors — we want to collect everything

BOLD='\033[1m'
DIM='\033[2m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
RESET='\033[0m'

section() { printf '\n%b══ %s ══%b\n' "$BOLD" "$1" "$RESET"; }
ok()      { printf '  %b✓ %s%b\n' "$GREEN" "$*" "$RESET"; }
bad()     { printf '  %b✗ %s%b\n' "$RED" "$*" "$RESET"; }
warn()    { printf '  %b⚠ %s%b\n' "$YELLOW" "$*" "$RESET"; }
info()    { printf '  %b%s%b\n' "$DIM" "$*" "$RESET"; }
kv()      { printf '  %-24s %s\n' "$1:" "$2"; }

# ── System ───────────────────────────────────────────────────────────────────
section "System"
kv "OS" "$(uname -s) $(uname -r)"
kv "Architecture" "$(uname -m)"
kv "Hostname" "$(scutil --get ComputerName 2>/dev/null || hostname)"
kv "macOS version" "$(sw_vers -productVersion 2>/dev/null || echo unknown)"

# ── Homebrew ─────────────────────────────────────────────────────────────────
section "Homebrew"
if command -v brew >/dev/null 2>&1; then
  ok "Installed at $(command -v brew)"
  kv "Python 3.13" "$(brew --prefix python@3.13 2>/dev/null)/bin/python3.13 $([ -x "$(brew --prefix python@3.13 2>/dev/null)/bin/python3.13" ] && echo '(exists)' || echo '(missing)')"
  kv "Git" "$(command -v git 2>/dev/null || echo missing)"
elif [ -x /opt/homebrew/bin/brew ]; then
  ok "Installed at /opt/homebrew/bin/brew (not on PATH)"
else
  bad "Not installed"
fi

# ── Tailscale ────────────────────────────────────────────────────────────────
section "Tailscale"
TS_BIN=""
for candidate in "$(command -v tailscale 2>/dev/null)" /usr/local/bin/tailscale /Applications/Tailscale.app/Contents/MacOS/Tailscale; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then
    TS_BIN="$candidate"
    break
  fi
done

if [ -n "$TS_BIN" ]; then
  ok "Binary at $TS_BIN"
  kv "Version" "$($TS_BIN version 2>/dev/null | head -1)"

  TS_IP="$($TS_BIN ip -4 2>/dev/null | head -1)"
  if [ -n "$TS_IP" ]; then
    ok "Connected — IP: $TS_IP"
  else
    bad "Not connected (no IPv4 address)"
  fi

  SELF_TAGS="$($TS_BIN status --json 2>/dev/null | python3 -c 'import json,sys; print(",".join(json.load(sys.stdin).get("Self",{}).get("Tags",[])))' 2>/dev/null)"
  if echo "$SELF_TAGS" | grep -qF "tag:omlx-node"; then
    ok "Tag: tag:omlx-node is advertised"
  else
    bad "Tag: tag:omlx-node NOT advertised (current tags: ${SELF_TAGS:-none})"
  fi

  info "Peers with tag:omlx-node:"
  $TS_BIN status --json 2>/dev/null > /tmp/omlx-debug-ts.json 2>/dev/null
  python3 << 'PYEOF'
import json
try:
    with open("/tmp/omlx-debug-ts.json") as f:
        d = json.load(f)
    found = False
    for pid, p in d.get("Peer", {}).items():
        tags = p.get("Tags") or []
        if "tag:omlx-node" in tags:
            found = True
            ips = p.get("TailscaleIPs") or []
            ip = ips[0] if ips else "?"
            online = "online" if p.get("Online") else "offline"
            name = p.get("HostName", "?")
            print(f"    {name} ({ip}) -- {online}")
    if not found:
        print("    (none found)")
except Exception as e:
    print(f"    (could not query: {e})")
PYEOF
else
  bad "Not installed"
fi

# ── oMLX ─────────────────────────────────────────────────────────────────────
section "oMLX (local inference server)"
OMLX_HEALTH="$(curl -sf --connect-timeout 3 http://127.0.0.1:5741/health 2>/dev/null)"
if [ -n "$OMLX_HEALTH" ]; then
  ok "Running on 127.0.0.1:5741"
  kv "Status" "$(echo "$OMLX_HEALTH" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status","?"))' 2>/dev/null)"
  OMLX_MODELS="$(curl -sf --connect-timeout 3 -H "Authorization: Bearer $(python3 -c 'import json; print(json.load(open("/Users/'$USER'/.omlx/settings.json")).get("auth",{}).get("api_key",""))' 2>/dev/null)" http://127.0.0.1:5741/v1/models 2>/dev/null)"
  if [ -n "$OMLX_MODELS" ]; then
    info "Models:"
    echo "$OMLX_MODELS" | python3 -c '
import json, sys
d = json.load(sys.stdin)
items = d.get("data", d.get("models", []))
for m in items:
    name = m.get("id","?") if isinstance(m,dict) else str(m)
    print(f"    {name}")
if not items:
    print("    (none)")
' 2>/dev/null
  fi
else
  warn "Not running on 127.0.0.1:5741"
fi

if [ -f "$HOME/.omlx/settings.json" ]; then
  ok "Settings at ~/.omlx/settings.json"
  kv "API key" "$(python3 -c 'import json; k=json.load(open("'$HOME'/.omlx/settings.json")).get("auth",{}).get("api_key",""); print(k[:4]+"..." if len(k)>4 else k or "(not set)")' 2>/dev/null)"
else
  info "No ~/.omlx/settings.json"
fi

# Check LaunchAgent
OMLX_PLIST=""
for plist in "$HOME"/Library/LaunchAgents/*omlx*.plist; do
  if [ -f "$plist" ]; then
    OMLX_PLIST="$plist"
    ok "LaunchAgent: $plist"
  fi
done
[ -z "$OMLX_PLIST" ] && info "No oMLX LaunchAgent found"

# ── PrivateNet State ─────────────────────────────────────────────────────────
section "PrivateNet state directory"
STATE="$HOME/.omlx-privatenet"
if [ -d "$STATE" ]; then
  ok "Exists at $STATE"
  [ -d "$STATE/venv" ] && ok "Python venv exists" || bad "Python venv missing"
  [ -f "$STATE/router.json" ] && ok "Router config exists" || bad "Router config missing"
  [ -f "$STATE/start-router.sh" ] && ok "Start script exists" || bad "Start script missing"
  [ -f "$STATE/disabled" ] && warn "Node is DISABLED (disabled file present)" || ok "Node is enabled"

  if [ -f "$STATE/start-router.sh" ]; then
    if grep -q '/opt/homebrew/bin\|/usr/local/bin' "$STATE/start-router.sh"; then
      ok "Start script has PATH for tailscale"
    else
      bad "Start script MISSING PATH — tailscale won't be found by LaunchAgent"
      info "Fix: re-run the installer"
    fi
  fi

  if [ -f "$STATE/router.json" ]; then
    info "Router config:"
    python3 << PYEOF
import json
with open("$STATE/router.json") as f:
    d = json.load(f)
print("    node_id:         ", d.get("local_node_id","?"))
print("    tailscale_ip:    ", d.get("local_tailscale_ip","?"))
print("    tailscale_tag:   ", d.get("tailscale_tag","?"))
print("    local_models:    ", len(d.get("local_models",[])), "model(s)")
print("    local_omlx_url:  ", d.get("local_omlx_url","?"))
k = d.get("local_omlx_api_key","")
print("    local_omlx_key:  ", (k[:4]+"...") if len(k)>4 else k or "(not set)")
print("    router_api_key:  ", "set" if d.get("api_key") else "not set")
PYEOF
  fi
else
  bad "State directory missing at $STATE"
fi

# ── Router ───────────────────────────────────────────────────────────────────
section "PrivateNet Router"
ROUTER_PLIST="$HOME/Library/LaunchAgents/com.omlx-privatenet.router.plist"
if [ -f "$ROUTER_PLIST" ]; then
  ok "LaunchAgent exists"
else
  bad "LaunchAgent missing at $ROUTER_PLIST"
fi

ROUTER_HEALTH="$(curl -sf --connect-timeout 3 http://127.0.0.1:8741/health 2>/dev/null)"
if [ -n "$ROUTER_HEALTH" ]; then
  ok "Running on 127.0.0.1:8741"
  echo "$ROUTER_HEALTH" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(f"    Status:     {d.get(\"status\",\"?\")}")
print(f"    Node ID:    {d.get(\"router\",{}).get(\"node_id\",\"?\")}")
print(f"    Cluster:    {len(d.get(\"cluster\",[]))} node(s)")
for n in d.get("cluster", []):
    local = " (local)" if n.get("local") else ""
    healthy = "healthy" if n.get("healthy") else "UNHEALTHY"
    err = n.get("last_error") or ""
    err_str = f" — {err}" if err else ""
    print(f"      {n.get(\"node_id\",\"?\")} ({n.get(\"tailscale_ip\",\"?\")}) {healthy}, {len(n.get(\"models\",[]))} models{local}{err_str}")
print(f"    Models:     {len(d.get(\"models\",[]))} total")
for m in d.get("models", []):
    print(f"      {m}")
' 2>/dev/null
else
  bad "Not running on 127.0.0.1:8741"

  if [ -f "$STATE/logs/router.stderr.log" ]; then
    info "Last 10 lines of router.stderr.log:"
    tail -10 "$STATE/logs/router.stderr.log" 2>/dev/null | while IFS= read -r line; do
      printf '    %s\n' "$line"
    done
  fi
fi

# ── Peer connectivity ────────────────────────────────────────────────────────
if [ -n "$TS_BIN" ] && [ -n "$ROUTER_HEALTH" ]; then
  section "Peer connectivity"
  $TS_BIN status --json 2>/dev/null | python3 -c '
import json, sys, urllib.request
d = json.load(sys.stdin)
self_ip = ""
for ip in d.get("Self",{}).get("TailscaleIPs",[]):
    if ":" not in ip:
        self_ip = ip
        break
found = False
for pid, p in d.get("Peer", {}).items():
    tags = p.get("Tags") or []
    if "tag:omlx-node" not in tags:
        continue
    found = True
    ips = p.get("TailscaleIPs") or []
    ip = next((i for i in ips if ":" not in i), None)
    if not ip:
        continue
    name = p.get("HostName", "?")
    try:
        req = urllib.request.Request(f"http://{ip}:8741/v1/node-info", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            info = json.loads(resp.read())
        models = info.get("models", [])
        healthy = info.get("healthy", False)
        status = "healthy" if healthy else "UNHEALTHY"
        print(f"  ✓ {name} ({ip}) — {status}, {len(models)} models")
    except Exception as e:
        print(f"  ✗ {name} ({ip}) — unreachable: {e}")
if not found:
    print("  (no tagged peers found)")
' 2>/dev/null
fi

# ── OpenClaw ─────────────────────────────────────────────────────────────────
section "OpenClaw"
OC_BIN=""
for candidate in "$(command -v openclaw 2>/dev/null)" "$HOME/.local/bin/openclaw" /usr/local/bin/openclaw; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then
    OC_BIN="$candidate"
    break
  fi
done

if [ -n "$OC_BIN" ]; then
  ok "Installed at $OC_BIN"
  OC_CONFIG="$HOME/.openclaw/openclaw.json"
  if [ -f "$OC_CONFIG" ]; then
    python3 -c '
import json
with open("'$OC_CONFIG'") as f:
    d = json.load(f)
omlx = d.get("plugins",{}).get("entries",{}).get("omlx",{})
if omlx:
    enabled = omlx.get("enabled", False)
    cfg = omlx.get("config", {})
    url = cfg.get("baseUrl", "(not set)")
    key = cfg.get("apiKey", "")
    key_display = key[:4]+"..." if len(key)>4 else key or "(not set)"
    print(f"  omlx plugin: {\"enabled\" if enabled else \"DISABLED\"}")
    print(f"    baseUrl: {url}")
    print(f"    apiKey:  {key_display}")
else:
    print("  omlx plugin: not configured")
' 2>/dev/null
  else
    info "No openclaw.json found"
  fi
else
  info "OpenClaw not installed"
fi

# ── privatenet CLI ───────────────────────────────────────────────────────────
section "privatenet CLI"
if command -v privatenet >/dev/null 2>&1; then
  ok "Available at $(command -v privatenet)"
else
  PN_BIN="$HOME/.omlx-privatenet/bin/privatenet"
  if [ -x "$PN_BIN" ]; then
    warn "Exists at $PN_BIN but not on PATH"
  else
    bad "Not installed"
  fi
fi

printf '\n%b── End of diagnostic report ──%b\n\n' "$DIM" "$RESET"
