#!/usr/bin/env python3
"""CLI for managing an oMLX PrivateNet node.

Usage:
    privatenet status              Show node status
    privatenet disable             Take this node out of service
    privatenet enable              Bring this node back into service
    privatenet models              Show which models are advertised
    privatenet models set A B C    Only advertise these models
    privatenet models reset        Advertise all models (default)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

def _state_dir() -> Path:
    return Path(os.environ.get("OMLX_PRIVATENET_STATE_DIR", Path.home() / ".omlx-privatenet"))


def _disabled_file() -> Path:
    return _state_dir() / "disabled"


def _router_config() -> Path:
    return _state_dir() / "router.json"

GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"
RED = "\033[0;31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def _load_node_id() -> str:
    """Read the node ID from the router config, or fall back to hostname."""
    try:
        with _router_config().open() as f:
            return json.load(f).get("local_node_id", "unknown")
    except Exception:
        import socket
        return socket.gethostname()


def _check_router_health() -> dict | None:
    """Quick health check against the local router."""
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8741/health", timeout=3) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def cmd_status(_args: argparse.Namespace) -> int:
    """Show current node status."""
    node_id = _load_node_id()
    disabled = _disabled_file().exists()

    print(f"\n  {BOLD}oMLX PrivateNet Node Status{RESET}")
    print(f"  {'─' * 40}")
    print(f"  Node ID:   {node_id}")

    if disabled:
        print(f"  Status:    {RED}disabled{RESET}")
        print(f"  {DIM}Peers will not route requests to this node.{RESET}")
    else:
        print(f"  Status:    {GREEN}enabled{RESET}")

    config = _load_router_config()
    advertise = config.get("advertise_models")
    if advertise is not None:
        print(f"  Models:    {YELLOW}filtered ({len(advertise)}){RESET} — {', '.join(advertise)}")
    else:
        print(f"  Models:    all (default)")

    health = _check_router_health()
    if health is None:
        print(f"  Router:    {RED}not responding{RESET}")
    else:
        router_status = health.get("status", "unknown")
        cluster = health.get("cluster", [])
        models = health.get("models", [])
        color = GREEN if router_status == "ok" else YELLOW
        print(f"  Router:    {color}{router_status}{RESET}")
        print(f"  Cluster:   {len(cluster)} node(s), {len(models)} model(s)")

    print()
    return 0


def cmd_disable(_args: argparse.Namespace) -> int:
    """Take this node out of service."""
    node_id = _load_node_id()

    if _disabled_file().exists():
        print(f"\n  {YELLOW}Node {node_id} is already disabled.{RESET}\n")
        return 0

    _state_dir().mkdir(parents=True, exist_ok=True, mode=0o700)
    _disabled_file().write_text("Disabled by CLI\n", encoding="utf-8")
    _disabled_file().chmod(0o600)

    print(f"\n  {RED}Node {node_id} is now disabled.{RESET}")
    print(f"  {DIM}Peers will stop routing requests here within ~30 seconds.{RESET}")
    print(f"  {DIM}To re-enable: privatenet enable{RESET}\n")
    return 0


def cmd_enable(_args: argparse.Namespace) -> int:
    """Bring this node back into service."""
    node_id = _load_node_id()

    if not _disabled_file().exists():
        print(f"\n  {GREEN}Node {node_id} is already enabled.{RESET}\n")
        return 0

    _disabled_file().unlink()

    print(f"\n  {GREEN}Node {node_id} is now enabled.{RESET}")
    print(f"  {DIM}Peers will start routing requests here within ~30 seconds.{RESET}\n")
    return 0


def _load_router_config() -> dict:
    """Load router.json as a dict."""
    try:
        with _router_config().open() as f:
            return json.load(f)
    except Exception:
        return {}


def _save_router_config(config: dict) -> None:
    """Write router.json back."""
    _router_config().parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _router_config().open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def cmd_models(args: argparse.Namespace) -> int:
    """Show or configure which models this node advertises."""
    config = _load_router_config()

    # privatenet models set model1 model2 ...
    if args.models_action == "set":
        if not args.model_names:
            print(f"\n  {RED}No models specified.{RESET}")
            print(f"  {DIM}Usage: privatenet models set model-a model-b{RESET}\n")
            return 1
        config["advertise_models"] = args.model_names
        _save_router_config(config)
        print(f"\n  {GREEN}This node will now only advertise:{RESET}")
        for m in args.model_names:
            print(f"  - {m}")
        print(f"\n  {DIM}Restart the router or wait ~30s for changes to take effect.{RESET}\n")
        return 0

    # privatenet models reset
    if args.models_action == "reset":
        config.pop("advertise_models", None)
        _save_router_config(config)
        print(f"\n  {GREEN}This node will now advertise all available models (default).{RESET}\n")
        return 0

    # privatenet models (no subcommand — show current state)
    advertise = config.get("advertise_models")
    health = _check_router_health()

    print(f"\n  {BOLD}Model Configuration{RESET}")
    print(f"  {'─' * 40}")

    if advertise is not None:
        print(f"  Filter:    {YELLOW}allowlist ({len(advertise)} model(s)){RESET}")
        for m in advertise:
            print(f"    - {m}")
    else:
        print(f"  Filter:    {GREEN}all models (default){RESET}")

    if health:
        cluster = health.get("cluster", [])
        local = next((n for n in cluster if n.get("local")), None)
        if local:
            models = local.get("models", [])
            print(f"\n  {BOLD}Currently advertised:{RESET} {len(models)} model(s)")
            for m in models:
                print(f"    - {m}")
        all_models = health.get("models", [])
        if all_models:
            print(f"\n  {BOLD}Cluster total:{RESET} {len(all_models)} model(s)")

    print()
    return 0


def _registry_path() -> Path:
    return _state_dir() / "registry.json"


def cmd_registry(args: argparse.Namespace) -> int:
    """Manage the shared model registry."""
    from router.registry import Registry, RegistryModel

    registry = Registry(path=_registry_path())
    registry.load()

    action = args.registry_action

    if action == "list":
        models = registry.models
        if not models:
            print(f"\n  {DIM}No models in the registry.{RESET}\n")
            return 0
        print(f"\n  {BOLD}Registry Models{RESET}")
        print(f"  {'─' * 60}")
        print(f"  {'ID':<30} {'REPO':<30} {'PRI':>3}  {'ADDED BY'}")
        print(f"  {'─' * 60}")
        for m in sorted(models, key=lambda x: x.priority, reverse=True):
            print(f"  {m.id:<30} {m.repo:<30} {m.priority:>3}  {m.added_by}")
        print()
        return 0

    if action == "add":
        repo = args.repo
        parts = repo.split("/", 1)
        if len(parts) != 2 or not parts[1]:
            print(f"\n  {RED}Invalid repo format. Expected: org/model-name{RESET}\n")
            return 1
        model_id = parts[1]
        node_id = _load_node_id()
        model = RegistryModel(
            repo=repo,
            id=model_id,
            priority=args.priority,
            added_by=node_id,
            added_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            registry.add(model)
        except ValueError as exc:
            print(f"\n  {RED}{exc}{RESET}\n")
            return 1
        registry.save()
        print(f"\n  {GREEN}Added {model_id} (from {repo}) to the registry.{RESET}\n")
        return 0

    if action == "remove":
        model_id = args.repo  # positional arg reused as model_id
        if registry.remove(model_id):
            registry.save()
            print(f"\n  {GREEN}Removed {model_id} from the registry.{RESET}\n")
            return 0
        else:
            print(f"\n  {YELLOW}Model {model_id} not found in the registry.{RESET}\n")
            return 1

    # No action — show help-like output
    print(f"\n  {BOLD}Usage:{RESET}")
    print(f"    privatenet registry list")
    print(f"    privatenet registry add <org/model-name> [--priority N]")
    print(f"    privatenet registry remove <model-id>\n")
    return 1


def cmd_config(args: argparse.Namespace) -> int:
    """Get or set config values in router.json."""
    action = args.config_action

    if action == "get":
        config = _load_router_config()
        key = args.key
        if key in config:
            print(config[key])
        else:
            print(f"\n  {YELLOW}Key '{key}' not found in router.json.{RESET}\n")
            return 1
        return 0

    if action == "set":
        config = _load_router_config()
        key = args.key
        value = args.value
        # Try to parse as JSON for non-string types
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            parsed = value
        config[key] = parsed
        _save_router_config(config)
        print(f"\n  {GREEN}Set {key} = {json.dumps(parsed)}{RESET}\n")
        return 0

    print(f"\n  {BOLD}Usage:{RESET}")
    print(f"    privatenet config get <key>")
    print(f"    privatenet config set <key> <value>\n")
    return 1


def cmd_update(args: argparse.Namespace) -> int:
    """Check for or apply updates."""
    from router.updater import check_for_update, drain_and_run, run_update

    info = check_for_update()

    if args.check:
        print(f"\n  {BOLD}Update Check{RESET}")
        print(f"  {'─' * 40}")
        print(f"  Local version:   {info.local_version} ({info.local_sha})")
        print(f"  Remote version:  {info.remote_version} ({info.remote_sha})")
        if info.available:
            print(f"  Status:          {YELLOW}update available{RESET}")
        else:
            print(f"  Status:          {GREEN}up to date{RESET}")
        print()
        return 0

    if not info.available:
        print(f"\n  {GREEN}Already up to date ({info.local_version}).{RESET}\n")
        return 0

    print(f"\n  Updating {info.local_version} -> {info.remote_version} ...")
    print(f"  {DIM}Draining in-flight requests...{RESET}")
    result = drain_and_run(run_update)
    if result.success:
        print(f"  {GREEN}Updated successfully.{RESET}")
        print(f"  {DIM}{result.previous_sha} -> {result.new_sha}{RESET}\n")
        return 0
    else:
        print(f"  {RED}Update failed: {result.error}{RESET}\n")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="privatenet",
        description="Manage your oMLX PrivateNet node.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show whether this node is enabled or disabled")
    sub.add_parser("disable", help="Take this node out of service")
    sub.add_parser("enable", help="Bring this node back into service")

    models_parser = sub.add_parser("models", help="Show or configure which models to advertise")
    models_parser.add_argument("models_action", nargs="?", choices=["set", "reset"], default=None)
    models_parser.add_argument("model_names", nargs="*", default=[])

    # registry subcommand
    registry_parser = sub.add_parser("registry", help="Manage the shared model registry")
    registry_parser.add_argument("registry_action", nargs="?", choices=["list", "add", "remove"], default=None)
    registry_parser.add_argument("repo", nargs="?", default=None, help="Repo (for add) or model ID (for remove)")
    registry_parser.add_argument("--priority", type=int, default=5, help="Model priority (default: 5)")

    # config subcommand
    config_parser = sub.add_parser("config", help="Get or set router configuration")
    config_parser.add_argument("config_action", nargs="?", choices=["get", "set"], default=None)
    config_parser.add_argument("key", nargs="?", default=None, help="Config key")
    config_parser.add_argument("value", nargs="?", default=None, help="Config value (for set)")

    # update subcommand
    update_parser = sub.add_parser("update", help="Check for or apply updates")
    update_parser.add_argument("--check", action="store_true", help="Only check for updates, don't apply")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    commands = {
        "status": cmd_status,
        "disable": cmd_disable,
        "enable": cmd_enable,
        "models": cmd_models,
        "registry": cmd_registry,
        "config": cmd_config,
        "update": cmd_update,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
