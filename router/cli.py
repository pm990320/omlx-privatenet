#!/usr/bin/env python3
"""CLI for managing an oMLX PrivateNet node.

Usage:
    privatenet status    Show whether this node is enabled or disabled
    privatenet disable   Take this node out of service (peers stop routing here)
    privatenet enable    Bring this node back into service
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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

    health = _check_router_health()
    if health is None:
        print(f"  Router:    {RED}not responding{RESET}")
    else:
        router_status = health.get("status", "unknown")
        cluster = health.get("cluster", [])
        models = health.get("models", [])
        color = GREEN if router_status == "ok" else YELLOW
        print(f"  Router:    {color}{router_status}{RESET}")
        print(f"  Models:    {len(models)}")
        print(f"  Cluster:   {len(cluster)} node(s)")

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="privatenet",
        description="Manage your oMLX PrivateNet node.",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show whether this node is enabled or disabled")
    sub.add_parser("disable", help="Take this node out of service")
    sub.add_parser("enable", help="Bring this node back into service")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    commands = {
        "status": cmd_status,
        "disable": cmd_disable,
        "enable": cmd_enable,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
