from __future__ import annotations

"""Peer discovery via ``tailscale status --json``."""

import json
import subprocess
from dataclasses import dataclass
from typing import Any

from .config import RouterConfig


@dataclass(slots=True)
class DiscoveredPeer:
    """A peer advertised on the PrivateNet tailnet."""

    node_id: str
    tailscale_ip: str
    router_url: str
    host_name: str | None = None
    online: bool = True
    local: bool = False


class TailscaleDiscovery:
    """Discover router peers from local Tailscale status output."""

    def __init__(self, config: RouterConfig) -> None:
        self.config = config

    def discover(self) -> list[DiscoveredPeer]:
        """Return the current local node plus any tagged peers.

        Discovery is intentionally forgiving: if Tailscale output is unavailable
        or malformed, the router falls back to a best-effort local-only view.
        """
        try:
            payload = self._status_json()
        except RuntimeError:
            payload = {}

        peers: dict[str, DiscoveredPeer] = {}
        self_info = payload.get("Self") if isinstance(payload.get("Self"), dict) else {}
        self_ip = self._pick_ip(self_info.get("TailscaleIPs")) or self.config.local_tailscale_ip
        if self_ip:
            self_peer = DiscoveredPeer(
                node_id=self.config.local_node_id,
                tailscale_ip=self_ip,
                router_url=f"http://{self_ip}:{self.config.port}",
                host_name=str(self_info.get("HostName") or self.config.local_node_id),
                online=True,
                local=True,
            )
            peers[self_peer.node_id] = self_peer

        raw_peers = payload.get("Peer") if isinstance(payload, dict) else None
        if isinstance(raw_peers, list):
            iterable = raw_peers
        elif isinstance(raw_peers, dict):
            iterable = raw_peers.values()
        else:
            iterable = []

        for raw_peer in iterable:
            if not isinstance(raw_peer, dict):
                continue
            tags = raw_peer.get("Tags") or []
            if isinstance(tags, str):
                tags = [tags]
            if not isinstance(tags, list) or self.config.tailscale_tag not in tags:
                continue
            peer_ip = self._pick_ipv4(raw_peer.get("TailscaleIPs"))
            if not peer_ip or peer_ip == self_ip:
                continue
            host_name = str(raw_peer.get("HostName") or raw_peer.get("DNSName") or peer_ip)
            node_id = host_name.split(".", 1)[0]
            peers[node_id] = DiscoveredPeer(
                node_id=node_id,
                tailscale_ip=peer_ip,
                router_url=f"http://{peer_ip}:{self.config.port}",
                host_name=host_name,
                online=bool(raw_peer.get("Online", False)),
                local=False,
            )

        return sorted(peers.values(), key=lambda item: (not item.local, item.node_id))

    def _status_json(self) -> dict[str, Any]:
        command = [self.config.tailscale_bin, "status", "--json"]
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
            payload = json.loads(completed.stdout)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            raise RuntimeError("Unable to read `tailscale status --json`.") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("`tailscale status --json` returned unexpected data.")
        return payload

    @staticmethod
    def _pick_ip(value: Any) -> str | None:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and ":" not in item:
                    return item
            for item in value:
                if isinstance(item, str):
                    return item
        if isinstance(value, str):
            return value
        return None

    @staticmethod
    def _pick_ipv4(value: Any) -> str | None:
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str) and ":" not in item:
                    return item
            return None
        if isinstance(value, str) and ":" not in value:
            return value
        return None
