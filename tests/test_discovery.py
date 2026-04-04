from __future__ import annotations

import json
import subprocess

from router.config import RouterConfig
from router.discovery import TailscaleDiscovery


def test_discovery_parses_tailscale_status_output(monkeypatch, tailscale_status_payload):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=json.dumps(tailscale_status_payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    discovery = TailscaleDiscovery(RouterConfig(local_node_id="local-node", local_tailscale_ip="100.64.0.1"))

    peers = discovery.discover()

    assert [peer.node_id for peer in peers] == ["local-node", "peer-a"]
    assert peers[0].local is True
    assert peers[1].tailscale_ip == "100.64.0.2"


def test_discovery_filters_by_omlx_tag(monkeypatch, tailscale_status_payload):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=json.dumps(tailscale_status_payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    discovery = TailscaleDiscovery(RouterConfig(local_node_id="local-node", local_tailscale_ip="100.64.0.1"))

    peers = discovery.discover()

    assert {peer.node_id for peer in peers} == {"local-node", "peer-a"}
    assert all(peer.local or peer.host_name.startswith("peer-a") for peer in peers)


def test_discovery_includes_self_node(monkeypatch, tailscale_status_payload):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=json.dumps(tailscale_status_payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    discovery = TailscaleDiscovery(RouterConfig(local_node_id="my-router", local_tailscale_ip="100.64.0.1"))

    peers = discovery.discover()

    assert peers[0].local is True
    assert peers[0].node_id == "my-router"
    assert peers[0].router_url == "http://100.64.0.1:8741"


def test_discovery_handles_missing_or_malformed_output_gracefully(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="{not json}", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    discovery = TailscaleDiscovery(RouterConfig(local_node_id="local-node", local_tailscale_ip="100.64.0.10"))

    peers = discovery.discover()

    assert len(peers) == 1
    assert peers[0].local is True
    assert peers[0].tailscale_ip == "100.64.0.10"


def test_discovery_skips_tagged_peers_without_ipv4(monkeypatch, tailscale_status_payload):
    tailscale_status_payload["Peer"]["peer-3"] = {
        "HostName": "peer-c.tailnet.ts.net",
        "TailscaleIPs": ["fd7a:115c:a1e0::9"],
        "Tags": ["tag:omlx-node"],
        "Online": True,
    }

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=json.dumps(tailscale_status_payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    discovery = TailscaleDiscovery(RouterConfig(local_node_id="local-node", local_tailscale_ip="100.64.0.1"))

    peers = discovery.discover()

    assert {peer.node_id for peer in peers} == {"local-node", "peer-a"}


def test_discovery_rejects_non_tailscale_ips(monkeypatch, tailscale_status_payload):
    """IPs outside the Tailscale CGNAT range (100.64.0.0/10) must be rejected to prevent SSRF."""
    tailscale_status_payload["Peer"]["rogue"] = {
        "HostName": "rogue-node.tailnet.ts.net",
        "TailscaleIPs": ["192.168.1.100"],
        "Tags": ["tag:omlx-node"],
        "Online": True,
    }
    tailscale_status_payload["Peer"]["sneaky"] = {
        "HostName": "sneaky-node.tailnet.ts.net",
        "TailscaleIPs": ["127.0.0.1"],
        "Tags": ["tag:omlx-node"],
        "Online": True,
    }

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=json.dumps(tailscale_status_payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    discovery = TailscaleDiscovery(RouterConfig(local_node_id="local-node", local_tailscale_ip="100.64.0.1"))

    peers = discovery.discover()

    node_ids = {peer.node_id for peer in peers}
    assert "rogue-node" not in node_ids
    assert "sneaky-node" not in node_ids
    assert node_ids == {"local-node", "peer-a"}
