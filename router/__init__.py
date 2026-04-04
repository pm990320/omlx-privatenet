"""oMLX PrivateNet router package."""

from .config import RouterConfig, load_config, resolve_config_path
from .discovery import DiscoveredPeer, TailscaleDiscovery
from .health import NodeHealthMonitor
from .router import ConsistentHashRouter, NodeInfo, RouteDecision
from .server import create_app

__all__ = [
    "__version__",
    "ConsistentHashRouter",
    "DiscoveredPeer",
    "NodeHealthMonitor",
    "NodeInfo",
    "RouteDecision",
    "RouterConfig",
    "TailscaleDiscovery",
    "create_app",
    "load_config",
    "resolve_config_path",
]

__version__ = "0.2.0"
