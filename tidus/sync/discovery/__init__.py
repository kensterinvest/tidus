"""Vendor model discovery — scans `/v1/models` endpoints to surface
new vendor models for human review.

This package is intentionally separate from `tidus.sync.pricing`:
discovery is a SURFACE-only flow (output: report + JSON sidecar),
never auto-routes traffic to unverified models. Promotion of a
discovered model to the active routing catalog still requires a
human edit to `config/models.yaml` + `tidus/sync/pricing/hardcoded_source.py`.
"""

from tidus.sync.discovery.base import DiscoveredModel, DiscoverySource
from tidus.sync.discovery.factory import build_discovery_sources
from tidus.sync.discovery.runner import DiscoveryReport, DiscoveryRunner

__all__ = [
    "DiscoveredModel",
    "DiscoverySource",
    "DiscoveryReport",
    "DiscoveryRunner",
    "build_discovery_sources",
]
