"""Helpers for standardized EpisodeInfo.extra payloads."""

from __future__ import annotations

from dataclasses import asdict

from robodeploy.core.types import MultiAgentInfo


def build_multi_agent_extra(payload: MultiAgentInfo) -> MultiAgentInfo:
    return payload


def build_assets_extra(asset_selections: dict) -> dict:
    try:
        return {rid: asdict(sel) for rid, sel in asset_selections.items()}
    except Exception:
        return {}


def build_viz_extra(tasks: dict[str, list[dict]]) -> dict:
    return {"tasks": tasks}


def build_diagnostics_extra(payload: dict | None) -> dict:
    return dict(payload or {})
