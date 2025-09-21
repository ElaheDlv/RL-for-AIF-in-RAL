"""Shared helpers for configuring CARLA route start/goal pairs."""

from __future__ import annotations

import json
from typing import Iterable, List, Dict


def parse_route_entries(entries: Iterable[str]) -> List[Dict[str, int]]:
    routes: List[Dict[str, int]] = []
    for item in entries:
        if ":" not in item:
            raise ValueError(f"Route '{item}' must be formatted as start_idx:goal_idx")
        start_str, goal_str = item.split(":", 1)
        try:
            start_idx = int(start_str)
            goal_idx = int(goal_str)
        except ValueError as exc:
            raise ValueError(f"Route '{item}' contains non-integer indices") from exc
        routes.append({"start": start_idx, "goal": goal_idx})
    return routes


def load_routes_from_json(path: str) -> List[Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("routes", [])
    routes: List[Dict[str, int]] = []
    for entry in data:
        if isinstance(entry, dict):
            start_idx = entry.get("start")
            goal_idx = entry.get("goal")
        else:
            start_idx, goal_idx = entry
        if start_idx is None or goal_idx is None:
            raise ValueError(f"Invalid route entry: {entry}")
        routes.append({"start": int(start_idx), "goal": int(goal_idx)})
    return routes


def combine_route_sources(route_args, routes_json) -> List[Dict[str, int]]:
    routes: List[Dict[str, int]] = []
    if routes_json:
        routes.extend(load_routes_from_json(routes_json))
    if route_args:
        routes.extend(parse_route_entries(route_args))
    if not routes:
        raise ValueError("No routes specified. Use --routes or --routes-json.")
    return routes


def validate_route_indices(routes: List[Dict[str, int]], total_spawn_points: int) -> List[Dict[str, int]]:
    for route in routes:
        start = route["start"]
        goal = route["goal"]
        if not (0 <= start < total_spawn_points):
            raise ValueError(f"Route start index {start} outside available spawn points (0-{total_spawn_points-1})")
        if not (0 <= goal < total_spawn_points):
            raise ValueError(f"Route goal index {goal} outside available spawn points (0-{total_spawn_points-1})")
    return routes
