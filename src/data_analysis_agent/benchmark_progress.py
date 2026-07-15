"""Terminal-safe presentation for live benchmark progress events."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TextIO, TypeAlias, TypedDict


class ProgressEvent(TypedDict, total=False):
    """Presentation facts emitted by benchmark orchestration and agent runs."""

    type: str
    task_id: str
    approach: str
    model: str
    repeat_index: int
    repeats: int
    role: str
    elapsed: float
    error: str
    goals: list[dict[str, object]]
    completed_goal_ids: list[str]
    current_goal_id: str | None
    scientific_replan_count: int
    goal_id: str
    objective: str
    attempt: int
    maximum: int
    message: str


ProgressCallback: TypeAlias = Callable[[ProgressEvent], None]
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CONTROLS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def sanitize_terminal_text(value: object, *, limit: int = 240) -> str:
    """Remove terminal controls from model-derived display text."""
    text = _CONTROLS.sub(" ", _ANSI.sub("", str(value))).replace("\r", " ")
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class BenchmarkProgressRenderer:
    """Render structured progress dynamically for TTYs and append-only otherwise."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        interactive: bool | None = None,
        artifact_path: Path | None = None,
    ) -> None:
        self.stream = stream or sys.stdout
        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.interactive = (
            is_tty and not os.getenv("CI") if interactive is None else interactive
        )
        self.artifact_path = artifact_path
        self.header: list[str] = []
        self.goals: list[dict[str, str]] = []
        self.completed: set[str] = set()
        self.current_goal: str | None = None
        self.scientific_replan_count = 0
        self.details: list[str] = []
        self.events: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        """Record and render one event without interpreting workflow semantics."""
        safe_event = {
            key: sanitize_terminal_text(value)
            if key
            in {"error", "message", "objective", "goal_id", "current_goal_id"}
            and value is not None
            else value
            for key, value in event.items()
        }
        self.events.append(safe_event)
        if self.artifact_path is not None:
            self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
            with self.artifact_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(safe_event, ensure_ascii=False) + "\n")
        self._apply(safe_event)
        if self.interactive:
            self._redraw()
        else:
            self._append(safe_event)

    def _apply(self, event: ProgressEvent) -> None:
        kind = event["type"]
        if kind == "benchmark_started":
            self.header = [
                "=" * 60,
                f"BENCHMARK RUN {event['repeat_index']}/{event['repeats']}",
                f"Task       : {event['task_id']}",
                f"Approach   : {event['approach']}",
                f"Model      : {event['model']}",
                "=" * 60,
            ]
        elif kind == "plan_available":
            self.goals = [
                {
                    "goal_id": sanitize_terminal_text(goal["goal_id"], limit=64),
                    "objective": sanitize_terminal_text(goal["objective"], limit=100),
                }
                for goal in event.get("goals", [])
            ]
            self.completed = {
                str(goal_id) for goal_id in event.get("completed_goal_ids", [])
            }
            current_goal_id = event.get("current_goal_id")
            self.current_goal = (
                str(current_goal_id) if current_goal_id is not None else None
            )
            self.scientific_replan_count = int(
                event.get("scientific_replan_count", 0)
            )
        elif kind == "goal_started":
            self.current_goal = str(event["goal_id"])
            self.details = []
        elif kind == "goal_completed":
            self.completed.add(str(event["goal_id"]))
        elif kind in {"activity", "error", "workflow_failed"}:
            message = event.get("message") or event.get("error")
            if message:
                self.details.append(str(message))

    def _progress_lines(self) -> list[str]:
        if not self.goals:
            return []
        visible_goal_ids = {goal["goal_id"] for goal in self.goals}
        visible_completed = self.completed & visible_goal_ids
        lines = [f"Planner proposed {len(self.goals)} steps"]
        for goal in self.goals:
            marker = (
                "✓"
                if goal["goal_id"] in visible_completed
                else "→"
                if goal["goal_id"] == self.current_goal
                else " "
            )
            lines.append(f"{marker} {goal['goal_id']} — {goal['objective']}")
        lines.append(f"Progress: [{len(visible_completed)}/{len(self.goals)}]")
        return lines

    def _current_title(self) -> str | None:
        if self.current_goal is None:
            return None
        goal = next(
            (item for item in self.goals if item["goal_id"] == self.current_goal), None
        )
        if goal is None:
            return None
        return f"Current step: {goal['goal_id']} — {goal['objective']}"

    def _redraw(self) -> None:
        lines = [*self.header]
        lines.extend(self._progress_lines())
        title = self._current_title()
        if title:
            lines.extend(["", title, "-" * 60, *self.details[-8:]])
        self.stream.write("\x1b[2J\x1b[H" + "\n".join(lines) + "\n")
        self.stream.flush()

    def _append(self, event: ProgressEvent) -> None:
        kind = event["type"]
        if kind == "benchmark_started":
            lines = [*self.header, ""]
        elif kind == "workflow_started":
            lines = ["Agent workflow started"]
        elif kind == "plan_available":
            lines = self._progress_lines()
        elif kind == "goal_started":
            title = self._current_title()
            lines = [*self._progress_lines(), "", title, "-" * 60] if title else []
        elif kind == "goal_completed":
            goal_id = str(event["goal_id"])
            goal = next(
                (item for item in self.goals if item["goal_id"] == goal_id), None
            )
            lines = (
                [f"✓ {goal_id} — {goal['objective']}", *self._progress_lines()[-1:]]
                if goal
                else []
            )
        elif kind in {"activity", "error", "workflow_failed"}:
            message = event.get("message") or event.get("error")
            lines = [str(message)] if message else []
        else:
            lines = []
        if lines:
            self.stream.write("\n".join(lines) + "\n")
            self.stream.flush()
