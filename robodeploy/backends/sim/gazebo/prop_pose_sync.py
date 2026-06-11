"""Best-effort Gazebo entity pose updates for kinematic carry (follow mode)."""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Sequence

logger = logging.getLogger(__name__)


class PropPoseSyncer:
    """Sync prop poses to Gazebo via gz-transport (preferred) or ``gz service`` fallback."""

    def __init__(self, *, gz_node=None) -> None:
        self._gz_node = gz_node

    def set_entity_pose(
        self,
        *,
        world_name: str,
        entity_name: str,
        position: Sequence[float],
        orientation: Sequence[float],
    ) -> bool:
        if self._try_transport(
            world_name=world_name,
            entity_name=entity_name,
            position=position,
            orientation=orientation,
        ):
            return True
        return _set_entity_pose_subprocess(
            world_name=world_name,
            entity_name=entity_name,
            position=position,
            orientation=orientation,
        )

    def _try_transport(
        self,
        *,
        world_name: str,
        entity_name: str,
        position: Sequence[float],
        orientation: Sequence[float],
    ) -> bool:
        node = self._gz_node
        if node is None:
            return False
        request = getattr(node, "request", None)
        if not callable(request):
            return False
        x, y, z = (float(v) for v in position)
        w, qx, qy, qz = (float(v) for v in orientation)
        service = f"/world/{world_name}/set_pose"
        for msgs_mod in ("gz.msgs13", "gz.msgs12", "gz.msgs"):
            try:
                msgs = __import__(msgs_mod, fromlist=["Pose", "Boolean"])
                pose = msgs.Pose()
                pose.name = str(entity_name)
                pose.position.x = x
                pose.position.y = y
                pose.position.z = z
                pose.orientation.w = w
                pose.orientation.x = qx
                pose.orientation.y = qy
                pose.orientation.z = qz
                response = msgs.Boolean()
                ok, _ = request(service, pose, 1000, response, True)
                return bool(ok)
            except Exception as exc:
                logger.debug(
                    "gz.transport set_pose via %s failed for '%s': %s",
                    msgs_mod,
                    entity_name,
                    exc,
                )
                continue
        logger.debug(
            "gz.transport set_pose unavailable for '%s' (world=%s); falling back to subprocess",
            entity_name,
            world_name,
        )
        return False


def _set_entity_pose_subprocess(
    *,
    world_name: str,
    entity_name: str,
    position: Sequence[float],
    orientation: Sequence[float],
) -> bool:
    """Move a Gazebo model via ``gz service`` (Harmonic). Returns True on success."""
    gz = shutil.which("gz")
    if not gz:
        logger.warning("gz binary not found; cannot sync prop pose for '%s'", entity_name)
        return False
    x, y, z = (float(v) for v in position)
    w, qx, qy, qz = (float(v) for v in orientation)
    service = f"/world/{world_name}/set_pose"
    req = (
        f'name: "{entity_name}", '
        f"position: {{x: {x}, y: {y}, z: {z}}}, "
        f"orientation: {{w: {w}, x: {qx}, y: {qy}, z: {qz}}}"
    )
    try:
        result = subprocess.run(
            [
                gz,
                "service",
                "-s",
                service,
                "--reqtype",
                "gz.msgs.Pose",
                "--reptype",
                "gz.msgs.Boolean",
                "--timeout",
                "1000",
                "--req",
                req,
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(
                "gz service set_pose failed for '%s' (rc=%s): %s",
                entity_name,
                result.returncode,
                (result.stdout or "").strip(),
            )
        return result.returncode == 0
    except Exception as exc:
        logger.warning("gz service set_pose raised for '%s': %s", entity_name, exc)
        return False


def set_entity_pose(
    *,
    world_name: str,
    entity_name: str,
    position: Sequence[float],
    orientation: Sequence[float],
    gz_node=None,
) -> bool:
    """Module-level helper; prefer ``PropPoseSyncer`` when syncing many props per step."""
    return PropPoseSyncer(gz_node=gz_node).set_entity_pose(
        world_name=world_name,
        entity_name=entity_name,
        position=position,
        orientation=orientation,
    )
