"""Ros2NodeAdapter — base class for ROS 2 controller / sensor adapters.

Hides lifecycle plumbing so adapter authors do not import rclpy or call
`Ros2Runtime` directly. Subclasses override `_on_node_ready(node)` to wire
publishers, subscriptions, TF, etc.

Lifecycle handled by the base class:
  - `Ros2Runtime.ensure_started()` so rclpy is initialised exactly once.
  - Node creation (`rclpy.node.Node`) with the configured name.
  - Registering / removing the node from the executor.
  - Destroying the node on `stop()`.

Subclasses are responsible only for adapter-specific wiring inside
`_on_node_ready(node)` and runtime behaviour (`send_action`, `get_obs`, etc.).
"""

from __future__ import annotations

from typing import Optional

from .runtime import Ros2Runtime


class Ros2NodeAdapter:
    """Base class for any ROS 2 adapter that needs a managed Node."""

    #: Override per subclass.
    node_name: str = "robodeploy_adapter"
    #: When set, overrides ``Ros2Runtime.use_sim_time`` for this adapter only.
    use_sim_time: bool | None = None

    def __init__(self) -> None:
        self._node = None  # rclpy.node.Node when started

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._node is not None:
            return
        try:
            import rclpy
            import rclpy.node
        except ImportError as exc:
            raise ImportError(
                "ROS 2 packages not found. Source your ROS 2 (Jazzy) environment "
                "and ensure `rclpy` is on PYTHONPATH before constructing this adapter."
            ) from exc

        Ros2Runtime.ensure_started()
        sim_time = self.use_sim_time if self.use_sim_time is not None else Ros2Runtime.use_sim_time
        if sim_time:
            from rclpy.parameter import Parameter

            self._node = rclpy.create_node(
                self.node_name,
                parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
            )
        else:
            self._node = rclpy.node.Node(self.node_name)
        Ros2Runtime.add_node(self._node)
        try:
            self._on_node_ready(self._node)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        if self._node is None:
            return
        try:
            self._on_node_stopping(self._node)
        except Exception:
            pass
        try:
            Ros2Runtime.remove_node(self._node)
            self._node.destroy_node()
        except Exception:
            pass
        self._node = None

    # ------------------------------------------------------------------
    # Subclass extension points
    # ------------------------------------------------------------------

    def _on_node_ready(self, node) -> None:
        """Called once after the node is created and registered.

        Subclasses set up publishers, subscriptions, TF listeners, etc. here.
        Use the supplied `node` (an `rclpy.node.Node`) — do not store rclpy
        imports at module level; import inside this method if needed.
        """
        del node

    def _on_node_stopping(self, node) -> None:
        """Optional cleanup hook called before the node is destroyed."""
        del node

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def node(self):
        """Return the underlying rclpy node, or None if not started."""
        return self._node

    @property
    def is_started(self) -> bool:
        return self._node is not None
