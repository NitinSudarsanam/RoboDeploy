"""
RemotePolicy — an IPolicy that calls a PolicyServer over a network transport.

From RoboBridge's perspective this is identical to any local policy.
The swap from local to distributed inference is one constructor change:

    # Local inference (default)
    policy = DiffusionPolicy(checkpoint="ckpt.pt")

    # Distributed inference — same interface, same env/bridge code
    policy = RemotePolicy(
        transport=ZmqTransport(host="gpu-server.local", port=5555),
        action_space=ActionSpace.JOINT_POS,
    )

    # Either policy drops into RoboBridge unchanged:
    bridge = RoboBridge(env=RoboEnv(policy=policy, ...), ...)

RemotePolicy handles:
  - Connecting to the server on first reset().
  - Sending reset signals at episode boundaries.
  - Sending observations and receiving actions on every get_action() call.
  - Reconnecting on transient failures (configurable retries).
  - Closing the connection in close().
"""

from __future__ import annotations

import time
from typing import Optional

from robodeploy.core.interfaces.policy  import IPolicy
from robodeploy.core.spaces             import ActionSpace
from robodeploy.core.types              import Action, Observation
from robodeploy.policies.base           import PolicyBase
from robodeploy.policies.remote.transport import IPolicyTransport


class RemotePolicy(PolicyBase):
    """Policy that forwards inference to a remote PolicyServer.

    Implements IPolicy fully — drop-in replacement for any local policy.

    Args:
        transport:    Transport that handles serialization and wire protocol.
                      Use ZmqTransport for development, GrpcTransport for production.
        action_space: ActionSpace this policy outputs. Must match what the
                      remote policy produces. Used by SafetyFilter and
                      RoboEnv for space compatibility checking.
        max_retries:  Number of reconnect attempts on connection failure before
                      raising. Default 3.
        retry_delay:  Seconds to wait between reconnect attempts. Default 1.0.
    """

    def __init__(
        self,
        transport:    IPolicyTransport,
        action_space: ActionSpace = ActionSpace.JOINT_POS,
        max_retries:  int   = 3,
        retry_delay:  float = 1.0,
    ) -> None:
        super().__init__(action_space=action_space)
        self._transport   = transport
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._connected   = False

    # ------------------------------------------------------------------
    # IPolicy
    # ------------------------------------------------------------------

    def _reset_impl(self) -> None:
        """Connect on first episode; send reset signal on subsequent episodes."""
        if not self._connected:
            self._connect_with_retry()
        else:
            self._transport.send_reset()

    def get_action(self, obs: Observation) -> Action:
        """Send observation to server, return action. Retries on transient failure.

        Args:
            obs: Current robot observation (already processed by ObsPipeline).

        Returns:
            Action from the remote policy.

        Raises:
            RuntimeError: If the server is unreachable after max_retries.
        """
        last_err: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                return self._transport.send_obs_recv_action(obs)
            except (TimeoutError, ConnectionError, RuntimeError) as e:
                last_err = e
                if attempt < self._max_retries - 1:
                    print(
                        f"[RemotePolicy] Attempt {attempt + 1}/{self._max_retries} failed: {e}. "
                        f"Retrying in {self._retry_delay}s..."
                    )
                    time.sleep(self._retry_delay)
                    self._connect_with_retry()   # attempt reconnect

        raise RuntimeError(
            f"RemotePolicy: server unreachable after {self._max_retries} attempts. "
            f"Last error: {last_err}"
        )

    def close(self) -> None:
        """Close the transport connection."""
        self._transport.close()
        self._connected = False

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _connect_with_retry(self) -> None:
        last_err: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                self._transport.connect()
                self._connected = True
                return
            except Exception as e:
                last_err = e
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay)

        raise ConnectionRefusedError(
            f"RemotePolicy: could not connect after {self._max_retries} attempts. "
            f"Last error: {last_err}"
        )
