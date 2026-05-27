"""
PolicyServer — hosts any IPolicy over a network transport on the GPU machine.

The server is the other half of RemotePolicy. It:
  1. Loads any IPolicy (DiffusionPolicy, VLAPolicy, RobomimicPolicy, etc.).
  2. Listens for observations over the configured transport.
  3. Calls policy.get_action(obs) locally (on the GPU).
  4. Returns the action over the same transport.
  5. Handles reset signals by calling policy.reset().

Usage on the GPU server machine:

    from robodeploy.policies.remote.server    import PolicyServer
    from robodeploy.policies.remote.transport import ZmqTransport
    from my_project.policies.diffusion        import DiffusionPolicy

    policy = DiffusionPolicy(checkpoint="ckpt.pt")
    server = PolicyServer(
        policy    = policy,
        transport = ZmqTransport(host="0.0.0.0", port=5555),
    )
    server.serve_forever()   # blocks until Ctrl+C

On the robot machine, connect with:

    from robodeploy.policies.remote import RemotePolicy, ZmqTransport

    policy = RemotePolicy(
        transport    = ZmqTransport(host="gpu-server.local", port=5555),
        action_space = ActionSpace.JOINT_POS,
    )

The server is transport-agnostic: swap ZmqTransport for GrpcTransport
without changing any policy or robot code.
"""

from __future__ import annotations

import signal
import sys
from typing import Optional

from robodeploy.core.interfaces.policy    import IPolicy
from robodeploy.policies.remote.transport import (
    IPolicyTransport,
    action_to_bytes,
    bytes_to_obs,
)


class PolicyServer:
    """Serves an IPolicy over a transport. Runs on the GPU inference machine.

    Args:
        policy:    Any IPolicy. Loaded and warmed up before serving starts.
        transport: Transport that matches the RemotePolicy's transport on the client.
        verbose:   Print per-request timing information.
    """

    def __init__(
        self,
        policy:    IPolicy,
        transport: IPolicyTransport,
        verbose:   bool = True,
    ) -> None:
        self._policy    = policy
        self._transport = transport
        self._verbose   = verbose
        self._running   = False

    def serve_forever(self) -> None:
        """Start listening and serving. Blocks until stop() or SIGINT/SIGTERM.

        Calls policy.reset() once at startup to initialize episode state.
        Registers SIGINT/SIGTERM handlers for clean shutdown.
        """
        self._policy.reset()
        self._transport.connect()   # for servers this opens the listening socket
        self._running = True

        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        print(f"[PolicyServer] Serving {type(self._policy).__name__}. Press Ctrl+C to stop.")

        try:
            self._serve_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._transport.close()
            print("[PolicyServer] Stopped.")

    def stop(self) -> None:
        """Request the server to stop after the current request."""
        self._running = False

    def infer(self, obs) -> "Action":  # noqa: ANN001
        """Run one local inference step (for tests and non-ZMQ integrations)."""
        return self._policy.get_action(obs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _serve_loop(self) -> None:
        """Main request/response loop. Dispatches ACT and RST messages."""
        import time

        # Import the server-side ZMQ socket directly.
        # This loop is transport-specific for ZMQ. GrpcTransport will have
        # its own serve loop (gRPC servicer). When GrpcTransport is fully
        # implemented, extract this into IPolicyTransport.serve_loop(handler_fn).
        try:
            import zmq  # type: ignore
        except ImportError:
            raise ImportError(
                "PolicyServer with ZmqTransport requires pyzmq.\n"
                "Install with: pip install pyzmq"
            )

        ctx    = zmq.Context.instance()
        socket = ctx.socket(zmq.REP)
        socket.bind(getattr(self._transport, "_addr", "tcp://0.0.0.0:5555"))

        while self._running:
            if not socket.poll(timeout=100):   # 100ms poll so stop() is responsive
                continue

            msg = socket.recv()

            if msg[:3] == b"RST":
                self._policy.reset()
                socket.send(b"OK")
                if self._verbose:
                    print("[PolicyServer] Episode reset.")

            elif msg[:3] == b"ACT":
                t0  = time.perf_counter()
                obs = bytes_to_obs(msg[3:])
                action = self._policy.get_action(obs)
                socket.send(action_to_bytes(action))
                if self._verbose:
                    ms = (time.perf_counter() - t0) * 1000
                    print(f"[PolicyServer] Inference: {ms:.1f}ms")

            else:
                socket.send(b"ERR:unknown_message")

        socket.close()

    def _handle_signal(self, signum, frame) -> None:
        print(f"\n[PolicyServer] Signal {signum} received — stopping.")
        self._running = False


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def serve(
    policy:    IPolicy,
    host:      str = "0.0.0.0",
    port:      int = 5555,
    transport: str = "zmq",
    verbose:   bool = True,
) -> None:
    """One-line server launch for the common case.

    Args:
        policy:    Any IPolicy to serve.
        host:      Interface to bind (0.0.0.0 = all interfaces).
        port:      Port to listen on.
        transport: "zmq" (default) or "grpc".
        verbose:   Print per-request timing.

    Example:
        from robodeploy.policies.remote.server import serve
        from my_project.policies.diffusion     import DiffusionPolicy

        serve(DiffusionPolicy(checkpoint="ckpt.pt"), host="0.0.0.0", port=5555)
    """
    if transport == "zmq":
        from robodeploy.policies.remote.transport import ZmqTransport
        t = ZmqTransport(host=host, port=port)
    elif transport == "grpc":
        from robodeploy.policies.remote.transport import GrpcTransport
        t = GrpcTransport(host=host, port=port)
    else:
        raise ValueError(f"Unknown transport '{transport}'. Use 'zmq' or 'grpc'.")

    PolicyServer(policy=policy, transport=t, verbose=verbose).serve_forever()
