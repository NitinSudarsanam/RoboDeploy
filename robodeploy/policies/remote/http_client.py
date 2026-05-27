"""HTTP JSON client for optional remote VLA/diffusion predictor endpoints."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Callable
from urllib import request

from robodeploy.core.interop import to_numpy

Transport = Callable[[str, dict[str, Any]], Any]


def to_jsonable(value):  # noqa: ANN001
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "shape") or hasattr(value, "dtype"):
        return to_numpy(value).tolist()
    return value


class HttpRemotePolicyClient:
  """POST JSON packets to HTTP endpoints and return parsed JSON responses."""

  def __init__(
      self,
      endpoint: str,
      *,
      batch_endpoint: str | None = None,
      timeout_s: float = 5.0,
      transport: Transport | None = None,
  ) -> None:
    self._endpoint = endpoint
    self._batch_endpoint = batch_endpoint or endpoint
    self._timeout_s = float(timeout_s)
    self._transport = transport or self._default_transport

  def predict(self, packet: dict[str, Any]):  # noqa: ANN201
    return self._transport(self._endpoint, {"inputs": to_jsonable(packet)})

  def predict_batch(self, packets: list[dict[str, Any]]):  # noqa: ANN201
    return self._transport(
        self._batch_endpoint,
        {"inputs": [to_jsonable(packet) for packet in packets]},
    )

  def _default_transport(self, endpoint: str, payload: dict[str, Any]):  # noqa: ANN001
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=self._timeout_s) as resp:  # noqa: S310
      return json.loads(resp.read().decode("utf-8"))
