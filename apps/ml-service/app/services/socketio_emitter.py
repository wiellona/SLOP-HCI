# app/services/socketio_emitter.py

import httpx
import structlog
from app.core.config import get_settings

logger = structlog.get_logger()


class SocketIOEmitter:
    """
    HTTP client untuk mengirim event inferensi ke Node.js server,
    yang nantinya akan diteruskan ke klien via Socket.io.
    """

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.node_server_url
        self.headers = {
            "Authorization": f"Bearer {self.settings.ml_service_api_key}",
            "Content-Type": "application/json",
        }

    async def emit_inference_started(
        self,
        session_token: str,
        inference_id: str,
        modality: str,
    ) -> None:
        payload = {
            "event": "inference_started",
            "session_token": session_token,
            "inference_id": inference_id,
            "modality": modality,
        }
        await self._post("/api/v1/relay/emit", payload)

    async def emit_inference_complete(
        self,
        session_token: str,
        payload: dict,
    ) -> None:
        relay_payload = {
            "event": "inference_complete",
            "session_token": session_token,
            **payload,
        }
        await self._post("/api/v1/relay/emit", relay_payload)

    async def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    url, json=payload, headers=self.headers
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                "Node server HTTP error",
                status=e.response.status_code,
                path=path,
            )
            return {}
        except httpx.RequestError as e:
            logger.error(
                "Node server connection error", error=str(e), path=path
            )
            return {}