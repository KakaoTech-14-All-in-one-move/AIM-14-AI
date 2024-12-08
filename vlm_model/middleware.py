# vlm_model/middleware.py

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send
from vlm_model.context_var import request_id_ctx_var
import logging

logger = logging.getLogger(__name__)

# Request ID 미들웨어: 각 요청에 고유한 ID와 클라이언트 IP를 설정
class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    요청 헤더에서 'X-Request-ID'를 추출하여 ContextVar에 설정하는 미들웨어.
    """
    async def dispatch(self, request: Request, call_next):
        # 요청 헤더에서 'X-Request-ID' 추출 (헤더 이름은 클라이언트와 협의하여 설정)
        request_id = request.headers.get("X-Request-ID", "unknown")
        # ContextVar에 설정
        token = request_id_ctx_var.set(request_id)
        logger.debug(f"RequestIDMiddleware: set request_id to {request_id}")

        try:
            response = await call_next(request)
            return response
        finally:
            # 요청 처리 후 ContextVar 복구
            request_id_ctx_var.reset(token)
            logger.debug("RequestIDMiddleware: reset request_id to previous value")