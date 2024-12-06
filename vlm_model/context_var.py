# vlm_model/context_var.py

import contextvars

# 요청별 고유 ID 및 클라이언트 IP를 저장하는 ContextVar
request_id_ctx_var = contextvars.ContextVar('request_id', default=None)
# client_ip_ctx_var = contextvars.ContextVar('client_ip', default=None)