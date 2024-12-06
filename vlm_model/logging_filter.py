# vlm_model/logging_filter.py

import logging
import inspect
from vlm_model.context_var import request_id_ctx_var

class ContextFilter(logging.Filter):
    """
    로그 레코드에 service, request_id, class, method를 추가하는 커스텀 필터.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # 모든 로그에 대해 'service'를 'VLM_API'로 설정
        record.service = "VLM_API"

        # ContextVar에서 request_id와 client_ip 가져오기
        record.request_id = request_id_ctx_var.get() or "unknown"

        # 로거 이름을 기반으로 class_name 설정
        logger_name = record.name
        prefix = 'vlm_model.'

        if logger_name == 'vlm_model':
            # 최상위 로거인 경우
            record.class_name = 'root'
        elif logger_name.startswith(prefix):
            # 'vlm_model.'을 제외한 나머지 부분을 class_name으로 설정
            record.class_name = logger_name[len(prefix):]
        else:
            # 기타 로거의 경우 로거 이름 전체를 class_name으로 설정
            record.class_name = logger_name

        # method_name을 LogRecord의 funcName 속성으로 설정 (실제 함수 이름)
        record.method_name = record.funcName or "unknown"

        # ERROR 레벨에만 추가 필드 설정
        if record.levelno >= logging.ERROR:
            record.error_type = getattr(record, 'error_type', "N/A")
            record.message = getattr(record, 'message', "N/A")
        else:
            # ERROR 레벨이 아닐 때는 error_type과 message를 삭제
            if hasattr(record, 'error_type'):
                del record.error_type
            if hasattr(record, 'message'):
                del record.message

        return True