# vlm_model/logging_filter.py

import logging
import inspect
from vlm_model.context import request_id_ctx_var, client_ip_ctx_var

class ContextFilter(logging.Filter):
    """
    로그 레코드에 service, request_id, client_ip, class, method를 추가하는 커스텀 필터.
    """
    # 서비스 매핑 설정: 로거 이름의 접두사에 따라 서비스 결정
    service_mapping = {
        "vlm_model": "vlm_model"
        # 추가적인 서비스 매핑을 여기에 정의
    }

    def filter(self, record: logging.LogRecord) -> bool:
        # ContextVar에서 request_id와 client_ip 가져오기
        record.request_id = request_id_ctx_var.get() or "unknown"
        record.client_ip = client_ip_ctx_var.get() or "unknown"  # user_id를 client_ip로 대체

        # 로거 이름을 기반으로 서비스 결정
        record.service = "unknown"
        for service, prefix in self.service_mapping.items():
            if record.name.startswith(prefix):
                record.service = service
                break

        # 호출자 정보 설정
        frame = inspect.currentframe()
        if frame is not None:
            # 두 단계 위의 프레임 (현재 필터 함수 -> 로거 호출 함수)
            caller_frame = frame.f_back.f_back
            if caller_frame is not None:
                # 클래스 내부에서 호출되었는지 확인
                cls = caller_frame.f_locals.get('self', None)
                if cls:
                    record.class_name = cls.__class__.__name__
                else:
                    record.class_name = caller_frame.f_globals.get('__name__', 'N/A')
                record.method_name = caller_frame.f_code.co_name
            else:
                record.class_name = "unknown"
                record.method_name = "unknown"
        else:
            record.class_name = "unknown"
            record.method_name = "unknown"

        # ERROR 및 WARNING 레벨에만 추가 필드 설정
        if record.levelno >= logging.WARNING:
            record.errorType = getattr(record, 'errorType', "N/A")
            record.error_message = getattr(record, 'error_message', "N/A")
        else:
            record.errorType = ""
            record.error_message = ""

        return True