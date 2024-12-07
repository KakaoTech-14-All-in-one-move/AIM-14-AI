# vlm_model/logging_filter.py

import logging
from vlm_model.context_var import request_id_ctx_var
from pythonjsonlogger import jsonlogger

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
            record.errorType = getattr(record, 'errorType', "N/A")
            record.message = getattr(record, 'message', "N/A")
        else:
            # ERROR 레벨이 아닐 때는 error_type과 message를 삭제
            if hasattr(record, 'errorType'):
                del record.errorType
            if hasattr(record, 'message'):
                del record.message

        return True
    
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def __init__(self, *args, **kwargs):
        # ensure_ascii를 False로 설정
        if 'json_ensure_ascii' not in kwargs:
            kwargs['json_ensure_ascii'] = False
        super().__init__(*args, **kwargs)
        
    def process_log_record(self, log_record):
        # asctime -> @timestamp
        if 'asctime' in log_record:
            log_record['@timestamp'] = log_record.pop('asctime')

        # levelname -> level
        if 'levelname' in log_record:
            log_record['level'] = log_record.pop('levelname')

        # class_name -> class
        if 'class_name' in log_record:
            log_record['class'] = log_record.pop('class_name')

        # method_name -> method
        if 'method_name' in log_record:
            log_record['method'] = log_record.pop('method_name')
        
        # errorType -> error_type
        if 'errorType' in log_record:
            log_record['error_type'] = log_record.pop('errorType')

        # error_message -> message
        # error_message가 None일 경우에도 일단 message로 변환
        if 'error_message' in log_record:
            msg = log_record.pop('error_message')
            log_record['message'] = msg if msg is not None else "N/A"

        # 레벨이 ERROR가 아닌 경우 error_type와 message 필드를 제거
        if 'level' in log_record and log_record['level'] != 'ERROR':
            if 'error_type' in log_record:
                del log_record['error_type']
            if 'message' in log_record:
                del log_record['message']

        # 원하는 순서에 따라 최종 필드를 재구성
        allowed_keys = ['@timestamp', 'level', 'service', 'request_id', 'class', 'method', 'error_type', 'message']
        final_record = {}
        for key in allowed_keys:
            if key in log_record:
                final_record[key] = log_record[key]

        return final_record