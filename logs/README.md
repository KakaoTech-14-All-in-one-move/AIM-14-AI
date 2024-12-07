# Logs Directory

이 디렉토리는 애플리케이션에서 생성되는 다양한 로그 파일을 저장하는 용도로 사용됩니다. 로그 파일들은 애플리케이션의 상태 모니터링, 문제 해결, 및 성능 분석에 중요한 역할을 합니다.

## 로그 파일 종류

- **`app.log`**
  - **설명:** 일반 애플리케이션 로그를 기록합니다.
  - **내용:** 정보, 디버그 메시지, 경고 등 애플리케이션의 전반적인 동작 상태를 추적합니다.

- **`access.log`**
  - **설명:** 사용자 접근 요청에 대한 로그를 기록합니다.
  - **내용:** HTTP 요청 정보, 응답 상태 코드, 요청 시간 등을 포함합니다.

- **`error.log`**
  - **설명:** 애플리케이션에서 발생하는 오류 및 예외를 기록합니다.
  - **내용:** 오류 메시지, 스택 트레이스, 발생 시각 등을 포함하여 문제 해결에 필요한 정보를 제공합니다.

## 로그 파일 구조

로그 파일은 JSON 형식으로 기록되며, 각 로그 항목은 다음과 같은 필드를 포함합니다:

- `@timestamp`: 로그가 기록된 시각 (ISO 8601 형식)
- `level`: 로그의 레벨 (`DEBUG`, `INFO`, `ERROR` 등)
- `service`: 로그를 생성한 서비스의 이름 (예: `SPRING_API`, `AI_API`)
- `request_id`: 관련된 요청 ID (선택 사항)
- `class`: 로그를 생성한 클래스의 이름
- `method`: 로그를 생성한 메서드의 이름
- `error_type`: 발생한 오류의 유형 (오류 로그에만 해당)
- `message`: 로그 메시지 또는 오류 설명 (오류 로그에만 해당)

## 로그 레벨별 설명 및 예시

### 1. DEBUG 레벨

**목적:** 개발 및 디버깅 과정에서 상세한 정보를 기록합니다. 애플리케이션의 내부 동작을 추적하는 데 유용합니다.

**예시:**

```json
{
  "@timestamp": "2024-12-04T10:15:30.000Z",
  "level": "DEBUG",
  "service": "SPRING_API",
  "request_id": "req-456",
  "class": "com.example.UserController",
  "method": "getUser"
}
```

**설명:**

- **`@timestamp`**: 로그가 생성된 정확한 시각을 나타냅니다.
- **`level`**: 로그의 심각도를 나타내며, 여기서는 `DEBUG` 레벨입니다.
- **`service`**: 로그를 생성한 서비스의 이름입니다. 예시에서는 `SPRING_API`입니다.
- **`request_id`**: 관련된 요청 ID로, 특정 요청의 행동을 추적할 때 유용합니다.
- **`class`**: 로그를 생성한 클래스의 이름입니다. 예시에서는 `com.example.UserController`입니다.
- **`method`**: 로그를 생성한 메서드의 이름입니다. 예시에서는 `getUser`입니다.

### 2. INFO 레벨

**목적:** 일반적인 애플리케이션 동작에 대한 정보를 기록합니다. 시스템의 정상적인 동작 상태를 모니터링하는 데 유용합니다.

**예시:**

```json
{
  "@timestamp": "2024-12-04T10:15:30.123Z",
  "level": "INFO",
  "service": "AI_API",
  "request_id": "req-789",
  "class": "com.example.UserController",
  "method": "getUser"
}
```

**설명:**

- **`@timestamp`**: 로그가 생성된 시각입니다.
- **`level`**: 로그의 심각도를 나타내며, 여기서는 `INFO` 레벨입니다.
- **`service`**: 로그를 생성한 서비스의 이름으로, 예시에서는 `AI_API`입니다.
- **`request_id`**: 관련된 요청 ID입니다.
- **`class`**: 로그를 생성한 클래스의 이름입니다.
- **`method`**: 로그를 생성한 메서드의 이름입니다.

### 3. ERROR 레벨

**목적:** 애플리케이션에서 발생한 오류나 예외를 기록합니다. 문제 해결과 장애 대응에 필수적인 정보를 제공합니다.

**예시:**

```json
{
  "@timestamp": "2024-12-04T10:15:30.123Z",
  "level": "ERROR",
  "service": "SPRING_API",
  "request_id": "req-789",
  "class": "com.example.UserController",
  "method": "getUser",
  "error_type": "EntityNotFoundException",
  "message": "User not found with id"
}
```

**설명:**

- **`@timestamp`**: 오류가 발생한 시각을 나타냅니다.
- **`level`**: 로그의 심각도를 나타내며, 여기서는 `ERROR` 레벨입니다.
- **`service`**: 로그를 생성한 서비스의 이름으로, 예시에서는 `SPRING_API`입니다.
- **`request_id`**: 관련된 요청 ID입니다.
- **`class`**: 로그를 생성한 클래스의 이름입니다.
- **`method`**: 로그를 생성한 메서드의 이름입니다.
- **`error_type`**: 발생한 오류의 유형을 명시합니다. 예시에서는 `EntityNotFoundException`입니다.
- **`message`**: 오류에 대한 상세한 설명을 제공합니다. 예시에서는 `"User not found with id"`입니다.

## 로그 관리

효율적인 로그 관리를 위해 다음과 같은 정책을 적용하고 있습니다:

### 로그 회전 (Log Rotation)

- **정책:** 로그 파일은 매일 회전되며, 파일 크기가 일정 기준을 초과하면 자동으로 새로운 파일로 교체됩니다.
- **설정:** `logging_config.json` 파일에서 `RotatingFileHandler` 또는 `TimedRotatingFileHandler`를 사용하여 설정합니다.

### 로그 보존 (Log Retention)

- **정책:** 로그 파일은 최대 30일 동안 보존되며, 그 이후의 로그는 자동으로 삭제되거나 아카이브됩니다.
- **설정:** 로그 회전 설정에서 보존 일수를 지정합니다.

### 로그 형식 (Log Format)

- **형식:** 로그는 JSON 형식으로 저장되어, 로그 분석 도구와의 호환성을 높입니다.
- **설정:** `logging_config.json` 파일에서 `CustomJsonFormatter`를 사용하여 설정합니다.