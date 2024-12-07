# vlm_model/routers/delete_files.py

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from pathlib import Path
import logging

from vlm_model.config import FEEDBACK_DIR, UPLOAD_DIR

router = APIRouter()

logger = logging.getLogger(__name__)

@router.delete("/delete_files/{video_id}", response_class=JSONResponse)
async def delete_files(video_id: str):
    """
    특정 video_id와 연관된 파일들을 삭제하는 API.
    """
    try:
        # video_id를 포함한 파일 찾기
        input_files = list(UPLOAD_DIR.glob(f"*{video_id}*.webm"))
        output_files = list(FEEDBACK_DIR.glob(f"*{video_id}*.jpg"))

        if not input_files and not output_files:
            logger.error(f"{video_id}와 관련된 파일이 없는것 같습니다.", extra={
                    "errorType": "FileNotFoundError",
                    "error_message": f"{video_id}와 관련 파일 찾는중 오류 발생",
                })
            raise HTTPException(status_code=404, detail="해당 video_id와 관련된 파일을 찾을 수 없습니다.")

        # 파일 삭제
        deleted_files = []
        for file in input_files + output_files:
            try:
                file.unlink()
                deleted_files.append(str(file))
                logger.debug(f"Deleted files related to the {video_id}")
            except Exception as e:
                logger.error(f"Failed to delete file related to the {video_id}: {file} {e}", extra={
                    "errorType": type(e).__name__,
                    "error_message": str(e)
                })
                raise HTTPException(status_code=500, detail=f"{file.name} 파일 삭제에 실패했습니다.") from e

        logger.info(f"{video_id}와 관련된 파일 삭제에 성공했습니다.")
        return {"status": "success", "deleted_files": deleted_files}

    except Exception as e:
        logger.error(f"Error in delete_files API: {e}", extra={
            "errorType": type(e).__name__,
            "error_message": str(e)
        })
        raise HTTPException(status_code=500, detail="내부 서버 오류가 발생했습니다.") from e
