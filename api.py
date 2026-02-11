"""
FastAPI wrapper around the FastALPR pipeline.

POST /detect accepts an image file (multipart/form-data) and returns detections plus OCR text.
"""

import base64
import json
import logging
import os
import time
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from fast_alpr import ALPR, ALPRResult
from filemaker_client import FileMakerClient, FileMakerError


class BoundingBoxModel(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class DetectionModel(BaseModel):
    label: str
    confidence: float
    bounding_box: BoundingBoxModel


class OCRModel(BaseModel):
    text: str
    confidence: float | list[float]


class ALPRItem(BaseModel):
    detection: DetectionModel
    ocr: OCRModel | None


class DetectResponse(BaseModel):
    count: int
    results: list[ALPRItem]


class FileMakerCallResult(BaseModel):
    plate_text: str
    frame_number: int
    success: bool
    error_message: str | None = None
    fm_response: dict | None = None


class ScanPlateRecord(BaseModel):
    plate_text: str
    confidence: float
    bounding_box: BoundingBoxModel
    frame_number: int
    fm_call: FileMakerCallResult


class VideoScanResponse(BaseModel):
    total_frames: int
    processing_time_seconds: float
    plates_detected: int
    fm_calls_made: int
    fm_calls_succeeded: int
    fm_calls_failed: int
    detections: list[ScanPlateRecord]


app = FastAPI(title="FastALPR API", version="0.1.0")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    force=True,
)
for _logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    logging.getLogger(_logger_name).setLevel(logging.INFO)
for _logger_name in ("httpx", "httpcore"):
    logging.getLogger(_logger_name).setLevel(logging.WARNING)
logger = logging.getLogger("fast_alpr.api")


def _decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw image bytes into a BGR ndarray.
    """
    frame_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Unable to decode image bytes. Please send a valid image file.")
    return frame


def _encode_image(frame: np.ndarray) -> bytes:
    """
    Encode a BGR ndarray as PNG bytes.
    """
    ok, buffer = cv2.imencode(".png", frame)
    if not ok:
        raise ValueError("Failed to encode image.")
    return buffer.tobytes()


def _normalize_confidence(confidence: float | Sequence[float]) -> float | list[float]:
    if isinstance(confidence, Sequence) and not isinstance(confidence, (str, bytes)):
        return [float(x) for x in confidence]
    return float(confidence)


def _serialize_result(result: ALPRResult) -> ALPRItem:
    detection = result.detection
    ocr = result.ocr

    detection_model = DetectionModel(
        label=detection.label,
        confidence=float(detection.confidence),
        bounding_box=BoundingBoxModel(
            x1=int(detection.bounding_box.x1),
            y1=int(detection.bounding_box.y1),
            x2=int(detection.bounding_box.x2),
            y2=int(detection.bounding_box.y2),
        ),
    )

    ocr_model = None
    if ocr is not None:
        ocr_model = OCRModel(
            text=ocr.text,
            confidence=_normalize_confidence(ocr.confidence),
        )

    return ALPRItem(detection=detection_model, ocr=ocr_model)


def _get_alpr() -> ALPR:
    """
    Lazily initialize the ALPR pipeline once and reuse it across requests.
    """
    if not hasattr(app.state, "alpr"):
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            logger.error("CUDAExecutionProvider not available. Providers: %s", available)
            raise RuntimeError(
                "CUDAExecutionProvider not available. Ensure GPU drivers and CUDA are installed."
            )
        logger.info("Initializing ALPR pipeline (device=cuda)")
        app.state.alpr = ALPR(
            detector_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            ocr_device="cuda",
            ocr_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _log_providers(app.state.alpr)
    return app.state.alpr


def _get_fm_client() -> FileMakerClient:
    """
    Lazily initialize the FileMaker client once and reuse across requests.
    """
    if not hasattr(app.state, "fm_client"):
        required_vars = ["FM_HOST", "FM_DATABASE", "FM_USERNAME", "FM_PASSWORD",
                         "FM_LAYOUT", "FM_SCRIPT"]
        missing = [v for v in required_vars if not os.environ.get(v)]
        if missing:
            raise RuntimeError(
                f"Missing required FileMaker environment variables: {', '.join(missing)}"
            )

        verify_ssl = os.environ.get("FM_VERIFY_SSL", "true").lower() in ("true", "1", "yes")

        app.state.fm_client = FileMakerClient(
            host=os.environ["FM_HOST"],
            database=os.environ["FM_DATABASE"],
            username=os.environ["FM_USERNAME"],
            password=os.environ["FM_PASSWORD"],
            layout=os.environ["FM_LAYOUT"],
            script=os.environ["FM_SCRIPT"],
            verify_ssl=verify_ssl,
        )
        logger.info(
            "FileMaker client initialized (host=%s, database=%s)",
            os.environ["FM_HOST"],
            os.environ["FM_DATABASE"],
        )
    return app.state.fm_client


def _log_providers(alpr: ALPR) -> None:
    """
    Log available ONNX Runtime providers for detector and OCR sessions.
    """
    detector_providers = _detect_providers(getattr(alpr.detector, "detector", None))
    if detector_providers:
        logger.info("Detector providers: %s", detector_providers)

    ocr_model = getattr(alpr.ocr, "ocr_model", None)
    ocr_providers = _detect_providers(ocr_model)
    if ocr_providers:
        logger.info("OCR providers: %s", ocr_providers)


def _detect_providers(model: object) -> list[str]:
    """
    Best-effort extraction of ONNX Runtime providers from a model/session.
    """
    if model is None:
        return []
    for attr in ("session", "ort_session", "ort_sessions", "onnx_session"):
        session = getattr(model, attr, None)
        if session is None:
            continue
        try:
            if isinstance(session, list):
                if session:
                    return list(session[0].get_providers())
            else:
                return list(session.get_providers())
        except Exception:  # pragma: no cover - best effort logging only
            continue
    return []


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)) -> DetectResponse:
    """
    Run ALPR on the uploaded image and return detections with OCR.
    """
    start = time.perf_counter()
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        frame = _decode_image(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        alpr = _get_alpr()
        results = alpr.predict(frame)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail="Failed to run detection.") from exc

    response = DetectResponse(
        count=len(results),
        results=[_serialize_result(res) for res in results],
    )
    elapsed = time.perf_counter() - start
    logger.info("Detect request processed in %.3fs", elapsed)
    return response


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)) -> StreamingResponse:
    """
    Run ALPR on the uploaded image and return the annotated image.
    """
    start = time.perf_counter()
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        frame = _decode_image(contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        alpr = _get_alpr()
        annotated = alpr.draw_predictions(frame)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail="Failed to run detection.") from exc

    try:
        img_bytes = _encode_image(annotated)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response = StreamingResponse(
        content=BytesIO(img_bytes),
        media_type="image/png",
        headers={"Content-Disposition": 'inline; filename="annotated.png"'},
    )
    elapsed = time.perf_counter() - start
    logger.info("Detect image request processed in %.3fs", elapsed)
    return response


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)) -> StreamingResponse:
    """
    Run ALPR on an uploaded video and return an annotated MP4.
    """
    start = time.perf_counter()
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    input_suffix = Path(file.filename or "").suffix or ".mp4"

    temp_in = NamedTemporaryFile(delete=False, suffix=input_suffix)
    temp_out = NamedTemporaryFile(delete=False, suffix=".mp4")
    writer = None
    frames_processed = 0
    try:
        temp_in.write(contents)
        temp_in.flush()
        cap = cv2.VideoCapture(temp_in.name)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Unable to read uploaded video.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        alpr = _get_alpr()

        while True:
            success, frame = cap.read()
            if not success:
                break
            if frame is None:
                continue

            annotated = alpr.draw_predictions(frame)
            frames_processed += 1

            if writer is None:
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    temp_out.name,
                    fourcc,
                    fps or 30.0,
                    (width, height),
                )
                if not writer.isOpened():
                    raise HTTPException(status_code=500, detail="Failed to create video writer.")

            writer.write(annotated)

        cap.release()
        if writer is None:
            raise HTTPException(status_code=400, detail="No frames found in uploaded video.")
        writer.release()

        video_bytes = Path(temp_out.name).read_bytes()
    except HTTPException:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail="Failed to process video.") from exc
    finally:
        try:
            temp_in.close()
            temp_out.close()
        finally:
            Path(temp_in.name).unlink(missing_ok=True)
            Path(temp_out.name).unlink(missing_ok=True)

    elapsed = time.perf_counter() - start
    fps = frames_processed / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Detect video processed %d frames in %.3fs (%.2f FPS)",
        frames_processed,
        elapsed,
        fps,
    )
    return StreamingResponse(
        content=BytesIO(video_bytes),
        media_type="video/mp4",
        headers={"Content-Disposition": 'inline; filename="annotated.mp4"'},
    )


@app.post("/detect/video/scan", response_model=VideoScanResponse)
async def detect_video_scan(file: UploadFile = File(...)) -> VideoScanResponse:
    """
    Scan an uploaded video for license plates and notify FileMaker for each detection.

    A 5-second cooldown per plate text prevents duplicate notifications.
    Returns a JSON summary of all detections and FileMaker call results.
    """
    start = time.perf_counter()
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    logger.info(
        "Video scan request received (filename=%s, size_bytes=%d)",
        file.filename,
        len(contents),
    )

    input_suffix = Path(file.filename or "").suffix or ".mp4"

    logger.info("Initializing ALPR and FileMaker clients for video scan")
    alpr = _get_alpr()
    try:
        fm_client = _get_fm_client()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    temp_in = NamedTemporaryFile(delete=False, suffix=input_suffix)
    frames_processed = 0
    detections: list[ScanPlateRecord] = []

    # Throttle state: plate_text -> last trigger time (monotonic)
    plate_cooldowns: dict[str, float] = {}
    cooldown_seconds = 5.0

    try:
        temp_in.write(contents)
        temp_in.flush()
        cap = cv2.VideoCapture(temp_in.name)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Unable to read uploaded video.")
        logger.info("Video opened successfully for scanning (temp_file=%s)", temp_in.name)

        while True:
            success, frame = cap.read()
            if not success:
                break
            if frame is None:
                continue

            frames_processed += 1
            if frames_processed % 30 == 0:
                logger.info(
                    "Video scan progress: processed %d frames, detections recorded=%d",
                    frames_processed,
                    len(detections),
                )
            results = alpr.predict(frame)
            if results:
                logger.info(
                    "Frame %d: ALPR produced %d candidate(s)",
                    frames_processed,
                    len(results),
                )

            for result in results:
                if result.ocr is None or not result.ocr.text:
                    logger.info("Frame %d: skipped candidate without OCR text", frames_processed)
                    continue

                plate_text = result.ocr.text
                now = time.monotonic()

                # Skip if same plate was triggered within cooldown
                last_trigger = plate_cooldowns.get(plate_text)
                if last_trigger is not None and (now - last_trigger) < cooldown_seconds:
                    logger.info(
                        "Frame %d: skipped plate '%s' due to cooldown (%.2fs remaining)",
                        frames_processed,
                        plate_text,
                        cooldown_seconds - (now - last_trigger),
                    )
                    continue

                plate_cooldowns[plate_text] = now
                logger.info(
                    "Frame %d: accepted plate '%s' for FileMaker call",
                    frames_processed,
                    plate_text,
                )

                # Crop the plate image for base64 encoding
                bbox = result.detection.bounding_box
                h, w = frame.shape[:2]
                x1 = max(bbox.x1, 0)
                y1 = max(bbox.y1, 0)
                x2 = min(bbox.x2, w)
                y2 = min(bbox.y2, h)
                cropped = frame[y1:y2, x1:x2]

                ok, buf = cv2.imencode(".jpg", cropped)
                plate_b64 = base64.b64encode(buf.tobytes()).decode() if ok else ""

                # Compute scalar confidence
                conf = result.ocr.confidence
                scalar_conf = float(
                    conf if isinstance(conf, float) else sum(conf) / len(conf)
                )

                # Build the script parameter as a single JSON string
                script_param = json.dumps({
                    "plate_text": plate_text,
                    "confidence": scalar_conf,
                    "bounding_box": {
                        "x1": int(bbox.x1),
                        "y1": int(bbox.y1),
                        "x2": int(bbox.x2),
                        "y2": int(bbox.y2),
                    },
                    "frame_number": frames_processed,
                    "plate_image_base64": plate_b64,
                })
                # Call FileMaker — continue processing on failure
                try:
                    fm_response = await fm_client.execute_script(script_param)
                    fm_call = FileMakerCallResult(
                        plate_text=plate_text,
                        frame_number=frames_processed,
                        success=True,
                        fm_response=fm_response,
                    )
                    logger.info(
                        "FM script call succeeded for plate '%s' at frame %d",
                        plate_text,
                        frames_processed,
                    )
                except FileMakerError as exc:
                    fm_call = FileMakerCallResult(
                        plate_text=plate_text,
                        frame_number=frames_processed,
                        success=False,
                        error_message=str(exc),
                    )
                    logger.warning(
                        "FM script call failed for plate '%s' at frame %d: %s",
                        plate_text,
                        frames_processed,
                        exc,
                    )

                detections.append(ScanPlateRecord(
                    plate_text=plate_text,
                    confidence=scalar_conf,
                    bounding_box=BoundingBoxModel(
                        x1=int(bbox.x1),
                        y1=int(bbox.y1),
                        x2=int(bbox.x2),
                        y2=int(bbox.y2),
                    ),
                    frame_number=frames_processed,
                    fm_call=fm_call,
                ))

        cap.release()

    except HTTPException:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Unexpected error during /detect/video/scan")
        raise HTTPException(
            status_code=500, detail="Failed to process video for scanning."
        ) from exc
    finally:
        try:
            temp_in.close()
        finally:
            Path(temp_in.name).unlink(missing_ok=True)

    elapsed = time.perf_counter() - start
    fm_succeeded = sum(1 for d in detections if d.fm_call.success)
    fm_failed = sum(1 for d in detections if not d.fm_call.success)

    logger.info(
        "Video scan: %d frames, %d plates, %d FM calls (%d ok, %d failed) in %.3fs",
        frames_processed,
        len(detections),
        len(detections),
        fm_succeeded,
        fm_failed,
        elapsed,
    )

    return VideoScanResponse(
        total_frames=frames_processed,
        processing_time_seconds=round(elapsed, 3),
        plates_detected=len(detections),
        fm_calls_made=len(detections),
        fm_calls_succeeded=fm_succeeded,
        fm_calls_failed=fm_failed,
        detections=detections,
    )


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up FileMaker session on application shutdown."""
    if hasattr(app.state, "fm_client"):
        await app.state.fm_client.close()
        logger.info("FileMaker client closed on shutdown")
