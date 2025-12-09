"""
FastAPI wrapper around the FastALPR pipeline.

POST /detect accepts an image file (multipart/form-data) and returns detections plus OCR text.
"""

import logging
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


app = FastAPI(title="FastALPR API", version="0.1.0")
logging.basicConfig(level=logging.INFO)
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
