import os
import shutil
import subprocess
import threading
import uuid
import zipfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from faster_whisper import WhisperModel

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
WORK_DIR = BASE_DIR / "work"

for p in (UPLOAD_DIR, OUTPUT_DIR, WORK_DIR):
    p.mkdir(exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://web-seven-eta-56.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}

WHISPER_MODEL = "tiny"
DETECT_EVERY = 6
DETECT_WIDTH = 480


def ffmpeg():
    return os.getenv("FFMPEG_BIN") or imageio_ffmpeg.get_ffmpeg_exe()


def run(cmd):
    subprocess.run(cmd, check=True)


def update_job(job_id, **kwargs):
    jobs[job_id].update(kwargs)

def public_base_url():
    return os.getenv("PUBLIC_BASE_URL", "https://auto-reels-api-production.up.railway.app").rstrip("/")


def probe_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return width, height, fps, frame_count


def smooth_centers(sample_indices, centers, total_frames, fallback_center):
    if total_frames <= 0:
        return [fallback_center]
    if not sample_indices:
        return [fallback_center] * total_frames

    sample_indices = np.array(sample_indices, dtype=np.int32)
    centers = np.array(centers, dtype=np.float32)

    if len(sample_indices) == 1:
        return [int(centers[0])] * total_frames

    full_idx = np.arange(total_frames, dtype=np.int32)
    interp = np.interp(full_idx, sample_indices, centers)

    def ema_pass(arr, alpha=0.15):
        out = arr.copy()
        for i in range(1, len(out)):
            out[i] = (1 - alpha) * out[i - 1] + alpha * out[i]
        return out

    smooth = interp
    for _ in range(2):
        smooth = ema_pass(smooth, alpha=0.18)
        smooth = ema_pass(smooth[::-1], alpha=0.18)[::-1]

    return [int(x) for x in smooth]


def detect_centers_mediapipe(video_path: Path, crop_w: int):
    width, height, fps, frame_count = probe_video(video_path)
    fallback_center = width // 2

    face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.55,
    )

    cap = cv2.VideoCapture(str(video_path))
    sample_indices = []
    centers = []
    idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if idx % DETECT_EVERY == 0:
                h, w = frame.shape[:2]
                scale = DETECT_WIDTH / max(w, 1)
                small = cv2.resize(frame, (int(w * scale), int(h * scale)))
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                result = face_detection.process(rgb)

                detected_center = fallback_center
                best_score = -1.0

                if result.detections:
                    for det in result.detections:
                        score = float(det.score[0]) if det.score else 0.0
                        bbox = det.location_data.relative_bounding_box
                        cx_small = (bbox.xmin + bbox.width / 2.0) * small.shape[1]
                        cx = cx_small / scale
                        if score > best_score:
                            best_score = score
                            detected_center = int(cx)

                min_c = crop_w // 2
                max_c = width - crop_w // 2
                if min_c > max_c:
                    min_c = max_c = fallback_center
                detected_center = max(min_c, min(max_c, detected_center))

                sample_indices.append(idx)
                centers.append(detected_center)

            idx += 1
    finally:
        cap.release()
        face_detection.close()

    return smooth_centers(sample_indices, centers, frame_count, fallback_center)


def trim_video(src: Path, dst: Path, trim_start: float, trim_end: float):
    cmd = [ffmpeg(), "-y"]
    if trim_start > 0:
        cmd += ["-ss", f"{trim_start:.3f}"]
    cmd += ["-i", str(src)]
    if trim_end > 0 and trim_end > trim_start:
        cmd += ["-t", f"{(trim_end - trim_start):.3f}"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-c:a", "aac", str(dst)]
    run(cmd)


def render_vertical_video(video_path: Path, out_path: Path, face_tracking: bool, export_preset: str):
    width, height, fps, frame_count = probe_video(video_path)
    target_aspect = 9 / 16
    crop_w = min(width, int(round(height * target_aspect)))

    if export_preset == "hq1080":
        target_w, target_h = 1080, 1920
    else:
        target_w, target_h = 720, 1280

    if face_tracking:
        crop_positions = detect_centers_mediapipe(video_path, crop_w)
    else:
        crop_positions = [width // 2] * max(frame_count, 1)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео для рендера")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (target_w, target_h))
    if not writer.isOpened():
        raise RuntimeError("Не удалось открыть writer для вертикального видео")

    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            center_x = crop_positions[min(idx, len(crop_positions) - 1)] if crop_positions else w // 2
            x1 = max(0, min(w - crop_w, center_x - crop_w // 2))
            crop = frame[:, x1:x1 + crop_w]
            resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            writer.write(resized)
            idx += 1
    finally:
        cap.release()
        writer.release()


def sec_to_ass(ts: float):
    hours = int(ts // 3600)
    minutes = int((ts % 3600) // 60)
    seconds = ts % 60
    return f"{hours}:{minutes:02d}:{seconds:05.2f}"


def ass_color(hex_rgb: str) -> str:
    hex_rgb = hex_rgb.lstrip("#")
    if len(hex_rgb) != 6:
        return "&H00FFFFFF"
    rr, gg, bb = hex_rgb[0:2], hex_rgb[2:4], hex_rgb[4:6]
    return f"&H00{bb}{gg}{rr}"


def make_ass(video_path: Path, ass_path: Path, font_name: str, font_size: int, subtitle_position: str, color: str):
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8", cpu_threads=2)
    segments, _ = model.transcribe(
        str(video_path),
        language="ru",
        vad_filter=True,
        word_timestamps=False,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
    )
    segments = list(segments)

    alignment = 2 if subtitle_position == "bottom" else 8
    margin_v = 170 if subtitle_position == "bottom" else 160
    primary = ass_color(color)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Default,{font_name},{font_size},{primary},&H000000FF,&H00101010,&H78000000,1,0,0,0,100,100,0,0,1,3.2,0,{alignment},60,60,{margin_v},1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""

    def clean_text(text: str) -> str:
        return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header)
        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            line = clean_text(text)
            f.write(
                f"Dialogue: 0,{sec_to_ass(seg.start)},{sec_to_ass(seg.end)},Default,,0,0,0,,{line}\n"
            )


def ffmpeg_sub_path(path: Path) -> str:
    p = path.resolve().as_posix()
    p = p.replace("\\", r"\\").replace(":", r"\:").replace("'", r"\'").replace(",", r"\,")
    return p


def burn_ass_and_mux(video_no_audio: Path, source_video: Path, ass_path: Path, out_path: Path, export_preset: str):
    crf = "18" if export_preset == "hq1080" else "21"
    cmd = [
        ffmpeg(),
        "-y",
        "-threads", "1",
        "-i", str(video_no_audio),
        "-i", str(source_video),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-filter_threads", "1",
        "-vf", f"ass={ffmpeg_sub_path(ass_path)},format=yuv420p",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", crf,
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        "-b:a", "160k",
        "-shortest",
        str(out_path),
    ]
    run(cmd)


def mux_audio(video_no_audio: Path, source_video: Path, out_path: Path, export_preset: str):
    cmd = [
        ffmpeg(),
        "-y",
        "-i", str(video_no_audio),
        "-i", str(source_video),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "160k",
        "-movflags", "+faststart",
        "-shortest",
        str(out_path),
    ]
    run(cmd)


def split_video(input_path: Path, out_dir: Path, split_mode: str):
    segment_time = 59 if split_mode == "59" else 179
    pattern = out_dir / "part_%03d.mp4"
    cmd = [
        ffmpeg(),
        "-y",
        "-i", str(input_path),
        "-force_key_frames", f"expr:gte(t,n_forced*{segment_time})",
        "-f", "segment",
        "-segment_time", str(segment_time),
        "-reset_timestamps", "1",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "20",
        "-c:a", "aac",
        str(pattern),
    ]
    run(cmd)
    return sorted(out_dir.glob("part_*.mp4"))


def zip_files(files, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)


def process_job(job_id: str, input_path: Path, options: dict):
    try:
        workdir = WORK_DIR / job_id
        workdir.mkdir(exist_ok=True)

        trimmed_path = workdir / "trimmed.mp4"
        visual_path = workdir / "visual.mp4"
        ass_path = workdir / "subs.ass"
        final_path = OUTPUT_DIR / f"{job_id}_result.mp4"
        zip_path = OUTPUT_DIR / f"{job_id}_split.zip"

        update_job(job_id, status="processing", progress=5, message="Обрезаю исходник")
        trim_video(input_path, trimmed_path, options["trim_start"], options["trim_end"])

        if options["auto_vertical"]:
            update_job(job_id, progress=20, message="Считаю плавный автокадр")
            render_vertical_video(trimmed_path, visual_path, options["face_tracking"], options["export_preset"])
        else:
            shutil.copy(trimmed_path, visual_path)

        if options["subtitles"]:
            update_job(job_id, progress=65, message="Делаю русские субтитры")
            make_ass(
                trimmed_path,
                ass_path,
                options["subtitle_font"],
                options["subtitle_size"],
                options["subtitle_position"],
                options["subtitle_color"],
            )
            update_job(job_id, progress=82, message="Собираю финальный экспорт")
            burn_ass_and_mux(visual_path, trimmed_path, ass_path, final_path, options["export_preset"])
        else:
            update_job(job_id, progress=82, message="Собираю финальный экспорт")
            mux_audio(visual_path, trimmed_path, final_path, options["export_preset"])

        if options["split_mode"] in {"59", "179"}:
            update_job(job_id, progress=92, message="Нарезаю ролик на части")
            split_dir = workdir / "split"
            split_dir.mkdir(exist_ok=True)
            files = split_video(final_path, split_dir, options["split_mode"])
            zip_files(files, zip_path)
            download_url = f"{public_base_url()}/api/download/{job_id}?kind=zip"
            output_filename = zip_path.name
        else:
            download_url = f"{public_base_url()}/api/download/{job_id}?kind=video"
            output_filename = final_path.name

        update_job(
            job_id,
            status="done",
            progress=100,
            message="Готово",
            download_url=download_url,
            output_filename=output_filename,
        )
    except Exception as e:
        update_job(job_id, status="error", progress=100, message="Ошибка обработки", error=str(e))


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    auto_vertical: str = Form("true"),
    face_tracking: str = Form("true"),
    subtitles: str = Form("true"),
    subtitle_size: int = Form(36),
    subtitle_position: str = Form("bottom"),
    subtitle_font: str = Form("Helvetica Neue"),
    subtitle_color: str = Form("#FFFFFF"),
    trim_start: float = Form(0),
    trim_end: float = Form(0),
    export_preset: str = Form("fast720"),
    split_mode: str = Form("none"),
):
    job_id = str(uuid.uuid4())
    input_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Файл загружен",
        "error": None,
        "download_url": None,
        "output_filename": None,
        "original_filename": file.filename,
    }

    options = {
        "auto_vertical": auto_vertical.lower() == "true",
        "face_tracking": face_tracking.lower() == "true",
        "subtitles": subtitles.lower() == "true",
        "subtitle_size": subtitle_size,
        "subtitle_position": subtitle_position,
        "subtitle_font": subtitle_font,
        "subtitle_color": subtitle_color,
        "trim_start": trim_start,
        "trim_end": trim_end,
        "export_preset": export_preset,
        "split_mode": split_mode,
    }

    threading.Thread(target=process_job, args=(job_id, input_path, options), daemon=True).start()
    return JSONResponse(jobs[job_id])


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse(jobs[job_id])


@app.get("/api/download/{job_id}")
async def download_file(job_id: str, kind: str = "video"):
    if kind == "zip":
        path = OUTPUT_DIR / f"{job_id}_split.zip"
        if not path.exists():
            return JSONResponse({"error": "File not found"}, status_code=404)
        return FileResponse(path, media_type="application/zip", filename=path.name)

    path = OUTPUT_DIR / f"{job_id}_result.mp4"
    if not path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/health")
async def health():
    return {"ok": True}
