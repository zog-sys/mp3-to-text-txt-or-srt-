"""
main.py — Serveur principal VidTXT

Lance FastAPI, sert le frontend statique et expose les routes :
  GET  /                          → index.html
  POST /api/upload                → reçoit le fichier, crée la tâche
  WS   /ws/{task_id}              → flux de progression temps réel
  GET  /api/download/{id}/{fmt}   → télécharge .txt ou .srt
  GET  /api/status/{task_id}      → état de la tâche

Lancement : cd backend && python main.py
"""

import asyncio
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Ajout du répertoire backend au path pour l'import relatif
sys.path.insert(0, str(Path(__file__).parent))
from transcriber import TranscriptionManager


# ── Chemins ───────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).parent.parent   # racine du projet
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR   = BASE_DIR / "uploads"
OUTPUT_DIR   = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Gestionnaire de tâches (singleton) ───────────────────────────────────────

transcription_manager = TranscriptionManager()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VidTXT démarré → http://localhost:8000")
    yield
    transcription_manager.cleanup()
    logger.info("VidTXT arrêté.")


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "VidTXT",
    description = "Transcription vidéo/audio locale via faster-whisper",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Fichiers statiques — le dossier frontend/ est accessible sous /static
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Sert l'interface utilisateur principale."""
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.post("/api/upload")
async def upload_file(
    file:        UploadFile = File(...),
    model:       str        = Form(default="base"),
    language:    str        = Form(default="auto"),
    use_gpu_str: str        = Form(default="false"),
):
    """
    Reçoit le fichier audio/vidéo, le sauvegarde temporairement
    et crée la tâche de transcription.

    Retourne : {"task_id": str, "filename": str}
    """
    use_gpu = use_gpu_str.lower() in ("true", "1", "yes", "on")

    # ── Validation de l'extension ─────────────────────────────────────────────
    allowed_ext = {".mp4", ".mp3"}
    file_ext    = Path(file.filename or "").suffix.lower()

    if file_ext not in allowed_ext:
        raise HTTPException(
            status_code = 400,
            detail      = "Format non supporté. Utilisez un fichier .mp4 ou .mp3.",
        )

    # ── Validation du modèle ──────────────────────────────────────────────────
    allowed_models = {"tiny", "base", "small", "medium", "large-v3"}
    if model not in allowed_models:
        raise HTTPException(
            status_code = 400,
            detail      = f"Modèle invalide : {model!r}. Choix possibles : {', '.join(sorted(allowed_models))}",
        )

    # ── Sauvegarde avec limite de taille (500 Mo) ─────────────────────────────
    MAX_BYTES = 500 * 1024 * 1024  # 500 Mo
    task_id   = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{task_id}{file_ext}"

    total = 0
    try:
        with open(save_path, "wb") as f:
            while chunk := await file.read(1_048_576):  # lecture par blocs de 1 Mo
                total += len(chunk)
                if total > MAX_BYTES:
                    raise HTTPException(
                        status_code = 413,
                        detail      = "Fichier trop volumineux (maximum 500 Mo).",
                    )
                f.write(chunk)
    except HTTPException:
        save_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        save_path.unlink(missing_ok=True)
        logger.error(f"Erreur lors de la sauvegarde du fichier : {exc}")
        raise HTTPException(status_code=500, detail="Erreur lors de la réception du fichier.")

    # ── Création de la tâche ──────────────────────────────────────────────────
    loop = asyncio.get_running_loop()
    transcription_manager.create_task(
        task_id    = task_id,
        file_path  = str(save_path),
        model      = model,
        language   = None if language == "auto" else language,
        use_gpu    = use_gpu,
        output_dir = str(OUTPUT_DIR),
        loop       = loop,
    )

    logger.info(f"Upload reçu : {file.filename!r} ({total / 1_048_576:.1f} Mo) → tâche {task_id}")
    return JSONResponse({"task_id": task_id, "filename": file.filename})


@app.websocket("/ws/{task_id}")
async def websocket_transcription(websocket: WebSocket, task_id: str):
    """
    WebSocket de suivi de progression en temps réel.

    Flux de messages JSON envoyés au client :
      {"type": "status",   "message": str}
      {"type": "language", "language": str, "duration": float}
      {"type": "segment",  "segment": {start, end, text}, "progress": int, "timestamp_display": str}
      {"type": "done",     "progress": 100, "message": str, "segment_count": int}
      {"type": "error",    "message": str}
    """
    await websocket.accept()

    task = transcription_manager.get_task(task_id)
    if not task:
        await websocket.send_json({"type": "error", "message": "Tâche introuvable."})
        await websocket.close()
        return

    loop = asyncio.get_running_loop()

    # Lance la transcription dans le pool de threads (non bloquant pour le loop)
    executor_future = loop.run_in_executor(
        None, transcription_manager.run_transcription, task_id
    )

    try:
        while True:
            # Attente du prochain message (timeout 10 min)
            try:
                msg = await asyncio.wait_for(task.queue.get(), timeout=600.0)
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type":    "error",
                    "message": "Délai d'attente dépassé (10 min). La transcription a peut-être planté.",
                })
                break

            await websocket.send_json(msg)

            # Fin de la tâche (succès ou erreur)
            if msg.get("type") in ("done", "error"):
                break

    except WebSocketDisconnect:
        logger.info(f"[{task_id}] Client WebSocket déconnecté.")

    except Exception as exc:
        logger.error(f"[{task_id}] Erreur WebSocket : {exc}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass

    finally:
        # Attente courte pour que le thread se termine proprement
        try:
            await asyncio.wait_for(asyncio.shield(executor_future), timeout=3.0)
        except (asyncio.TimeoutError, Exception):
            pass
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/api/download/{task_id}/{fmt}")
async def download_result(task_id: str, fmt: str):
    """
    Télécharge le fichier de transcription généré.
    fmt : "txt" ou "srt"
    """
    if fmt not in ("txt", "srt"):
        raise HTTPException(
            status_code = 400,
            detail      = "Format invalide. Utilisez 'txt' ou 'srt'.",
        )

    # Sécurité : s'assurer que task_id ne contient pas de traversée de chemin
    if "/" in task_id or "\\" in task_id or ".." in task_id:
        raise HTTPException(status_code=400, detail="Identifiant de tâche invalide.")

    path = OUTPUT_DIR / f"{task_id}.{fmt}"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Fichier non trouvé.")

    media_types = {
        "txt": "text/plain; charset=utf-8",
        "srt": "application/x-subrip",
    }
    return FileResponse(
        str(path),
        media_type = media_types[fmt],
        filename   = f"transcription.{fmt}",
    )


@app.get("/api/status/{task_id}")
async def task_status(task_id: str):
    """Retourne l'état courant d'une tâche."""
    status = transcription_manager.get_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Tâche introuvable.")
    return status


# ── Lancement direct ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host      = "0.0.0.0",
        port      = 8000,
        reload    = False,
        log_level = "info",
    )
