"""
transcriber.py — Module de transcription VidTXT

Gère les tâches de transcription via faster-whisper.
Chaque tâche tourne dans un thread séparé et communique
avec le WebSocket via une asyncio.Queue thread-safe.
"""

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Utilitaires de formatage ──────────────────────────────────────────────────

def fmt_srt(seconds: float) -> str:
    """Convertit des secondes en timestamp SRT : HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_display(seconds: float) -> str:
    """Convertit des secondes en timestamp lisible : [HH:MM:SS]"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"[{h:02d}:{m:02d}:{s:02d}]"


# ── Modèle de données d'une tâche ─────────────────────────────────────────────

@dataclass
class TaskInfo:
    task_id:    str
    file_path:  str
    model_name: str
    language:   Optional[str]          # None = détection automatique
    use_gpu:    bool
    output_dir: str
    loop:       asyncio.AbstractEventLoop
    queue:      asyncio.Queue
    status:     str = "pending"        # pending | running | done | error
    segments:   List[Dict] = field(default_factory=list)


# ── Gestionnaire des tâches ───────────────────────────────────────────────────

class TranscriptionManager:
    """
    Gère la création, l'exécution et le suivi des tâches de transcription.

    Chaque tâche possède une asyncio.Queue pour la communication
    thread-safe entre le worker (thread) et le WebSocket (event loop).
    """

    def __init__(self) -> None:
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = threading.Lock()

    # ── API publique ──────────────────────────────────────────────────────────

    def create_task(
        self,
        *,
        task_id:    str,
        file_path:  str,
        model:      str,
        language:   Optional[str],
        use_gpu:    bool,
        output_dir: str,
        loop:       asyncio.AbstractEventLoop,
    ) -> TaskInfo:
        """Crée et enregistre une nouvelle tâche."""
        task = TaskInfo(
            task_id    = task_id,
            file_path  = file_path,
            model_name = model,
            language   = language,
            use_gpu    = use_gpu,
            output_dir = output_dir,
            loop       = loop,
            queue      = asyncio.Queue(),
        )
        with self._lock:
            self._tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[TaskInfo]:
        return self._tasks.get(task_id)

    def get_status(self, task_id: str) -> Optional[Dict]:
        task = self._tasks.get(task_id)
        return {"task_id": task_id, "status": task.status} if task else None

    def run_transcription(self, task_id: str) -> None:
        """
        Lance la transcription — doit être appelé depuis un thread séparé
        (via loop.run_in_executor).
        """
        task = self._tasks.get(task_id)
        if not task:
            logger.error(f"Tâche {task_id!r} introuvable.")
            return

        task.status = "running"

        try:
            self._execute(task)
        except Exception as exc:
            logger.error(f"[{task_id}] Erreur inattendue : {exc}", exc_info=True)
            task.status = "error"
            self._put(task, {"type": "error", "message": f"Erreur : {exc}"})
        finally:
            # Suppression du fichier uploadé (temporaire)
            try:
                Path(task.file_path).unlink(missing_ok=True)
            except Exception:
                pass

    def cleanup(self) -> None:
        """Libère les ressources (appelé à l'arrêt du serveur)."""
        self._tasks.clear()

    # ── Logique interne ───────────────────────────────────────────────────────

    def _put(self, task: TaskInfo, msg: Dict) -> None:
        """
        Envoie un message dans la queue asyncio depuis un thread quelconque.
        Utilise run_coroutine_threadsafe pour la sécurité thread.
        """
        asyncio.run_coroutine_threadsafe(task.queue.put(msg), task.loop)

    def _execute(self, task: TaskInfo) -> None:
        """Logique principale de transcription."""
        # Forcer le mode CPU pur — évite le chargement des DLL CUDA sur Windows
        if not task.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        from faster_whisper import WhisperModel  # import tardif pour ne pas bloquer au démarrage

        # ── Chargement du modèle ──────────────────────────────────────────────
        self._put(task, {
            "type":    "status",
            "message": f"Chargement du modèle « {task.model_name} »…",
        })

        device       = "cuda" if task.use_gpu else "cpu"
        compute_type = "float16" if task.use_gpu else "float32"

        try:
            model = WhisperModel(
                task.model_name,
                device       = device,
                compute_type = compute_type,
                cpu_threads  = 4,
                num_workers  = 1,
            )
        except Exception as e:
            if task.use_gpu:
                # Fallback CPU si CUDA indispo
                self._put(task, {"type": "status", "message": "GPU indisponible, bascule sur CPU…"})
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                model = WhisperModel(
                    task.model_name,
                    device       = "cpu",
                    compute_type = "float32",
                    cpu_threads  = 4,
                    num_workers  = 1,
                )
            else:
                raise

        # ── Lancement de la transcription ─────────────────────────────────────
        self._put(task, {
            "type":    "status",
            "message": "Analyse et transcription en cours…",
        })

        segments_iter, info = model.transcribe(
            task.file_path,
            language       = task.language,    # None = détection automatique
            beam_size      = 5,
            vad_filter     = True,             # Voice Activity Detection
            vad_parameters = {"min_silence_duration_ms": 500},
        )

        # Durée totale pour le calcul de progression
        duration = getattr(info, "duration", 0.0) or 0.0

        self._put(task, {
            "type":     "language",
            "language": info.language,
            "duration": round(duration, 1),
        })

        # ── Itération sur les segments ────────────────────────────────────────
        segments_list: List[Dict] = []

        try:
            for segment in segments_iter:
                seg = {
                    "start": round(segment.start, 2),
                    "end":   round(segment.end,   2),
                    "text":  segment.text.strip(),
                }
                segments_list.append(seg)
                task.segments.append(seg)

                # Progression (0–99 % pendant la transcription, 100 % à la fin)
                progress = (
                    min(99, int((segment.end / duration) * 100))
                    if duration > 0 else 0
                )

                self._put(task, {
                    "type":              "segment",
                    "segment":           seg,
                    "progress":          progress,
                    "timestamp_display": fmt_display(segment.start),
                })
        except RuntimeError as e:
            if "cuda" in str(e).lower() or "cublas" in str(e).lower() or "dll" in str(e).lower():
                raise RuntimeError(
                    f"Erreur GPU/CUDA : {e}\n\n"
                    "Désactivez l'option GPU dans l'interface ou installez CUDA 12.x."
                )
            raise

        # ── Génération des fichiers de sortie ─────────────────────────────────
        out_dir = Path(task.output_dir)

        # Fichier TXT — une ligne par segment avec timestamp lisible
        txt_path = out_dir / f"{task.task_id}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for seg in segments_list:
                f.write(f"{fmt_display(seg['start'])} {seg['text']}\n")

        # Fichier SRT — format sous-titres standard
        srt_path = out_dir / f"{task.task_id}.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments_list, 1):
                f.write(f"{i}\n")
                f.write(f"{fmt_srt(seg['start'])} --> {fmt_srt(seg['end'])}\n")
                f.write(f"{seg['text']}\n\n")

        # ── Fin ───────────────────────────────────────────────────────────────
        task.status = "done"
        self._put(task, {
            "type":          "done",
            "progress":      100,
            "message":       f"Transcription terminée — {len(segments_list)} segment(s).",
            "segment_count": len(segments_list),
        })

        logger.info(f"[{task.task_id}] Terminé : {len(segments_list)} segments.")
