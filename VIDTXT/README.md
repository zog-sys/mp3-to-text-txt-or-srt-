# VidTXT — Transcription Vidéo/Audio Locale

Transcription automatique de fichiers **MP4** et **MP3** via [faster-whisper](https://github.com/SYSTRAN/faster-whisper), entièrement en local — zéro API cloud, zéro connexion externe.

---

## Installation en 3 étapes

**1. Créer et activer l'environnement virtuel**

```bash
# Linux / macOS
python -m venv .venv && source .venv/bin/activate

# Windows
python -m venv .venv && .venv\Scripts\activate
```

**2. Installer les dépendances**

```bash
pip install -r requirements.txt
```

**3. Lancer l'application**

```bash
cd backend && python main.py
```

Ouvrir dans le navigateur : **http://localhost:8000**

---

## GPU (optionnel, CUDA)

Pour activer l'accélération GPU, installer PyTorch avec support CUDA **avant** d'installer les dépendances :

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Puis cocher l'option **"Utiliser le GPU"** dans l'interface.

---

## Modèles disponibles

| Modèle     | Taille   | Vitesse     | Précision   |
|------------|----------|-------------|-------------|
| `tiny`     | ~39 Mo   | Très rapide | Basique     |
| `base`     | ~74 Mo   | Rapide      | Correcte    |
| `small`    | ~244 Mo  | Moyen       | Bonne       |
| `medium`   | ~769 Mo  | Lent        | Très bonne  |
| `large-v3` | ~3 Go    | Très lent   | Excellente  |

Les modèles sont téléchargés automatiquement au premier lancement et mis en cache localement.

---

## Structure du projet

```
VIDTXT/
├── backend/
│   ├── main.py          # Serveur FastAPI
│   └── transcriber.py   # Module de transcription
├── frontend/
│   └── index.html       # Interface utilisateur
├── uploads/             # Fichiers uploadés (temporaires)
├── outputs/             # Transcriptions générées (.txt / .srt)
├── requirements.txt
└── README.md
```
"# mp3-to-text-txt-or-srt-"  
