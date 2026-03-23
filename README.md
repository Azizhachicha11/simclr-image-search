# 🔍 SimCLR Image Search Engine

Moteur de recherche d'images par **similarité visuelle** utilisant le **Contrastive Learning (SimCLR)**.  
Upload une image → le modèle ResNet-18 encode un vecteur 512d → FAISS trouve les images les plus similaires.

---

## 📐 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        UTILISATEUR                               │
│                   Upload une image query                         │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                   FRONTEND (GitHub Pages)                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                │
│  │ index.html │  │  style.css │  │   app.js   │                │
│  └────────────┘  └────────────┘  └─────┬──────┘                │
└────────────────────────────────────────┬────────────────────────┘
                                         │ POST /search
                                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                   BACKEND (Render / Docker)                      │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │  api.py │→ │ model.py │→ │ SimCLR   │→ │ FAISS search   │  │
│  │ FastAPI │  │ ResNet18 │  │ 512d vec │  │ Top-K results  │  │
│  └─────────┘  └──────────┘  └──────────┘  └────────────────┘  │
│                                                                  │
│  Fichiers chargés au démarrage :                                │
│  • checkpoints/simclr_best.pt  (42 MB, depuis HF Hub)          │
│  • index/faiss.index           (10 MB, depuis HF Hub)          │
│  • index/labels.npy                                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Prérequis

| Outil         | Version   | Obligatoire |
|---------------|-----------|:-----------:|
| Python        | 3.12+     | ✅          |
| pip           | 23+       | ✅          |
| Docker        | 24+       | Optionnel   |
| CUDA          | 12.x      | Optionnel   |
| Git           | 2.40+     | ✅          |

---

## 🚀 Installation & Lancement en local

### 1. Cloner le repo

```bash
git clone https://github.com/<votre-username>/simclr-image-search.git
cd simclr-image-search
```

### 2. Installer les dépendances

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r backend/requirements.txt
```

### 3. Télécharger le modèle depuis Hugging Face

```bash
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('index', exist_ok=True)
hf_hub_download(repo_id='<votre-username>/simclr-stl10', filename='simclr_best.pt', local_dir='checkpoints')
hf_hub_download(repo_id='<votre-username>/simclr-stl10', filename='faiss.index', local_dir='index')
hf_hub_download(repo_id='<votre-username>/simclr-stl10', filename='labels.npy', local_dir='index')
"
```

### 4. Lancer l'API

```bash
cd backend
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 5. Ouvrir le frontend

Ouvrir `frontend/index.html` dans un navigateur, ou :

```bash
cd frontend && python -m http.server 3000
```

Puis visiter `http://localhost:3000`.

---

## 🐳 Lancement avec Docker

```bash
# Construire et lancer les deux services
docker compose up --build

# Frontend → http://localhost:80
# Backend  → http://localhost:8000
```

Pour arrêter :
```bash
docker compose down
```

---

## 🧪 Lancer les tests

```bash
pytest tests/ -v
```

| Test                          | Vérifie                                   |
|-------------------------------|-------------------------------------------|
| `test_health`                 | `/health` retourne 200                    |
| `test_search_valid_image`     | Image valide → 5 résultats               |
| `test_search_invalid_file`    | Fichier invalide → erreur 400             |
| `test_search_top_k`           | `top_k=3` → exactement 3 résultats       |
| `test_model_loads`            | Le modèle se charge sans erreur           |
| `test_encode_shape`           | `encode()` → shape `(1, 512)`             |
| `test_encode_normalized`      | Vecteurs normalisés L2 (norme ≈ 1.0)     |

---

## 🔄 CI/CD — Configuration

### Secrets GitHub à ajouter

Allez dans **Settings → Secrets and variables → Actions → New repository secret** et ajoutez :

| Secret               | Valeur                                                                 |
|----------------------|------------------------------------------------------------------------|
| `RENDER_DEPLOY_HOOK` | URL du deploy hook Render (Settings → Deploys → Deploy Hook)           |
| `API_URL`            | URL publique de l'API Render (ex: `https://simclr-api.onrender.com`)   |
| `HF_TOKEN`           | Token Hugging Face (Settings → Access Tokens → New token, scope: read) |

### Comment obtenir chaque secret

**1. `RENDER_DEPLOY_HOOK`**
1. Connectez-vous sur [render.com](https://render.com)
2. Créez un **Web Service** avec Docker
3. Dans **Settings → Deploy Hook**, copiez l'URL

**2. `API_URL`**
1. Après le premier déploiement Render, copiez l'URL publique
2. Format : `https://votre-service.onrender.com`

**3. `HF_TOKEN`**
1. Connectez-vous sur [huggingface.co](https://huggingface.co)
2. **Settings → Access Tokens → New token**
3. Scope : `read` (suffisant pour télécharger le modèle)

### Activer GitHub Pages
1. **Settings → Pages → Source** : sélectionnez **GitHub Actions**

---

## 🔌 Endpoints API

| Méthode | Endpoint  | Description                                | Paramètres                            |
|---------|-----------|--------------------------------------------|---------------------------------------|
| `GET`   | `/health` | Statut de l'API                            | —                                     |
| `POST`  | `/search` | Recherche d'images similaires              | `image` (file), `top_k` (int, opt.)   |

### Exemple avec curl

```bash
curl -X POST http://localhost:8000/search \
  -F "image=@photo.jpg" \
  -F "top_k=5"
```

### Réponse JSON

```json
{
  "results": [
    {"rank": 1, "class": "horse", "similarity": 0.8934, "index": 142},
    {"rank": 2, "class": "horse", "similarity": 0.8521, "index": 87},
    ...
  ],
  "total_indexed": 5000
}
```

---

## 📂 Structure du projet

```
simclr-image-search/
├── .github/workflows/
│   ├── test.yml              ← CI : tests automatiques
│   └── deploy.yml            ← CD : déploiement Render + GitHub Pages
├── backend/
│   ├── api.py                ← API FastAPI
│   ├── model.py              ← SimCLR (ResNet18 + Projection Head)
│   ├── dataset.py            ← Augmentations SimCLR + DataLoaders
│   ├── train.py              ← Boucle d'entraînement NT-Xent
│   ├── index.py              ← Construction de l'index FAISS
│   ├── search.py             ← Recherche CLI
│   └── requirements.txt      ← Dépendances Python
├── frontend/
│   ├── index.html            ← Interface web
│   ├── style.css             ← Design dark mode
│   └── app.js                ← Logique JS (upload + fetch)
├── tests/
│   ├── test_api.py           ← Tests d'intégration API
│   └── test_model.py         ← Tests unitaires modèle
├── checkpoints/              ← (gitignored) simclr_best.pt
├── index/                    ← (gitignored) faiss.index + labels.npy
├── Dockerfile                ← Container API
├── docker-compose.yml        ← Frontend + Backend
├── .gitignore
└── README.md
```

---

## 🔀 Workflow Git

```
feature branch ──→ Pull Request ──→ CI (tests) ──→ Merge → main
                                                         │
                                                         ▼
                                              CD (deploy.yml)
                                              ├─→ Render (backend)
                                              └─→ GitHub Pages (frontend)
```

1. Créez une branche : `git checkout -b feat/new-feature`
2. Commitez vos changements : `git commit -m "feat: add ..."`
3. Ouvrez une Pull Request → les tests CI tournent automatiquement
4. Après merge sur `main` → déploiement automatique

---

## 📄 Licence

MIT
