# 📋 Documentation Complète — CI/CD & Déploiement
## Projet : SimCLR Image Search Engine
### Date : 23 Mars 2026

---

## 🎯 Objectif

Prendre un projet de **moteur de recherche d'images par similarité visuelle** (SimCLR + FAISS),
qui fonctionnait uniquement en local, et le déployer entièrement sur Internet avec :
- Un pipeline **CI/CD automatisé** (GitHub Actions)
- Un **backend API** en production (Render)
- Un **frontend** accessible publiquement (GitHub Pages)
- Un **stockage de modèles** sur Hugging Face Hub

---

## 🏗️ Architecture Finale

```
INTERNET
   │
   ├─► https://azizhachicha11.github.io/simclr-image-search/
   │        │  (GitHub Pages — Frontend statique)
   │        │  HTML + CSS + JavaScript
   │        │
   │        │  POST /search (fetch)
   │        ▼
   └─► https://simclr-api.onrender.com
            │  (Render — Backend Docker)
            │  FastAPI + ResNet18 + FAISS
            │
            │  Au démarrage : télécharge les fichiers
            ▼
        https://huggingface.co/ElBOOH55/simclr-stl10
            │  (Hugging Face Hub — Stockage de modèles)
            ├── simclr_best.pt   (42 MB — Poids SimCLR)
            ├── faiss.index      (10 MB — Index de recherche)
            └── labels.npy       (Labels STL-10)
```

---

## 📁 Fichiers Créés (9 nouveaux fichiers)

### 1. `backend/requirements.txt`
Liste toutes les dépendances Python avec versions fixées :
- `torch==2.2.2` / `torchvision==0.17.2` — Deep learning
- `faiss-cpu==1.8.0` — Recherche par similarité vectorielle
- `fastapi==0.110.0` / `uvicorn` — API REST
- `huggingface-hub==0.21.0` — Téléchargement modèles
- `pytest==8.0.0` / `httpx==0.27.0` — Tests

### 2. `tests/test_api.py`
4 tests d'intégration pour l'API FastAPI :
- `test_health()` — Vérifie que `/health` retourne 200
- `test_search_valid_image()` — Image valide → 5 résultats
- `test_search_invalid_file()` — Fichier invalide → erreur 400
- `test_search_top_k()` — Paramètre `top_k=3` → exactement 3 résultats

### 3. `tests/test_model.py`
3 tests unitaires pour le modèle SimCLR :
- `test_model_loads()` — Le modèle se charge sans erreur
- `test_encode_shape()` — La sortie fait bien (1, 512)
- `test_encode_normalized()` — Les vecteurs sont normalisés L2 (norme ≈ 1.0)

### 4. `.github/workflows/test.yml` — Pipeline CI
Déclenché sur chaque push/PR vers `main` :
1. Checkout du code
2. Installation Python 3.12
3. Installation des dépendances (`pip install -r backend/requirements.txt`)
4. Téléchargement de `simclr_best.pt` depuis HuggingFace (via `HF_TOKEN`)
5. Téléchargement de `faiss.index` + `labels.npy` (via `HF_TOKEN`)
6. Exécution de `pytest tests/ -v --tb=short`

### 5. `.github/workflows/deploy.yml` — Pipeline CD
Déclenché sur chaque push vers `main` (après tests réussis) :
- **Job 1 — Tests** : Répète les tests (même steps que test.yml)
- **Job 2 — Deploy Backend** : Envoie une requête HTTP au Deploy Hook Render → déclenche un redéploiement automatique
- **Job 3 — Deploy Frontend** :
  - `sed` remplace `http://localhost:8000` par l'URL Render dans `app.js`
  - Upload le dossier `frontend/` comme artifact GitHub Pages
  - Déploie sur GitHub Pages

### 6. `Dockerfile`
Containerise l'API FastAPI :
```
Base image : python:3.12-slim
WORKDIR    : /app
COPY       : backend/requirements.txt → installe les deps
COPY       : backend/ → copie le code Python
EXPOSE     : 8000
HEALTHCHECK: GET /health toutes les 30s
CMD        : uvicorn api:app --host 0.0.0.0 --port 8000
```

### 7. `docker-compose.yml`
Lance les deux services localement :
- `backend` — Construit depuis le Dockerfile, port 8000
- `frontend` — Image `nginx:alpine`, sert `./frontend/`, port 80

### 8. `.gitignore`
Exclut les fichiers lourds du repo Git :
- `checkpoints/` — Poids du modèle (42 MB)
- `index/` — FAISS index (10 MB)
- `data/` — Dataset STL-10 (2.5 GB)
- `__pycache__/`, `.env`, `.venv/`, `.pytest_cache/`

### 9. `frontend/` (3 fichiers)
Dossier statique déployé sur GitHub Pages :
- `index.html` — Structure HTML (chemins relatifs, pas `/static/`)
- `style.css` — Copie de `static/style.css`
- `app.js` — JavaScript avec `const API_URL = "http://localhost:8000"` qui est remplacé par CI/CD

---

## 🔧 Étapes Réalisées — Chronologie Complète

### PHASE 1 — Préparation du code (local)

**Étape 1 : Analyse du projet existant**
- Lecture de tous les fichiers : `model.py`, `train.py`, `dataset.py`, `index.py`, `search.py`, `app.py`
- Compréhension de l'architecture : Flask → ResNet18 → FAISS → JSON
- Identification des fichiers à NE PAS modifier (code métier existant)

**Étape 2 : Création des dépendances**
- Créé `backend/requirements.txt` avec toutes les versions fixées

**Étape 3 : Création des tests**
- Créé `tests/test_api.py` — 4 tests API
- Créé `tests/test_model.py` — 3 tests modèle

**Étape 4 : Création des workflows GitHub Actions**
- Créé `.github/workflows/test.yml` — CI automatique
- Créé `.github/workflows/deploy.yml` — CD automatique

**Étape 5 : Création du Dockerfile**
- Créé `Dockerfile` basé sur `python:3.12-slim`
- Créé `docker-compose.yml` pour lancement local

**Étape 6 : Création du .gitignore**
- Exclu les fichiers > 1 MB du repo Git

**Étape 7 : Création du README.md**
- Documentation complète du projet

**Étape 8 : Création de `frontend/`**
- Nécessaire car le projet utilise Flask (chemins `/static/`) incompatible avec GitHub Pages
- Adapté pour hosting statique (chemins relatifs)
- Ajout de `const API_URL = "http://localhost:8000"` pour le remplacement par sed

---

### PHASE 2 — Configuration Git

**Étape 9 : Initialisation Git**
```bash
git init
git config --global user.name "Azizhachicha11"
git config --global user.email "azizh@users.noreply.github.com"
git add .
git commit -m "feat: SimCLR image search engine with CI/CD pipeline"
```

---

### PHASE 3 — GitHub

**Étape 10 : Création du repo GitHub**
- Navigé vers github.com/new via le navigateur
- Repo créé : `Azizhachicha11/simclr-image-search` (Public)
- Pas de README initial (pour éviter les conflits)

**Étape 11 : Push du code**
```bash
git branch -M main
git remote add origin https://github.com/Azizhachicha11/simclr-image-search.git
git push -u origin main
```
→ 19 fichiers envoyés sur GitHub
→ GitHub Actions déclenché automatiquement (mais échoue car secrets manquants)

---

### PHASE 4 — Hugging Face Hub

**Étape 12 : Connexion HuggingFace**
- Utilisateur connecté sur huggingface.co

**Étape 13 : Création du repo modèle**
- Navigé vers huggingface.co/new-model
- Repo créé : `ElBOOH55/simclr-stl10` (Public)

**Étape 14 : Génération du token Read**
- Settings → Access Tokens → New token
- Nom : `github-actions`, Type : Read
- Token : `hf_****************************` (Read)
- **Utilisé comme secret `HF_TOKEN` dans GitHub Actions**

**Étape 15 : Génération du token Write (pour l'upload)**
- Nom : `upload-token`, Type : Write
- Token : `hf_****************************` (Write, local uniquement pour l'upload initial)

**Étape 16 : Upload des 3 fichiers lourds**
```python
from huggingface_hub import HfApi
api.upload_file('checkpoints/simclr_best.pt', 'simclr_best.pt', 'ElBOOH55/simclr-stl10')
api.upload_file('index/faiss.index',          'faiss.index',    'ElBOOH55/simclr-stl10')
api.upload_file('index/labels.npy',           'labels.npy',     'ElBOOH55/simclr-stl10')
```
✔ 42 MB + 10 MB + quelques KB uploadés sur HuggingFace

---

### PHASE 5 — Render (Backend)

**Étape 17 : Connexion Render**
- Connexion via Google OAuth (compte : azizhach98@gmail.com)

**Étape 18 : Création du Web Service**
- New → Web Service
- GitHub repo connecté : `Azizhachicha11/simclr-image-search`
- Nom : `simclr-api`
- Runtime : Docker (auto-détecté via Dockerfile)
- Plan : Free
- Déploiement initié

**Étape 19 : Récupération des infos Render**
- URL du service : `https://simclr-api.onrender.com`
- Deploy Hook : `https://api.render.com/deploy/srv-d70ocu49c44c73b50j30?key=X1sOVKfokIc`

---

### PHASE 6 — Secrets GitHub

**Étape 20 : Ajout des 3 secrets**
Navigé vers : Settings → Secrets and variables → Actions

| Secret | Valeur |
|--------|--------|
| `HF_TOKEN` | `hf_****************************` |
| `API_URL` | `https://simclr-api.onrender.com` |
| `RENDER_DEPLOY_HOOK` | `https://api.render.com/deploy/srv-***?key=***` |

---

### PHASE 7 — GitHub Pages

**Étape 21 : Activation de GitHub Pages**
- Settings → Pages → Source : **GitHub Actions**
- Message confirmé : "GitHub Pages source saved."

---

### PHASE 8 — Corrections & Debug

**Étape 22 : Fix repo HuggingFace dans test.yml**
- Problème : `repo_id='${{ github.repository_owner }}/simclr-stl10'` — variable non interpolée dans Python
- Fix : remplacé par la valeur en dur `'ElBOOH55/simclr-stl10'`
- Commit : `fix: use correct HF repo id`

**Étape 23 : Fix version faiss-cpu**
- Problème : `faiss-cpu==1.7.4` n'existe pas pour Python 3.12 sur GitHub Actions Ubuntu
- Fix : mis à jour vers `faiss-cpu==1.8.0`

**Étape 24 : Fix deploy.yml**
- Problème : le workflow appelait `test.yml` comme workflow réutilisable (`uses: ./.github/workflows/test.yml`) — cela nécessite le trigger `workflow_call` qui n'était pas défini
- Fix : les steps de test ont été inlinés directement dans `deploy.yml`

**Étape 25 : Création de frontend/**
- Problème : le dossier `frontend/` n'existait pas (le projet utilisait `static/` + `templates/` pour Flask)
- Fix : créé `frontend/index.html` (chemins relatifs), `frontend/style.css` (copie), `frontend/app.js` (avec `API_URL`)
- Commit : `fix: add frontend/ dir, fix faiss-cpu version (1.8.0), fix deploy workflow`

---

## 🔄 Comment Fonctionne le CI/CD Maintenant

```
Développeur : git push origin main
                      │
                      ▼
         ┌────────────────────────┐
         │   GitHub Actions       │
         │                        │
         │  test.yml (CI)         │
         │  ├─ pip install        │
         │  ├─ télécharge modèle  │
         │  └─ pytest tests/      │
         └────────────┬───────────┘
                      │ Si OK ✅
                      ▼
         ┌────────────────────────┐
         │   deploy.yml (CD)      │
         │                        │
         │  Job 1: Tests (idem)   │
         │                        │
         │  Job 2: Backend        │
         │  └─ curl RENDER_HOOK   │──► Render rebuild Docker
         │                        │
         │  Job 3: Frontend       │
         │  ├─ sed localhost→URL  │
         │  ├─ upload artifact    │
         │  └─ deploy-pages       │──► GitHub Pages live
         └────────────────────────┘
```

---

## 🌐 URLs de Production

| Service | URL |
|---------|-----|
| 🌐 **Frontend** | https://azizhachicha11.github.io/simclr-image-search/ |
| 🔌 **API Backend** | https://simclr-api.onrender.com |
| ❤️ **Health Check** | https://simclr-api.onrender.com/health |
| 📦 **Modèle HF** | https://huggingface.co/ElBOOH55/simclr-stl10 |
| 💻 **Code Source** | https://github.com/Azizhachicha11/simclr-image-search |
| ⚙️ **CI/CD Logs** | https://github.com/Azizhachicha11/simclr-image-search/actions |

---

## ⚠️ Points Importants

1. **`simclr_best.pt` n'est PAS dans Git** — il est sur HuggingFace Hub, téléchargé automatiquement lors du CI et au démarrage de Render

2. **Le backend tourne sur CPU** — Render Free ne propose pas de GPU, le modèle est configuré pour CPU uniquement

3. **Render Free = cold start** — Sur le plan gratuit, Render éteint le service après 15 min d'inactivité. La première requête peut prendre 30-60 secondes (démarrage + chargement du modèle)

4. **Secrets jamais en dur dans le code** — Tous les tokens et URLs sensibles sont dans GitHub Actions Secrets

5. **Pour mettre à jour le site** : il suffit de faire `git push origin main` — le CI/CD fait tout automatiquement

---

## 🧪 Tester en Local

```bash
# 1. Activer l'environnement
venv\Scripts\activate

# 2. Lancer l'API
cd backend && uvicorn api:app --port 8000

# 3. Ouvrir le frontend
# Ouvrir frontend/index.html dans le navigateur
# (ou python -m http.server 3000 dans le dossier frontend/)

# 4. Lancer les tests
python -m pytest tests/ -v
```

---

## 🐳 Tester avec Docker

```bash
docker compose up --build
# Frontend → http://localhost:80
# Backend  → http://localhost:8000
```
