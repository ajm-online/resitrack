# ResiTrack — Cancer Drug Resistance Prediction System

## Folder Structure
```
resitrack/
├── backend/          → FastAPI server + database + JSON files
├── ml_models/        → Trained LightGBM model + encoders
├── frontend/         → HTML frontend (index.html)
├── requirements.txt  → Python packages for Render
├── render.yaml       → Render deployment config
└── .gitignore
```

## Deploy Backend on Render
1. Push this folder to GitHub
2. Go to render.com → New Web Service → Connect GitHub repo
3. Render reads render.yaml automatically
4. Wait 3-5 min → get your URL: https://resitrack-api.onrender.com

## Deploy Frontend on Vercel
1. Go to vercel.com → New Project → Import GitHub repo
2. Set Root Directory = frontend
3. Framework = Other
4. Deploy → get your URL: https://resitrack.vercel.app

## Connect Frontend to Backend
In frontend/index.html, change:
  const API_BASE = '';
To:
  const API_BASE = 'https://YOUR-RENDER-URL.onrender.com';
