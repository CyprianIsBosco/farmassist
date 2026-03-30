# AgroScan – Deployment Guide

## Project Structure
```
plant_disease_app/
├── app.py               ← Flask backend
├── plant_disease.h5     ← your model (copy here!)
├── requirements.txt
├── Procfile             ← for Render / Railway
├── .env                 ← secrets (never commit this)
└── templates/
    └── index.html       ← full frontend
```

---

## Step 1 — Set Up Locally

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy your model into the project folder
cp /path/to/plant_disease.h5 .

# Create .env file
echo "ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx" > .env
```

---

## Step 2 — Load API Key in app.py

Add this near the top of `app.py` (already done if you use python-dotenv):

```python
from dotenv import load_dotenv
load_dotenv()
```

The `anthropic.Anthropic()` client automatically reads `ANTHROPIC_API_KEY` from the environment.

---

## Step 3 — Run Locally

```bash
python app.py
# Visit http://localhost:5000
```

---

## Step 4 — Deploy to Render (Free Tier)

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial AgroScan commit"
   gh repo create agroscan --public --push
   ```

2. **Create account** at https://render.com

3. **New → Web Service → Connect your GitHub repo**

4. **Settings:**
   | Field | Value |
   |---|---|
   | Runtime | Python 3 |
   | Build Command | `pip install -r requirements.txt` |
   | Start Command | `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120` |

5. **Environment Variables** (in Render dashboard → Environment):
   ```
   ANTHROPIC_API_KEY = sk-ant-xxxxxxxxxxxx
   ```

6. **Upload your model** — since Render's free tier has ephemeral storage, store `plant_disease.h5` using one of these options:
   - **Option A (simplest):** Commit the `.h5` file directly to GitHub if < 100 MB (use Git LFS for larger files)
   - **Option B:** Upload to Google Drive, download in app startup with `gdown`
   - **Option C:** Use Render's Persistent Disk (paid, ~$1/month)

7. Click **Deploy** — your site will be live at `https://your-app.onrender.com`

---

## Step 5 — Deploy to Railway (Alternative, also free tier)

```bash
# Install Railway CLI
npm install -g @railway/cli
railway login
railway init
railway up
# Set env variable
railway variables set ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx
```

---

## Step 6 — Custom Domain (Optional)

In Render or Railway dashboard:
- Go to Settings → Custom Domains
- Add your domain (e.g. `agroscan.co.ke`)
- Update your DNS A/CNAME records as instructed

---

## Storing the Model on Render (Recommended Method)

Since the H5 file can be large, use this pattern in `app.py`:

```python
import os, urllib.request

MODEL_URL = "https://your-storage-url/plant_disease.h5"  # Google Drive direct link
MODEL_PATH = "plant_disease.h5"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")
```

Call `ensure_model()` at startup before `get_model()`.

---

## Getting a Google Drive Direct Download Link

1. Upload `plant_disease.h5` to Google Drive
2. Right-click → Share → Anyone with the link
3. Copy the file ID from the share URL: `https://drive.google.com/file/d/FILE_ID/view`
4. Your direct link: `https://drive.google.com/uc?export=download&id=FILE_ID`

Install gdown for large files: `pip install gdown`
```python
import gdown
gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
```

---

## Environment Variables Summary

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Your Anthropic API key from console.anthropic.com |
| `PORT` | Auto-set by Render/Railway |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Model not found` | Make sure `plant_disease.h5` is in the same folder as `app.py` |
| `ANTHROPIC_API_KEY not set` | Add it to `.env` locally or Render environment variables |
| Timeout on prediction | Increase `--timeout 120` in Procfile |
| TF version errors | Ensure `tensorflow==2.16.1` matches your model's save version |
| Large model on free tier | Use Google Drive download method above |
