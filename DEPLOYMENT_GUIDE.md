# AgriGuard — Deployment Guide

## PROJECT STRUCTURE
```
plant_disease_app/
├── app.py                  Flask backend + prediction API
├── plant_disease.h5        Your AI model (already here)
├── requirements.txt        Python dependencies
├── Procfile                For Render/Railway deployment
├── templates/
│   └── index.html          Full website (farming UI + chatbot)
└── uploads/                Temp image storage (auto-created)
```

---

## STEP 1 — LOCAL TESTING

### Install Python 3.10
Download from https://python.org

### Create Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```
(TensorFlow is large — takes ~10 minutes first time)

### Run Locally
```bash
python app.py
```
Open: http://localhost:5000

---

## STEP 2 — ADD ANTHROPIC API KEY (Chatbot)

1. Get your key at: https://console.anthropic.com
2. In templates/index.html, find the sendChat() function
3. The API is called directly from the frontend for now
4. For production security, use the backend proxy approach (see Step 5)

---

## STEP 3 — DEPLOY FREE ON RENDER.COM

### 3.1 Push to GitHub
```bash
git init
git add .
git commit -m "AgriGuard plant disease app"
# Create repo on github.com then:
git remote add origin https://github.com/YOUR_USERNAME/agriguard.git
git push -u origin main
```

### 3.2 Deploy on Render
1. Go to https://render.com — create free account
2. New → Web Service → Connect GitHub repo
3. Settings:
   - Runtime: Python 3
   - Build Command: pip install -r requirements.txt
   - Start Command: gunicorn app:app
4. Add Environment Variable: ANTHROPIC_API_KEY = sk-ant-...
5. Click Deploy!

Live URL: https://agriguard.onrender.com

---

## STEP 4 — ALTERNATIVE: RAILWAY.APP

1. Go to https://railway.app
2. New Project → Deploy from GitHub
3. Add ANTHROPIC_API_KEY in Variables
4. Live in ~3 minutes

---

## STEP 5 — SECURE API KEY (Production)

Add this route to app.py to proxy API calls from the backend:

```python
@app.route('/api/chat', methods=['POST'])
def chat_proxy():
    import requests, os
    data = request.json
    res = requests.post(
        'https://api.anthropic.com/v1/messages',
        headers={
            'x-api-key': os.environ.get('ANTHROPIC_API_KEY'),
            'anthropic-version': '2023-06-01',
            'Content-Type': 'application/json'
        },
        json={
            'model': 'claude-sonnet-4-20250514',
            'max_tokens': 1000,
            'system': data.get('system',''),
            'messages': data.get('messages',[])
        }
    )
    return jsonify(res.json())
```

Then in index.html, change the fetch to /api/chat.

---

## CUSTOMIZATION

- Update contact phone/email in index.html (search for +254 712 345 678)
- Add more diseases: update CLASS_NAMES and DISEASE_INFO in app.py
- Change the office address in the Contact section of index.html

---

## TROUBLESHOOTING

| Problem | Fix |
|---------|-----|
| tensorflow install fails | pip install tensorflow-cpu |
| Model not found | Ensure plant_disease.h5 is in app.py folder |
| Chatbot silent | Check Anthropic API key is valid |
| Port 5000 busy | Change app.run(port=5001) in app.py |

