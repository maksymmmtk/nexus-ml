from fastapi import FastAPI

app = FastAPI(
    title="MLOps Inference API",
    description="Універсальна платформа для інференсу ML-моделей",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "MLOps Inference API is running"}

@app.get("/health")
async def health_check():
    # Згодом тут буде перевірка підключення до Redis та MinIO
    return {"status": "healthy", "dependencies": "not_configured"}