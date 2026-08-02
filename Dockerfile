# ── Image de base ─────────────────────────────────────────────────────
FROM python:3.11-slim

# Empêche Python de bufferiser stdout/stderr (logs visibles immédiatement)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# LightGBM dépend de libgomp (runtime OpenMP), absent de l'image slim par défaut.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Dépendances (couche mise en cache tant que requirements.txt ne change pas) ─
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Code de l'application et artefacts du modèle ─────────────────────────
COPY api/ ./api/
COPY model/ ./model/

# ── Utilisateur non-root (bonne pratique de sécurité) ────────────────────
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Vérifie que l'API répond, sans dépendance à curl (non présent dans l'image slim)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]