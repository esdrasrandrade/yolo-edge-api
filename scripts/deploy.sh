#!/bin/bash
set -euo pipefail
DEPLOY_PATH="${DEPLOY_PATH:-~/yolo-api}"
cd "$DEPLOY_PATH"

echo "[1/4] Baixando nova imagem..."
docker compose pull
echo "[2/4] Iniciando nova versão..."
docker compose up -d
echo "[3/4] Aguardando health check..."
sleep 10
if curl -sf http://localhost:8000/health > /dev/null; then
    echo "[OK] Deploy bem-sucedido."
    exit 0
else
    echo "[ERRO] Health check falhou."
    exit 1
fi
