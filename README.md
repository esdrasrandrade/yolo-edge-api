# YOLOv8 Edge Vision — Detecção de EPIs em Embedded Hardware

Sistema end-to-end de Visão Computacional para detecção em tempo real de Equipamentos de Proteção Individual (EPIs), otimizado para execução na **Raspberry Pi 5**.

## Arquitetura do Sistema
* **Modelo:** YOLOv8n (redimensionado para 320x320 para otimização de CPU/ARM NEON)
* **Dataset:** EPI Hard Hat Universe via Roboflow (6 classes)
* **Backend:** FastAPI / Uvicorn empacotado via Docker Container
* **Orquestração & MLOps:** Docker Compose, DVC e GitHub Actions CI/CD

## Como Executar Localmente

### Pré-requisitos
* Docker e Docker Compose instalados.

### Passos
1. Clone o repositório:
   ```bash
   git clone [https://github.com/esdrasrodrigues/yolo-api.git](https://github.com/esdrasrodrigues/yolo-api.git)
   cd yolo-api
