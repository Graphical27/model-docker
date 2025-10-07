# 🧠 Brain Tumor Detection API Docker

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?logo=docker)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-EE4C2C.svg?logo=pytorch)](https://pytorch.org)

## 📚 Learning Docker - My First Container Project

This project serves as my learning journey into Docker containerization, featuring a Brain Tumor Detection API powered by PyTorch and FastAPI. As a first-time Docker user, I've documented the process and challenges faced during development.

### 🌟 Key Learning Points

- Setting up a Python environment in Docker
- Handling large dependencies (PyTorch)
- Managing Docker cache for efficient builds
- Exposing API ports
- Working with ML models in containers

## 🚀 Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Git (for cloning the repository)

### Building the Container

```bash
# Clone the repository
git clone https://github.com/Graphical27/model-docker.git
cd model-docker

# Build the Docker image
docker build -t brain-tumor-api .

# Run the container
docker run -p 8000:8000 brain-tumor-api
```

## 💡 Pro Tips I Learned

### Optimizing Docker Builds

1. **Handling Large Dependencies:**
   - Pre-download large wheel files
   - Use BuildKit cache for faster builds
   ```dockerfile
   RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
   ```

2. **Layer Caching:**
   - Copy requirements.txt first
   - Install dependencies before copying the rest of the code
   - Use .dockerignore for unnecessary files

## 🛠️ Project Structure

```
model-docker/
│
├── api.py              # FastAPI application
├── Dockerfile          # Container configuration
├── requirements.txt    # Python dependencies
│
└── best_model/        # PyTorch model files
    └── ResNet.pth     # Trained model weights
```

## 🔍 API Endpoints

- `GET /` - Health check
- `POST /predict` - Submit image for tumor detection

## 📝 Notes for Fellow Learners

- Start with a slim base image
- Understand layer caching
- Use multi-stage builds for smaller images
- Keep an eye on Docker Desktop resources

## 🤝 Contributing

Feel free to open issues and PRs! As I'm learning Docker, I appreciate any feedback or suggestions for improvement.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
Made with ❤️ while learning Docker | [@Graphical27](https://github.com/Graphical27)