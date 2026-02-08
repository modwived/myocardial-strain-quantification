# Myocardial Strain Quantification

Artificial intelligence fully automated myocardial strain quantification for risk stratification following acute myocardial infarction.

## Prerequisites

- Python 3.10+
- pip
- Docker and Docker Compose (for containerized deployment)
- (Optional) NVIDIA GPU with CUDA for accelerated inference

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/modwived/myocardial-strain-quantification.git
cd myocardial-strain-quantification
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env to set MODEL_PATH, DATA_DIR, and other settings
```

### 4. Prepare data and model weights

Place your trained model weights in the `models/` directory and input cardiac imaging data (DICOM or NIfTI) in the `data/` directory:

```bash
mkdir -p models data
# Copy your model weights into models/
# Copy your imaging data into data/
```

### 5. Start the application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Interactive API docs are served at `http://localhost:8000/docs`.

## Running with Docker

### 1. Build and start

```bash
docker compose up --build
```

This will:
- Build the container with all dependencies
- Mount `./data` and `./models` as volumes
- Expose the API on port 8000

### 2. Stop

```bash
docker compose down
```

## Running with GPU Support

For GPU-accelerated inference, use the NVIDIA Container Toolkit:

```bash
docker compose up --build  # ensure nvidia-container-toolkit is installed
```

Or run directly:

```bash
docker run --gpus all -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  myocardial-strain-quantification
```

## Environment Variables

| Variable     | Description                        | Default     |
|--------------|------------------------------------|-------------|
| `MODEL_PATH` | Path to trained model weights      | `./models`  |
| `DATA_DIR`   | Directory for input imaging data   | `./data`    |
| `LOG_LEVEL`  | Logging level                      | `info`      |
| `HOST`       | Server bind address                | `0.0.0.0`   |
| `PORT`       | Server port                        | `8000`      |

## Project Structure

```
myocardial-strain-quantification/
├── app/                  # Application source code
│   └── main.py           # FastAPI entry point
├── models/               # Trained model weights (not tracked in git)
├── data/                 # Input imaging data (not tracked in git)
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container image definition
├── docker-compose.yml    # Multi-service orchestration
├── .env.example          # Environment variable template
└── README.md
```
