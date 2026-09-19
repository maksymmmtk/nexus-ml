# NexusML

High-performance, Cloud-Native platform for deploying and monitoring machine learning models. The architecture is built on a microservices approach utilizing **FastAPI**, **ONNX Runtime**, and **Redis**, deployed in a serverless **AWS ECS Fargate** environment with fully automated CI/CD pipelines.

## Key Features

*   **Dynamic Model Scaling:** Automated parsing of S3 buckets paired with GitHub Actions Matrix to parallel-deploy independent microservices for every single ML model.
*   **Fail-Fast Architecture:** Models are loaded directly into RAM during container startup (FastAPI Lifespan). The container refuses incoming traffic until the model is fully initialized and verified.
*   **Inference Optimization:** Powered by ONNX Runtime for ultra-fast predictions, heavily optimized with **Redis** caching (using SHA-256 hashing for input features).
*   **Advanced Monitoring:** Automated Service Discovery via AWS Cloud Map. Seamless collection of "Golden Signals" (Latency, Error Rate, Traffic) and Cache Hit Rate using **Prometheus** and **Grafana**.
*   **ML Validation Pipeline:** Automated testing for new models (Tensor Shape, Accuracy, Latency) in a staging environment prior to automatic promotion to production.

## Infrastructure Architecture

![NexusML Architecture](./assets/architecture.png)

The project consists of four primary microservices deployed within an AWS ECS cluster:
1.  **NexusML API** (Dynamic auto-scaling container count based on the number of production models in S3).
2.  **Redis** (In-memory database for inference caching and latency reduction).
3.  **Prometheus** (Time-series database for metrics with automatic API container discovery).
4.  **Grafana** (Visualization platform with automated dashboard provisioning).

## Project Structure

```text
.
├── .github/workflows/       # CI/CD Pipelines (App, Model, Infra)
├── app/                     # FastAPI application source code
│   ├── main.py              # Entry point, Lifespan events, Endpoints
│   ├── ml_service.py        # ONNX inference logic
│   ├── model_loader.py      # AWS S3 integration
│   ├── cache.py             # Redis caching logic
│   └── schemas.py           # Pydantic models (data validation)
├── infrastructure/          # Infrastructure configurations and Dockerfiles
│   ├── grafana/             # Dashboards (Golden Signals) and Datasources
│   ├── prometheus/          # Metrics collection rules (Service Discovery)
│   └── redis/               # In-memory caching setup
├── scripts/                 # Utility scripts (S3 Sync, Training)
├── tests/                   # Pytest suites (Coverage > 94%)
├── docker-compose.yml       # Local development environment
├── Dockerfile               # Production image configuration for FastAPI
├── requirements.txt         # Python package dependencies
└── task-def-template.json   # AWS ECS Task Definition template for dynamic deployment
```

## CI/CD Pipelines (GitHub Actions)

The project utilizes a decoupled pipeline strategy for maximum flexibility:

*   **app_pipeline.yml (App CI/CD):**
Triggered on changes in app/ or tests/. Runs linting (Flake8), executes Pytest suites, builds the API Docker image, pushes it to Amazon ECR, dynamically scans production/ in S3, and updates all corresponding ECS services in parallel.

*   **model_pipeline.yml (Model Promotion):**
Triggered manually (Workflow Dispatch). Downloads .zip archives from S3 (staging/), runs validation tests (Accuracy, Latency constraints), promotes validated models to production/, and orchestrates ECS container updates via AWS Cloud Map.

*   **infra_pipeline.yml (Infra CI/CD):**
Manages core underlying services (Redis, Prometheus, Grafana). Automatically registers them into AWS Service Discovery.

## API Endpoints

### **GET /health**
Returns the readiness status of the service (checks if the model is loaded in memory and Redis is reachable). Primarily used by the AWS Application Load Balancer.\
**Example Response:**

```json
{
  "status": "healthy",
  "model": "rf_iris_v1",
  "services": {
    "api": "connected",
    "redis": "connected",
    "model_loaded": true
  }
}
```

### **POST /predict**
Executes an inference prediction based on the provided input features.\
**Example Request:**

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

**Example Response (Cache Miss):**

```json
{
  "model_used": "rf_iris_v1",
  "prediction": 0.0,
  "status": "success (computed)"
}
```

**Example Response (Cache Hit):**

```json
{
  "model_used": "rf_iris_v1",
  "prediction": 0.0,
  "status": "success (from cache)"
}
```

## Monitoring (Grafana)

![NexusML Architecture](./assets/grafana_dashboard.png)

The platform features an auto-provisioned Golden Signals dashboard that tracks:

1) **Total Requests:** The absolute volume of inference requests.

2) **Requests per Second:** RPS broken down by HTTP status codes (200, 400, 500).

3) **Latency 95th Percentile:** Core API and ML inference performance.

4) **Error Rate:** Percentage of 5xx internal server errors.

5) **Cache Hit Rate:** Operational efficiency of the Redis caching layer.

The dashboard supports dynamic filtering for individual active models via the $model variable dropdown.