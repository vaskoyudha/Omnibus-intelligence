# Deployment Guide

Follow these instructions to deploy the Omnibus Legal Compass in a production or staging environment.

## Vector Database (Qdrant)

The system requires Qdrant for vector storage. We recommend running it via Docker with persistent storage:

```bash
docker run -d --name omnibus-qdrant \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

Ensure the Qdrant port (6333) is accessible to the backend server but protected from the public internet.

## Backend Deployment

The backend is a FastAPI application. In production, use an ASGI server like `uvicorn` or `gunicorn`.

1.  **Environment**: Set up a Python virtual environment and install dependencies.
2.  **Environment Variables**: Ensure `NVIDIA_API_KEY` and `QDRANT_URL` are set in the environment.
3.  **Run**:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```

## Frontend Deployment

The frontend is built with Next.js.

1.  **Build**:
    ```bash
    cd frontend
    npm install
    npm run build
    ```
2.  **Run**:
    ```bash
    npm run start
    ```

## Production Considerations

-   **Reverse Proxy**: Use Nginx or Apache as a reverse proxy to handle SSL/TLS termination and load balancing.
-   **Security**: Ensure all API keys are stored securely (e.g., using secret managers) and not committed to version control.
-   **Monitoring**: Implement logging and monitoring for both the FastAPI backend and Qdrant to track system health and performance.
-   **Data Persistence**: Regularly back up the Qdrant storage volume to prevent data loss.

## CI/CD

The project includes GitHub Actions workflows for continuous integration and automated documentation deployment.
-   **CI**: `.github/workflows/ci.yml` runs tests on every push.
-   **Docs**: `.github/workflows/docs.yml` deploys this documentation site to GitHub Pages.
    
Documentation is automatically built and deployed whenever changes are pushed to the `docs/` directory on the `main` branch.
