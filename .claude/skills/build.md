---
description: Build the Docker image for production deployment
user_invocable: true
---

# Build for Production

1. Build the Docker image:
   ```bash
   docker build -t zscore-app .
   ```

2. Optionally test the production image locally:
   ```bash
   docker run -p 8080:8080 zscore-app
   ```

3. The production server uses Gunicorn with 1 worker and 8 threads on port 8080.

4. Report the build status and any errors to the user.
