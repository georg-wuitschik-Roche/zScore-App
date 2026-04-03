---
description: Build the frontend for production
---

# Build for Production

1. Install dependencies if needed:
   ```bash
   cd frontend && npm install
   ```

2. Run the production build (includes TypeScript check):
   ```bash
   cd frontend && npm run build
   ```

3. Output goes to `frontend/dist/`. This is a static site — no server needed.

4. Report the build status and any errors to the user.
