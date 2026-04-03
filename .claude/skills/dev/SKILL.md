---
description: Start the development server and run verification
---

# Development Workflow

1. Install dependencies if needed:
   ```bash
   cd frontend && npm install
   ```

2. Start the Vite dev server:
   ```bash
   cd frontend && npm run dev
   ```

3. The app will be available at http://localhost:5173

4. Report the URL to the user and confirm the server is running.

Note: Vite provides hot module replacement (HMR) — changes appear instantly without full reload.

## Verification Steps
After making changes, always:
1. Run TypeScript check: `cd frontend && npx tsc --noEmit`
2. Run tests: `cd frontend && npx vitest run`
