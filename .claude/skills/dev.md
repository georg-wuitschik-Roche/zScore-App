---
description: Start the development server
user_invocable: true
---

# Start Development Server

1. Check that dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the Dash dev server with debug mode:
   ```bash
   python app.py
   ```

3. The app will be available at http://localhost:8050

4. Report the URL to the user and confirm the server is running.

Note: The dev server runs with `debug=True` which enables hot-reloading on code changes.
