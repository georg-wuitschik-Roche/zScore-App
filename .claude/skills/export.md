---
description: Run batch export of boxplots and supplementary figures
user_invocable: true
---

# Export Visualizations

1. Export all boxplots by reaction type:
   ```bash
   python export_boxplots.py
   ```
   Output goes to `exports/boxplots/` and `exports/paper_boxplots/`.

2. Generate supplementary statistical figures:
   ```bash
   python generate_supplementary_figures.py
   ```
   Output goes to `exports/supplementary/`.

3. Report the number of files generated and any errors.
