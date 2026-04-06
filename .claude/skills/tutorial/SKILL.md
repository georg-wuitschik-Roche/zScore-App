---
name: tutorial
description: Guidelines for modifying the 21-step guided tutorial walkthrough.
---

# Tutorial System

The tutorial is a multi-step guided walkthrough defined across several tightly coupled files. Changes require careful coordination.

## File Locations

| File | What it contains |
|------|-----------------|
| `frontend/src/hooks/useTutorial.ts` | `TUTORIAL_STEPS[]` array (step definitions) + `useIsStepSatisfied()` (gating conditions) |
| `frontend/src/components/TutorialOverlay.tsx` | Step sets, step-specific `useEffect` hooks, highlight logic |
| `frontend/src/components/Dashboard.tsx` | Hardcoded step range for `tutorial-step-settings` CSS class |
| `frontend/src/styles/app.css` | `.tutorial-highlight` styles and variants |

## Adding or Removing Steps

Step indices are hardcoded in **all four files**. When inserting or removing a step, update every reference:

1. **`useTutorial.ts`** — add/remove entry in `TUTORIAL_STEPS[]`, update all `case` numbers in `useIsStepSatisfied()`
2. **`TutorialOverlay.tsx`** — update these step sets to match new indices:
   - `STEPS_REQUIRING_PANEL_OPEN` — steps that need the options panel open
   - `STEPS_IN_NAVBAR` — steps targeting navbar elements
   - `SETTINGS_STEPS` — steps that show the settings modal
3. **`TutorialOverlay.tsx`** — update `step !== N` checks in **every** step-specific `useEffect` hook (tab cycling, split mode setup, reactant selection, etc.)
4. **`Dashboard.tsx`** — update the `tutorialStep >= N && tutorialStep <= M` range for settings steps

**Checklist after any step change:**
- [ ] `TUTORIAL_STEPS[]` array updated
- [ ] `useIsStepSatisfied()` cases renumbered
- [ ] Step sets (`STEPS_REQUIRING_PANEL_OPEN`, `STEPS_IN_NAVBAR`, `SETTINGS_STEPS`) updated
- [ ] Step-specific `useEffect` hooks reference correct step numbers
- [ ] `Dashboard.tsx` settings step range updated
- [ ] Comments on effects match their new step numbers
- [ ] Run `npx tsc --noEmit` and `npx vitest run`

## Tutorial Highlight CSS

Highlights are applied by adding the `tutorial-highlight` class to the target element via its `targetId`.

### Key rules:

1. **Always use background fill, not just outline.** Outline alone is too subtle — users won't notice it. Use `background: var(--color-primary-light)` for light-background elements.

2. **Navbar elements need inverted highlights.** The navbar has a dark background, so `--color-primary-light` makes text unreadable. Use `background: rgba(255, 255, 255, 0.2)` instead. The `.navbar .tutorial-highlight` rule handles this.

3. **Don't add extra padding to highlighted elements.** It shifts layout and the user will reject it. Use `outline-offset` and matching `box-shadow` spread to extend the highlight area without affecting layout.

4. **Match `box-shadow` spread to `outline-offset`.** If `outline-offset: 3px`, use `box-shadow: 0 0 0 3px`. Mismatched values look wrong.

5. **Check for inline styles before adding CSS.** Some elements (e.g., `.filter-panel`) have inline styles in their component that override CSS. If your CSS change has no effect, check the component's JSX for inline `style={}`.

### Highlight style by element type:

| Element | Background | Outline |
|---------|-----------|---------|
| Default (`.tutorial-highlight`) | none | `2px solid var(--color-primary)` |
| Buttons (`button.tutorial-highlight`) | `var(--color-primary-light)` | `2px solid var(--color-primary)` |
| Dropdowns (`.control-col.tutorial-highlight`) | `var(--color-primary-light)` | inherited + `outline-offset: 3px` |
| Sliders/checkboxes (`.slider-group`, `label`) | `var(--color-primary-light)` | inherited + `outline-offset: 3px` |
| Download buttons (`#download-buttons`) | `var(--color-primary-light)` | inherited |
| Navbar elements (`.navbar .tutorial-highlight`) | `rgba(255, 255, 255, 0.2)` | white outline |

## Plotly Integration (Zoom Reset)

- **Never use key-based remount** to reset Plotly zoom. Plotly mutates layout objects in place — remounting reuses the mutated layout with zoom ranges baked in, causing data loss.
- **Use `Plotly.relayout()`** to reset zoom: `Plotly.relayout(div, { 'xaxis.autorange': true, 'yaxis.autorange': true })`
- **Plotly uses its own event emitter**, not DOM events. `addEventListener('plotly_relayout', ...)` does NOT work. Use the `onInitialized` callback from react-plotly.js to get the graph div, then register via `graphDiv.on('plotly_relayout', handler)`.
- The `useZoomReset()` hook in `DistributionView.tsx` encapsulates this pattern — reuse it in any new plot component.
