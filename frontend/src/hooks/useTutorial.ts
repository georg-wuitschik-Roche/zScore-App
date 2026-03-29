/**
 * Tutorial state machine — 19-step walkthrough.
 *
 * Each step has a target element ID, title, body text, and a gating
 * condition that determines whether the user has completed the step.
 */

import { create } from 'zustand';
import { useFilterStore } from '../stores/filterStore';

export interface TutorialStep {
  targetId: string | null;
  title: string;
  body: string;
}

export const TUTORIAL_STEPS: TutorialStep[] = [
  {
    targetId: 'reaction-type-dropdown',
    title: 'Select Reaction Type(s)',
    body: 'Pick one or more reaction classes to focus the analysis.',
  },
  {
    targetId: 'reactant-types-dropdown',
    title: 'Select Reactant Type(s)',
    body: 'Choose the reagent categories to analyze (e.g., Catalyst, Base).',
  },
  {
    targetId: 'fg-a-dropdown',
    title: 'Choose Functional Group A',
    body: 'Optionally filter by reacting functional groups (side A).',
  },
  {
    targetId: 'fg-b-dropdown',
    title: 'Choose Functional Group B',
    body: 'Optionally filter by reacting functional groups (side B).',
  },
  {
    targetId: 'options-toggle',
    title: 'Open Options',
    body: 'Click Options to reveal advanced filters.',
  },
  {
    targetId: 'min-eln-slider',
    title: 'Minimum ELNs',
    body: 'Drag to require a minimum number of ELNs (reactions) per selection.',
  },
  {
    targetId: 'topn-slider',
    title: 'Top-N z-Score',
    body: 'Limit to the top-N z-scores per ELN and selected reactant type(s).',
  },
  {
    targetId: 'max-comp-slider',
    title: 'Max Components',
    body: 'Cap how many components are displayed in plots.',
  },
  {
    targetId: 'exclude-cui-checkbox',
    title: 'Exclude CuI as Catalyst',
    body: 'Toggle to include/exclude CuI catalyst entries.',
  },
  {
    targetId: 'exclude-scaleup-checkbox',
    title: 'Exclude Scale-Up Plates',
    body: 'Toggle to exclude scale-up plate experiments from the analysis.',
  },
  {
    targetId: 'include-null-checkbox',
    title: 'Include Null Reactant Types',
    body: 'Toggle to include combinations where a reactant type has no value (null).',
  },
  {
    targetId: 'download-buttons',
    title: 'Download Data',
    body: 'Export the filtered data as CSV or save the current plot as a high-resolution PNG.',
  },
  {
    targetId: 'view-toggle',
    title: 'Explore Results',
    body: 'Switch between Boxplot, Violin, Heatmap, and Statistics views. Violin plots show the full distribution shape with individual data points.',
  },
  {
    targetId: 'split-toggle',
    title: 'Split / Combined Mode',
    body: 'When 2+ values are selected in a dropdown, use Split to compare them side by side. Combined overlays everything in one plot.',
  },
  {
    targetId: 'settings-section-data',
    title: 'Settings — Data',
    body: 'Switch between dataset versions or upload your own CSV file.',
  },
  {
    targetId: 'settings-section-comparison',
    title: 'Settings — Comparison',
    body: 'Enable comparison mode to see how results changed between dataset versions.',
  },
  {
    targetId: 'settings-section-appearance',
    title: 'Settings — Appearance',
    body: 'Toggle dark mode or turn on Presentation Mode for cleaner visuals.',
  },
  {
    targetId: 'reset-btn',
    title: 'Reset Filters',
    body: 'Click Reset to clear all filters and return to the default state.',
  },
  {
    targetId: null,
    title: "You're all set!",
    body: "That's the tour. You can restart anytime via Start Tutorial.",
  },
];

interface TutorialState {
  active: boolean;
  step: number;
  start: () => void;
  next: () => void;
  back: () => void;
  skip: () => void;
  finish: () => void;
}

export const useTutorialStore = create<TutorialState>((set) => ({
  active: false,
  step: 0,
  start: () => set({ active: true, step: 0 }),
  next: () =>
    set((s) => {
      if (s.step >= TUTORIAL_STEPS.length - 1) {
        return { active: false, step: 0 };
      }
      return { step: s.step + 1 };
    }),
  back: () => set((s) => ({ step: Math.max(0, s.step - 1) })),
  skip: () =>
    set((s) => {
      if (s.step >= TUTORIAL_STEPS.length - 1) {
        return { active: false, step: 0 };
      }
      return { step: s.step + 1 };
    }),
  finish: () => set({ active: false, step: 0 }),
}));

/**
 * Check if the current tutorial step's gating condition is satisfied.
 */
export function useIsStepSatisfied(): boolean {
  const step = useTutorialStore((s) => s.step);
  const reactionTypes = useFilterStore((s) => s.reactionTypes);
  const reactantTypes = useFilterStore((s) => s.reactantTypes);
  const fgA = useFilterStore((s) => s.fgA);
  const fgB = useFilterStore((s) => s.fgB);
  const optionsPanelOpen = useFilterStore((s) => s.optionsPanelOpen);
  const minEln = useFilterStore((s) => s.minEln);
  const topnZscore = useFilterStore((s) => s.topnZscore);
  const maxComponents = useFilterStore((s) => s.maxComponents);
  const excludeCui = useFilterStore((s) => s.excludeCui);
  const excludeScaleup = useFilterStore((s) => s.excludeScaleup);
  const includeNullCategories = useFilterStore((s) => s.includeNullCategories);
  switch (step) {
    case 0:
      return reactionTypes.length > 0;
    case 1:
      return reactantTypes.length > 0;
    case 2:
      return fgA.length > 0;
    case 3:
      return fgB.length > 0;
    case 4:
      return optionsPanelOpen;
    case 5:
      return minEln !== 10;
    case 6:
      return topnZscore !== 3;
    case 7:
      return maxComponents !== 10;
    case 8:
      return !excludeCui;
    case 9:
      return !excludeScaleup;
    case 10:
      return includeNullCategories;
    case 11:
      return true; // downloads — informational, click Next to proceed
    case 12:
      return true; // tabs — auto-cycled, click Next to proceed
    case 13:
      return true; // split — auto-enabled, click Next to proceed
    case 14:
      return true; // settings: data
    case 15:
      return true; // settings: comparison
    case 16:
      return true; // settings: appearance
    case 17:
      return true; // reset
    default:
      return true;
  }
}
