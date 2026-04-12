/**
 * Tutorial overlay — 22-step guided walkthrough.
 *
 * Shows a floating panel with step title/body, highlights the target
 * element, and gates progression on user interaction.
 */

import { useEffect, useRef, useMemo } from 'react';
import {
  useTutorialStore,
  useIsStepSatisfied,
  TUTORIAL_STEPS,
} from '../hooks/useTutorial';
import { useFilterStore } from '../stores/filterStore';
import { useEffectiveDataset } from '../hooks/useEffectiveDataset';
import { getReactantOptions } from '../data/dropdownOptions';

// Steps 5-12 target elements inside the options panel
const STEPS_REQUIRING_PANEL_OPEN = new Set([5, 6, 7, 8, 9, 10, 11, 12]);

// Steps 17-19 are settings sub-panels, step 20 is reset — all in navbar
const STEPS_IN_NAVBAR = new Set([17, 18, 19, 20]);

// Steps 17-19 show settings modal side-by-side with tutorial
const SETTINGS_STEPS = new Set([17, 18, 19]);

export function TutorialOverlay() {
  const active = useTutorialStore((s) => s.active);
  const step = useTutorialStore((s) => s.step);
  const next = useTutorialStore((s) => s.next);
  const back = useTutorialStore((s) => s.back);
  const finish = useTutorialStore((s) => s.finish);

  const isSatisfied = useIsStepSatisfied();
  const currentStep = TUTORIAL_STEPS[step];
  const isLastStep = step >= TUTORIAL_STEPS.length - 1;

  // Track whether the condition was already satisfied when the step started
  // so we don't auto-advance past pre-filled steps
  const wasSatisfiedOnEntry = useRef(false);

  // Record whether condition is satisfied when entering a new step
  useEffect(() => {
    wasSatisfiedOnEntry.current = isSatisfied;
    // Only re-run when step changes, not when isSatisfied changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, active]);

  const optionsPanelOpen = useFilterStore((s) => s.optionsPanelOpen);
  const setFilters = useFilterStore((s) => s.setFilters);
  const reactantTypes = useFilterStore((s) => s.reactantTypes);
  const setReactantTypes = useFilterStore((s) => s.setReactantTypes);
  const setActiveTab = useFilterStore((s) => s.setActiveTab);
  const setSplitSelector = useFilterStore((s) => s.setSplitSelector);
  const reactionTypes = useFilterStore((s) => s.reactionTypes);

  const sourceData = useEffectiveDataset();
  const availableReactants = useMemo(
    () => getReactantOptions(sourceData, reactionTypes),
    [sourceData, reactionTypes],
  );

  // Auto-open options panel when tutorial reaches slider/checkbox/download steps
  useEffect(() => {
    if (!active) return;
    if (STEPS_REQUIRING_PANEL_OPEN.has(step) && !optionsPanelOpen) {
      setFilters({ optionsPanelOpen: true });
    }
  }, [active, step, optionsPanelOpen, setFilters]);

  // Lift navbar z-index when highlighting elements inside it
  useEffect(() => {
    if (!active) return;
    const navbar = document.querySelector('.navbar');
    if (STEPS_IN_NAVBAR.has(step)) {
      navbar?.classList.add('tutorial-lift');
    }
    return () => {
      navbar?.classList.remove('tutorial-lift');
    };
  }, [active, step]);

  // Step 13a — ensure 2+ reactant types so heatmap tab is visible
  useEffect(() => {
    if (!active || step !== 13) return;
    if (reactantTypes.length < 2) {
      const preferred = ['Base', 'Solvent'].filter((r) => availableReactants.includes(r));
      if (preferred.length >= 2) {
        setReactantTypes(preferred);
      } else {
        const second = availableReactants.find((r) => !reactantTypes.includes(r));
        if (second) setReactantTypes([...reactantTypes, second]);
      }
    }
  }, [active, step, reactantTypes, availableReactants, setReactantTypes]);

  // Step 13b — cycle through tabs (separate effect to avoid restart on reactantTypes change)
  useEffect(() => {
    if (!active || step !== 13) return;
    const timers = [
      setTimeout(() => setActiveTab('boxplot'), 800),
      setTimeout(() => setActiveTab('heatmap'), 3600),
      setTimeout(() => setActiveTab('stats'), 6400),
      setTimeout(() => setActiveTab('violin'), 9200),
    ];
    return () => timers.forEach(clearTimeout);
  }, [active, step, setActiveTab]);

  // Step 15 — set reactants to Catalyst + Base + Solvent, show violin, enable split
  useEffect(() => {
    if (!active || step !== 15) return;
    setActiveTab('violin');
    const preferred = ['Catalyst', 'Base', 'Solvent'].filter((r) => availableReactants.includes(r));
    const needsUpdate =
      preferred.length >= 2
        ? reactantTypes.length !== preferred.length ||
          preferred.some((r) => !reactantTypes.includes(r))
        : reactantTypes.length < 2;
    if (needsUpdate) {
      if (preferred.length >= 2) {
        setReactantTypes(preferred);
      } else {
        const second = availableReactants.find((r) => !reactantTypes.includes(r));
        if (second) setReactantTypes([...reactantTypes, second]);
      }
    }
    const timer = setTimeout(() => setSplitSelector('reactantTypes'), 400);
    return () => clearTimeout(timer);
  }, [active, step, reactantTypes, availableReactants, setReactantTypes, setSplitSelector]);

  // Steps 16-18 — auto-open settings modal, keep open across sub-steps
  const isSettingsStep = SETTINGS_STEPS.has(step);
  useEffect(() => {
    if (!active || !isSettingsStep) return;
    if (!document.querySelector('.settings-modal')) {
      document.getElementById('settings-toggle')?.click();
    }
  }, [active, isSettingsStep]);

  // Close settings when leaving settings range or tutorial ends
  useEffect(() => {
    if (!active || !isSettingsStep) {
      const closeBtn = document.querySelector('.settings-modal-close') as HTMLButtonElement | null;
      closeBtn?.click();
    }
  }, [active, isSettingsStep]);

  // Add/remove highlight class on target element
  useEffect(() => {
    if (!active || !currentStep?.targetId) return;

    // Small delay to let panel animation finish before highlighting
    const timer = setTimeout(() => {
      const el = document.getElementById(currentStep.targetId!);
      if (el) {
        el.classList.add('tutorial-highlight');
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, STEPS_REQUIRING_PANEL_OPEN.has(step) ? 350 : SETTINGS_STEPS.has(step) ? 100 : 0);

    return () => {
      clearTimeout(timer);
      const el = currentStep.targetId
        ? document.getElementById(currentStep.targetId)
        : null;
      if (el) el.classList.remove('tutorial-highlight');
    };
  }, [active, step, currentStep?.targetId]);

  // Auto-advance only when condition *becomes* satisfied (wasn't on entry)
  useEffect(() => {
    if (!active || isLastStep) return;
    if (isSatisfied && !wasSatisfiedOnEntry.current) {
      const timer = setTimeout(() => next(), 500);
      return () => clearTimeout(timer);
    }
  }, [active, isSatisfied, isLastStep, next]);

  if (!active) return null;

  return (
    <div className="tutorial-overlay">
      <div className={`tutorial-panel${SETTINGS_STEPS.has(step) ? ' tutorial-panel-settings' : ''}`}>
        <div className="tutorial-step-indicator">
          Step {step + 1} of {TUTORIAL_STEPS.length}
        </div>
        <h3 className="tutorial-title">{currentStep.title}</h3>
        <p className="tutorial-body">{currentStep.body}</p>
        <div className="tutorial-btn-row">
          <button
            className="tutorial-btn tutorial-btn-back"
            onClick={back}
            disabled={step === 0}
          >
            Back
          </button>
          <button
            className="tutorial-btn tutorial-btn-skip"
            onClick={finish}
          >
            Skip Tour
          </button>
          <button
            className="tutorial-btn tutorial-btn-next"
            onClick={isLastStep ? finish : next}
          >
            {isLastStep ? 'Finish' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
}
