/**
 * Tutorial overlay — 11-step guided walkthrough.
 *
 * Shows a floating panel with step title/body, highlights the target
 * element, and gates progression on user interaction.
 */

import { useEffect, useRef } from 'react';
import {
  useTutorialStore,
  useIsStepSatisfied,
  TUTORIAL_STEPS,
} from '../hooks/useTutorial';
import { useFilterStore } from '../stores/filterStore';

// Steps 5-8 target elements inside the options panel
const STEPS_REQUIRING_PANEL_OPEN = new Set([5, 6, 7, 8]);

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

  // Auto-open options panel when tutorial reaches slider/checkbox steps
  useEffect(() => {
    if (!active) return;
    if (STEPS_REQUIRING_PANEL_OPEN.has(step) && !optionsPanelOpen) {
      setFilters({ optionsPanelOpen: true });
    }
  }, [active, step, optionsPanelOpen, setFilters]);

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
    }, STEPS_REQUIRING_PANEL_OPEN.has(step) ? 350 : 0);

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
      <div className="tutorial-panel">
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
            {isLastStep ? 'Finish' : isSatisfied ? 'Next' : 'Skip Step'}
          </button>
        </div>
      </div>
    </div>
  );
}
