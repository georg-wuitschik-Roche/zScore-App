import { Navbar } from './Navbar';
import { FilterControls } from './FilterControls';
import { OptionsPanel } from './OptionsPanel';
import { AnalysisTabs } from './AnalysisTabs';
import { TutorialOverlay } from './TutorialOverlay';
import { useUrlState } from '../hooks/useUrlState';
import { useTutorialStore } from '../hooks/useTutorial';

export function Dashboard() {
  useUrlState();
  const tutorialActive = useTutorialStore((s) => s.active);
  const tutorialStep = useTutorialStore((s) => s.step);

  return (
    <div className={`dashboard-content${tutorialActive ? ' tutorial-active' : ''}${tutorialActive && tutorialStep >= 14 && tutorialStep <= 16 ? ' tutorial-step-settings' : ''}`}>
      <Navbar />
      <FilterControls />
      <OptionsPanel />
      <AnalysisTabs />
      <TutorialOverlay />
    </div>
  );
}
