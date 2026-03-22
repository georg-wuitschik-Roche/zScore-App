import { Navbar } from './Navbar';
import { FilterControls } from './FilterControls';
import { OptionsPanel } from './OptionsPanel';
import { AnalysisTabs } from './AnalysisTabs';
import { TutorialOverlay } from './TutorialOverlay';
import { useUrlState } from '../hooks/useUrlState';

export function Dashboard() {
  useUrlState();

  return (
    <div className="dashboard-content">
      <Navbar />
      <FilterControls />
      <OptionsPanel />
      <AnalysisTabs />
      <TutorialOverlay />
    </div>
  );
}
