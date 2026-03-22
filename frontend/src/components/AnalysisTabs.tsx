import { useFilterStore } from '../stores/filterStore';
import { DistributionView } from './DistributionView';
import { createBoxplotConfig } from '../plots/boxplot';
import { createViolinConfig } from '../plots/violin';
import { HeatmapView } from './HeatmapView';
import { StatsTable } from './StatsTable';

type TabId = 'boxplot' | 'violin' | 'heatmap' | 'stats';

interface TabDef {
  id: TabId;
  label: string;
  requiresMultiReactant: boolean;
}

const TABS: TabDef[] = [
  { id: 'boxplot', label: 'Boxplot', requiresMultiReactant: false },
  { id: 'violin', label: 'Violin', requiresMultiReactant: false },
  { id: 'heatmap', label: 'Heatmap', requiresMultiReactant: true },
  { id: 'stats', label: 'Stats', requiresMultiReactant: false },
];

export function AnalysisTabs() {
  const activeTab = useFilterStore((s) => s.activeTab);
  const setActiveTab = useFilterStore((s) => s.setActiveTab);
  const reactantTypes = useFilterStore((s) => s.reactantTypes);

  const showHeatmap = reactantTypes.length >= 2;

  const visibleTabs = TABS.filter(
    (tab) => !tab.requiresMultiReactant || showHeatmap,
  );

  return (
    <div className="analysis-view">
      <div className="view-toggle" id="view-toggle">
        {visibleTabs.map((tab) => (
          <button
            key={tab.id}
            className={`view-toggle-btn${activeTab === tab.id ? ' active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
            title={tab.label}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="view-content">
        {activeTab === 'boxplot' && <DistributionView buildConfig={createBoxplotConfig} label="boxplot" />}
        {activeTab === 'violin' && <DistributionView buildConfig={createViolinConfig} label="violin plot" />}
        {activeTab === 'heatmap' && <HeatmapView />}
        {activeTab === 'stats' && <StatsTable />}
      </div>
    </div>
  );
}
