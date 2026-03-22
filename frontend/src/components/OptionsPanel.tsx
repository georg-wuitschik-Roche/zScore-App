import { useFilterStore } from '../stores/filterStore';
import { useFilteredData } from '../hooks/useFilteredData';

export function OptionsPanel() {
  const optionsPanelOpen = useFilterStore((s) => s.optionsPanelOpen);
  const toggleOptionsPanel = useFilterStore((s) => s.toggleOptionsPanel);
  const minEln = useFilterStore((s) => s.minEln);
  const topnZscore = useFilterStore((s) => s.topnZscore);
  const maxComponents = useFilterStore((s) => s.maxComponents);
  const excludeCui = useFilterStore((s) => s.excludeCui);
  const excludeScaleup = useFilterStore((s) => s.excludeScaleup);
  const includeNullCategories = useFilterStore(
    (s) => s.includeNullCategories,
  );
  const setMinEln = useFilterStore((s) => s.setMinEln);
  const setTopnZscore = useFilterStore((s) => s.setTopnZscore);
  const setMaxComponents = useFilterStore((s) => s.setMaxComponents);
  const setExcludeCui = useFilterStore((s) => s.setExcludeCui);
  const setExcludeScaleup = useFilterStore((s) => s.setExcludeScaleup);
  const setIncludeNullCategories = useFilterStore(
    (s) => s.setIncludeNullCategories,
  );

  const { rows, stats } = useFilteredData();

  const maxComponentsCap = stats.maxComponentsCap ?? 10;

  function handleDownloadCSV() {
    if (rows.length === 0) return;
    const headers = Object.keys(rows[0]);
    const csvContent = [
      headers.join(','),
      ...rows.map((row) =>
        headers
          .map((h) => {
            const val = row[h];
            if (val === null || val === undefined) return '';
            const str = String(val);
            return str.includes(',') ? `"${str}"` : str;
          })
          .join(','),
      ),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'zscore_filtered_data.csv';
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleDownloadPNG() {
    const plotEl = document.querySelector('.js-plotly-plot') as HTMLElement;
    if (!plotEl) return;
    // @ts-expect-error — plotly.js-dist-min has no type declarations
    import('plotly.js-dist-min').then((Plotly) => {
      Plotly.downloadImage(plotEl, {
        format: 'png',
        width: 1600,
        height: 800,
        scale: 4,
        filename: 'zscore_plot',
      });
    });
  }

  return (
    <>
      {/* Toggle button */}
      <div className="filter-toggle-container">
        <div className="filter-toggle-line" />
        <button className="filter-toggle-btn" id="toggle-filters-btn" onClick={toggleOptionsPanel}>
          <svg
            width="14"
            height="14"
            viewBox="0 0 16 16"
            fill="currentColor"
            style={{ marginRight: 8 }}
          >
            <path d="M1 2h14v2H1zM3 7h10v2H3zM5 12h6v2H5z" />
          </svg>
          <span>Options</span>
        </button>
      </div>

      {/* Collapsible panel */}
      <div
        className="filter-panel"
        style={{
          display: optionsPanelOpen ? 'block' : 'none',
          maxHeight: optionsPanelOpen ? '500px' : '0',
          padding: optionsPanelOpen ? '20px' : '0 20px',
        }}
      >
        {/* Sliders row */}
        <div className="filter-options-row sliders">
          <label>Minimum Number of ELNs:</label>
          <div className="slider-wrap min-eln">
            <input
              type="range"
              min={1}
              max={20}
              step={1}
              value={minEln}
              onChange={(e) => setMinEln(Number(e.target.value))}
            />
            <span className="slider-value">{minEln}</span>
          </div>

          <label>Top-N z-Score per (ELN_ID, selected reactant type(s)):</label>
          <div className="slider-wrap topn">
            <input
              type="range"
              min={1}
              max={10}
              step={1}
              value={topnZscore}
              onChange={(e) => setTopnZscore(Number(e.target.value))}
            />
            <span className="slider-value">{topnZscore}</span>
          </div>

          <label>Max Components to Display:</label>
          <div className="slider-wrap max-comp">
            <input
              type="range"
              min={1}
              max={Math.max(maxComponentsCap, 1)}
              step={1}
              value={Math.min(maxComponents, maxComponentsCap)}
              onChange={(e) => setMaxComponents(Number(e.target.value))}
            />
            <span className="slider-value">
              {Math.min(maxComponents, maxComponentsCap)}
            </span>
          </div>
        </div>

        {/* Checkboxes row */}
        <div className="filter-options-row">
          <label className="checklist-item">
            <input
              type="checkbox"
              checked={excludeCui}
              onChange={(e) => setExcludeCui(e.target.checked)}
            />
            {' '}Exclude CuI as Catalyst
          </label>
          <label className="checklist-item">
            <input
              type="checkbox"
              checked={excludeScaleup}
              onChange={(e) => setExcludeScaleup(e.target.checked)}
            />
            {' '}Exclude Scale-Up Plates
          </label>
          <label className="checklist-item">
            <input
              type="checkbox"
              checked={includeNullCategories}
              onChange={(e) => setIncludeNullCategories(e.target.checked)}
            />
            {' '}Include combinations with null reactant types
          </label>
        </div>

        {/* Download buttons row */}
        <div className="filter-options-row downloads">
          <button className="download-btn-gap" onClick={handleDownloadCSV}>
            Download CSV
          </button>
          <button onClick={handleDownloadPNG}>Download PNG</button>
        </div>
      </div>
    </>
  );
}
