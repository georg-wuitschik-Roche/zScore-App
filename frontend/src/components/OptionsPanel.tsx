import { useState, useRef, useCallback, useEffect } from 'react';
import { useFilterStore } from '../stores/filterStore';
import { useFilteredData } from '../hooks/useFilteredData';
import { CATALYST_FILTER_OPTIONS } from '../data/types';
import type { CatalystFilterMode } from '../data/types';

const SLIDER_DEBOUNCE_MS = 120;

/** Local slider state that debounces store updates for smooth dragging. */
function useDebouncedSlider(
  storeValue: number,
  storeSetter: (v: number) => void,
): [number, (v: number) => void] {
  const [local, setLocal] = useState(storeValue);
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  // Sync local state when store value changes externally (e.g. URL restore)
  useEffect(() => { setLocal(storeValue); }, [storeValue]);

  const update = useCallback(
    (v: number) => {
      setLocal(v);
      clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => storeSetter(v), SLIDER_DEBOUNCE_MS);
    },
    [storeSetter],
  );

  return [local, update];
}

/** Clickable slider value that becomes an editable number input. */
function EditableSliderValue({
  value,
  min,
  max,
  onChange,
}: {
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(String(value));
  const inputRef = useCallback((el: HTMLInputElement | null) => {
    if (el) requestAnimationFrame(() => el.select());
  }, []);

  function commit() {
    setEditing(false);
    const parsed = parseInt(draft, 10);
    if (!isNaN(parsed)) {
      onChange(Math.min(max, Math.max(min, parsed)));
    }
  }

  if (editing) {
    return (
      <input
        ref={inputRef}
        className="slider-value slider-value-input"
        type="number"
        min={min}
        max={max}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') commit();
          if (e.key === 'Escape') setEditing(false);
        }}
      />
    );
  }

  return (
    <span
      className="slider-value slider-value-clickable"
      title="Click to type a value"
      onClick={() => { setDraft(String(value)); setEditing(true); }}
    >
      {value}
    </span>
  );
}

function CatalystFilterToggle({ id, label, value, onChange }: {
  id: string;
  label: string;
  value: CatalystFilterMode;
  onChange: (val: CatalystFilterMode) => void;
}) {
  return (
    <div className="catalyst-filter-group" id={id}>
      <span className="catalyst-filter-label">{label}</span>
      <div className="catalyst-filter-toggle">
        {CATALYST_FILTER_OPTIONS.map((mode) => (
          <button
            key={mode}
            className={`catalyst-filter-btn${value === mode ? ' active' : ''}`}
            onClick={() => onChange(mode)}
          >
            {mode.charAt(0).toUpperCase() + mode.slice(1)}
          </button>
        ))}
      </div>
    </div>
  );
}

export function OptionsPanel() {
  const optionsPanelOpen = useFilterStore((s) => s.optionsPanelOpen);
  const toggleOptionsPanel = useFilterStore((s) => s.toggleOptionsPanel);
  const minEln = useFilterStore((s) => s.minEln);
  const topnZscore = useFilterStore((s) => s.topnZscore);
  const maxComponents = useFilterStore((s) => s.maxComponents);
  const copperFilter = useFilterStore((s) => s.copperFilter);
  const precomplexedFilter = useFilterStore((s) => s.precomplexedFilter);
  const excludeScaleup = useFilterStore((s) => s.excludeScaleup);
  const includeNullCategories = useFilterStore(
    (s) => s.includeNullCategories,
  );
  const setMinEln = useFilterStore((s) => s.setMinEln);
  const setTopnZscore = useFilterStore((s) => s.setTopnZscore);
  const setMaxComponents = useFilterStore((s) => s.setMaxComponents);
  const setCopperFilter = useFilterStore((s) => s.setCopperFilter);
  const setPrecomplexedFilter = useFilterStore((s) => s.setPrecomplexedFilter);
  const setExcludeScaleup = useFilterStore((s) => s.setExcludeScaleup);
  const setIncludeNullCategories = useFilterStore(
    (s) => s.setIncludeNullCategories,
  );
  const resetOptions = useFilterStore((s) => s.resetOptions);

  const { rows, stats } = useFilteredData();

  const maxComponentsCap = Math.min(stats.maxComponentsCap ?? 10, 50);

  const [localMinEln, setLocalMinEln] = useDebouncedSlider(minEln, setMinEln);
  const [localTopn, setLocalTopn] = useDebouncedSlider(topnZscore, setTopnZscore);
  const [localMaxComp, setLocalMaxComp] = useDebouncedSlider(maxComponents, setMaxComponents);

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
        <button className="filter-toggle-btn" id="options-toggle" onClick={toggleOptionsPanel}>
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
          padding: optionsPanelOpen ? '32px 20px 20px' : '0 20px',
        }}
      >
        {/* Sliders row */}
        <div className="filter-options-row sliders">
          <div className="slider-group" id="min-eln-slider">
            <label>Minimum Number of ELNs:</label>
            <div className="slider-wrap min-eln">
              <input
                type="range"
                min={1}
                max={20}
                step={1}
                value={localMinEln}
                onChange={(e) => setLocalMinEln(Number(e.target.value))}
              />
              <EditableSliderValue value={localMinEln} min={1} max={20} onChange={setLocalMinEln} />
            </div>
          </div>

          <div className="slider-group" id="topn-slider">
            <label>Top-N z-Score per (ELN_ID, selected reactant type(s)):</label>
            <div className="slider-wrap topn">
              <input
                type="range"
                min={1}
                max={10}
                step={1}
                value={localTopn}
                onChange={(e) => setLocalTopn(Number(e.target.value))}
              />
              <EditableSliderValue value={localTopn} min={1} max={10} onChange={setLocalTopn} />
            </div>
          </div>

          <div className="slider-group" id="max-comp-slider">
            <label>Max Components to Display:</label>
            <div className="slider-wrap max-comp">
            <input
              type="range"
              min={1}
              max={Math.max(maxComponentsCap, 1)}
              step={1}
              value={Math.min(localMaxComp, maxComponentsCap)}
              onChange={(e) => setLocalMaxComp(Number(e.target.value))}
            />
            <EditableSliderValue
              value={Math.min(localMaxComp, maxComponentsCap)}
              min={1}
              max={Math.max(maxComponentsCap, 1)}
              onChange={setLocalMaxComp}
            />
            </div>
          </div>
        </div>

        {/* Checkboxes row */}
        <div className="filter-options-row">
          <CatalystFilterToggle
            id="copper-filter-control"
            label="Copper Catalysts:"
            value={copperFilter}
            onChange={setCopperFilter}
          />
          <CatalystFilterToggle
            id="precomplexed-filter-control"
            label="Pre-Complexed Catalysts:"
            value={precomplexedFilter}
            onChange={setPrecomplexedFilter}
          />
          <label className="checklist-item" id="exclude-scaleup-checkbox">
            <input
              type="checkbox"
              checked={excludeScaleup}
              onChange={(e) => setExcludeScaleup(e.target.checked)}
            />
            {' '}Exclude Scale-Up Plates
          </label>
          <label className="checklist-item" id="include-null-checkbox">
            <input
              type="checkbox"
              checked={includeNullCategories}
              onChange={(e) => setIncludeNullCategories(e.target.checked)}
            />
            {' '}Include combinations with null reactant types
          </label>
        </div>

        {/* Action buttons row */}
        <div className="filter-options-row downloads">
          <span id="download-buttons" className="options-actions">
            <button className="options-btn" onClick={handleDownloadCSV}>Download CSV</button>
            <button className="options-btn" onClick={handleDownloadPNG}>Download PNG</button>
          </span>
          <button className="options-btn options-btn-reset" id="reset-options-btn" onClick={resetOptions}>Reset Options</button>
        </div>
      </div>
    </>
  );
}
