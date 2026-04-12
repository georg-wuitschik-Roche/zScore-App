import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Plot, { Plotly } from './Plot';
import { useFilterStore } from '../stores/filterStore';
import type { PlotConfig } from '../plots/types';
import type { Row, RankDelta, ComparisonInfo } from '../data/types';
import { wrapTickLabel, RANK_BADGE_TICK_PAD } from '../plots/helpers';

const PLOT_CONFIG = { responsive: true, displayModeBar: false } as const;
const PLOT_STYLE = { width: '100%' } as const;

export function useZoomReset() {
  const [isZoomed, setIsZoomed] = useState(false);
  const plotDivRef = useRef<ReturnType<typeof Plotly.newPlot> extends Promise<infer R> ? R : unknown>(null);

  const handleInit = useCallback((_figure: unknown, graphDiv: HTMLElement) => {
    plotDivRef.current = graphDiv;
    (graphDiv as unknown as { on: (e: string, h: (d: Record<string, unknown>) => void) => void }).on(
      'plotly_relayout',
      (data: Record<string, unknown>) => {
        const keys = Object.keys(data);
        if (keys.some(k => /[xy]axis\d*\.range/.test(k))) setIsZoomed(true);
        else if (keys.some(k => /[xy]axis\d*\.autorange/.test(k))) setIsZoomed(false);
      },
    );
  }, []);

  const resetZoom = useCallback(() => {
    if (plotDivRef.current) {
      Plotly.relayout(plotDivRef.current, { 'xaxis.autorange': true, 'yaxis.autorange': true });
    }
  }, []);

  return { isZoomed, setIsZoomed, handleInit, resetZoom };
}

type ConfigBuilder = (
  rows: Row[],
  reactantTypes: string[],
  presentationMode: boolean,
  rankMap?: Map<string, RankDelta> | null,
  isDark?: boolean,
  comparisonInfo?: ComparisonInfo | null,
  showElnLegend?: boolean,
) => PlotConfig;

interface Props {
  buildConfig: ConfigBuilder;
  label: string;
  rows: Row[];
  reactantTypes: string[];
  noDataHint?: string;
  rankMap?: Map<string, RankDelta> | null;
  comparisonInfo?: ComparisonInfo | null;
  heightOverride?: number;
  panelId?: string;
}

/**
 * Build a reverse lookup from rendered SVG tick text → original category name.
 * Handles label wrapping via wrapTickLabel and optional rank badge padding.
 */
function buildTickLabelLookup(categories: string[], hasRankBadges: boolean): Map<string, string> {
  const map = new Map<string, string>();
  for (const cat of categories) {
    // Also store the raw category for exact matches
    map.set(cat, cat);
    const wrapped = wrapTickLabel(cat);
    // Normalize: strip <br> → space, collapse whitespace, trim
    const norm = wrapped.replace(/<br>/g, ' ').replace(/\s+/g, ' ').trim();
    map.set(norm, cat);
    if (hasRankBadges) {
      const padded = wrapped
        .split('<br>')
        .map((line) => line + RANK_BADGE_TICK_PAD)
        .join(' ')
        .replace(/\s+/g, ' ')
        .trim();
      map.set(padded, cat);
    }
  }
  return map;
}

/** Extract text content from a ytick SVG element, normalizing tspan joins. */
function getTickText(el: Element): string {
  const tspans = el.querySelectorAll('tspan');
  if (tspans.length > 0) {
    return Array.from(tspans)
      .map((t) => t.textContent ?? '')
      .join(' ')
      .replace(/\s+/g, ' ')
      .trim();
  }
  return (el.textContent ?? '').replace(/\s+/g, ' ').trim();
}

export const DistributionView = memo(function DistributionView({ buildConfig, label, rows, reactantTypes, noDataHint, rankMap, comparisonInfo, heightOverride, panelId }: Props) {
  const presentationMode = useFilterStore((s) => s.presentationMode);
  const reactionTypes = useFilterStore((s) => s.reactionTypes);
  const isDark = useFilterStore((s) => s.theme) === 'dark';
  const showElnLegend = useFilterStore((s) => s.showElnLegend);
  const toggleCrossFilterValue = useFilterStore((s) => s.toggleCrossFilterValue);
  const panelSelection = useFilterStore((s) =>
    panelId ? s.crossFilterSelections[panelId] ?? null : null,
  );

  const { isZoomed, setIsZoomed, handleInit, resetZoom } = useZoomReset();

  const config = useMemo(() => {
    if (rows.length === 0) return null;
    const c = buildConfig(rows, reactantTypes, presentationMode, rankMap, isDark, comparisonInfo, showElnLegend);
    if (heightOverride && c.layout) {
      c.layout.height = heightOverride;
    }
    return c;
  }, [buildConfig, rows, reactantTypes, presentationMode, rankMap, isDark, comparisonInfo, showElnLegend, heightOverride]);

  // Build lookup for reverse-mapping SVG tick text → category name
  const tickLabelLookup = useMemo(() => {
    if (!panelId || !config?.layout?.yaxis) return null;
    const cats = (config.layout.yaxis as { tickvals?: string[] }).tickvals;
    if (!cats) return null;
    const hasRankBadges = !!rankMap && rankMap.size > 0;
    return buildTickLabelLookup(cats, hasRankBadges);
  }, [panelId, config, rankMap]);

  // Ref to the container div wrapping the Plot — used for ytick click delegation
  const containerRef = useRef<HTMLDivElement>(null);
  const configRef = useRef(config);
  configRef.current = config;

  // Ref for current selection — read by applyStyles without being a dependency
  const panelSelectionRef = useRef(panelSelection);
  panelSelectionRef.current = panelSelection;

  // Stable ref for tickLabelLookup — read inside applyStyles and click handler
  const tickLabelLookupRef = useRef(tickLabelLookup);
  tickLabelLookupRef.current = tickLabelLookup;

  // Apply bold styling to ytick labels based on current selection.
  // Called by onUpdate (after Plotly renders) and the selection effect.
  const applyStyles = useCallback(() => {
    const container = containerRef.current;
    const lookup = tickLabelLookupRef.current;
    if (!container || !lookup) return;
    const selected = panelSelectionRef.current ?? [];
    const ticks = container.querySelectorAll('.ytick');
    ticks.forEach((g) => {
      const textEl = g.querySelector('text') as SVGTextElement | null;
      if (textEl) {
        const text = getTickText(textEl);
        const cat = lookup.get(text);
        textEl.style.fontWeight = cat && selected.includes(cat) ? 'bold' : '';
      }
    });
  }, []);

  // Boolean gate: set up once when lookup becomes available, never tear down on value changes.
  // The effect body reads the latest lookup from tickLabelLookupRef.
  const hasTickLookup = !!tickLabelLookup;

  // Stable effect: click handler (does NOT depend on panelSelection or tickLabelLookup value)
  useEffect(() => {
    if (!panelId || !containerRef.current || !hasTickLookup) return;

    const container = containerRef.current;

    // Click delegation: handle clicks on ytick text elements
    const pid = panelId; // narrowed to string after guard
    function handleTickClick(e: MouseEvent) {
      const target = e.target as Element;
      const tickGroup = target.closest('.ytick');
      if (tickGroup) {
        const textEl = tickGroup.querySelector('text');
        if (!textEl) return;
        const text = getTickText(textEl);
        const cat = tickLabelLookupRef.current?.get(text);
        if (!cat) return;
        e.stopPropagation();
        const isMulti = e.ctrlKey || e.metaKey;
        toggleCrossFilterValue(pid, cat, isMulti);
        return;
      }
      // Click on y-axis background (margin area) → clear this panel's selection
      const svg = container.querySelector('svg.main-svg');
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const plotMarginLeft = (configRef.current?.layout?.margin as { l?: number } | undefined)?.l ?? 200;
      const clickX = e.clientX - rect.left;
      const state = useFilterStore.getState();
      if (clickX < plotMarginLeft && state.crossFilterSelections[pid]?.length > 0) {
        e.stopPropagation();
        const selections = { ...state.crossFilterSelections };
        delete selections[pid];
        const order = state.crossFilterOrder.filter((p) => p !== pid);
        useFilterStore.setState({ crossFilterSelections: selections, crossFilterOrder: order });
      }
    }

    container.addEventListener('click', handleTickClick, true);

    return () => {
      container.removeEventListener('click', handleTickClick, true);
    };
  }, [panelId, hasTickLookup, toggleCrossFilterValue]);

  // Lightweight effect: re-apply bold styling when selection changes (no observer churn)
  useEffect(() => {
    if (panelId) applyStyles();
  }, [panelId, panelSelection, applyStyles]);

  useEffect(() => { setIsZoomed(false); }, [config, setIsZoomed]);

  if (reactionTypes.length === 0 || reactantTypes.length === 0) {
    const missing: string[] = [];
    if (reactionTypes.length === 0) missing.push('reaction type');
    if (reactantTypes.length === 0) missing.push('reactant type');
    return (
      <div className="plot-container empty-state">
        <img src="/assets/logo.svg" alt="" className="empty-state-logo" />
        <p className="no-data-message">
          Select a {missing.join(' and ')} to display the {label}.
        </p>
      </div>
    );
  }

  if (!config) {
    return (
      <div className="plot-container">
        <p className="no-data-message">
          {noDataHint ?? 'No data available for the current filter selection.'}
        </p>
      </div>
    );
  }

  return (
    <div className={`plot-container plot-container--zoomable${panelId ? ' cross-filter-enabled' : ''}`} ref={containerRef}>
      {isZoomed && (
        <button className="reset-zoom-btn" onClick={resetZoom} title="Reset zoom">
          Reset Zoom
        </button>
      )}
      <Plot
        key={showElnLegend ? 'legend' : 'no-legend'}
        data={config.data}
        layout={config.layout}
        config={PLOT_CONFIG}
        style={PLOT_STYLE}
        useResizeHandler
        onInitialized={handleInit}
        onUpdate={applyStyles}
      />
    </div>
  );
});
