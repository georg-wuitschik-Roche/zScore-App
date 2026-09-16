/**
 * Shared furniture around a Plotly plot: zoom tracking and the hover tooltip.
 *
 * DistributionView and HeatmapView both render a `.plot-container--zoomable` with
 * a reset-zoom button, a <Plot>, and a tooltip. This hook owns that wiring so the
 * views only supply the plot itself — and so a third plot view gets it for free.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { Plotly } from '../components/Plot';
import { PlotTooltip } from '../components/PlotTooltip';
import { bindPlotlyEvent } from '../plots/plotlyEvents';
import { usePlotTooltip } from './usePlotTooltip';

/** Tracks whether the user has zoomed/panned, and can put the axes back. */
function useZoom() {
  const [isZoomed, setIsZoomed] = useState(false);
  const plotDivRef = useRef<HTMLElement | null>(null);

  const attach = useCallback((graphDiv: HTMLElement) => {
    plotDivRef.current = graphDiv;
    bindPlotlyEvent<Record<string, unknown>>(graphDiv, 'plotly_relayout', (data) => {
      const keys = Object.keys(data);
      if (keys.some((k) => /[xy]axis\d*\.range/.test(k))) setIsZoomed(true);
      else if (keys.some((k) => /[xy]axis\d*\.autorange/.test(k))) setIsZoomed(false);
    });
  }, []);

  const resetZoom = useCallback(() => {
    if (plotDivRef.current) {
      Plotly.relayout(plotDivRef.current, { 'xaxis.autorange': true, 'yaxis.autorange': true });
    }
  }, []);

  /** Forget any tracked zoom — the plot underneath has been replaced. */
  const clear = useCallback(() => setIsZoomed(false), []);

  return { isZoomed, attach, resetZoom, clear };
}

/**
 * @param config - the plot config; a new object means a fresh plot, which resets
 *                 zoom state and dismisses any open tooltip.
 */
export function usePlotChrome(config: unknown) {
  const containerRef = useRef<HTMLDivElement>(null);
  const zoom = useZoom();
  const tooltip = usePlotTooltip();

  const { attach: attachZoom } = zoom;
  const { attach: attachTooltip } = tooltip;

  const onInitialized = useCallback(
    (_figure: unknown, graphDiv: HTMLElement) => {
      attachZoom(graphDiv);
      attachTooltip(graphDiv);
    },
    [attachZoom, attachTooltip],
  );

  const { clear: clearZoom } = zoom;
  const { close: closeTooltip } = tooltip;
  useEffect(() => {
    clearZoom();
    closeTooltip();
  }, [config, clearZoom, closeTooltip]);

  const tooltipNode = tooltip.tooltip ? (
    <PlotTooltip
      key={tooltip.tooltip.key}
      model={tooltip.tooltip.model}
      anchor={tooltip.tooltip.anchor}
      container={containerRef}
      onMouseEnter={tooltip.onTooltipEnter}
      onMouseLeave={tooltip.onTooltipLeave}
    />
  ) : null;

  return {
    containerRef,
    isZoomed: zoom.isZoomed,
    resetZoom: zoom.resetZoom,
    onInitialized,
    tooltipNode,
  };
}
