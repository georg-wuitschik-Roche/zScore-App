/**
 * Hover state for the custom plot tooltip.
 *
 * Plot traces set `hoverinfo: 'none'` so Plotly fires `plotly_hover` without
 * drawing its own SVG label. This hook turns those events into tooltip state and
 * keeps the tooltip alive while the cursor travels onto it: unhover only schedules
 * a close, which `onTooltipEnter` cancels.
 *
 * `attach` binds every event the tooltip cares about — including the relayout that
 * dismisses it on zoom/pan — so no other hook has to know the tooltip exists.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { PlotHoverEvent } from 'plotly.js';
import { bindPlotlyEvent } from '../plots/plotlyEvents';
import { buildTooltipModel, readTooltipHover, type TooltipAnchor, type TooltipModel } from '../plots/tooltip';

/** Grace period between leaving a point and the tooltip closing, in ms.
 *  Long enough to cross the gap to the tooltip, short enough to feel immediate. */
const CLOSE_DELAY_MS = 220;

interface PlotTooltipState {
  model: TooltipModel;
  anchor: TooltipAnchor;
  /** Bumped on every content swap — used as the React key so copy flashes reset. */
  key: number;
}

export function usePlotTooltip() {
  const [tooltip, setTooltip] = useState<PlotTooltipState | null>(null);
  const timerRef = useRef<number | null>(null);
  const keyRef = useRef(0);

  const cancelClose = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const close = useCallback(() => {
    cancelClose();
    setTooltip(null);
  }, [cancelClose]);

  const scheduleClose = useCallback(() => {
    cancelClose();
    timerRef.current = window.setTimeout(() => {
      timerRef.current = null;
      setTooltip(null);
    }, CLOSE_DELAY_MS);
  }, [cancelClose]);

  const onHover = useCallback(
    (event: Readonly<PlotHoverEvent>) => {
      // Scan rather than taking points[0]: a distribution point and its invisible
      // median marker can both be in range, and only one carries a payload.
      for (const point of event.points) {
        const hovered = readTooltipHover(point);
        if (!hovered) continue;
        cancelClose();
        keyRef.current += 1;
        setTooltip({
          model: buildTooltipModel(hovered.source),
          anchor: hovered.anchor,
          key: keyRef.current,
        });
        return;
      }
    },
    [cancelClose],
  );

  /** Bind the tooltip's Plotly events. Call from the plot's onInitialized. */
  const attach = useCallback(
    (graphDiv: HTMLElement) => {
      bindPlotlyEvent(graphDiv, 'plotly_hover', onHover);
      bindPlotlyEvent(graphDiv, 'plotly_unhover', scheduleClose);
      // Zoom/pan moves the points out from under the tooltip — dismiss it.
      bindPlotlyEvent(graphDiv, 'plotly_relayout', close);
    },
    [onHover, scheduleClose, close],
  );

  useEffect(() => cancelClose, [cancelClose]);

  return {
    tooltip,
    attach,
    onTooltipEnter: cancelClose,
    onTooltipLeave: scheduleClose,
    /** Close immediately — for data changes. */
    close,
  };
}
