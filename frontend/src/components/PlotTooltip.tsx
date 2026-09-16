/**
 * Custom HTML hover tooltip for the plots.
 *
 * Replaces Plotly's native SVG hover label, which is non-interactive and vanishes
 * the moment the cursor leaves the point. This one stays open while the cursor is
 * over it, its text is selectable, and clicking a row copies that value.
 */

import { useCallback, useEffect, useLayoutEffect, useRef, useState, type RefObject } from 'react';
import { copyText } from '../data/clipboard';
import type { TooltipAnchor, TooltipModel } from '../plots/tooltip';

/** Visual gap between the data point and the tooltip. Bridged by .plot-tooltip::before. */
const GAP = 10;
/** Minimum clearance from the container / viewport edge. */
const EDGE = 8;
/** How long the value stays flipped to "Copied" before flipping back.
 *  Covers the 0.6s flip each way plus a readable pause. */
const FLASH_MS = 2000;

interface Props {
  model: TooltipModel;
  anchor: TooltipAnchor;
  /** The `.plot-container` the anchor coordinates are relative to. */
  container: RefObject<HTMLDivElement | null>;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}

export function PlotTooltip({ model, anchor, container, onMouseEnter, onMouseLeave }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const flashRef = useRef<number | null>(null);
  const aliveRef = useRef(true);

  // Measure and place before paint. Written straight to the DOM rather than through
  // state: positioning is external-system sync, and hovers are frequent enough that
  // the extra render per hover is worth avoiding. The CSS keeps .plot-tooltip hidden
  // until this runs, so it never paints at the wrong position.
  useLayoutEffect(() => {
    const el = ref.current;
    const box = container.current;
    if (!el || !box) return;

    const { offsetWidth: width, offsetHeight: height } = el;
    const containerRect = box.getBoundingClientRect();

    // Horizontal: prefer right of the point, flip left only on *viewport* overflow.
    // The boundary is deliberately the window, not the container: in split mode each
    // panel is a narrow column, and measuring against it would flip the tooltip left
    // while the screen still has plenty of room. Nothing between here and the body
    // clips or creates a stacking context, so overhanging a column is safe.
    let left = anchor.x1 + GAP;
    let flipped = false;
    if (containerRect.left + left + width > window.innerWidth - EDGE) {
      left = anchor.x0 - GAP - width;
      flipped = true;
    }
    // Flipping may push it off the left of the screen — clamp to the viewport edge.
    if (containerRect.left + left < EDGE) {
      left = EDGE - containerRect.left;
    }

    // Vertical: centre on the point, then clamp into the container ∩ viewport.
    // The viewport term matters — plot containers are far taller than the screen.
    let top = (anchor.y0 + anchor.y1) / 2 - height / 2;
    const minTop = Math.max(EDGE, EDGE - containerRect.top);
    const maxTop = Math.min(
      box.clientHeight - height - EDGE,
      window.innerHeight - EDGE - height - containerRect.top,
    );
    top = maxTop < minTop ? minTop : Math.min(Math.max(top, minTop), maxTop);

    el.style.left = `${left}px`;
    el.style.top = `${top}px`;
    el.classList.toggle('plot-tooltip--left', flipped);
    el.style.visibility = 'visible';
  }, [anchor, model, container]);

  useEffect(() => {
    // Set on setup, not just cleared on teardown: StrictMode runs setup → cleanup →
    // setup, so a teardown-only flag would stay false for the live instance.
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
      if (flashRef.current !== null) window.clearTimeout(flashRef.current);
    };
  }, []);

  const handleRowClick = useCallback((event: React.MouseEvent, id: string, value: string) => {
    event.stopPropagation();
    // The user was drag-selecting text — don't hijack their click.
    if ((window.getSelection()?.toString() ?? '') !== '') return;

    void copyText(value).then((ok) => {
      // The next hover remounts this component; don't arm a timer on a dead one.
      if (!ok || !aliveRef.current) return;
      if (flashRef.current !== null) window.clearTimeout(flashRef.current);
      setCopiedId(id);
      flashRef.current = window.setTimeout(() => {
        setCopiedId(null);
        flashRef.current = null;
      }, FLASH_MS);
    });
  }, []);

  return (
    <div
      ref={ref}
      className="plot-tooltip"
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      role="tooltip"
    >
      <div className="plot-tooltip-head">
        <span className="plot-tooltip-title">{model.title}</span>
        {model.subtitle && <span className="plot-tooltip-subtitle">{model.subtitle}</span>}
      </div>
      <div className="plot-tooltip-body">
        {model.sections.map((section) => (
          <div className="plot-tooltip-section" key={section.heading}>
            <div className="plot-tooltip-heading">{section.heading}</div>
            {section.rows.map((row) => {
              const id = `${section.heading}:${row.label}`;
              return (
                <div
                  key={id}
                  className={`plot-tooltip-row${copiedId === id ? ' plot-tooltip-row--copied' : ''}`}
                  onClick={(e) => handleRowClick(e, id, row.value)}
                  title="Click to copy"
                >
                  <span className="plot-tooltip-label">{row.label}</span>
                  {/* Only the value flips over to "Copied" and back. */}
                  <span className="plot-tooltip-flip">
                    <span className="plot-tooltip-value plot-tooltip-face">{row.value}</span>
                    <span className="plot-tooltip-face plot-tooltip-face--back" aria-hidden="true">
                      Copied
                    </span>
                  </span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
