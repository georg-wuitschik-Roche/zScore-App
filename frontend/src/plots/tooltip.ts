/**
 * Tooltip data model for the custom HTML hover tooltip.
 *
 * Plot traces carry a `TooltipSource` per point inside `customdata`; Plotly hands
 * each element straight back on `plotly_hover`, where `buildTooltipModel()` turns
 * it into the flat, plain-text rows the tooltip renders. Values must stay markup
 * free — every row is individually copyable to the clipboard.
 */

import type { Datum } from 'plotly.js';
import { REAGENT_COLS, type Row } from '../data/types';

// ── Formatting helpers ────────────────────────────────────────────────

/** Safe string for display — null/undefined → '' */
function s(val: unknown): string {
  if (val === null || val === undefined || val === '') return '';
  return String(val);
}

/** Format z-Score to 3 decimal places */
function fmtZ(val: unknown): string {
  if (val === null || val === undefined) return '';
  const n = Number(val);
  return isNaN(n) ? '' : n.toFixed(3);
}

/** Format area to 2 decimal places with % */
function fmtArea(val: unknown): string {
  if (val === null || val === undefined) return '';
  const n = Number(val);
  return isNaN(n) ? '' : n.toFixed(2) + '%';
}

// ── Display model ─────────────────────────────────────────────────────

/** A single label/value pair. The value is plain text and copyable on click. */
export interface TooltipRow {
  label: string;
  value: string;
}

export interface TooltipSection {
  heading: string;
  rows: TooltipRow[];
}

export interface TooltipModel {
  title: string;
  subtitle?: string;
  sections: TooltipSection[];
}

// ── Payload carried in trace.customdata ───────────────────────────────

/** Per-point hover payload. Kept compact — the model is built lazily on hover. */
export type TooltipSource =
  | { kind: 'point'; row: Row; elnCount: number }
  | { kind: 'median'; category: string; medianVal: number; n: number; elnCount: number }
  | {
      kind: 'cell';
      yCol: string;
      yLabel: string;
      xCol: string;
      xLabel: string;
      medianVal: number;
      elnCount: number;
    };

const TOOLTIP_KINDS: readonly string[] = [
  'point',
  'median',
  'cell',
] satisfies readonly TooltipSource['kind'][];

/** Runtime guard for values coming back out of Plotly's hover event. */
export function isTooltipSource(val: unknown): val is TooltipSource {
  if (typeof val !== 'object' || val === null || Array.isArray(val)) return false;
  const kind = (val as { kind?: unknown }).kind;
  return typeof kind === 'string' && TOOLTIP_KINDS.includes(kind);
}

/** Point bounding box, relative to the graph div's offset parent (.plot-container). */
export interface TooltipAnchor {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

/**
 * `Data['customdata']` is typed as `Datum[] | Datum[][]`, but at runtime Plotly
 * passes each element through to `points[i].customdata` untouched.
 *
 * This module owns both ends of that round trip: `asCustomdata` on the way into a
 * trace, `readTooltipHover` on the way back out of a hover event. Those are the
 * only two places the Plotly typings are overridden.
 *
 * Overloaded to keep the rank narrow. Several members of Plotly's `Data` union
 * type `customdata` as `Datum[]` alone, so returning the wider
 * `Datum[] | Datum[][]` for a 1-D payload makes the whole trace unassignable to
 * `Data` — which is what broke `buildDistributionConfig` under `tsc -b`.
 */
export function asCustomdata(src: TooltipSource[]): Datum[];
export function asCustomdata(src: TooltipSource[][]): Datum[][];
export function asCustomdata(src: TooltipSource[] | TooltipSource[][]): Datum[] | Datum[][] {
  return src as unknown as Datum[] | Datum[][];
}

/**
 * Recover the payload and anchor from one Plotly hover point, or null if this
 * point carries no tooltip payload (a trace we don't decorate, e.g. the ELN
 * colorbar). Plotly sets `bbox` on every hover point, but @types/plotly.js
 * models neither it nor the passed-through `customdata`.
 */
export function readTooltipHover(
  point: unknown,
): { source: TooltipSource; anchor: TooltipAnchor } | null {
  const { customdata, bbox } = point as { customdata?: unknown; bbox?: TooltipAnchor };
  if (!isTooltipSource(customdata) || !bbox) return null;
  return { source: customdata, anchor: bbox };
}

// ── Model construction ────────────────────────────────────────────────

/** Reagent columns shown in the REAGENTS section, in display order.
 *  Constrained to REAGENT_COLS so a rename there breaks the build rather than
 *  silently dropping a row. FG A / FG B are deliberately absent — they lead the
 *  REACTION section. */
const TOOLTIP_REAGENT_COLS = [
  'Catalyst',
  'Solvent',
  'Base',
  'Ligand',
  'Additive',
  'Coupling Reagent',
  'Secondary Solvent',
] as const satisfies readonly (typeof REAGENT_COLS)[number][];

/** Append a label/value row, skipping empty values. */
function push(rows: TooltipRow[], label: string, value: string): void {
  if (value !== '') rows.push({ label, value });
}

function buildPointModel(row: Row, elnCount: number): TooltipModel {
  // REACTION first — the functional groups define what the reaction is.
  const reaction: TooltipRow[] = [];
  push(reaction, 'FG A', s(row['FG A']));
  push(reaction, 'FG B', s(row['FG B']));
  push(reaction, 'Type', s(row['Reaction Type']));

  const results: TooltipRow[] = [];
  push(results, 'z-Score', fmtZ(row['z-Score']));
  push(results, 'Area', fmtArea(row.AREA_TOTAL_REDUCED));
  push(results, 'ELNs', String(elnCount));

  const reagents: TooltipRow[] = [];
  for (const col of TOOLTIP_REAGENT_COLS) {
    push(reagents, col, s(row[col]));
  }

  const sections: TooltipSection[] = [
    { heading: 'REACTION', rows: reaction },
    { heading: 'RESULTS', rows: results },
    { heading: 'REAGENTS', rows: reagents },
  ].filter((section) => section.rows.length > 0);

  const plate = s(row.PLATENUMBER);
  const coordinate = s(row.Coordinate);
  const subtitleParts: string[] = [];
  if (plate !== '') subtitleParts.push(`Plate ${plate}`);
  if (coordinate !== '') subtitleParts.push(coordinate);

  return {
    title: s(row.ELN_ID),
    subtitle: subtitleParts.length > 0 ? subtitleParts.join(' · ') : undefined,
    sections,
  };
}

/** Turn a hover payload into the rows the tooltip renders. */
export function buildTooltipModel(src: TooltipSource): TooltipModel {
  switch (src.kind) {
    case 'point':
      return buildPointModel(src.row, src.elnCount);

    case 'median':
      return {
        title: src.category,
        sections: [
          {
            heading: 'GROUP',
            rows: [
              { label: 'Median', value: fmtZ(src.medianVal) },
              { label: 'n', value: String(src.n) },
              { label: 'ELNs', value: String(src.elnCount) },
            ],
          },
        ],
      };

    case 'cell':
      return {
        title: `${src.yLabel} × ${src.xLabel}`,
        sections: [
          {
            heading: 'REAGENTS',
            rows: [
              { label: src.yCol, value: src.yLabel },
              { label: src.xCol, value: src.xLabel },
            ],
          },
          {
            heading: 'RESULTS',
            rows: [
              { label: 'Median z-Score', value: fmtZ(src.medianVal) },
              { label: 'ELNs', value: String(src.elnCount) },
            ],
          },
        ],
      };
  }
}
