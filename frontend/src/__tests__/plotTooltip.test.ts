/**
 * Tests for the hover tooltip data model from plots/tooltip.ts.
 *
 * Covers section ordering (FG A / FG B lead), empty-value skipping, the three
 * payload variants, the runtime guard, and the invariant that every value is
 * plain copyable text — the tooltip replaced an HTML-in-customdata approach.
 */

import { describe, it, expect } from 'vitest';
import {
  buildTooltipModel,
  isTooltipSource,
  type TooltipRow,
  type TooltipSource,
} from '../plots/tooltip';
import { createHeatmapConfig } from '../plots/heatmap';
import type { Row } from '../data/types';

function makeRow(overrides: Partial<Row> = {}): Row {
  return {
    ELN_ID: 'ELN001',
    PLATENUMBER: '7',
    Coordinate: 'A3',
    AREA_TOTAL_REDUCED: 63.1,
    Additive: 'LiCl',
    Base: 'K3PO4',
    Catalyst: 'Pd(OAc)2',
    'Coupling Reagent': null,
    Solvent: 'DMF',
    Ligand: 'XPhos',
    'Secondary Solvent': null,
    'Reaction Type': 'Buchwald-Hartwig',
    'FG A': 'ArBr',
    'FG B': 'RNH2',
    'z-Score': 1.8423,
    ...overrides,
  };
}

/** Flatten every row of a model, for invariants that apply to all of them. */
function allRows(src: TooltipSource): TooltipRow[] {
  return buildTooltipModel(src).sections.flatMap((s) => s.rows);
}

const POINT: TooltipSource = { kind: 'point', row: makeRow(), elnCount: 14 };
const MEDIAN: TooltipSource = {
  kind: 'median', category: 'Pd(OAc)2', medianVal: 1.5, n: 12, elnCount: 4,
};
const CELL: TooltipSource = {
  kind: 'cell', yCol: 'Catalyst', yLabel: 'Pd(OAc)2', xCol: 'Solvent',
  xLabel: 'DMF', medianVal: 0.75, elnCount: 9,
};

describe('buildTooltipModel — point', () => {
  it('uses the ELN ID as title and plate/coordinate as subtitle', () => {
    const model = buildTooltipModel(POINT);
    expect(model.title).toBe('ELN001');
    expect(model.subtitle).toBe('Plate 7 · A3');
  });

  it('leads with FG A and FG B — they define the reaction', () => {
    const model = buildTooltipModel(POINT);
    const first = model.sections[0];
    expect(first.heading).toBe('REACTION');
    expect(first.rows[0]).toEqual({ label: 'FG A', value: 'ArBr' });
    expect(first.rows[1]).toEqual({ label: 'FG B', value: 'RNH2' });
    expect(first.rows[2]).toEqual({ label: 'Type', value: 'Buchwald-Hartwig' });
  });

  it('formats z-Score to 3 decimals and area to 2 with a percent sign', () => {
    const results = buildTooltipModel(POINT).sections.find((s) => s.heading === 'RESULTS');
    expect(results?.rows).toEqual([
      { label: 'z-Score', value: '1.842' },
      { label: 'Area', value: '63.10%' },
      { label: 'ELNs', value: '14' },
    ]);
  });

  it('lists reagents in display order, excluding the FG columns', () => {
    const reagents = buildTooltipModel(POINT).sections.find((s) => s.heading === 'REAGENTS');
    expect(reagents?.rows.map((r) => r.label)).toEqual([
      'Ligand', 'Catalyst', 'Base', 'Solvent', 'Additive',
    ]);
  });

  it('orders every reagent column, not just the ones the base fixture fills', () => {
    // POINT leaves Coupling Reagent and Secondary Solvent null, so the case
    // above silently skips them — fill all seven to pin the whole order.
    const src: TooltipSource = {
      kind: 'point',
      row: makeRow({ 'Coupling Reagent': 'HATU', 'Secondary Solvent': 'H2O' }),
      elnCount: 5,
    };
    const reagents = buildTooltipModel(src).sections.find((s) => s.heading === 'REAGENTS');
    expect(reagents?.rows.map((r) => r.label)).toEqual([
      'Ligand', 'Catalyst', 'Coupling Reagent', 'Base',
      'Solvent', 'Secondary Solvent', 'Additive',
    ]);
  });

  it('skips null and empty values rather than rendering blank rows', () => {
    const src: TooltipSource = {
      kind: 'point',
      row: makeRow({ Ligand: null, Additive: '', 'FG B': null, AREA_TOTAL_REDUCED: null }),
      elnCount: 3,
    };
    const labels = allRows(src).map((r) => r.label);
    expect(labels).not.toContain('Ligand');
    expect(labels).not.toContain('Additive');
    expect(labels).not.toContain('FG B');
    expect(labels).not.toContain('Area');
    expect(labels).toContain('FG A');
  });

  it('omits the subtitle when plate and coordinate are both missing', () => {
    const model = buildTooltipModel({
      kind: 'point',
      row: makeRow({ PLATENUMBER: '', Coordinate: '' }),
      elnCount: 1,
    });
    expect(model.subtitle).toBeUndefined();
  });
});

describe('buildTooltipModel — median and cell', () => {
  it('builds a median marker model', () => {
    const model = buildTooltipModel(MEDIAN);
    expect(model.title).toBe('Pd(OAc)2');
    expect(model.sections).toHaveLength(1);
    expect(model.sections[0].rows).toEqual([
      { label: 'Median', value: '1.500' },
      { label: 'n', value: '12' },
      { label: 'ELNs', value: '4' },
    ]);
  });

  it('builds a heatmap cell model labelled by the axis columns', () => {
    const model = buildTooltipModel(CELL);
    expect(model.title).toBe('Pd(OAc)2 × DMF');
    expect(model.sections[0].rows).toEqual([
      { label: 'Catalyst', value: 'Pd(OAc)2' },
      { label: 'Solvent', value: 'DMF' },
    ]);
    expect(model.sections[1].rows).toEqual([
      { label: 'Median z-Score', value: '0.750' },
      { label: 'ELNs', value: '9' },
    ]);
  });
});

describe('tooltip value invariants', () => {
  it.each([
    ['point', POINT],
    ['median', MEDIAN],
    ['cell', CELL],
  ])('%s values are non-empty plain text with no markup', (_kind, src) => {
    const rows = allRows(src);
    expect(rows.length).toBeGreaterThan(0);
    for (const row of rows) {
      expect(row.value).not.toBe('');
      // Values are copied verbatim to the clipboard — markup must never leak in.
      expect(row.value).not.toMatch(/<[^>]+>/);
    }
  });
});

describe('isTooltipSource', () => {
  it('accepts every payload variant', () => {
    expect(isTooltipSource(POINT)).toBe(true);
    expect(isTooltipSource(MEDIAN)).toBe(true);
    expect(isTooltipSource(CELL)).toBe(true);
  });

  it('rejects anything else Plotly might hand back', () => {
    for (const val of [null, undefined, 'str', 42, [], {}, { kind: 'nope' }, ['ELN001']]) {
      expect(isTooltipSource(val)).toBe(false);
    }
  });
});

describe('heatmap customdata', () => {
  const ROWS: Row[] = [
    makeRow({ Catalyst: 'A', Solvent: 'DMF', 'z-Score': 1.0, ELN_ID: 'E1' }),
    makeRow({ Catalyst: 'A', Solvent: 'THF', 'z-Score': 2.0, ELN_ID: 'E2' }),
    makeRow({ Catalyst: 'B', Solvent: 'DMF', 'z-Score': 3.0, ELN_ID: 'E3' }),
    makeRow({ Catalyst: 'B', Solvent: 'THF', 'z-Score': 4.0, ELN_ID: 'E4' }),
  ];

  it('is a cell matrix matching z, indexed [row][col]', () => {
    const config = createHeatmapConfig(ROWS, ['Catalyst', 'Solvent']);
    const trace = config.data[0] as Record<string, unknown>;

    expect(trace.hoverinfo).toBe('none');
    expect(trace.hovertemplate).toBeUndefined();

    const z = trace.z as (number | null)[][];
    const customdata = trace.customdata as unknown as TooltipSource[][];
    expect(customdata).toHaveLength(z.length);

    customdata.forEach((hoverRow, r) => {
      expect(hoverRow).toHaveLength(z[r].length);
      hoverRow.forEach((src, c) => {
        expect(src.kind).toBe('cell');
        if (src.kind !== 'cell') return;
        if (z[r][c] !== null) expect(src.medianVal).toBe(z[r][c]);
        expect(src.yLabel).toBe((trace.y as string[])[r]);
        expect(src.xLabel).toBe((trace.x as string[])[c]);
      });
    });
  });
});
