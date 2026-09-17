import { describe, it, expect } from 'vitest';
import {
  scaleFonts,
  resolveExportHeight,
  namespaceSvgIds,
  decodeSvgDataUrl,
  labelMetrics,
  EXPORT_WIDTH,
} from '../plots/export';

describe('scaleFonts', () => {
  it('scales font sizes in layout, including nested and annotation fonts', () => {
    const layout = {
      font: { size: 14, family: 'DM Sans' },
      xaxis: { title: { font: { size: 18 } }, tickfont: { size: 13 } },
      annotations: [{ text: 'NEW', font: { size: 15 } }],
    };

    const out = scaleFonts(layout, 2);

    expect(out.font.size).toBe(28);
    expect(out.xaxis.title.font.size).toBe(36);
    expect(out.xaxis.tickfont.size).toBe(26);
    expect(out.annotations[0].font.size).toBe(30);
    // Non-size font properties survive untouched
    expect(out.font.family).toBe('DM Sans');
  });

  it('scales trace-level fonts (heatmap textfont, colorbar)', () => {
    const data = [
      {
        type: 'heatmap',
        textfont: { size: 11 },
        colorbar: { title: { font: { size: 14 } }, tickfont: { size: 10 } },
      },
    ];

    const out = scaleFonts(data, 1.5);

    expect(out[0].textfont.size).toBe(16.5);
    expect(out[0].colorbar.title.font.size).toBe(21);
    expect(out[0].colorbar.tickfont.size).toBe(15);
  });

  it('leaves marker.size and line.width untouched', () => {
    const data = [
      {
        type: 'box',
        marker: { size: 6, color: '#fff' },
        line: { width: 1.5 },
        hoverlabel: { font: { size: 14 } },
      },
      { type: 'scatter', marker: { color: 'rgba(0,0,0,0)', size: 20 } },
    ];

    // Only the nested font is scaled; every other `size`/`width` is geometry.
    expect(scaleFonts(data, 3)).toEqual([
      {
        type: 'box',
        marker: { size: 6, color: '#fff' },
        line: { width: 1.5 },
        hoverlabel: { font: { size: 42 } },
      },
      { type: 'scatter', marker: { color: 'rgba(0,0,0,0)', size: 20 } },
    ]);
  });

  it('does not mutate its input', () => {
    const layout = { font: { size: 14 }, annotations: [{ font: { size: 10 } }] };

    const out = scaleFonts(layout, 2);

    expect(layout.font.size).toBe(14);
    expect(layout.annotations[0].font.size).toBe(10);
    expect(out).not.toBe(layout);
    expect(out.font).not.toBe(layout.font);
  });

  it('is a no-op at factor 1 but still returns a copy', () => {
    const layout = { font: { size: 14 } };
    const out = scaleFonts(layout, 1);
    expect(out).toEqual(layout);
    expect(out).not.toBe(layout);
  });
});

describe('resolveExportHeight', () => {
  function elementOf(clientWidth: number, clientHeight: number): HTMLElement {
    return { clientWidth, clientHeight } as HTMLElement;
  }

  it("preserves on-screen proportions for 'auto'", () => {
    // A tall 20-category boxplot: 800 wide on screen, 2200 tall
    expect(resolveExportHeight('auto', elementOf(800, 2200))).toBe(EXPORT_WIDTH * 2.75);
    expect(resolveExportHeight('auto', elementOf(1000, 500))).toBe(EXPORT_WIDTH * 0.5);
  });

  it('falls back to a sane height when the element has no measured size', () => {
    expect(resolveExportHeight('auto', elementOf(0, 0))).toBe(800);
  });

  it('derives height from preset ratios', () => {
    const el = elementOf(800, 2200);
    expect(resolveExportHeight('16:9', el)).toBe(900);
    expect(resolveExportHeight('4:3', el)).toBe(1200);
    expect(resolveExportHeight('1:1', el)).toBe(1600);
    expect(resolveExportHeight('3:4', el)).toBe(2133);
  });

  it('ignores the element for preset ratios', () => {
    expect(resolveExportHeight('1:1', elementOf(10, 10000))).toBe(EXPORT_WIDTH);
  });
});

describe('namespaceSvgIds', () => {
  // Mirrors real Plotly output: clip paths use url(#id), but the colorbar
  // gradient is referenced as style="fill: url('#id')" with quotes.
  const svg =
    '<svg><defs><clipPath id="clipABC"><rect/></clipPath>' +
    '<linearGradient id="gradXYZ"><stop/></linearGradient></defs>' +
    '<g clip-path="url(#clipABC)"/>' +
    '<rect class="cbfill" style="fill: url(\'#gradXYZ\');"/>' +
    '<use xlink:href="#clipABC"/></svg>';

  it('rewrites definitions and references together', () => {
    const out = namespaceSvgIds(svg, 'p1-');
    expect(out).toContain('id="p1-clipABC"');
    expect(out).toContain('url(#p1-clipABC)');
    expect(out).toContain('xlink:href="#p1-clipABC"');
  });

  it("rewrites quoted url('#id') refs — the colorbar gradient", () => {
    const out = namespaceSvgIds(svg, 'p1-');
    expect(out).toContain('id="p1-gradXYZ"');
    expect(out).toContain("url('#p1-gradXYZ')");
  });

  it('leaves no reference pointing at a missing id', () => {
    const out = namespaceSvgIds(svg, 'p0-');
    const defined = new Set([...out.matchAll(/id="([^"]+)"/g)].map((m) => m[1]));
    const referenced = [...out.matchAll(/url\(['"]?#([^'")]+)['"]?\)/g)].map((m) => m[1]);
    expect(referenced.length).toBeGreaterThan(0);
    expect(referenced.filter((r) => !defined.has(r))).toEqual([]);
  });

  it('produces no shared ids across panels', () => {
    const a = namespaceSvgIds(svg, 'p0-');
    const b = namespaceSvgIds(svg, 'p1-');
    const idsOf = (s: string) => [...s.matchAll(/id="([^"]+)"/g)].map((m) => m[1]);
    const shared = idsOf(a).filter((id) => idsOf(b).includes(id));
    expect(idsOf(a)).toHaveLength(2);
    expect(shared).toEqual([]);
  });
});

describe('labelMetrics', () => {
  it('scales the split-panel label with the font setting', () => {
    expect(labelMetrics(1).fontSize).toBe(22);
    expect(labelMetrics(1.5).fontSize).toBe(33);
    expect(labelMetrics(0.85).fontSize).toBeCloseTo(18.7);
  });

  it('grows the label band so larger labels still fit above the panel', () => {
    const base = labelMetrics(1);
    const big = labelMetrics(1.5);
    expect(big.height).toBeGreaterThan(base.height);
    // Baseline must leave descender room inside the band.
    for (const m of [base, big]) {
      expect(m.baseline).toBeLessThan(m.height);
      expect(m.height).toBeGreaterThanOrEqual(m.fontSize);
    }
  });
});

describe('decodeSvgDataUrl', () => {
  it('decodes a percent-encoded payload', () => {
    const svg = '<svg><text>a b&c</text></svg>';
    expect(decodeSvgDataUrl(`data:image/svg+xml,${encodeURIComponent(svg)}`)).toBe(svg);
  });

  it('decodes a base64 payload', () => {
    const svg = '<svg></svg>';
    expect(decodeSvgDataUrl(`data:image/svg+xml;base64,${btoa(svg)}`)).toBe(svg);
  });
});
