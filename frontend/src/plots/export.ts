/**
 * Plot image export — PNG (raster) and SVG (vector).
 *
 * Export settings (aspect ratio, font size) apply to the exported file only;
 * the on-screen plots are never re-rendered. That works because Plotly.toImage()
 * accepts a figure object `{ data, layout }` as well as a graph div — so we read
 * the live figure off the plot element, transform a deep copy, and export that.
 */

import type { Data, Layout } from 'plotly.js';
import { downloadBlob, downloadDataUrl, downloadTextFile } from '../data/download';
import { MONO_FONT } from './helpers';

export type ExportFormat = 'png' | 'svg';
export type AspectRatio = 'auto' | '16:9' | '4:3' | '1:1' | '3:4';

export interface ExportSettings {
  format: ExportFormat;
  aspectRatio: AspectRatio;
  fontScale: number;
}

/** Exported images are always this wide; height follows the aspect ratio. */
export const EXPORT_WIDTH = 1600;

export const ASPECT_RATIOS: { value: AspectRatio; label: string }[] = [
  { value: 'auto', label: 'Auto' },
  { value: '16:9', label: '16:9' },
  { value: '4:3', label: '4:3' },
  { value: '1:1', label: '1:1' },
  { value: '3:4', label: '3:4' },
];

export const FONT_SCALES: { value: number; label: string }[] = [
  { value: 0.85, label: 'Small' },
  { value: 1, label: 'Default' },
  { value: 1.25, label: 'Large' },
  { value: 1.5, label: 'X-Large' },
];

/** Split-panel grid geometry, shared by the PNG and SVG compositors. */
const GRID = { gap: 8, maxCols: 3, labelFontSize: 22 };
const LABEL_COLOR = '#6b7280';

// ── Pure helpers ──────────────────────────────────────────────────────

/** Keys whose value is a font spec — only `size` inside these gets scaled.
 *  Restricting by key name is what keeps marker.size and line.width intact. */
const FONT_KEYS = new Set(['font', 'tickfont', 'textfont', 'titlefont', 'insidetextfont', 'outsidetextfont']);

/**
 * Deep-copy a figure fragment with every font size multiplied by `factor`.
 * Walks traces and layout alike — heatmap `textfont` and colorbar fonts live
 * on traces, not layout. Never mutates the input.
 */
export function scaleFonts<T>(node: T, factor: number): T {
  return walk(node, factor, false) as T;
}

function walk(node: unknown, factor: number, inFont: boolean): unknown {
  if (Array.isArray(node)) return node.map((v) => walk(v, factor, inFont));
  if (node === null || typeof node !== 'object') return node;

  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(node as Record<string, unknown>)) {
    if (inFont && key === 'size' && typeof value === 'number') {
      out[key] = value * factor;
    } else {
      out[key] = walk(value, factor, FONT_KEYS.has(key));
    }
  }
  return out;
}

/**
 * Export height for a plot rendered at `width`. 'auto' preserves the on-screen
 * proportions, which matters because distribution plots grow taller with
 * category count.
 *
 * `width` is the width of the individual plot, which in split mode is the
 * panel width rather than the full canvas — passing EXPORT_WIDTH there would
 * stretch every panel by the column count.
 */
export function resolveExportHeight(ratio: AspectRatio, el: HTMLElement, width = EXPORT_WIDTH): number {
  if (ratio === 'auto') {
    const w = el.clientWidth;
    const h = el.clientHeight;
    if (!w || !h) return Math.round(width * 0.5);
    return Math.round(width * (h / w));
  }
  const [rw, rh] = ratio.split(':').map(Number);
  return Math.round(width * (rh / rw));
}

/**
 * Prefix every id and internal reference in an SVG document.
 *
 * Merging panel SVGs into one root puts their clip-path ids in a shared
 * namespace, where a duplicate would make one panel clip against another's
 * path and disappear. Plotly does seed these ids with a random per-export
 * string, so this is belt-and-braces — but it is cheap, and it makes the
 * compositor correct by construction rather than by Plotly's internals.
 */
export function namespaceSvgIds(svgText: string, prefix: string): string {
  return svgText
    .replace(/\bid="([^"]+)"/g, (_m, id: string) => `id="${prefix}${id}"`)
    // Plotly writes clip paths as url(#id) but colorbar gradients as
    // style="fill: url('#id')" — miss the quoted form and the gradient
    // reference dangles, so the ELN legend bar exports blank.
    .replace(/url\((['"]?)#([^'")]+)\1\)/g, (_m, q: string, id: string) => `url(${q}#${prefix}${id}${q})`)
    .replace(/\b(xlink:href|href)="#([^"]+)"/g, (_m, attr: string, id: string) => `${attr}="#${prefix}${id}"`);
}

/** Decode the `data:image/svg+xml,...` URI that Plotly.toImage returns. */
export function decodeSvgDataUrl(dataUrl: string): string {
  const comma = dataUrl.indexOf(',');
  const payload = dataUrl.slice(comma + 1);
  return dataUrl.slice(0, comma).includes(';base64') ? atob(payload) : decodeURIComponent(payload);
}

function escapeXml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** Panel label band, sized from the label font so it grows with the font setting. */
export function labelMetrics(fontScale: number) {
  const fontSize = GRID.labelFontSize * fontScale;
  return { fontSize, baseline: Math.round(fontSize), height: Math.round(fontSize * 1.6) };
}

/** Width of one panel in an n-panel grid. Needed before the height, since
 *  'auto' sizes each panel against its own width. */
export function panelWidthFor(n: number): number {
  const cols = Math.min(n, GRID.maxCols);
  return Math.round((EXPORT_WIDTH - GRID.gap * (cols - 1)) / cols);
}

/** Grid layout for n split panels at a given panel size. */
function gridGeometry(n: number, panelHeight: number, fontScale: number) {
  const cols = Math.min(n, GRID.maxCols);
  const rows = Math.ceil(n / cols);
  const panelWidth = panelWidthFor(n);
  const label = labelMetrics(fontScale);
  const cellHeight = panelHeight + label.height;
  const totalHeight = rows * cellHeight + (rows - 1) * GRID.gap;
  const position = (i: number) => ({
    x: (i % cols) * (panelWidth + GRID.gap),
    y: Math.floor(i / cols) * (cellHeight + GRID.gap),
  });
  return { cols, rows, panelWidth, cellHeight, totalHeight, label, position };
}

/** Pixel size the current selection would produce — shown in the export dialog
 *  so the aspect ratio choice is concrete before committing to a download. */
export function previewExportSize(
  plotEls: HTMLElement[],
  aspectRatio: AspectRatio,
  fontScale: number,
): { width: number; height: number } | null {
  if (plotEls.length === 0) return null;
  if (plotEls.length === 1) {
    return { width: EXPORT_WIDTH, height: resolveExportHeight(aspectRatio, plotEls[0]) };
  }
  const panelHeight = resolveExportHeight(aspectRatio, plotEls[0], panelWidthFor(plotEls.length));
  return { width: EXPORT_WIDTH, height: gridGeometry(plotEls.length, panelHeight, fontScale).totalHeight };
}

// ── Export orchestration ──────────────────────────────────────────────

/** Plotly attaches the current figure to the graph div itself. */
type PlotlyGraphDiv = HTMLElement & { data: Data[]; layout: Partial<Layout> };

/** Read the live figure off a plot element and apply the export settings.
 *
 *  `customdata` is dropped first: it carries a TooltipSource (holding a whole
 *  Row) per point, so deep-copying it costs ~100x the rest of the trace on a
 *  large category — and a static image never reads it. */
function figureFor(el: HTMLElement, width: number, height: number, fontScale: number) {
  const gd = el as PlotlyGraphDiv;
  const data = gd.data.map((trace) => {
    const shallow: Record<string, unknown> = { ...trace };
    delete shallow.customdata;
    return scaleFonts(shallow, fontScale) as Data;
  });
  return {
    data,
    layout: { ...scaleFonts(gd.layout, fontScale), width, height, autosize: false },
  };
}

/** Every rendered Plotly plot on the page, in document order. Keeps knowledge
 *  of Plotly's DOM class inside the module that owns the rest of it. */
export function currentPlotElements(): HTMLElement[] {
  return Array.from(document.querySelectorAll<HTMLElement>('.js-plotly-plot'));
}

/** Split-panel labels, read from the surrounding split-panel markup. The label
 *  div also nests a cross-filter badge, which would otherwise be concatenated
 *  into the heading with no separator. */
function panelLabels(plotEls: HTMLElement[]): string[] {
  return plotEls.map((el) => {
    const label = el.closest('.split-panel')?.querySelector('.split-panel-label');
    if (!label) return '';
    const text = label.cloneNode(true) as HTMLElement;
    text.querySelector('.cross-filter-badge')?.remove();
    return text.textContent?.trim() ?? '';
  });
}

/**
 * Export the currently rendered plot(s). A single plot is exported directly;
 * split-mode panels are composited into one image on a 3-column grid.
 */
export async function exportPlots(plotEls: HTMLElement[], settings: ExportSettings): Promise<void> {
  if (plotEls.length === 0) return;
  const { format, aspectRatio, fontScale } = settings;
  // Dynamic import keeps plotly.js-dist-min out of the components layer.
  // @ts-expect-error — plotly.js-dist-min has no type declarations
  const Plotly = await import('plotly.js-dist-min');
  // SVG is vector; scaling up only inflates the coordinate values.
  const scale = format === 'png' ? 4 : 1;

  if (plotEls.length === 1) {
    const height = resolveExportHeight(aspectRatio, plotEls[0]);
    const fig = figureFor(plotEls[0], EXPORT_WIDTH, height, fontScale);
    const uri: string = await Plotly.toImage(fig, { format, width: EXPORT_WIDTH, height, scale });
    if (format === 'svg') downloadTextFile('zscore_plot.svg', decodeSvgDataUrl(uri), 'image/svg+xml;charset=utf-8;');
    else downloadDataUrl('zscore_plot.png', uri);
    return;
  }

  // Each panel is sized against its own width, not the full canvas.
  const panelHeight = resolveExportHeight(aspectRatio, plotEls[0], panelWidthFor(plotEls.length));
  const geo = gridGeometry(plotEls.length, panelHeight, fontScale);
  const labels = panelLabels(plotEls);
  // Rendered one at a time: Plotly.toImage is main-thread bound either way, and
  // holding every panel's figure at once is what makes large exports run out of
  // memory.
  const images: string[] = [];
  for (const el of plotEls) {
    const opts = { format, width: geo.panelWidth, height: panelHeight, scale };
    images.push(await Plotly.toImage(figureFor(el, geo.panelWidth, panelHeight, fontScale), opts));
  }

  if (format === 'svg') downloadTextFile('zscore_plot.svg', compositeSvg(images, labels, geo, panelHeight), 'image/svg+xml;charset=utf-8;');
  else downloadBlob('zscore_plot.png', await compositePng(images, labels, geo, scale));
}

type Geometry = ReturnType<typeof gridGeometry>;

/** Merge panel SVGs into one document — the vector mirror of compositePng.
 *  Each panel is nested in a positioning <svg> rather than unwrapped into a
 *  translated <g>: nesting establishes its own viewport, so the panel keeps its
 *  coordinate system untouched and no attribute surgery is needed. */
function compositeSvg(images: string[], labels: string[], geo: Geometry, panelHeight: number): string {
  const parts = images.map((uri, i) => {
    const { x, y } = geo.position(i);
    // Ids are only unique per source document, so namespace them before merging.
    const panel = namespaceSvgIds(decodeSvgDataUrl(uri), `p${i}-`);
    const label = `<text x="${x}" y="${y + geo.label.baseline}" fill="${LABEL_COLOR}" font-family='${MONO_FONT}' font-size="${geo.label.fontSize}" font-weight="500">${escapeXml(labels[i])}</text>`;
    return `${label}<svg x="${x}" y="${y + geo.label.height}" width="${geo.panelWidth}" height="${panelHeight}" overflow="visible">${panel}</svg>`;
  });

  return [
    `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${EXPORT_WIDTH}" height="${geo.totalHeight}" viewBox="0 0 ${EXPORT_WIDTH} ${geo.totalHeight}">`,
    `<rect width="${EXPORT_WIDTH}" height="${geo.totalHeight}" fill="#ffffff"/>`,
    ...parts,
    '</svg>',
  ].join('');
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.src = src;
  });
}

/** Draw panel PNGs onto one canvas and return it as a PNG blob. */
async function compositePng(images: string[], labels: string[], geo: Geometry, scale: number): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = EXPORT_WIDTH * scale;
  canvas.height = geo.totalHeight * scale;
  const ctx = canvas.getContext('2d')!;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.font = `500 ${geo.label.fontSize * scale}px ${MONO_FONT}`;

  // Decoded sequentially so only one full-size bitmap is resident at a time;
  // at 4x scale each panel is tens of megabytes.
  for (const [i, src] of images.entries()) {
    const img = await loadImage(src);
    const { x, y } = geo.position(i);
    ctx.fillStyle = LABEL_COLOR;
    ctx.fillText(labels[i], x * scale, (y + geo.label.baseline) * scale);
    ctx.drawImage(img, x * scale, (y + geo.label.height) * scale);
  }

  // toBlob rather than toDataURL: base64 would add a ~33% larger copy of a
  // canvas that is already tens of megapixels.
  return new Promise<Blob>((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error('Canvas export failed'))), 'image/png'),
  );
}
