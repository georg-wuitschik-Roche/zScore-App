/**
 * Dataset loader — fetch Parquet file with hyparquet (pure JS, no WASM).
 *
 * Also supports CSV upload via PapaParse (for user-uploaded files).
 * The default dataset ships as Parquet (0.5 MB vs 15 MB CSV = 30x smaller).
 */

import Papa from 'papaparse';
import { parquetRead } from 'hyparquet';
import type { Row, DropdownIndex } from './types';

/** Default Parquet URL — served from public/. */
const DEFAULT_PARQUET_URL = '/data/z-score-peaks.parquet';

/** Pre-computed dropdown index URL. */
const DROPDOWN_INDEX_URL = '/data/dropdown-index.json';

/**
 * Compute sorted FG pair string: "ArBr, RNH2" format.
 */
function computeFgPairSorted(fgA: string | null, fgB: string | null): string | null {
  if (!fgA || !fgB) return null;
  const pair = [fgA, fgB].sort();
  return `${pair[0]}, ${pair[1]}`;
}

/**
 * Normalize a raw value from Parquet/CSV to null if it's empty/NaN.
 */
function normalizeNull(val: unknown): string | null {
  if (val === null || val === undefined) return null;
  const s = String(val);
  if (s === '' || s === 'nan' || s === 'NaN') return null;
  return s;
}

/**
 * Clean a raw row from Parquet or CSV into a typed Row.
 */
function cleanRow(raw: Record<string, unknown>): Row {
  const row = { ...raw } as Record<string, unknown>;

  // Convert BigInt values to regular numbers (Parquet stores INT64 as BigInt)
  for (const key of Object.keys(row)) {
    if (typeof row[key] === 'bigint') {
      row[key] = Number(row[key]);
    }
  }

  // Normalize empty/NaN strings to null for categorical columns
  for (const col of [
    'Additive', 'Base', 'Catalyst', 'Coupling Reagent',
    'Solvent', 'Ligand', 'Secondary Solvent', 'Tertiary Solvent',
  ]) {
    if (col in row) {
      row[col] = normalizeNull(row[col]);
    }
  }

  // Ensure numerics are numbers (Parquet already stores them as numbers,
  // but CSV upload may need conversion)
  if (typeof row['z-Score'] !== 'number') {
    row['z-Score'] = parseNumeric(row['z-Score']);
  }
  if (typeof row['AREA_TOTAL_REDUCED'] !== 'number') {
    row['AREA_TOTAL_REDUCED'] = parseNumeric(row['AREA_TOTAL_REDUCED']);
  }

  // Compute FG_PAIR_SORTED if not present
  if (!row.FG_PAIR_SORTED) {
    const fgSorted = row['FG_sorted'];
    if (fgSorted && typeof fgSorted === 'string') {
      row.FG_PAIR_SORTED = fgSorted;
    } else {
      row.FG_PAIR_SORTED = computeFgPairSorted(
        row['FG A'] as string | null,
        row['FG B'] as string | null,
      );
    }
  }

  return row as Row;
}

/**
 * Fetch the pre-computed dropdown index (tiny JSON, ~12KB).
 * Returns instantly-usable dropdown data while the full parquet loads.
 */
export async function fetchDropdownIndex(
  url: string = DROPDOWN_INDEX_URL,
): Promise<DropdownIndex> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch dropdown index: ${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<DropdownIndex>;
}

/**
 * Fetch a Parquet file as an ArrayBuffer.
 */
export async function fetchParquetBuffer(
  url: string = DEFAULT_PARQUET_URL,
): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch dataset: ${response.status} ${response.statusText}`);
  }
  return response.arrayBuffer();
}

/**
 * Parse an ArrayBuffer containing Parquet data into Row[].
 *
 * Uses hyparquet — a pure JavaScript Parquet reader (420KB, no WASM).
 * Parquet advantages: 30x smaller than CSV, types preserved, dictionary-encoded strings.
 */
export function parseDataset(buffer: ArrayBuffer): Promise<Row[]> {
  return new Promise<Row[]>((resolve, reject) => {
    try {
      parquetRead({
        file: buffer,
        rowFormat: 'object',
        onComplete: (data: Record<string, unknown>[]) => {
          const rows = data.map(cleanRow);
          resolve(rows);
        },
      });
    } catch (e) {
      reject(e);
    }
  });
}

/**
 * Load the default dataset from a Parquet file (convenience wrapper).
 */
export async function loadDataset(url: string = DEFAULT_PARQUET_URL): Promise<Row[]> {
  const buffer = await fetchParquetBuffer(url);
  return parseDataset(buffer);
}

/**
 * Parse a numeric string, handling comma-as-decimal separator.
 */
function parseNumeric(value: unknown): number | null {
  if (value === null || value === undefined || value === '') return null;
  const str = String(value).replace(',', '.').trim();
  const num = Number(str);
  return isNaN(num) ? null : num;
}

/**
 * Parse CSV text into Row[]. Used for user-uploaded CSV files.
 */
export function parseCSVText(csvText: string): Row[] {
  let result = Papa.parse<Record<string, string>>(csvText, {
    header: true,
    skipEmptyLines: true,
  });

  if (result.meta.fields && result.meta.fields.length <= 1) {
    result = Papa.parse<Record<string, string>>(csvText, {
      header: true,
      skipEmptyLines: true,
      delimiter: ';',
    });
  }

  if (result.meta.fields && result.meta.fields.length <= 1) {
    result = Papa.parse<Record<string, string>>(csvText, {
      header: true,
      skipEmptyLines: true,
      delimiter: '\t',
    });
  }

  return result.data.map((raw) => cleanRow(raw as Record<string, unknown>));
}
