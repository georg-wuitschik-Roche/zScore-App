/**
 * CSV loader — fetch and parse the dataset with PapaParse.
 *
 * Handles encoding detection, comma-as-decimal conversion, and
 * FG_PAIR_SORTED computation.
 */

import Papa from 'papaparse';
import type { Row } from './types';

/** Default CSV URL — served from public/ or fetched from GCS. */
const DEFAULT_CSV_URL = '/data/z-score-peaks.csv';

/**
 * Parse a numeric string, handling comma-as-decimal separator.
 * Returns null for non-numeric values.
 */
function parseNumeric(value: unknown): number | null {
  if (value === null || value === undefined || value === '') return null;
  const str = String(value).replace(',', '.').trim();
  const num = Number(str);
  return isNaN(num) ? null : num;
}

/** Compute sorted FG pair string: "ArBr, RNH2" format. */
function computeFgPairSorted(fgA: string | null, fgB: string | null): string | null {
  if (!fgA || !fgB) return null;
  const pair = [fgA, fgB].sort();
  return `${pair[0]}, ${pair[1]}`;
}

/**
 * Load and parse a CSV file from a URL.
 *
 * Performs the same cleaning as Python's _load_and_prepare():
 * - Converts z-Score and AREA_TOTAL_REDUCED to numbers (handles comma decimals)
 * - Computes FG_PAIR_SORTED if not present
 */
export async function loadDataset(url: string = DEFAULT_CSV_URL): Promise<Row[]> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch CSV: ${response.status} ${response.statusText}`);
  }

  const csvText = await response.text();
  return parseCSVText(csvText);
}

/**
 * Parse CSV text into Row[]. Used by both loadDataset and file upload.
 */
export function parseCSVText(csvText: string): Row[] {
  // Try comma delimiter first, fall back to semicolon, then tab
  let result = Papa.parse<Record<string, string>>(csvText, {
    header: true,
    skipEmptyLines: true,
  });

  // If we got 1 or fewer columns, try semicolon
  if (result.meta.fields && result.meta.fields.length <= 1) {
    result = Papa.parse<Record<string, string>>(csvText, {
      header: true,
      skipEmptyLines: true,
      delimiter: ';',
    });
  }

  // If still 1 or fewer columns, try tab
  if (result.meta.fields && result.meta.fields.length <= 1) {
    result = Papa.parse<Record<string, string>>(csvText, {
      header: true,
      skipEmptyLines: true,
      delimiter: '\t',
    });
  }

  // Coerce types and clean data
  const rows: Row[] = result.data.map((raw) => {
    const row = raw as unknown as Row;

    // Convert numeric columns (handles comma-as-decimal)
    row['z-Score'] = parseNumeric(raw['z-Score']);
    row['AREA_TOTAL_REDUCED'] = parseNumeric(raw['AREA_TOTAL_REDUCED']);

    // Normalize empty strings to null for categorical columns
    for (const col of [
      'Additive', 'Base', 'Catalyst', 'Coupling Reagent',
      'Solvent', 'Ligand', 'Secondary Solvent', 'Tertiary Solvent',
    ]) {
      if (raw[col] === '' || raw[col] === 'nan' || raw[col] === 'NaN') {
        (row as Record<string, unknown>)[col] = null;
      }
    }

    // Compute FG_PAIR_SORTED if not present
    if (!row.FG_PAIR_SORTED && row['FG A'] && row['FG B']) {
      if (raw['FG_sorted']) {
        row.FG_PAIR_SORTED = raw['FG_sorted'];
      } else {
        row.FG_PAIR_SORTED = computeFgPairSorted(
          row['FG A'],
          row['FG B'],
        );
      }
    }

    return row;
  });

  return rows;
}
