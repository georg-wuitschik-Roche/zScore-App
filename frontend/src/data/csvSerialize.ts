/**
 * CSV serialization — the counterpart to parseCSVText in loader.ts.
 *
 * Reagent names routinely contain commas (2,6-lutidine), and uploaded datasets
 * carry through arbitrary extra columns, so both cells and headers need RFC
 * 4180 quoting rather than a comma-only check.
 */

type CsvValue = string | number | null | undefined;

/** Quote a field when it contains a delimiter, quote or newline; double inner quotes. */
export function escapeCsvCell(value: CsvValue): string {
  if (value === null || value === undefined) return '';
  const str = String(value);
  return /[",\r\n]/.test(str) ? `"${str.replace(/"/g, '""')}"` : str;
}

/** Serialize rows to CSV text, one column per header, in the given order. */
export function toCSV(
  headers: readonly string[],
  rows: ReadonlyArray<Record<string, CsvValue>>,
): string {
  return [
    headers.map(escapeCsvCell).join(','),
    ...rows.map((row) => headers.map((h) => escapeCsvCell(row[h])).join(',')),
  ].join('\n');
}
