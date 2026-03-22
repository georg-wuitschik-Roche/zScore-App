import { useMemo } from 'react';
import type { Row } from '../data/types';

interface StatsTableProps {
  rows: Row[];
  reactantTypes: string[];
}

interface GroupStats {
  group: string;
  count: number;
  mean: number;
  std: number;
  min: number;
  q25: number;
  median: number;
  q75: number;
  max: number;
}

function computeStats(values: number[]): Omit<GroupStats, 'group'> {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const mean = values.reduce((s, v) => s + v, 0) / n;
  const variance =
    values.reduce((s, v) => s + (v - mean) ** 2, 0) / (n > 1 ? n - 1 : 1);
  const std = Math.sqrt(variance);

  function percentile(p: number): number {
    const idx = (p / 100) * (n - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    if (lo === hi) return sorted[lo];
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo);
  }

  return {
    count: n,
    mean: Number(mean.toFixed(4)),
    std: Number(std.toFixed(4)),
    min: sorted[0],
    q25: Number(percentile(25).toFixed(4)),
    median: Number(percentile(50).toFixed(4)),
    q75: Number(percentile(75).toFixed(4)),
    max: sorted[n - 1],
  };
}

function groupBy(rows: Row[], cols: string[]): Map<string, number[]> {
  const groups = new Map<string, number[]>();
  for (const row of rows) {
    const z = row['z-Score'];
    if (z === null || z === undefined || isNaN(z)) continue;

    const key = cols.map((c) => String(row[c] ?? 'N/A')).join(' | ');
    const existing = groups.get(key);
    if (existing) {
      existing.push(z);
    } else {
      groups.set(key, [z]);
    }
  }
  return groups;
}

/** Color a numeric cell based on its value relative to 0 */
function cellColor(val: number): string {
  if (val > 0.5) return 'var(--stats-positive)';
  if (val < -0.5) return 'var(--stats-negative)';
  return 'inherit';
}

export function StatsTable({ rows, reactantTypes }: StatsTableProps) {

  const tableData = useMemo((): GroupStats[] => {
    if (rows.length === 0 || reactantTypes.length === 0) return [];

    const groups = groupBy(rows, reactantTypes);
    const result: GroupStats[] = [];

    for (const [group, values] of groups) {
      if (values.length === 0) continue;
      result.push({
        group,
        ...computeStats(values),
      });
    }

    // Sort by median descending
    result.sort((a, b) => b.median - a.median);
    return result;
  }, [rows, reactantTypes]);

  const uniqueElns = useMemo(() => {
    const elns = new Set<string>();
    for (const row of rows) {
      if (row.ELN_ID) elns.add(row.ELN_ID);
    }
    return elns.size;
  }, [rows]);

  if (rows.length === 0) {
    return (
      <div className="stats-container">
        <p className="no-data-message">
          No data available for the current filter selection.
        </p>
      </div>
    );
  }

  return (
    <div className="stats-container">
      <div className="stats-summary-cards">
        <div className="stats-card">
          <span className="stats-card-value">
            {rows.length.toLocaleString()}
          </span>
          <span className="stats-card-label">Total Rows</span>
        </div>
        <div className="stats-card">
          <span className="stats-card-value">{uniqueElns}</span>
          <span className="stats-card-label">Unique ELNs</span>
        </div>
        <div className="stats-card">
          <span className="stats-card-value">{tableData.length}</span>
          <span className="stats-card-label">Categories</span>
        </div>
      </div>

      <div className="stats-table-wrapper">
        <table className="stats-table">
          <thead>
            <tr>
              <th className="stats-th-group">
                {reactantTypes.join(' / ')}
              </th>
              <th>n</th>
              <th>Mean</th>
              <th>Std</th>
              <th>Min</th>
              <th>25%</th>
              <th>50%</th>
              <th>75%</th>
              <th>Max</th>
            </tr>
          </thead>
          <tbody>
            {tableData.map((row, i) => (
              <tr key={row.group} className={i % 2 === 0 ? 'stats-row-even' : ''}>
                <td className="stats-group-cell">{row.group}</td>
                <td className="stats-num-cell">{row.count}</td>
                <td className="stats-num-cell" style={{ color: cellColor(row.mean) }}>
                  {row.mean.toFixed(4)}
                </td>
                <td className="stats-num-cell">{row.std.toFixed(4)}</td>
                <td className="stats-num-cell" style={{ color: cellColor(row.min) }}>
                  {row.min.toFixed(4)}
                </td>
                <td className="stats-num-cell" style={{ color: cellColor(row.q25) }}>
                  {row.q25.toFixed(4)}
                </td>
                <td className="stats-num-cell stats-median-cell" style={{ color: cellColor(row.median) }}>
                  {row.median.toFixed(4)}
                </td>
                <td className="stats-num-cell" style={{ color: cellColor(row.q75) }}>
                  {row.q75.toFixed(4)}
                </td>
                <td className="stats-num-cell" style={{ color: cellColor(row.max) }}>
                  {row.max.toFixed(4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
