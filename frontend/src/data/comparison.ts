/**
 * Rank comparison between two dataset versions.
 *
 * Groups rows by reactant type compound key, computes median z-Score
 * rankings, and returns a map of rank deltas for each category.
 */

import type { Row, RankDelta, VersionInfo } from './types';
import { median } from '../plots/helpers';

/**
 * Determine which version to compare against.
 * If explicitly set, use that. Otherwise pick the version before the active one.
 */
export function resolveComparisonVersion(
  availableVersions: VersionInfo[],
  activeVersion: string,
  explicitVersion: string | null,
  hasUpload = false,
): string | null {
  if (explicitVersion && (hasUpload || explicitVersion !== activeVersion)) return explicitVersion;
  // With an upload active, default to the active built-in version as baseline
  if (hasUpload) return activeVersion;
  const idx = availableVersions.findIndex((v) => v.id === activeVersion);
  if (idx < 0) return availableVersions.length > 0 ? availableVersions[0].id : null;
  // Pick the previous version, or the next one if already at the start
  if (idx === 0) return availableVersions.length > 1 ? availableVersions[1].id : null;
  return availableVersions[idx - 1].id;
}

/** Group rows by reactant compound key → median z-Score. */
function computeMedianRanking(
  rows: Row[],
  reactantTypes: string[],
): Map<string, number> {
  const groupScores = new Map<string, number[]>();
  for (const row of rows) {
    const z = row['z-Score'];
    if (z === null || z === undefined || isNaN(z)) continue;
    const key = reactantTypes
      .map((col) => String(row[col] ?? '(no value)'))
      .join(' / ');
    if (!groupScores.has(key)) groupScores.set(key, []);
    groupScores.get(key)!.push(z);
  }

  const medians = new Map<string, number>();
  for (const [key, scores] of groupScores) {
    medians.set(key, median(scores));
  }
  return medians;
}

/** Assign 1-based ranks from a median map (rank 1 = highest median, alphabetical tie-break). */
function assignRanks(medians: Map<string, number>): Map<string, number> {
  const sorted = Array.from(medians.entries()).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  const ranks = new Map<string, number>();
  for (let i = 0; i < sorted.length; i++) {
    ranks.set(sorted[i][0], i + 1);
  }
  return ranks;
}

/**
 * Compute rank deltas between current and comparison filtered rows.
 *
 * Returns a map from category name → RankDelta.
 * Positive rankChange = moved up, negative = moved down.
 */
export function computeRankDeltas(
  currentRows: Row[],
  comparisonRows: Row[],
  reactantTypes: string[],
): Map<string, RankDelta> {
  const currentMedians = computeMedianRanking(currentRows, reactantTypes);
  const comparisonMedians = computeMedianRanking(comparisonRows, reactantTypes);


  const currentRanks = assignRanks(currentMedians);
  const comparisonRanks = assignRanks(comparisonMedians);

  const result = new Map<string, RankDelta>();

  for (const [name] of currentMedians) {
    const currentMedian = currentMedians.get(name)!;
    const comparisonMedian = comparisonMedians.get(name);

    if (comparisonMedian === undefined) {
      const currentRank = currentRanks.get(name)!;
      result.set(name, {
        rankChange: 0, medianDelta: 0, isNew: true,
        currentRank, comparisonRank: 0,
        currentMedian, comparisonMedian: 0,
      });
    } else {
      const currentRank = currentRanks.get(name)!;
      const comparisonRank = comparisonRanks.get(name)!;
      result.set(name, {
        rankChange: comparisonRank - currentRank, // positive = moved up
        medianDelta: currentMedian - comparisonMedian,
        isNew: false,
        currentRank, comparisonRank,
        currentMedian, comparisonMedian,
      });
    }
  }

  return result;
}
