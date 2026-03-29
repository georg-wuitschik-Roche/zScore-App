/**
 * Vite plugin that auto-discovers versioned dataset files in public/data/
 * and emits a versions.json manifest at build time.
 *
 * Naming convention:
 *   v1.parquet + v1-dropdown-index.json
 *   v2.parquet + v2-dropdown-index.json
 *
 * If no versioned files are found, falls back to the legacy filenames.
 */

import { readdirSync, readFileSync, existsSync } from 'fs';
import { join } from 'path';
import { writeFileSync } from 'fs';
import type { Plugin } from 'vite';
import type { VersionsManifest, VersionInfo } from './src/data/types';

const LEGACY_PARQUET = '/data/z-score-peaks.parquet';
const LEGACY_INDEX = '/data/dropdown-index.json';

function loadExistingManifest(dataDir: string): Map<string, VersionInfo> {
  const manifestPath = join(dataDir, 'versions.json');
  const map = new Map<string, VersionInfo>();
  if (!existsSync(manifestPath)) return map;
  try {
    const raw = JSON.parse(readFileSync(manifestPath, 'utf-8')) as VersionsManifest;
    for (const v of raw.versions) {
      map.set(v.id, v);
    }
  } catch { /* ignore corrupt file */ }
  return map;
}

function scanVersions(dataDir: string): VersionsManifest {
  if (!existsSync(dataDir)) {
    return { versions: [{ id: 'default', parquet: LEGACY_PARQUET, index: LEGACY_INDEX, label: 'Default' }], latest: 'default' };
  }

  const existing = loadExistingManifest(dataDir);
  const files = readdirSync(dataDir);
  const parquetPattern = /^v(\d+)\.parquet$/;
  const versions: VersionInfo[] = [];

  for (const file of files) {
    const match = file.match(parquetPattern);
    if (!match) continue;
    const num = parseInt(match[1], 10);
    const indexFile = `v${num}-dropdown-index.json`;
    if (!files.includes(indexFile)) continue;
    const id = `v${num}`;
    const prev = existing.get(id);
    versions.push({
      id,
      parquet: `/data/v${num}.parquet`,
      index: `/data/v${num}-dropdown-index.json`,
      label: prev?.label ?? `Version ${num}`,
      ...(prev?.date ? { date: prev.date } : {}),
    });
  }

  // Sort by version number ascending
  versions.sort((a, b) => {
    const numA = parseInt(a.id.slice(1), 10);
    const numB = parseInt(b.id.slice(1), 10);
    return numA - numB;
  });

  if (versions.length === 0) {
    const defaultEntry = existing.get('default');
    return {
      versions: [{
        id: 'default',
        parquet: LEGACY_PARQUET,
        index: LEGACY_INDEX,
        label: defaultEntry?.label ?? 'Default',
        ...(defaultEntry?.date ? { date: defaultEntry.date } : {}),
      }],
      latest: 'default',
    };
  }

  return { versions, latest: versions[versions.length - 1].id };
}

export default function versionsPlugin(): Plugin {
  let dataDir: string;

  return {
    name: 'zscore-versions',

    configResolved(config) {
      dataDir = join(config.publicDir, 'data');
    },

    // At build time, write versions.json into public/data/
    buildStart() {
      const manifest = scanVersions(dataDir);
      const outPath = join(dataDir, 'versions.json');
      writeFileSync(outPath, JSON.stringify(manifest, null, 2));
    },

    // In dev mode, serve versions.json dynamically so new files are picked up without restart
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url === '/data/versions.json') {
          const manifest = scanVersions(dataDir);
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify(manifest, null, 2));
          return;
        }
        next();
      });
    },
  };
}
