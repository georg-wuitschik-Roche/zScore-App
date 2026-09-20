/**
 * Deployment-base-aware URL resolution.
 *
 * The app is hosted under a sub-path on GitHub Pages (/zScore-App/), so the
 * root-relative dataset paths in versions.json ('/data/v1.parquet') have to be
 * rebased before they hit the network. Keeping the manifest root-relative means
 * the same file is valid in dev and in production.
 *
 * Images don't need this — they're imported from src/assets/, so Vite rebases
 * and content-hashes them at build time.
 */

/** Resolve a root-relative public path against the deployment base. */
export function publicUrl(path: string): string {
  const base = import.meta.env.BASE_URL;
  return path.startsWith('/') ? base + path.slice(1) : base + path;
}
