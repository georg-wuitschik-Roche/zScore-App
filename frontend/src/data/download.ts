/**
 * Browser download helpers.
 *
 * Sits alongside uploadStorage.ts — thin wrappers over browser APIs that the
 * components would otherwise hand-roll at each call site.
 */

/** Trigger a download of an already-formed URL (blob, data or object URL). */
function triggerDownload(filename: string, href: string) {
  const a = document.createElement('a');
  a.href = href;
  a.download = filename;
  a.click();
}

/** Trigger a browser download of in-memory text. */
export function downloadTextFile(filename: string, content: string, mime = 'text/csv;charset=utf-8;') {
  const url = URL.createObjectURL(new Blob([content], { type: mime }));
  triggerDownload(filename, url);
  URL.revokeObjectURL(url);
}

/** Trigger a browser download of a data URL, e.g. from canvas.toDataURL(). */
export function downloadDataUrl(filename: string, dataUrl: string) {
  triggerDownload(filename, dataUrl);
}

/** Trigger a browser download of binary data, e.g. from canvas.toBlob().
 *  Preferred over downloadDataUrl for large images — no base64 copy. */
export function downloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  triggerDownload(filename, url);
  URL.revokeObjectURL(url);
}
