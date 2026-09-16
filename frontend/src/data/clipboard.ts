/**
 * Clipboard helper.
 *
 * Sits alongside download.ts — a thin wrapper over a browser API that the
 * components would otherwise hand-roll at each call site.
 */

/**
 * Copy text to the clipboard. Returns whether it succeeded.
 *
 * Falls back to a hidden textarea for non-secure contexts, where
 * `navigator.clipboard` is unavailable.
 */
export async function copyText(text: string): Promise<boolean> {
  if (typeof navigator.clipboard?.writeText === 'function') {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fall through to the textarea fallback.
    }
  }

  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'fixed';
  textarea.style.top = '-1000px';
  textarea.style.opacity = '0';
  document.body.appendChild(textarea);

  // Preserve whatever the user had selected — they may be mid-selection.
  const selection = document.getSelection();
  const previous = selection && selection.rangeCount > 0 ? selection.getRangeAt(0) : null;

  textarea.select();
  let ok = false;
  try {
    ok = document.execCommand('copy');
  } catch {
    ok = false;
  }

  document.body.removeChild(textarea);
  if (previous && selection) {
    selection.removeAllRanges();
    selection.addRange(previous);
  }
  return ok;
}
