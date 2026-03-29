/**
 * localStorage persistence for uploaded datasets.
 *
 * Stores uploaded CSV data so it survives page refresh.
 * Handles QuotaExceededError gracefully (returns false on failure).
 */

import type { Row, UploadMode } from './types';

const STORAGE_KEY = 'zscore-upload';

interface StoredUpload {
  rows: Row[];
  fileName: string;
  mode: UploadMode;
  timestamp: number;
}

function hasLocalStorage(): boolean {
  try {
    return typeof localStorage !== 'undefined';
  } catch {
    return false;
  }
}

/** Save upload to localStorage. Returns true on success, false if quota exceeded or unavailable. */
export function saveUpload(rows: Row[], fileName: string, mode: UploadMode): boolean {
  if (!hasLocalStorage()) return false;
  try {
    const data: StoredUpload = { rows, fileName, mode, timestamp: Date.now() };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    return true;
  } catch {
    // QuotaExceededError or other storage failure
    return false;
  }
}

/** Load persisted upload. Returns null if missing or corrupted. */
export function loadUpload(): StoredUpload | null {
  if (!hasLocalStorage()) return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw) as StoredUpload;
    if (!Array.isArray(data.rows) || !data.fileName) return null;
    return data;
  } catch {
    // Corrupted entry — remove it
    localStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

/** Clear persisted upload from localStorage. */
export function clearUpload(): void {
  if (!hasLocalStorage()) return;
  localStorage.removeItem(STORAGE_KEY);
}
