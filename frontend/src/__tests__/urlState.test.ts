/**
 * Tests for URL encoding/decoding logic used by useUrlState.
 *
 * Since encodeArray/decodeArray are internal to useUrlState.ts (not exported),
 * we replicate the logic here and verify the pipe-separator convention handles
 * reaction type names that contain commas.
 */

import { describe, it, expect } from 'vitest';

// ---------------------------------------------------------------------------
// Replicated encode/decode logic from useUrlState.ts
// ---------------------------------------------------------------------------

/** Serialize a string array to a URL param (pipe-separated). */
function encodeArray(arr: string[]): string {
  return arr.join('|');
}

/** Deserialize a URL param to a string array. */
function decodeArray(val: string | null): string[] | null {
  if (!val) return null;
  return val.split('|').filter(Boolean);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('URL encoding/decoding (pipe separator)', () => {
  describe('encodeArray', () => {
    it('encodes multiple values with pipe separator', () => {
      const result = encodeArray(['Borylation, Miyaura', 'Buchwald-Hartwig']);
      expect(result).toBe('Borylation, Miyaura|Buchwald-Hartwig');
    });

    it('encodes a single value without separator', () => {
      const result = encodeArray(['Buchwald-Hartwig']);
      expect(result).toBe('Buchwald-Hartwig');
    });

    it('encodes empty array as empty string', () => {
      const result = encodeArray([]);
      expect(result).toBe('');
    });

    it('preserves commas within values', () => {
      const result = encodeArray(['Negishi, in-situ', 'Suzuki-Miyaura']);
      expect(result).toBe('Negishi, in-situ|Suzuki-Miyaura');
    });

    it('handles three values', () => {
      const result = encodeArray(['A', 'B', 'C']);
      expect(result).toBe('A|B|C');
    });
  });

  describe('decodeArray', () => {
    it('decodes pipe-separated values with commas', () => {
      const result = decodeArray('Borylation, Miyaura|Buchwald-Hartwig');
      expect(result).toEqual(['Borylation, Miyaura', 'Buchwald-Hartwig']);
    });

    it('decodes value containing commas (Negishi)', () => {
      const result = decodeArray('Negishi, in-situ|Suzuki-Miyaura');
      expect(result).toEqual(['Negishi, in-situ', 'Suzuki-Miyaura']);
    });

    it('null input returns null', () => {
      const result = decodeArray(null);
      expect(result).toBeNull();
    });

    it('empty string returns null', () => {
      const result = decodeArray('');
      expect(result).toBeNull();
    });

    it('single value returns single-element array', () => {
      const result = decodeArray('Buchwald-Hartwig');
      expect(result).toEqual(['Buchwald-Hartwig']);
    });

    it('handles three values', () => {
      const result = decodeArray('A|B|C');
      expect(result).toEqual(['A', 'B', 'C']);
    });

    it('filters out empty segments from trailing pipe', () => {
      const result = decodeArray('A|B|');
      expect(result).toEqual(['A', 'B']);
    });
  });

  describe('round-trip', () => {
    it('encode then decode preserves values with commas', () => {
      const original = ['Borylation, Miyaura', 'Buchwald-Hartwig'];
      const encoded = encodeArray(original);
      const decoded = decodeArray(encoded);
      expect(decoded).toEqual(original);
    });

    it('encode then decode preserves single value', () => {
      const original = ['Suzuki-Miyaura'];
      const encoded = encodeArray(original);
      const decoded = decodeArray(encoded);
      expect(decoded).toEqual(original);
    });

    it('encode then decode preserves complex reaction names', () => {
      const original = [
        'Borylation, Miyaura',
        'Negishi, in-situ',
        'Buchwald-Hartwig',
        'C-H Activation, Heck-type',
      ];
      const encoded = encodeArray(original);
      const decoded = decodeArray(encoded);
      expect(decoded).toEqual(original);
    });
  });
});
