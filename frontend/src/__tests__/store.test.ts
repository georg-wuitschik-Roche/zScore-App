/**
 * Tests for the Zustand filter store.
 *
 * Zustand stores work outside React — use getState()/setState() directly.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { useFilterStore } from '../stores/filterStore';
import type { FilterState } from '../stores/filterStore';

// ---------------------------------------------------------------------------
// Reset store before each test to avoid cross-contamination
// ---------------------------------------------------------------------------

const INITIAL_STATE: Partial<FilterState> = {
  dataset: [],
  uploadedDataset: null,
  isFullDataLoaded: false,
  dropdownIndex: null,
  loadError: null,
  reactionTypes: ['Buchwald-Hartwig'],
  reactantTypes: ['Catalyst'],
  fgA: ['RNH2 a-branch', 'RNH2'],
  fgB: ['ArBr', 'ArCl'],
  excludeCui: true,
  excludeScaleup: true,
  includeNullCategories: true,
  minEln: 5,
  topnZscore: 5,
  maxComponents: 10,
  activeTab: 'boxplot',
  presentationMode: false,
  optionsPanelOpen: false,
  uploadError: null,
  uploadFileName: null,
};

beforeEach(() => {
  useFilterStore.setState(INITIAL_STATE);
});

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

describe('initial state', () => {
  it('has correct default reactionTypes', () => {
    const state = useFilterStore.getState();
    expect(state.reactionTypes).toEqual(['Buchwald-Hartwig']);
  });

  it('has excludeCui=true by default', () => {
    expect(useFilterStore.getState().excludeCui).toBe(true);
  });

  it('has excludeScaleup=true by default', () => {
    expect(useFilterStore.getState().excludeScaleup).toBe(true);
  });

  it('has includeNullCategories=true by default', () => {
    expect(useFilterStore.getState().includeNullCategories).toBe(true);
  });

  it('has correct default slider values', () => {
    const state = useFilterStore.getState();
    expect(state.minEln).toBe(5);
    expect(state.topnZscore).toBe(5);
    expect(state.maxComponents).toBe(10);
  });

  it('has correct default reactantTypes', () => {
    expect(useFilterStore.getState().reactantTypes).toEqual(['Catalyst']);
  });

  it('has correct default fgA and fgB', () => {
    const state = useFilterStore.getState();
    expect(state.fgA).toEqual(['RNH2 a-branch', 'RNH2']);
    expect(state.fgB).toEqual(['ArBr', 'ArCl']);
  });

  it('has boxplot as default active tab', () => {
    expect(useFilterStore.getState().activeTab).toBe('boxplot');
  });

  it('has presentationMode=false by default', () => {
    expect(useFilterStore.getState().presentationMode).toBe(false);
  });

  it('has optionsPanelOpen=false by default', () => {
    expect(useFilterStore.getState().optionsPanelOpen).toBe(false);
  });

  it('has null uploadedDataset by default', () => {
    expect(useFilterStore.getState().uploadedDataset).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// setReactionTypes
// ---------------------------------------------------------------------------

describe('setReactionTypes', () => {
  it('updates reactionTypes', () => {
    useFilterStore.getState().setReactionTypes(['Suzuki-Miyaura']);
    expect(useFilterStore.getState().reactionTypes).toEqual(['Suzuki-Miyaura']);
  });

  it('clears fgA when reaction types change', () => {
    useFilterStore.setState({ fgA: ['ArBr'] });
    useFilterStore.getState().setReactionTypes(['Suzuki-Miyaura']);
    expect(useFilterStore.getState().fgA).toEqual([]);
  });

  it('clears fgB when reaction types change', () => {
    useFilterStore.setState({ fgB: ['RNH2'] });
    useFilterStore.getState().setReactionTypes(['Suzuki-Miyaura']);
    expect(useFilterStore.getState().fgB).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// resetFilters
// ---------------------------------------------------------------------------

describe('resetFilters', () => {
  it('keeps reactionTypes', () => {
    useFilterStore.setState({ reactionTypes: ['Suzuki-Miyaura'] });
    useFilterStore.getState().resetFilters();
    expect(useFilterStore.getState().reactionTypes).toEqual(['Suzuki-Miyaura']);
  });

  it('clears fgA and fgB', () => {
    useFilterStore.setState({ fgA: ['ArBr'], fgB: ['RNH2'] });
    useFilterStore.getState().resetFilters();
    const state = useFilterStore.getState();
    expect(state.fgA).toEqual([]);
    expect(state.fgB).toEqual([]);
  });

  it('clears reactantTypes', () => {
    useFilterStore.setState({ reactantTypes: ['Catalyst', 'Solvent'] });
    useFilterStore.getState().resetFilters();
    expect(useFilterStore.getState().reactantTypes).toEqual([]);
  });

  it('resets all sliders to defaults', () => {
    useFilterStore.setState({ minEln: 20, topnZscore: 50, maxComponents: 100 });
    useFilterStore.getState().resetFilters();
    const state = useFilterStore.getState();
    expect(state.minEln).toBe(5);
    expect(state.topnZscore).toBe(5);
    expect(state.maxComponents).toBe(10);
  });

  it('resets excludeCui to true', () => {
    useFilterStore.setState({ excludeCui: false });
    useFilterStore.getState().resetFilters();
    expect(useFilterStore.getState().excludeCui).toBe(true);
  });

  it('resets excludeScaleup to true', () => {
    useFilterStore.setState({ excludeScaleup: false });
    useFilterStore.getState().resetFilters();
    expect(useFilterStore.getState().excludeScaleup).toBe(true);
  });

  it('resets includeNullCategories to true', () => {
    useFilterStore.setState({ includeNullCategories: false });
    useFilterStore.getState().resetFilters();
    expect(useFilterStore.getState().includeNullCategories).toBe(true);
  });

  it('clears uploadedDataset', () => {
    useFilterStore.setState({
      uploadedDataset: [
        {
          ELN_ID: 'X', PLATENUMBER: '1', Coordinate: 'A1',
          AREA_TOTAL_REDUCED: null, Additive: null, Base: null,
          Catalyst: null, 'Coupling Reagent': null, Solvent: null,
          Ligand: null, 'Secondary Solvent': null,
          'Reaction Type': 'BH', 'FG A': null, 'FG B': null,
          FG_sorted: null, FG_PAIR_SORTED: null, 'z-Score': 1,
        },
      ],
    });
    useFilterStore.getState().resetFilters();
    expect(useFilterStore.getState().uploadedDataset).toBeNull();
  });

  it('resets activeTab to boxplot', () => {
    useFilterStore.setState({ activeTab: 'heatmap' });
    useFilterStore.getState().resetFilters();
    expect(useFilterStore.getState().activeTab).toBe('boxplot');
  });
});

// ---------------------------------------------------------------------------
// togglePresentationMode / toggleOptionsPanel
// ---------------------------------------------------------------------------

describe('togglePresentationMode', () => {
  it('toggles from false to true', () => {
    useFilterStore.setState({ presentationMode: false });
    useFilterStore.getState().togglePresentationMode();
    expect(useFilterStore.getState().presentationMode).toBe(true);
  });

  it('toggles from true to false', () => {
    useFilterStore.setState({ presentationMode: true });
    useFilterStore.getState().togglePresentationMode();
    expect(useFilterStore.getState().presentationMode).toBe(false);
  });
});

describe('toggleOptionsPanel', () => {
  it('toggles from false to true', () => {
    useFilterStore.setState({ optionsPanelOpen: false });
    useFilterStore.getState().toggleOptionsPanel();
    expect(useFilterStore.getState().optionsPanelOpen).toBe(true);
  });

  it('toggles from true to false', () => {
    useFilterStore.setState({ optionsPanelOpen: true });
    useFilterStore.getState().toggleOptionsPanel();
    expect(useFilterStore.getState().optionsPanelOpen).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// uploadCSV
// ---------------------------------------------------------------------------

describe('uploadCSV', () => {
  // Build a complete valid CSV string with all required columns
  const REQUIRED_HEADERS = [
    'ELN_ID', 'PLATENUMBER', 'Coordinate', 'AREA_TOTAL_REDUCED',
    'Base', 'Catalyst', 'Solvent', 'Ligand',
    'Reaction Type', 'FG A', 'FG B', 'FG_sorted', 'z-Score',
  ];

  /** Quote a CSV field if it contains a comma. */
  function csvField(val: string): string {
    return val.includes(',') ? `"${val}"` : val;
  }

  function validCsvRow(overrides: Record<string, string> = {}): string {
    const defaults: Record<string, string> = {
      ELN_ID: 'ELN001', PLATENUMBER: '1', Coordinate: 'A1',
      AREA_TOTAL_REDUCED: '50', Base: 'K3PO4', Catalyst: 'Pd(OAc)2',
      Solvent: 'DMF', Ligand: 'XPhos', 'Reaction Type': 'Buchwald-Hartwig',
      'FG A': 'ArBr', 'FG B': 'RNH2', FG_sorted: 'ArBr, RNH2',
      'z-Score': '1.23',
    };
    const merged = { ...defaults, ...overrides };
    return REQUIRED_HEADERS.map((h) => csvField(merged[h] ?? '')).join(',');
  }

  const VALID_CSV = [REQUIRED_HEADERS.join(','), validCsvRow()].join('\n');

  it('sets uploadedDataset with valid CSV', () => {
    useFilterStore.getState().uploadCSV(VALID_CSV, 'test.csv');
    const state = useFilterStore.getState();
    expect(state.uploadedDataset).not.toBeNull();
    expect(state.uploadedDataset).toHaveLength(1);
    expect(state.uploadError).toBeNull();
    expect(state.uploadFileName).toBe('test.csv');
  });

  it('sets uploadError with missing required columns', () => {
    const csv = 'ELN_ID,z-Score\nELN001,1.23';
    useFilterStore.getState().uploadCSV(csv);
    const state = useFilterStore.getState();
    expect(state.uploadError).not.toBeNull();
    expect(state.uploadError).toContain('Missing required columns');
    expect(state.uploadedDataset).toBeNull();
  });

  it('sets uploadError when z-Score has no numeric values', () => {
    const csv = [
      REQUIRED_HEADERS.join(','),
      validCsvRow({ 'z-Score': '' }),
    ].join('\n');

    useFilterStore.getState().uploadCSV(csv);
    const state = useFilterStore.getState();
    expect(state.uploadError).not.toBeNull();
    expect(state.uploadError).toContain('z-Score');
  });

  it('sets uploadError with empty CSV', () => {
    useFilterStore.getState().uploadCSV('');
    const state = useFilterStore.getState();
    expect(state.uploadError).not.toBeNull();
  });

  it('sets uploadError with header-only CSV (no data rows)', () => {
    useFilterStore.getState().uploadCSV(REQUIRED_HEADERS.join(','));
    const state = useFilterStore.getState();
    expect(state.uploadError).not.toBeNull();
    expect(state.uploadError).toContain('no data rows');
  });

  it('handles multiple rows correctly', () => {
    const csv = [
      REQUIRED_HEADERS.join(','),
      validCsvRow({ ELN_ID: 'ELN001', 'z-Score': '1.23' }),
      validCsvRow({ ELN_ID: 'ELN002', 'z-Score': '2.34' }),
    ].join('\n');

    useFilterStore.getState().uploadCSV(csv);
    const state = useFilterStore.getState();
    expect(state.uploadedDataset).toHaveLength(2);
    expect(state.uploadError).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// clearUploadError
// ---------------------------------------------------------------------------

describe('clearUploadError', () => {
  it('clears the error', () => {
    useFilterStore.setState({ uploadError: 'some error' });
    useFilterStore.getState().clearUploadError();
    expect(useFilterStore.getState().uploadError).toBeNull();
  });

  it('is a no-op when error is already null', () => {
    useFilterStore.setState({ uploadError: null });
    useFilterStore.getState().clearUploadError();
    expect(useFilterStore.getState().uploadError).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// setFilters
// ---------------------------------------------------------------------------

describe('setFilters', () => {
  it('bulk updates multiple fields', () => {
    useFilterStore.getState().setFilters({
      reactionTypes: ['Suzuki-Miyaura'],
      minEln: 10,
      maxComponents: 20,
      excludeCui: false,
    });

    const state = useFilterStore.getState();
    expect(state.reactionTypes).toEqual(['Suzuki-Miyaura']);
    expect(state.minEln).toBe(10);
    expect(state.maxComponents).toBe(20);
    expect(state.excludeCui).toBe(false);
  });

  it('does not affect unspecified fields', () => {
    useFilterStore.getState().setFilters({ minEln: 99 });
    const state = useFilterStore.getState();
    expect(state.minEln).toBe(99);
    // These should remain at defaults
    expect(state.reactionTypes).toEqual(['Buchwald-Hartwig']);
    expect(state.topnZscore).toBe(5);
    expect(state.excludeCui).toBe(true);
  });

  it('can set activeTab', () => {
    useFilterStore.getState().setFilters({ activeTab: 'stats' });
    expect(useFilterStore.getState().activeTab).toBe('stats');
  });
});
