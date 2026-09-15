/**
 * Example CSV offered as a starting point in the upload help panel.
 *
 * The template must always satisfy the validation in `filterStore.uploadCSV`,
 * so the row is keyed off REQUIRED_COLUMNS — adding a required column breaks
 * the build here rather than silently shipping a template that fails upload.
 */

import { OPTIONAL_COLUMNS, REQUIRED_COLUMNS } from './types';
import { toCSV } from './csvSerialize';

type RequiredColumn = (typeof REQUIRED_COLUMNS)[number];
type OptionalColumn = (typeof OPTIONAL_COLUMNS)[number];

const TEMPLATE_ROW: Record<RequiredColumn | OptionalColumn, string> = {
  ELN_ID: 'ELN001-001',
  PLATENUMBER: '1',
  Coordinate: 'A1',
  AREA_TOTAL_REDUCED: '1250000',
  Base: 'Cs2CO3',
  Catalyst: 'Pd(OAc)2',
  Solvent: 'DMF',
  Ligand: 'XPhos',
  'Reaction Type': 'Buchwald-Hartwig amination',
  'FG A': 'Aryl bromide',
  'FG B': 'Primary amine',
  'z-Score': '1.42',
  Additive: '',
  'Coupling Reagent': '',
  'Secondary Solvent': '',
};

/** Required first so the header order matches what the validator reports. */
export const TEMPLATE_COLUMNS = [...REQUIRED_COLUMNS, ...OPTIONAL_COLUMNS];

/** Header line plus one example row, ready to download. */
export function buildCSVTemplate(): string {
  return toCSV(TEMPLATE_COLUMNS, [TEMPLATE_ROW]);
}
