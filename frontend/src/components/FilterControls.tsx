import { useMemo } from 'react';
import { useFilterStore } from '../stores/filterStore';
import { useFilteredData } from '../hooks/useFilteredData';
import {
  getReactionTypes,
  getFgOptions,
  getFgBOptionsConditioned,
  getReactantOptions,
} from '../data/dropdownOptions';
import { MultiSelect } from './MultiSelect';

export function FilterControls() {
  const dataset = useFilterStore((s) => s.dataset);
  const uploadedDataset = useFilterStore((s) => s.uploadedDataset);
  const reactionTypes = useFilterStore((s) => s.reactionTypes);
  const fgA = useFilterStore((s) => s.fgA);
  const fgB = useFilterStore((s) => s.fgB);
  const reactantTypes = useFilterStore((s) => s.reactantTypes);
  const setReactionTypes = useFilterStore((s) => s.setReactionTypes);
  const setFgA = useFilterStore((s) => s.setFgA);
  const setFgB = useFilterStore((s) => s.setFgB);
  const setReactantTypes = useFilterStore((s) => s.setReactantTypes);

  const sourceData = uploadedDataset ?? dataset;
  const { stats } = useFilteredData();

  const reactionTypeOptions = useMemo(
    () => getReactionTypes(sourceData),
    [sourceData],
  );

  const fgAOptions = useMemo(
    () => ['All', ...getFgOptions(sourceData, reactionTypes)],
    [sourceData, reactionTypes],
  );

  const fgBOptions = useMemo(
    () => ['All', ...getFgBOptionsConditioned(sourceData, reactionTypes, fgA)],
    [sourceData, reactionTypes, fgA],
  );

  const reactantTypeOptions = useMemo(
    () => getReactantOptions(sourceData, reactionTypes),
    [sourceData, reactionTypes],
  );

  return (
    <div className="controls-row">
      {/* Reaction Type(s) */}
      <div className="control-col">
        <label>Reaction Type(s):</label>
        <MultiSelect
          options={reactionTypeOptions}
          value={reactionTypes}
          onChange={setReactionTypes}
          placeholder="Select reaction types..."
          autoClose
        />
        <div className="stats-badge">
          <div className="stats-badge-content">
            ELNs: {stats.wholeDataset?.elns ?? '--'}
          </div>
        </div>
      </div>

      {/* Functional Group(s) A */}
      <div className="control-col">
        <label>Functional Group(s) A:</label>
        <MultiSelect
          options={fgAOptions}
          value={fgA}
          onChange={(vals) => {
            // "All" means no filtering — clear specific selections
            if (vals.includes('All') && !fgA.includes('All')) {
              setFgA([]);
            } else {
              setFgA(vals.filter((v) => v !== 'All'));
            }
          }}
          placeholder="All (no filter)"
          className="fg-dropdown"
        />
        <div className="stats-badge">
          <div className="stats-badge-content">
            ELNs: {stats.afterFgA?.elns ?? '--'}
          </div>
        </div>
      </div>

      {/* Functional Group(s) B */}
      <div className="control-col">
        <label>Functional Group(s) B:</label>
        <MultiSelect
          options={fgBOptions}
          value={fgB}
          onChange={(vals) => {
            if (vals.includes('All') && !fgB.includes('All')) {
              setFgB([]);
            } else {
              setFgB(vals.filter((v) => v !== 'All'));
            }
          }}
          placeholder="All (no filter)"
          className="fg-dropdown"
        />
        <div className="stats-badge">
          <div className="stats-badge-content">
            ELNs: {stats.afterFgB?.elns ?? '--'}
          </div>
        </div>
      </div>

      {/* Reactant Type(s) */}
      <div className="control-col">
        <label>Reactant Type(s):</label>
        <MultiSelect
          options={reactantTypeOptions}
          value={reactantTypes}
          onChange={setReactantTypes}
          placeholder="Select reactant types..."
          autoClose
        />
      </div>
    </div>
  );
}
