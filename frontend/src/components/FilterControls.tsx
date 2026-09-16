import { useMemo } from 'react';
import { useFilterStore } from '../stores/filterStore';
import { useFilteredData } from '../hooks/useFilteredData';
import { useEffectiveDataset } from '../hooks/useEffectiveDataset';
import {
  getReactionTypes,
  getFgOptions,
  getFgBOptionsConditioned,
  getReactantOptions,
} from '../data/dropdownOptions';
import { useDropdownCounts } from '../hooks/useDropdownCounts';
import { MultiSelect } from './MultiSelect';
import type { SplitSelector } from '../data/types';

/**
 * One tab stop, not two: it's a radio group, so Tab lands on the selected
 * option and Arrow/Home/End move between them. Two stops would double the Tab
 * distance to the next dropdown every time a selector goes multi-value.
 */
function SplitToggle({ selectorKey, values, id }: { selectorKey: SplitSelector; values: string[]; id?: string }) {
  const splitSelector = useFilterStore((s) => s.splitSelector);
  const setSplitSelector = useFilterStore((s) => s.setSplitSelector);

  if (values.length < 2) return null;

  const isActive = splitSelector === selectorKey;

  // Arrows move focus only; Enter/Space commits. Selection-follows-focus would
  // rebuild every plot on each arrow press.
  function handleKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
    const btns = [...e.currentTarget.querySelectorAll('button')];
    const from = btns.indexOf(document.activeElement as HTMLButtonElement);
    const to =
      e.key === 'Home' ? 0
      : e.key === 'End' ? btns.length - 1
      : e.key.startsWith('Arrow') ? (from === 0 ? 1 : 0)
      : null;
    if (to === null) return;
    e.preventDefault();
    btns[to]?.focus();
  }

  return (
    <div
      className="split-toggle"
      id={id}
      role="radiogroup"
      aria-label="Panel layout"
      onKeyDown={handleKeyDown}
    >
      <button
        className={`split-toggle-btn${!isActive ? ' active' : ''}`}
        role="radio"
        aria-checked={!isActive}
        tabIndex={!isActive ? 0 : -1}
        onClick={() => setSplitSelector(null)}
      >
        Combined
      </button>
      <button
        className={`split-toggle-btn${isActive ? ' active' : ''}`}
        role="radio"
        aria-checked={isActive}
        tabIndex={isActive ? 0 : -1}
        onClick={() => setSplitSelector(selectorKey)}
      >
        Split
      </button>
    </div>
  );
}

export function FilterControls() {
  const reactionTypes = useFilterStore((s) => s.reactionTypes);
  const fgA = useFilterStore((s) => s.fgA);
  const fgB = useFilterStore((s) => s.fgB);
  const reactantTypes = useFilterStore((s) => s.reactantTypes);
  const setReactionTypes = useFilterStore((s) => s.setReactionTypes);
  const setFgA = useFilterStore((s) => s.setFgA);
  const setFgB = useFilterStore((s) => s.setFgB);
  const setReactantTypes = useFilterStore((s) => s.setReactantTypes);

  const sourceData = useEffectiveDataset();
  const counts = useDropdownCounts();
  const { stats } = useFilteredData();

  const reactionTypeOptions = useMemo(
    () => getReactionTypes(sourceData),
    [sourceData],
  );

  const fgAOptions = useMemo(
    () => getFgOptions(sourceData, reactionTypes),
    [sourceData, reactionTypes],
  );

  const fgBOptions = useMemo(
    () => getFgBOptionsConditioned(sourceData, reactionTypes, fgA),
    [sourceData, reactionTypes, fgA],
  );

  const reactantTypeOptions = useMemo(
    () => getReactantOptions(sourceData, reactionTypes),
    [sourceData, reactionTypes],
  );

  return (
    <div className="controls-row">
      {/* Reaction Type(s) */}
      <div className="control-col" id="reaction-type-dropdown">
        <label htmlFor="filter-reaction-types">Reaction Type(s):</label>
        <MultiSelect
          options={reactionTypeOptions}
          value={reactionTypes}
          onChange={setReactionTypes}
          inputId="filter-reaction-types"
          counts={counts.reactionTypes}
          countLabel="ELNs"
          placeholder="Select reaction types..."
          autoClose
        />
        <SplitToggle selectorKey="reactionTypes" values={reactionTypes} />
        <div className="stats-badge">
          <div className="stats-badge-content">
            ELNs: {stats.wholeDataset?.elns ?? '--'}
          </div>
        </div>
      </div>

      {/* Functional Group(s) A */}
      <div className="control-col" id="fg-a-dropdown">
        <label htmlFor="filter-fg-a">Functional Group(s) A:</label>
        <MultiSelect
          options={fgAOptions}
          value={fgA}
          onChange={setFgA}
          inputId="filter-fg-a"
          counts={counts.fgA}
          countLabel="ELNs"
          clearOption="All"
          placeholder="All (no filter)"
          className="fg-dropdown"
        />
        <SplitToggle selectorKey="fgA" values={fgA} />
        <div className="stats-badge">
          <div className="stats-badge-content">
            ELNs: {stats.afterFgA?.elns ?? '--'}
          </div>
        </div>
      </div>

      {/* Functional Group(s) B */}
      <div className="control-col" id="fg-b-dropdown">
        <label htmlFor="filter-fg-b">Functional Group(s) B:</label>
        <MultiSelect
          options={fgBOptions}
          value={fgB}
          onChange={setFgB}
          inputId="filter-fg-b"
          counts={counts.fgB}
          countLabel="ELNs"
          clearOption="All"
          placeholder="All (no filter)"
          className="fg-dropdown"
        />
        <SplitToggle selectorKey="fgB" values={fgB} />
        <div className="stats-badge">
          <div className="stats-badge-content">
            ELNs: {stats.afterFgB?.elns ?? '--'}
          </div>
        </div>
      </div>

      {/* Reactant Type(s) */}
      <div className="control-col" id="reactant-types-dropdown">
        <label htmlFor="filter-reactant-types">Reactant Type(s):</label>
        <MultiSelect
          options={reactantTypeOptions}
          value={reactantTypes}
          onChange={setReactantTypes}
          inputId="filter-reactant-types"
          counts={counts.reactantTypes}
          countLabel="ELNs"
          placeholder="Select reactant types..."
          autoClose
        />
        <SplitToggle selectorKey="reactantTypes" values={reactantTypes} id="split-toggle" />
      </div>
    </div>
  );
}
