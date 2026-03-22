import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useFilterStore } from '../stores/filterStore';
import { useTutorialStore } from '../hooks/useTutorial';
import {
  getReactionTypes,
  getFgOptions,
  getFgBOptionsConditioned,
  getReactantOptions,
  getReactionTypesFromIndex,
  getFgOptionsFromIndex,
  getFgBOptionsFromIndex,
  getReactantOptionsFromIndex,
} from '../data/dropdownOptions';
import { MultiSelect } from './MultiSelect';
import { Footer } from './Footer';

export function LandingPage() {
  const navigate = useNavigate();
  const dataset = useFilterStore((s) => s.dataset);
  const uploadedDataset = useFilterStore((s) => s.uploadedDataset);
  const dropdownIndex = useFilterStore((s) => s.dropdownIndex);
  const setFilters = useFilterStore((s) => s.setFilters);

  // Use index for instant dropdowns; fall back to row scanning for uploaded CSVs
  const useIndex = !uploadedDataset && dropdownIndex !== null;

  const [reactionTypes, setReactionTypes] = useState<string[]>([]);
  const [fgA, setFgA] = useState<string[]>([]);
  const [fgB, setFgB] = useState<string[]>([]);
  const [reactantTypes, setReactantTypes] = useState<string[]>([]);

  const rowData = uploadedDataset ?? dataset;

  const reactionTypeOptions = useMemo(
    () => useIndex
      ? getReactionTypesFromIndex(dropdownIndex)
      : getReactionTypes(rowData),
    [useIndex, dropdownIndex, rowData],
  );

  const fgAOptions = useMemo(
    () => ['All', ...(useIndex
      ? getFgOptionsFromIndex(dropdownIndex, reactionTypes)
      : getFgOptions(rowData, reactionTypes))],
    [useIndex, dropdownIndex, rowData, reactionTypes],
  );

  const fgBOptions = useMemo(
    () => ['All', ...(useIndex
      ? getFgBOptionsFromIndex(dropdownIndex, reactionTypes, fgA)
      : getFgBOptionsConditioned(rowData, reactionTypes, fgA))],
    [useIndex, dropdownIndex, rowData, reactionTypes, fgA],
  );

  const reactantTypeOptions = useMemo(
    () => useIndex
      ? getReactantOptionsFromIndex(dropdownIndex, reactionTypes)
      : getReactantOptions(rowData, reactionTypes),
    [useIndex, dropdownIndex, rowData, reactionTypes],
  );

  function handleExplore() {
    if (reactionTypes.length === 0) return;

    setFilters({
      reactionTypes,
      fgA,
      fgB,
      reactantTypes,
    });

    const params = new URLSearchParams();
    params.set('rt', reactionTypes.join('|'));
    if (fgA.length > 0) params.set('fga', fgA.join('|'));
    if (fgB.length > 0) params.set('fgb', fgB.join('|'));
    if (reactantTypes.length > 0) params.set('cat', reactantTypes.join('|'));
    navigate(`/dashboard?${params.toString()}`);
  }

  const startTutorial = useTutorialStore((s) => s.start);

  function handleStartTutorial() {
    setFilters({
      reactionTypes: ['Buchwald-Hartwig'],
      reactantTypes: ['Catalyst'],
      fgA: [],
      fgB: [],
    });
    startTutorial();
    navigate('/dashboard?rt=Buchwald-Hartwig&cat=Catalyst');
  }

  const showFilters = reactionTypes.length > 0;

  return (
    <div className="landing-container">
      <img
        src="/assets/logo.svg"
        alt="Z-Score Dashboard"
        className="landing-logo"
      />
      <h1 className="landing-title">Z-Score Dashboard</h1>
      <p className="landing-subtitle">
        Search for a reaction type to explore z-score analytics
      </p>

      <div className="landing-filters">
        <div className="landing-filter-row">
          <div className="landing-filter-col landing-filter-col-wide">
            <label>Reaction Type(s):</label>
            <MultiSelect
              options={reactionTypeOptions}
              value={reactionTypes}
              onChange={(vals) => {
                setReactionTypes(vals);
                setFgA([]);
                setFgB([]);
              }}
              placeholder="Search by reaction type..."
              autoClose
            />
          </div>
        </div>

        {showFilters && (
          <>
            <div className="landing-filter-row">
              <div className="landing-filter-col">
                <label>Functional Group(s) A:</label>
                <MultiSelect
                  options={fgAOptions}
                  value={fgA}
                  onChange={(vals) => {
                    if (vals.includes('All') && !fgA.includes('All')) {
                      setFgA([]);
                    } else {
                      setFgA(vals.filter((v) => v !== 'All'));
                    }
                  }}
                  placeholder="All (no filter)"
                />
              </div>
              <div className="landing-filter-col">
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
                />
              </div>
            </div>
            <div className="landing-filter-row">
              <div className="landing-filter-col">
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
            <button className="landing-explore-btn" onClick={handleExplore}>
              Explore
            </button>
          </>
        )}
      </div>

      <div className="landing-actions">
        <button className="landing-tutorial-btn" onClick={handleStartTutorial}>
          Start Tutorial
        </button>
        <a
          href="https://github.com/georg-wuitschik-Roche/zScore-App"
          target="_blank"
          rel="noopener noreferrer"
          className="landing-github-link"
        >
          <svg width="20" height="20" viewBox="0 0 16 16" fill="currentColor">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
          </svg>
          View on GitHub
        </a>
      </div>
      <Footer />
    </div>
  );
}
