import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useFilterStore } from '../stores/filterStore';
import { useTutorialStore } from '../hooks/useTutorial';
import {
  getReactionTypes,
  getFgOptions,
  getFgBOptionsConditioned,
  getReactantOptions,
} from '../data/dropdownOptions';
import { MultiSelect } from './MultiSelect';
import { Footer } from './Footer';

export function LandingPage() {
  const navigate = useNavigate();
  const dataset = useFilterStore((s) => s.dataset);
  const setFilters = useFilterStore((s) => s.setFilters);

  const [reactionTypes, setReactionTypes] = useState<string[]>([]);
  const [fgA, setFgA] = useState<string[]>([]);
  const [fgB, setFgB] = useState<string[]>([]);
  const [reactantTypes, setReactantTypes] = useState<string[]>([]);

  const reactionTypeOptions = useMemo(
    () => getReactionTypes(dataset),
    [dataset],
  );

  const fgAOptions = useMemo(
    () => ['All', ...getFgOptions(dataset, reactionTypes)],
    [dataset, reactionTypes],
  );

  const fgBOptions = useMemo(
    () => ['All', ...getFgBOptionsConditioned(dataset, reactionTypes, fgA)],
    [dataset, reactionTypes, fgA],
  );

  const reactantTypeOptions = useMemo(
    () => getReactantOptions(dataset, reactionTypes),
    [dataset, reactionTypes],
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

      <button className="landing-tutorial-btn" onClick={handleStartTutorial}>
        Start Tutorial
      </button>
      <Footer />
    </div>
  );
}
