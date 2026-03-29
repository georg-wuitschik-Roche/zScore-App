import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useFilterStore } from '../stores/filterStore';
import { useTutorialStore } from '../hooks/useTutorial';
import { useEffectiveDataset } from '../hooks/useEffectiveDataset';
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
import { SettingsMenu } from './SettingsMenu';
import { Footer } from './Footer';
import { DEFAULTS, SPLIT_URL_KEYS } from '../data/types';
import type { SplitSelector } from '../data/types';

export function LandingPage() {
  const navigate = useNavigate();
  const uploadedDataset = useFilterStore((s) => s.uploadedDataset);
  const dropdownIndex = useFilterStore((s) => s.dropdownIndex);
  const setFilters = useFilterStore((s) => s.setFilters);
  const reactionTypes = useFilterStore((s) => s.reactionTypes);
  const setReactionTypes = useFilterStore((s) => s.setReactionTypes);
  const fgA = useFilterStore((s) => s.fgA);
  const setFgA = useFilterStore((s) => s.setFgA);
  const fgB = useFilterStore((s) => s.fgB);
  const setFgB = useFilterStore((s) => s.setFgB);
  const reactantTypes = useFilterStore((s) => s.reactantTypes);
  const setReactantTypes = useFilterStore((s) => s.setReactantTypes);
  const availableVersions = useFilterStore((s) => s.availableVersions);
  const activeVersion = useFilterStore((s) => s.activeVersion);
  const switchVersion = useFilterStore((s) => s.switchVersion);
  const isLoadingVersion = useFilterStore((s) => s.isLoadingVersion);
  // Use index for instant dropdowns; fall back to row scanning for uploaded CSVs
  const useIndex = !uploadedDataset && dropdownIndex !== null;

  const rowData = useEffectiveDataset();

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

  // Determine which selector (if any) can be split
  const splittableSelector: SplitSelector | null = useMemo(() => {
    if (reactantTypes.length >= 2) return 'reactantTypes';
    if (reactionTypes.length >= 2) return 'reactionTypes';
    if (fgA.length >= 2) return 'fgA';
    if (fgB.length >= 2) return 'fgB';
    return null;
  }, [reactionTypes, fgA, fgB, reactantTypes]);

  function handleExplore(split: SplitSelector | null) {
    if (reactionTypes.length === 0) return;

    setFilters({
      ...DEFAULTS,
      reactionTypes,
      fgA,
      fgB,
      reactantTypes,
      splitSelector: split,
      activeTab: 'boxplot',
      optionsPanelOpen: false,
    });

    const params = new URLSearchParams();
    params.set('rt', reactionTypes.join('|'));
    if (fgA.length > 0) params.set('fga', fgA.join('|'));
    if (fgB.length > 0) params.set('fgb', fgB.join('|'));
    if (reactantTypes.length > 0) params.set('cat', reactantTypes.join('|'));
    if (split) params.set('split', SPLIT_URL_KEYS[split]);
    navigate(`/dashboard?${params.toString()}`);
  }

  const startTutorial = useTutorialStore((s) => s.start);

  function handleStartTutorial() {
    setFilters({
      ...DEFAULTS,
      reactionTypes: ['Buchwald-Hartwig'],
      reactantTypes: ['Catalyst'],
      splitSelector: null,
      activeTab: 'boxplot',
      optionsPanelOpen: false,
    });
    startTutorial();
    navigate('/dashboard?rt=Buchwald-Hartwig&cat=Catalyst');
  }

  const [showAbout, setShowAbout] = useState(false);

  const showFilters = reactionTypes.length > 0;

  return (
    <div className="landing-container">
      <div className="landing-settings">
        <SettingsMenu variant="light" />
      </div>
      <img
        src="/assets/logo.svg"
        alt="Z-Score Dashboard"
        className="landing-logo"
      />
      <h1 className="landing-title">Z-Score Dashboard</h1>
      <p className="landing-subtitle">
        Search for a reaction type to explore z-score analytics
      </p>

      {availableVersions.length > 1 && (
        <div className="version-picker">
          <span className="version-picker-label">Dataset:</span>
          {availableVersions.map((v) => (
            <button
              key={v.id}
              className={`version-pill${v.id === activeVersion ? ' active' : ''}`}
              onClick={() => switchVersion(v.id)}
              disabled={isLoadingVersion}
            >
              {v.label}{v.date ? ` (${v.date})` : ''}
            </button>
          ))}
        </div>
      )}

      {isLoadingVersion && (
        <div className="version-loading">Loading dataset...</div>
      )}

      <div className="landing-filters">
        <div className="landing-filter-row">
          <div className="landing-filter-col landing-filter-col-wide">
            <label>Reaction Type(s):</label>
            <MultiSelect
              options={reactionTypeOptions}
              value={reactionTypes}
              onChange={setReactionTypes}
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
            {splittableSelector ? (
              <div className="landing-explore-split">
                <button className="landing-explore-btn" onClick={() => handleExplore(null)}>
                  Combined
                </button>
                <button className="landing-explore-btn landing-explore-btn-split" onClick={() => handleExplore(splittableSelector)}>
                  Split
                </button>
              </div>
            ) : (
              <button className="landing-explore-btn" onClick={() => handleExplore(null)}>
                Explore
              </button>
            )}
          </>
        )}
      </div>

      <div className="landing-actions">
        <button className="landing-tutorial-btn" onClick={handleStartTutorial}>
          Start Tutorial
        </button>
        <button className="landing-about-btn" onClick={() => setShowAbout(true)}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/>
            <line x1="12" y1="16" x2="12" y2="12"/>
            <line x1="12" y1="8" x2="12.01" y2="8"/>
          </svg>
          About the Data
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
        <a
          href="https://pubs.acs.org/doi/10.1021/acscentsci.5c02031"
          target="_blank"
          rel="noopener noreferrer"
          className="landing-github-link"
        >
          <img src="/assets/acs-logo.png" alt="ACS" width="20" height="20" />
          View on ACS Central Science
        </a>
      </div>

      {showAbout && (
        <div className="about-modal" onClick={() => setShowAbout(false)}>
          <div className="about-panel" onClick={(e) => e.stopPropagation()}>
            <div className="about-header">
              <h2>About the Data</h2>
              <button className="about-close" onClick={() => setShowAbout(false)}>
                &times;
              </button>
            </div>
            <div className="about-body">
              <p>
                This dashboard analyzes <strong>over 66,000 high-throughput experimentation (HTE) reactions</strong> across
                42 reaction types, focusing on transformations of drug-like molecules at Roche.
              </p>
              <h3>From LC-MS to z-Scores</h3>
              <p>
                Reaction outcomes are measured via LC-MS peak area percentages, normalized to exclude
                reagent and solvent peaks. For each substrate pair, a <strong>z-score</strong> is calculated:
              </p>
              <p className="about-formula">
                z = (x &minus; &mu;) / &sigma;
              </p>
              <p>
                where <em>x</em> is the observed area %, <em>&mu;</em> the mean, and <em>&sigma;</em> the
                standard deviation across all conditions for that transformation. This normalization
                makes results comparable across different reaction types: a 30% yield in a difficult
                transformation can score higher than 80% in an easy one.
              </p>
              <h3>Reagent Ranking</h3>
              <p>
                For each ELN (experiment), the top <em>n</em> z-scores containing a given reagent
                are selected. The reagent is then ranked by the <strong>median</strong> of those
                top-<em>n</em> values. With a large <em>n</em>, reagents that perform robustly
                across many conditions are favored. With a small <em>n</em>, reagents with
                extraordinary performance under specific conditions are highlighted. The
                default (<em>n</em> = 5) strikes a balance between both.
              </p>
              <p className="about-caveat">
                Note: The underlying distributions are non-normal (median skewness = 1.36).
                z-Scores are used here solely for normalization, not for probabilistic inference.
              </p>
              <p className="about-citation">
                Ahlbrecht, J.; Lutz, M.{'\u2009'}D.{'\u2009'}R.; Jost, V.;
                F{'\u00e4'}rber, M.; Br{'\u00e4'}se, S.; Wuitschik, G.{' '}
                <em>ACS Cent. Sci.</em> <strong>2026</strong>, 12 (2), 222–232.{' '}
                <a href="https://doi.org/10.1021/acscentsci.5c02031" target="_blank" rel="noopener noreferrer">
                  DOI: 10.1021/acscentsci.5c02031
                </a>
              </p>
            </div>
          </div>
        </div>
      )}
      <Footer />
    </div>
  );
}
