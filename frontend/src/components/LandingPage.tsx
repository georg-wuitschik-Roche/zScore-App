import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useFilterStore } from '../stores/filterStore';
import { getReactionTypes } from '../data/dropdownOptions';
import { MultiSelect } from './MultiSelect';
import { Footer } from './Footer';

export function LandingPage() {
  const navigate = useNavigate();
  const dataset = useFilterStore((s) => s.dataset);
  const setReactionTypes = useFilterStore((s) => s.setReactionTypes);

  const [selected, setSelected] = useState<string[]>([]);

  const reactionTypeOptions = useMemo(
    () => getReactionTypes(dataset),
    [dataset],
  );

  function handleStartTutorial() {
    setReactionTypes(['Buchwald-Hartwig']);
    navigate('/dashboard?rt=Buchwald-Hartwig');
  }

  return (
    <div className="landing-container">
      <img
        src="/assets/logo.png"
        alt="Z-Score Dashboard"
        className="landing-logo"
      />
      <h1 className="landing-title">Z-Score Dashboard</h1>
      <p className="landing-subtitle">
        Search for a reaction type to explore z-score analytics
      </p>
      <div className="landing-search-wrapper">
        <MultiSelect
          options={reactionTypeOptions}
          value={selected}
          onChange={(values) => {
            setSelected(values);
            if (values.length > 0) {
              setReactionTypes(values);
              const params = new URLSearchParams();
              params.set('rt', values.join('|'));
              navigate(`/dashboard?${params.toString()}`);
            }
          }}
          placeholder="Search by reaction type..."
          className="landing-search"
        />
      </div>
      <button className="landing-tutorial-btn" onClick={handleStartTutorial}>
        Start Tutorial
      </button>
      <Footer />
    </div>
  );
}
