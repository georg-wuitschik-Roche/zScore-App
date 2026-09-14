import { useNavigate } from 'react-router-dom';
import { useFilterStore } from '../stores/filterStore';
import { useEffectiveDataset } from '../hooks/useEffectiveDataset';
import { SettingsMenu } from './SettingsMenu';

export function Navbar() {
  const navigate = useNavigate();
  const resetFilters = useFilterStore((s) => s.resetFilters);
  const uploadFileName = useFilterStore((s) => s.uploadFileName);
  const uploadedDataset = useFilterStore((s) => s.uploadedDataset);
  const effectiveData = useEffectiveDataset();

  return (
    <nav className="navbar">
      <div className="navbar-inner">
        <img
          src="/assets/hiker.png"
          alt="Home"
          className="logo"
          onClick={() => navigate('/')}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') navigate('/');
          }}
        />
        <h1 className="title">
          Lessons from {effectiveData.length > 0 ? effectiveData.length.toLocaleString() : '...'} High-Throughput Experiments
          {uploadedDataset && uploadFileName && (
            <span className="title-dataset-name"> — {uploadFileName}</span>
          )}
        </h1>

        <SettingsMenu />

        {/* Reset button */}
        <button className="reset-btn-subtle" id="reset-btn" onClick={() => resetFilters()}>
          <svg
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            style={{ marginRight: 6, verticalAlign: 'middle' }}
          >
            <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
            <path d="M3 3v5h5" />
          </svg>
          Reset
        </button>
      </div>
    </nav>
  );
}
