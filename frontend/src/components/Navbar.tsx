import { useNavigate } from 'react-router-dom';
import { useFilterStore } from '../stores/filterStore';
import { useEffectiveDataset } from '../hooks/useEffectiveDataset';
import { SettingsMenu } from './SettingsMenu';

export function Navbar() {
  const navigate = useNavigate();
  const resetFilters = useFilterStore((s) => s.resetFilters);
  const uploadError = useFilterStore((s) => s.uploadError);
  const uploadFileName = useFilterStore((s) => s.uploadFileName);
  const uploadedDataset = useFilterStore((s) => s.uploadedDataset);
  const clearUploadError = useFilterStore((s) => s.clearUploadError);
  const effectiveData = useEffectiveDataset();

  return (
    <>
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

      {/* Upload error modal */}
      {uploadError && (
        <div className="upload-error-modal" style={{ display: 'flex' }}>
          <div className="upload-error-panel">
            <div className="upload-error-header">
              <h3>Upload Error</h3>
              <button
                className="upload-error-close-btn"
                onClick={clearUploadError}
              >
                &times;
              </button>
            </div>
            <div className="upload-error-body">
              <p>{uploadError}</p>
            </div>
            <div className="upload-error-footer">
              <h4>Required Columns:</h4>
              <ul>
                {[
                  'ELN_ID', 'PLATENUMBER', 'Coordinate', 'AREA_TOTAL_REDUCED',
                  'Base', 'Catalyst', 'Solvent', 'Ligand',
                  'Reaction Type', 'FG A', 'FG B', 'FG_sorted', 'z-Score',
                ].map((col) => (
                  <li key={col}><code>{col}</code></li>
                ))}
              </ul>
              <button className="close-btn-full" onClick={clearUploadError}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
