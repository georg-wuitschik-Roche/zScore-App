import { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useFilterStore } from '../stores/filterStore';

export function Navbar() {
  const navigate = useNavigate();
  const resetFilters = useFilterStore((s) => s.resetFilters);
  const togglePresentationMode = useFilterStore(
    (s) => s.togglePresentationMode,
  );
  const presentationMode = useFilterStore((s) => s.presentationMode);
  const uploadCSV = useFilterStore((s) => s.uploadCSV);
  const dataset = useFilterStore((s) => s.dataset);
  const uploadError = useFilterStore((s) => s.uploadError);
  const uploadFileName = useFilterStore((s) => s.uploadFileName);
  const uploadedDataset = useFilterStore((s) => s.uploadedDataset);
  const clearUploadError = useFilterStore((s) => s.clearUploadError);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const settingsRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleClickOutside = useCallback((e: MouseEvent) => {
    if (
      settingsRef.current &&
      !settingsRef.current.contains(e.target as Node)
    ) {
      setSettingsOpen(false);
    }
  }, []);

  useEffect(() => {
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [handleClickOutside]);

  function handleUploadClick() {
    fileInputRef.current?.click();
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > 50 * 1024 * 1024) {
      alert('File too large (max 50 MB)');
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target?.result;
      if (typeof text === 'string') {
        uploadCSV(text, file.name);
        setSettingsOpen(false);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  }

  function handleReset() {
    resetFilters();
  }

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
            Lessons from {dataset.length > 0 ? dataset.length.toLocaleString() : '...'} High-Throughput Experiments
          </h1>

          {/* Upload status indicator */}
          {uploadedDataset && uploadFileName && (
            <span className="upload-status">
              Using: {uploadFileName} ({uploadedDataset.length.toLocaleString()} rows)
            </span>
          )}

          {/* Settings gear + dropdown */}
          <div className="settings-wrapper" ref={settingsRef}>
            <button
              className="settings-toggle"
              onClick={() => setSettingsOpen((prev) => !prev)}
              aria-label="Settings"
            >
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
                <circle cx="12" cy="12" r="3" />
              </svg>
            </button>
            <div
              className={`settings-dropdown${settingsOpen ? '' : ' hidden'}`}
            >
              <div className="upload-container settings-dropdown-item">
                <button
                  className="settings-dropdown-btn"
                  onClick={handleUploadClick}
                >
                  Upload Dataset
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  style={{ display: 'none' }}
                  onChange={handleFileChange}
                />
              </div>
              <button
                className={`settings-dropdown-btn${presentationMode ? ' active' : ''}`}
                onClick={() => {
                  togglePresentationMode();
                  setSettingsOpen(false);
                }}
              >
                Presentation Mode
              </button>
            </div>
          </div>

          {/* Reset button */}
          <button className="reset-btn-subtle" onClick={handleReset}>
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
