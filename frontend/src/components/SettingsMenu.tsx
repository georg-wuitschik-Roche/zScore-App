import { useState, useRef } from 'react';
import { useFilterStore } from '../stores/filterStore';
import { resolveComparisonVersion } from '../data/comparison';
import type { UploadMode } from '../data/types';

export function SettingsMenu({ variant = 'dark' }: { variant?: 'dark' | 'light' }) {
  const uploadCSV = useFilterStore((s) => s.uploadCSV);
  const uploadedDataset = useFilterStore((s) => s.uploadedDataset);
  const uploadMode = useFilterStore((s) => s.uploadMode);
  const setUploadMode = useFilterStore((s) => s.setUploadMode);
  const clearUploadData = useFilterStore((s) => s.clearUploadData);
  const togglePresentationMode = useFilterStore((s) => s.togglePresentationMode);
  const presentationMode = useFilterStore((s) => s.presentationMode);
  const availableVersions = useFilterStore((s) => s.availableVersions);
  const activeVersion = useFilterStore((s) => s.activeVersion);
  const switchVersion = useFilterStore((s) => s.switchVersion);
  const isLoadingVersion = useFilterStore((s) => s.isLoadingVersion);
  const themePreference = useFilterStore((s) => s.themePreference);
  const setTheme = useFilterStore((s) => s.setTheme);
  const comparisonMode = useFilterStore((s) => s.comparisonMode);
  const setComparisonMode = useFilterStore((s) => s.setComparisonMode);
  const comparisonVersion = useFilterStore((s) => s.comparisonVersion);
  const setComparisonVersion = useFilterStore((s) => s.setComparisonVersion);

  const [open, setOpen] = useState(false);
  const [pendingUpload, setPendingUpload] = useState<{ text: string; name: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
        setPendingUpload({ text, name: file.name });
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  }

  function handleConfirmUpload(mode: UploadMode) {
    if (!pendingUpload) return;
    setUploadMode(mode);
    uploadCSV(pendingUpload.text, pendingUpload.name, mode);
    setPendingUpload(null);
  }

  return (
    <>
      <button
        className={`settings-toggle ${variant}`}
        id="settings-toggle"
        onClick={() => setOpen(true)}
        aria-label="Settings"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
          <circle cx="12" cy="12" r="3" />
        </svg>
      </button>

      {/* Settings modal */}
      {open && (
        <div className="settings-modal-backdrop" onClick={() => setOpen(false)}>
          <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
            <div className="settings-modal-header">
              <h2>Settings</h2>
              <button className="settings-modal-close" onClick={() => setOpen(false)}>&times;</button>
            </div>

            <div className="settings-modal-body">
              {/* DATA section */}
              <div className="settings-section" id="settings-section-data">
                <h3 className="settings-section-title">Data</h3>

                {availableVersions.length > 1 && (
                  <div className="settings-row">
                    <span className="settings-row-label">Dataset</span>
                    <div className="settings-pills">
                      {availableVersions.map((v) => (
                        <button
                          key={v.id}
                          className={`settings-pill settings-pill-with-sub${v.id === activeVersion ? ' active' : ''}`}
                          onClick={() => switchVersion(v.id)}
                          disabled={isLoadingVersion}
                        >
                          {v.label}
                          {v.date && <span className="settings-pill-date">{v.date}</span>}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                <div className="settings-row">
                  <span className="settings-row-label">Upload</span>
                  <button className="settings-action-btn" onClick={() => fileInputRef.current?.click()}>
                    {uploadedDataset ? 'Replace Dataset' : 'Upload Dataset'}
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv"
                    style={{ display: 'none' }}
                    onChange={handleFileChange}
                  />
                </div>

                {uploadedDataset && (
                  <>
                    <div className="settings-row">
                      <span className="settings-row-label">Mode</span>
                      <div className="settings-pills">
                        <button
                          className={`settings-pill${uploadMode === 'replace' ? ' active' : ''}`}
                          onClick={() => setUploadMode('replace')}
                        >
                          My data
                        </button>
                        <button
                          className={`settings-pill${uploadMode === 'combine' ? ' active' : ''}`}
                          onClick={() => setUploadMode('combine')}
                        >
                          Combined
                        </button>
                      </div>
                    </div>
                    <div className="settings-row">
                      <span className="settings-row-label" />
                      <button className="settings-remove-btn" onClick={() => clearUploadData()}>
                        Remove uploaded data
                      </button>
                    </div>
                  </>
                )}
              </div>

              {/* COMPARISON section */}
              {availableVersions.length > 1 && (
                <div className="settings-section" id="settings-section-comparison">
                  <h3 className="settings-section-title">Version Comparison</h3>

                  <div className="settings-row">
                    <span className="settings-row-label">Compare</span>
                    <div className="settings-pills">
                      <button
                        className={`settings-pill${!comparisonMode ? ' active' : ''}`}
                        onClick={() => setComparisonMode(false)}
                      >
                        Off
                      </button>
                      <button
                        className={`settings-pill${comparisonMode ? ' active' : ''}`}
                        onClick={() => setComparisonMode(true)}
                      >
                        On
                      </button>
                    </div>
                  </div>

                  {comparisonMode && (
                    <div className="settings-row">
                      <span className="settings-row-label">Compare with</span>
                      <div className="settings-pills">
                        {availableVersions
                          .filter((v) => v.id !== activeVersion)
                          .map((v) => {
                            const resolved = resolveComparisonVersion(availableVersions, activeVersion, comparisonVersion);
                            const isSelected = v.id === resolved;
                            return (
                              <button
                                key={v.id}
                                className={`settings-pill settings-pill-with-sub${isSelected ? ' active' : ''}`}
                                onClick={() => setComparisonVersion(v.id)}
                              >
                                {v.label}
                                {v.date && <span className="settings-pill-date">{v.date}</span>}
                              </button>
                            );
                          })}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* APPEARANCE section */}
              <div className="settings-section" id="settings-section-appearance">
                <h3 className="settings-section-title">Appearance</h3>

                <div className="settings-row">
                  <span className="settings-row-label">Theme</span>
                  <div className="settings-pills">
                    <button
                      className={`settings-pill${themePreference === 'auto' ? ' active' : ''}`}
                      onClick={() => setTheme('auto')}
                    >
                      Auto
                    </button>
                    <button
                      className={`settings-pill${themePreference === 'light' ? ' active' : ''}`}
                      onClick={() => setTheme('light')}
                    >
                      Light
                    </button>
                    <button
                      className={`settings-pill${themePreference === 'dark' ? ' active' : ''}`}
                      onClick={() => setTheme('dark')}
                    >
                      Dark
                    </button>
                  </div>
                </div>

                <div className="settings-row">
                  <span className="settings-row-label">Presentation</span>
                  <div className="settings-pills">
                    <button
                      className={`settings-pill${!presentationMode ? ' active' : ''}`}
                      onClick={() => { if (presentationMode) togglePresentationMode(); }}
                    >
                      Off
                    </button>
                    <button
                      className={`settings-pill${presentationMode ? ' active' : ''}`}
                      onClick={() => { if (!presentationMode) togglePresentationMode(); }}
                    >
                      On
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Upload mode selection modal */}
      {pendingUpload && (
        <div className="upload-mode-modal" onClick={() => setPendingUpload(null)}>
          <div className="upload-mode-panel" onClick={(e) => e.stopPropagation()}>
            <div className="upload-mode-header">
              <h3>How would you like to use this data?</h3>
              <button className="upload-mode-close-btn" onClick={() => setPendingUpload(null)}>
                &times;
              </button>
            </div>
            <p className="upload-mode-filename">{pendingUpload.name}</p>
            <div className="upload-mode-choices">
              <button className="upload-mode-choice" onClick={() => handleConfirmUpload('replace')}>
                <strong>My data only</strong>
                <span>Replace the built-in dataset with your uploaded data</span>
              </button>
              <button className="upload-mode-choice" onClick={() => handleConfirmUpload('combine')}>
                <strong>Combined with built-in</strong>
                <span>Merge your data with the built-in dataset</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
