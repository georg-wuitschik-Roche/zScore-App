import { useState, useRef, useEffect, useCallback } from 'react';
import { useFilterStore } from '../stores/filterStore';
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

  const [open, setOpen] = useState(false);
  const [pendingUpload, setPendingUpload] = useState<{ text: string; name: string } | null>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleClickOutside = useCallback((e: MouseEvent) => {
    if (wrapperRef.current && !wrapperRef.current.contains(e.target as Node)) {
      setOpen(false);
    }
  }, []);

  useEffect(() => {
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [handleClickOutside]);

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
        setOpen(false);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  }

  function handleConfirmUpload(mode: UploadMode) {
    if (!pendingUpload) return;
    setUploadMode(mode);
    uploadCSV(pendingUpload.text, pendingUpload.name);
    setPendingUpload(null);
  }

  return (
    <>
      <div className="settings-wrapper" ref={wrapperRef}>
        <button
          className={`settings-toggle ${variant}`}
          onClick={() => setOpen((prev) => !prev)}
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
        <div className={`settings-dropdown${open ? '' : ' hidden'}`}>
          {availableVersions.length > 1 && (
            <>
              <div className="settings-dropdown-row">
                <span className="settings-dropdown-row-label">Dataset</span>
                <div className="upload-mode-btns">
                  {availableVersions.map((v) => (
                    <button
                      key={v.id}
                      className={`upload-mode-btn${v.id === activeVersion ? ' active' : ''}`}
                      onClick={() => switchVersion(v.id)}
                      disabled={isLoadingVersion}
                      title={v.date ?? ''}
                    >
                      {v.label}
                    </button>
                  ))}
                </div>
              </div>
              <div className="settings-dropdown-divider" />
            </>
          )}
          <button
            className="settings-dropdown-btn"
            onClick={() => fileInputRef.current?.click()}
          >
            {uploadedDataset ? 'Replace Dataset' : 'Upload Dataset'}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />
          {uploadedDataset && (
            <>
              <div className="settings-dropdown-row">
                <span className="settings-dropdown-row-label">Mode</span>
                <div className="upload-mode-btns">
                  <button
                    className={`upload-mode-btn${uploadMode === 'replace' ? ' active' : ''}`}
                    onClick={() => setUploadMode('replace')}
                  >
                    My data
                  </button>
                  <button
                    className={`upload-mode-btn${uploadMode === 'combine' ? ' active' : ''}`}
                    onClick={() => setUploadMode('combine')}
                  >
                    Combined
                  </button>
                </div>
              </div>
              <button
                className="settings-dropdown-btn settings-dropdown-btn-danger"
                onClick={() => {
                  clearUploadData();
                  setOpen(false);
                }}
              >
                Remove Data
              </button>
              <div className="settings-dropdown-divider" />
            </>
          )}
          <button
            className={`settings-dropdown-btn${presentationMode ? ' active' : ''}`}
            onClick={() => {
              togglePresentationMode();
              setOpen(false);
            }}
          >
            {presentationMode ? 'Exit Presentation Mode' : 'Presentation Mode'}
          </button>
        </div>
      </div>

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
