import { useMemo, useState } from 'react';
import { useFilterStore } from '../stores/filterStore';
import { exportPlots, previewExportSize, currentPlotElements, ASPECT_RATIOS, FONT_SCALES } from '../plots/export';
import { Modal } from './Modal';
import type { ExportFormat } from '../plots/export';

const FORMATS: { value: ExportFormat; label: string; hint: string }[] = [
  { value: 'png', label: 'PNG', hint: 'High-resolution bitmap (4x), ready to paste into slides.' },
  { value: 'svg', label: 'SVG', hint: 'Vector — scales without blurring and stays editable in Illustrator.' },
];

/** One labelled row of mutually exclusive pills. */
function PillRow<T>({ label, options, value, onChange }: {
  label: string;
  options: readonly { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <div className="settings-row">
      <span className="settings-row-label">{label}</span>
      <div className="settings-pills">
        {options.map((o) => (
          <button
            key={String(o.value)}
            className={`settings-pill${value === o.value ? ' active' : ''}`}
            onClick={() => onChange(o.value)}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

interface Props {
  open: boolean;
  onClose: () => void;
}

/** Download settings popup: pick format, aspect ratio and font size, then export.
 *  Settings persist in the store, so they carry over to the next download. */
export function ExportDialog({ open, onClose }: Props) {
  const settings = useFilterStore((s) => s.exportSettings);
  const setSettings = useFilterStore((s) => s.setExportSettings);

  const [busy, setBusy] = useState(false);

  // Measured while the dialog is open — the plots behind it are still mounted.
  const size = useMemo(
    () => (open ? previewExportSize(currentPlotElements(), settings.aspectRatio, settings.fontScale) : null),
    [open, settings.aspectRatio, settings.fontScale],
  );

  if (!open) return null;

  async function handleDownload() {
    setBusy(true);
    try {
      await exportPlots(currentPlotElements(), settings);
      onClose();
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal title="Download Plots" className="export-dialog" onClose={onClose}>
      <div className="settings-section">
        <PillRow label="Format" options={FORMATS} value={settings.format} onChange={(format) => setSettings({ format })} />
        <p className="export-dialog-hint">{FORMATS.find((f) => f.value === settings.format)?.hint}</p>
        <PillRow label="Aspect ratio" options={ASPECT_RATIOS} value={settings.aspectRatio} onChange={(aspectRatio) => setSettings({ aspectRatio })} />
        <PillRow label="Font size" options={FONT_SCALES} value={settings.fontScale} onChange={(fontScale) => setSettings({ fontScale })} />
      </div>

      <div className="export-dialog-footer">
        <span className="export-dialog-size">
          {size ? `${size.width} × ${size.height} px` : 'No plot to export'}
        </span>
        <button className="options-btn export-dialog-download" onClick={handleDownload} disabled={!size || busy}>
          {busy ? 'Preparing…' : `Download ${settings.format.toUpperCase()}`}
        </button>
      </div>
    </Modal>
  );
}
