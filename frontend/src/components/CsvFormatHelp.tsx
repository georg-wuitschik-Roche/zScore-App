import { MAX_UPLOAD_MB, OPTIONAL_COLUMNS, REPO_URL, REQUIRED_COLUMNS } from '../data/types';
import { buildCSVTemplate } from '../data/csvTemplate';
import { downloadTextFile } from '../data/download';

const FORMAT_DOCS_URL = `${REPO_URL}#csv-upload-format`;

function handleDownloadTemplate() {
  downloadTextFile('zscore_template.csv', buildCSVTemplate());
}

/**
 * What an uploaded CSV has to look like. Shown up front from the Settings
 * upload row, and again in the upload error modal.
 */
export function CsvFormatHelp() {
  return (
    <div className="csv-help">
      <p className="csv-help-label">Required columns</p>
      <ul className="csv-help-columns">
        {REQUIRED_COLUMNS.map((col) => (
          <li key={col}><code>{col}</code></li>
        ))}
      </ul>

      <p className="csv-help-label">Optional columns</p>
      <ul className="csv-help-columns">
        {OPTIONAL_COLUMNS.map((col) => (
          <li key={col}><code>{col}</code></li>
        ))}
      </ul>

      <ul className="csv-help-rules">
        <li>Comma, semicolon or tab separated — the delimiter is detected automatically.</li>
        <li><code>z-Score</code> must contain numeric values; a decimal comma is accepted.</li>
        <li>Maximum file size {MAX_UPLOAD_MB} MB.</li>
      </ul>

      <div className="csv-help-actions">
        <button className="settings-action-btn" onClick={handleDownloadTemplate}>
          Download template CSV
        </button>
        <a
          className="csv-help-link"
          href={FORMAT_DOCS_URL}
          target="_blank"
          rel="noopener noreferrer"
        >
          Full details on GitHub ↗
        </a>
      </div>
    </div>
  );
}
