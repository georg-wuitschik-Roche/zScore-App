import { useState, useRef, useEffect, useCallback } from 'react';

interface MultiSelectProps {
  options: string[];
  value: string[];
  onChange: (values: string[]) => void;
  placeholder?: string;
  className?: string;
  /** Close dropdown after each selection (useful for single-purpose selectors). */
  autoClose?: boolean;
  /**
   * Label for an option that clears the selection (e.g. "All"). Only offered
   * while something is selected — with an empty selection it would be a no-op.
   */
  clearOption?: string;
  /**
   * Per-option counts, shown dimmed at the right of each row. Options absent
   * from the map count as 0. The clearOption row never shows a count.
   */
  counts?: Record<string, number>;
  /** Unit for the count tooltip, e.g. "ELNs". */
  countLabel?: string;
}

export function MultiSelect({
  options,
  value,
  onChange,
  placeholder = 'Select...',
  className = '',
  autoClose = false,
  clearOption,
  counts,
  countLabel,
}: MultiSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [highlightIndex, setHighlightIndex] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const handleClickOutside = useCallback((e: MouseEvent) => {
    if (
      containerRef.current &&
      !containerRef.current.contains(e.target as Node)
    ) {
      setIsOpen(false);
      setSearch('');
    }
  }, []);

  useEffect(() => {
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [handleClickOutside]);

  const selectable =
    clearOption && value.length > 0 ? [clearOption, ...options] : options;

  const filtered = selectable.filter(
    (opt) =>
      !value.includes(opt) &&
      opt.toLowerCase().includes(search.toLowerCase()),
  );

  // Reset highlight when filtered options change
  useEffect(() => {
    setHighlightIndex(-1);
  }, [search, filtered.length]);

  function handleRemove(item: string) {
    onChange(value.filter((v) => v !== item));
  }

  function handleAdd(item: string) {
    if (item === clearOption) {
      onChange([]);
      setSearch('');
      setIsOpen(false);
      return;
    }
    onChange([...value, item]);
    setSearch('');
    if (autoClose) {
      setIsOpen(false);
    } else {
      inputRef.current?.focus();
    }
  }

  function handleInputFocus() {
    setIsOpen(true);
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Backspace' && search === '' && value.length > 0) {
      onChange(value.slice(0, -1));
    }
    if (e.key === 'Escape') {
      setIsOpen(false);
      setSearch('');
    }
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (!isOpen) {
        setIsOpen(true);
      }
      setHighlightIndex((i) => (i < filtered.length - 1 ? i + 1 : 0));
    }
    if (e.key === 'ArrowUp') {
      e.preventDefault();
      setHighlightIndex((i) => (i > 0 ? i - 1 : filtered.length - 1));
    }
    if (e.key === 'Enter' && highlightIndex >= 0 && highlightIndex < filtered.length) {
      e.preventDefault();
      handleAdd(filtered[highlightIndex]);
    }
  }

  return (
    <div
      ref={containerRef}
      className={`multi-select ${className}`}
    >
      <div
        className="multi-select-control"
        onClick={() => {
          setIsOpen(true);
          inputRef.current?.focus();
        }}
      >
        <div className="multi-select-values">
          {value.map((v) => (
            <span key={v} className="multi-select-pill">
              {v}
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  handleRemove(v);
                }}
                aria-label={`Remove ${v}`}
              >
                ×
              </button>
            </span>
          ))}
          <input
            ref={inputRef}
            type="text"
            className="multi-select-input"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            onFocus={handleInputFocus}
            onBlur={() => { setIsOpen(false); setSearch(''); }}
            onKeyDown={handleKeyDown}
            placeholder={value.length === 0 ? placeholder : ''}
            size={Math.max(1, search.length || (value.length === 0 ? placeholder.length : 1))}
          />
        </div>
      </div>

      {isOpen && filtered.length > 0 && (
        <div className="multi-select-dropdown" ref={dropdownRef}>
          {filtered.map((opt, i) => {
            // Match on the label, not index — search can filter out clearOption
            const count =
              !counts || opt === clearOption ? undefined : (counts[opt] ?? 0);
            return (
              <div
                key={opt}
                className={`multi-select-option${i === highlightIndex ? ' highlighted' : ''}`}
                onMouseDown={(e) => { e.preventDefault(); handleAdd(opt); }}
                onMouseEnter={() => setHighlightIndex(i)}
                role="option"
                aria-selected={i === highlightIndex}
                ref={i === highlightIndex ? (el) => el?.scrollIntoView({ block: 'nearest' }) : undefined}
              >
                <span className="multi-select-option-label">{opt}</span>
                {count !== undefined && (
                  <span
                    className="multi-select-option-count"
                    title={countLabel ? `${count} ${countLabel}` : undefined}
                  >
                    {count}
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
