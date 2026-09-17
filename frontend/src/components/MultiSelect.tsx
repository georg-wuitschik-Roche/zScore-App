import { useState, useRef, useEffect, useCallback, useId } from 'react';

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
  /**
   * Id for the text input, so a visible <label htmlFor> can name the combobox.
   * Without it the control announces as an unnamed combobox.
   */
  inputId?: string;
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
  inputId,
}: MultiSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [highlightIndex, setHighlightIndex] = useState(-1);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const listboxId = useId();
  const optionId = (i: number) => `${listboxId}-opt-${i}`;

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

  // Drives both the popup and the combobox's aria-expanded, so the two can't
  // disagree about whether there's a listbox to point at.
  const expanded = isOpen && filtered.length > 0;

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
                // Not a tab stop — one per pill would push the input further
                // out of reach with every selection. Backspace on an empty
                // search removes the last pill for keyboard users.
                tabIndex={-1}
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
            id={inputId}
            type="text"
            className="multi-select-input"
            role="combobox"
            aria-autocomplete="list"
            aria-expanded={expanded}
            // Only while the listbox exists — a dangling IDREF names nothing.
            aria-controls={expanded ? listboxId : undefined}
            // Arrow keys move a visual highlight rather than DOM focus, so this
            // is what tells a screen reader which option is current.
            aria-activedescendant={
              expanded && highlightIndex >= 0 ? optionId(highlightIndex) : undefined
            }
            value={search}
            // Typing always reopens the list: after an autoClose selection, a
            // clearOption pick, or Escape, the input keeps focus with isOpen
            // false, and nothing else would bring the suggestions back.
            onChange={(e) => {
              setSearch(e.target.value);
              setIsOpen(true);
            }}
            onFocus={handleInputFocus}
            onBlur={() => { setIsOpen(false); setSearch(''); }}
            onKeyDown={handleKeyDown}
            placeholder={value.length === 0 ? placeholder : ''}
            size={Math.max(1, search.length || (value.length === 0 ? placeholder.length : 1))}
          />
        </div>
        {value.length > 0 && (
          <button
            type="button"
            className="multi-select-clear"
            // Not a tab stop — it would cost a second Tab press to leave any
            // dropdown that has a selection. Backspace still clears by keyboard.
            tabIndex={-1}
            // Keeps focus on the input, so its onBlur doesn't reset the control
            // out from under the click.
            onMouseDown={(e) => e.preventDefault()}
            onClick={(e) => {
              e.stopPropagation();
              onChange([]);
              setSearch('');
              inputRef.current?.focus();
            }}
            aria-label="Clear all selections"
          >
            Clear
          </button>
        )}
      </div>

      {expanded && (
        // tabIndex opts a long, scrolling list out of Chrome's focusable-
        // scroller heuristic: Tab used to land here, the blur unmounted the
        // list, and focus fell back to <body> — costing a phantom Tab press.
        <div
          id={listboxId}
          className="multi-select-dropdown"
          ref={dropdownRef}
          role="listbox"
          tabIndex={-1}
        >
          {filtered.map((opt, i) => {
            // Match on the label, not index — search can filter out clearOption
            const count =
              !counts || opt === clearOption ? undefined : (counts[opt] ?? 0);
            return (
              <div
                key={opt}
                id={optionId(i)}
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
