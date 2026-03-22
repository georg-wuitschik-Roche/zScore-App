import { useState, useRef, useEffect, useCallback } from 'react';

interface MultiSelectProps {
  options: string[];
  value: string[];
  onChange: (values: string[]) => void;
  placeholder?: string;
  className?: string;
}

export function MultiSelect({
  options,
  value,
  onChange,
  placeholder = 'Select...',
  className = '',
}: MultiSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

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

  const filtered = options.filter(
    (opt) =>
      !value.includes(opt) &&
      opt.toLowerCase().includes(search.toLowerCase()),
  );

  function handleRemove(item: string) {
    onChange(value.filter((v) => v !== item));
  }

  function handleAdd(item: string) {
    onChange([...value, item]);
    setSearch('');
    inputRef.current?.focus();
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
            onKeyDown={handleKeyDown}
            placeholder={value.length === 0 ? placeholder : ''}
            size={Math.max(1, search.length || (value.length === 0 ? placeholder.length : 1))}
          />
        </div>
      </div>

      {isOpen && filtered.length > 0 && (
        <div className="multi-select-dropdown">
          {filtered.map((opt) => (
            <div
              key={opt}
              className="multi-select-option"
              onClick={() => handleAdd(opt)}
              role="option"
              aria-selected={false}
            >
              {opt}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
