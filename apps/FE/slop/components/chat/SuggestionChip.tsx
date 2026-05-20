'use client';

interface CafeSuggestionChipsProps {
  currentWord: string;
  suggestions: string[];
  onSelect: (suggestion: string) => void;
}

export function CafeSuggestionChips({
  currentWord,
  suggestions,
  onSelect,
}: CafeSuggestionChipsProps) {
  if (suggestions.length === 0) return null;

  return (
    <div className="retro-window">
      <div className="retro-titlebar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span className="retro-titlebar-icon" />
          <span className="retro-titlebar-title">suggestions.exe</span>
        </div>
      </div>
      <div style={{ padding: 8 }}>
        <p style={{ fontFamily: 'VT323, monospace', fontSize: 16, color: '#a08060', margin: '0 0 6px 0' }}>
          BUKAN "{currentWord.toUpperCase()}"? COBA:
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {suggestions.map((suggestion, idx) => (
            <button
              key={idx}
              onClick={() => onSelect(suggestion)}
              className="retro-chip"
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
