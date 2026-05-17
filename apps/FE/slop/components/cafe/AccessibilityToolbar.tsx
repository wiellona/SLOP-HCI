'use client';

import { useState } from 'react';
import { Type, Contrast } from 'lucide-react';

export function AccessibilityToolbar() {
  const [fontSize, setFontSize] = useState(16);
  const [highContrast, setHighContrast] = useState(false);

  const handleIncreaseFont = () => {
    const newSize = Math.min(fontSize + 2, 24);
    setFontSize(newSize);
    document.documentElement.style.fontSize = `${newSize}px`;
  };

  const handleDecreaseFont = () => {
    const newSize = Math.max(fontSize - 2, 12);
    setFontSize(newSize);
    document.documentElement.style.fontSize = `${newSize}px`;
  };

  const toggleHighContrast = () => {
    setHighContrast(!highContrast);
    document.body.classList.toggle('high-contrast', !highContrast);
  };

  return (
    <div
      className="retro-window"
      style={{
        position: 'fixed', bottom: 16, right: 16, zIndex: 50,
        display: 'flex', flexDirection: 'column', minWidth: 0,
      }}
    >
      <div className="retro-titlebar" style={{ fontSize: 14, padding: '3px 8px' }}>
        <span>ACCESS.EXE</span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '4px 6px', background: '#f4dcc8' }}>
        <button
          onClick={handleDecreaseFont}
          className="retro-btn"
          style={{ padding: '2px 7px', fontSize: 14 }}
          title="Perkecil teks"
        >
          A-
        </button>
        <span style={{ fontFamily: 'VT323, monospace', fontSize: 16, minWidth: 28, textAlign: 'center', color: '#2b1d1d' }}>
          {fontSize}
        </span>
        <button
          onClick={handleIncreaseFont}
          className="retro-btn"
          style={{ padding: '2px 7px', fontSize: 14 }}
          title="Perbesar teks"
        >
          A+
        </button>
        <div style={{ width: 2, height: 20, background: '#2b1d1d', margin: '0 2px' }} />
        <button
          onClick={toggleHighContrast}
          className={highContrast ? 'retro-btn-primary' : 'retro-btn'}
          style={{ padding: '2px 7px', fontSize: 14, display: 'flex', alignItems: 'center', gap: 3 }}
          title="Kontras tinggi"
        >
          <Contrast size={11} />
          HC
        </button>
      </div>
    </div>
  );
}
