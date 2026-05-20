'use client';

import { useState } from 'react';

interface EditPanelProps {
  initialText: string;
  onSave: (text: string) => void;
  onClose: () => void;
}

export function EditPanel({ initialText, onSave, onClose }: EditPanelProps) {
  const [text, setText] = useState(initialText);

  return (
    <div
      style={{
        position: 'fixed', inset: 0, zIndex: 50,
        background: 'rgba(42,29,29,0.6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
      onClick={onClose}
    >
      <div
        className="retro-window"
        style={{ width: '100%', maxWidth: 440 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="retro-titlebar">
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span className="retro-titlebar-icon" />
            <span className="retro-titlebar-title">edit_translation.exe</span>
          </div>
          <div className="retro-winctrls">
            <button className="retro-winctrl" onClick={onClose} style={{ background: '#e05528', color: '#fff8f0' }}>x</button>
          </div>
        </div>

        <div style={{ padding: 12, display: 'flex', flexDirection: 'column', gap: 10 }}>
          <label style={{ fontFamily: 'VT323, monospace', fontSize: 16, color: '#a08060', textTransform: 'uppercase' }}>
            EDIT TERJEMAHAN:
          </label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="retro-textarea"
            rows={4}
            placeholder="Ketik terjemahan yang benar..."
            autoFocus
          />
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="retro-btn" onClick={onClose} style={{ flex: 1 }}>
              BATAL
            </button>
            <button className="retro-btn-primary" onClick={() => onSave(text)} style={{ flex: 1 }}>
              SIMPAN
            </button>
          </div>
        </div>

        {/* Retro status bar */}
        <div className="retro-statusbar">
          <span>Ln 1, Col {text.length + 1}</span>
          <span className="retro-statusbar-cell">{text.length} chars</span>
        </div>
      </div>
    </div>
  );
}
