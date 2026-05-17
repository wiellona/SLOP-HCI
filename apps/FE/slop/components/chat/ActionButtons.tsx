'use client';

import { Send } from 'lucide-react';

interface CafeActionButtonsProps {
  onSend: () => void;
  disabled?: boolean;
  sendLabel?: string;
}

export function CafeActionButtons({
  onSend,
  disabled = false,
  sendLabel = 'KIRIM PESANAN',
}: CafeActionButtonsProps) {
  return (
    <div style={{ display: 'flex', gap: 8 }}>
      <button
        onClick={onSend}
        disabled={disabled}
        className="retro-btn-primary"
        style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, padding: '10px 0', fontSize: 20 }}
      >
        <Send size={16} />
        {sendLabel}
      </button>
    </div>
  );
}
