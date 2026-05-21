'use client';

import { ReactNode } from 'react';
import { RotateCcw } from 'lucide-react';
import Image from 'next/image';

interface CafeHeaderProps {
  title: string;
  subtitle: string;
  status?: 'online' | 'connecting' | 'offline';
  rightActions?: ReactNode;
  onReset?: () => void;
  iconUrl?: string;
  iconSize?: number;
}

export function CafeHeader({ title, subtitle, status, rightActions, onReset, iconUrl = '/SLOP.png', iconSize = 100 }: CafeHeaderProps) {
  const statusColors: Record<string, string> = {
    online: '#6B8C42',
    connecting: '#C4A77D',
    offline: '#B85C4A',
  };

  const statusText: Record<string, string> = {
    online: 'ONLINE',
    connecting: 'CONNECTING...',
    offline: 'OFFLINE',
  };

  return (
    <div className="retro-window" style={{ borderRadius: 0, boxShadow: '0 4px 0px #2b1d1d', borderLeft: 'none', borderRight: 'none', borderTop: 'none' }}>
      {/* Title bar */}
      <div className="retro-titlebar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          {iconUrl && (
            <div style={{
              position: 'relative',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <Image 
                src={iconUrl}
                alt={title}
                width={iconSize}
                height={iconSize}
                style={{ objectFit: 'contain' }}
              />
            </div>
          )}
          <span className="retro-titlebar-title">- {subtitle}</span>
        </div>
        <div className="retro-winctrls">
          <button className="retro-winctrl" title="Minimize">-</button>
          <button className="retro-winctrl" title="Maximize">+</button>
          <button className="retro-winctrl" title="Close" style={{ background: '#e05528', color: '#fff8f0' }}>x</button>
        </div>
      </div>

      {/* Menu bar */}
      <div className="retro-menubar">
        <span className="retro-menuitem">File</span>
        <span className="retro-menuitem">View</span>
        <span className="retro-menuitem">Tools</span>
        <span className="retro-menuitem">Help</span>

        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 12 }}>
          {status && (
            <span style={{
              fontFamily: 'VT323, monospace',
              fontSize: 17,
              color: statusColors[status],
              display: 'flex',
              alignItems: 'center',
              gap: 5,
            }}>
              <span style={{
                display: 'inline-block',
                width: 8,
                height: 8,
                background: statusColors[status],
                border: '1px solid #2b1d1d',
                animation: status === 'connecting' ? 'blink 1s step-end infinite' : 'none',
              }} />
              {statusText[status]}
            </span>
          )}
          {rightActions}
          {onReset && (
            <button className="retro-btn" onClick={onReset} style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '2px 10px', fontSize: 16 }}>
              <RotateCcw size={12} />
              RESET
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
