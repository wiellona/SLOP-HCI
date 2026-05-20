'use client';

import { Wifi, WifiOff, Clock } from 'lucide-react';

interface StatusBadgeProps {
  isOnline: boolean;
  currentTime: Date;
}

export function StatusBadge({ isOnline, currentTime }: StatusBadgeProps) {
  return (
    <div className="fixed bottom-4 left-4 z-50 flex items-center gap-3 rounded-full bg-white/90 px-3 py-1.5 text-xs shadow-md backdrop-blur-sm border border-[#E8DCC8]">
      <div className="flex items-center gap-1.5">
        {isOnline ? (
          <Wifi className="h-3 w-3 text-[#6B8C42]" />
        ) : (
          <WifiOff className="h-3 w-3 text-[#B85C4A]" />
        )}
        <span className="text-[#4A3728]">
          {isOnline ? 'Sistem Online' : 'Sistem Offline'}
        </span>
      </div>
      <div className="flex items-center gap-1.5">
        <Clock className="h-3 w-3 text-[#C4A77D]" />
        <span className="text-[#4A3728]">
          {currentTime.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </div>
  );
}