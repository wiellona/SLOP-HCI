'use client';

import { ReactNode } from 'react';

interface CafeCardProps {
  title?: string;
  children: ReactNode;
  className?: string;
}

export function CafeCard({ title, children, className = '' }: CafeCardProps) {
  return (
    <div className={`cafe-card overflow-hidden ${className}`}>
      {title && (
        <div className="border-b border-[#E8DCC8] bg-[#FDF8F0] px-4 py-2">
          <h3 className="text-xs font-medium uppercase tracking-wide text-[#9B7B5C]">
            {title}
          </h3>
        </div>
      )}
      <div className={title ? '' : ''}>{children}</div>
    </div>
  );
}