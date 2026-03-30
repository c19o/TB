'use client';

import type { OverlayType } from '@/types';

interface OverlayToggleProps {
  type: OverlayType;
  label: string;
  color: string;
  icon: string;
  active: boolean;
  onToggle: () => void;
}

export default function OverlayToggle({ type, label, color, icon, active, onToggle }: OverlayToggleProps) {
  return (
    <button
      onClick={onToggle}
      className={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-left transition-all group ${
        active ? 'toggle-active' : 'hover:bg-white/[0.03]'
      }`}
    >
      {/* Custom checkbox */}
      <div
        className={`w-3.5 h-3.5 rounded-sm border flex items-center justify-center shrink-0 transition-all ${
          active ? 'border-transparent' : 'border-slate-600'
        }`}
        style={active ? { backgroundColor: color } : {}}
      >
        {active && (
          <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
          </svg>
        )}
      </div>

      {/* Icon */}
      <span
        className="text-xs shrink-0 w-4 text-center"
        style={{ color: active ? color : '#475569' }}
      >
        {icon}
      </span>

      {/* Label */}
      <span className={`text-xs truncate ${active ? 'text-slate-300' : 'text-slate-500 group-hover:text-slate-400'}`}>
        {label}
      </span>
    </button>
  );
}
