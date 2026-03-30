'use client';

import type { OverlayType } from '@/types';
import OverlayToggle from './OverlayToggle';

interface SidebarProps {
  overlays: Record<OverlayType, boolean>;
  onToggle: (type: OverlayType) => void;
  signals: any;
}

const OVERLAY_CONFIG: { type: OverlayType; label: string; color: string; icon: string; description: string }[] = [
  { type: 'lunar', label: 'Lunar Phases', color: '#a78bfa', icon: '\u263D', description: 'New/Full Moon markers' },
  { type: 'caution', label: 'Caution Days', color: '#ef4444', icon: '\u26A0', description: 'Bearish number dates' },
  { type: 'pump', label: 'Pump Days', color: '#22c55e', icon: '\u2191', description: 'Bullish number dates' },
  { type: 'ritual', label: 'Ritual Dates', color: '#8b5cf6', icon: '\u2726', description: 'Master numbers & 22nd' },
  // Tweet overlay hidden — SHAP showed 0 tweet features matter. Data kept intact.
  // { type: 'tweet', label: 'Tweet Markers', color: '#06b6d4', icon: '\u2709', description: 'Decoded tweet signals' },
  { type: 'wyckoff', label: 'Wyckoff Phases', color: '#f59e0b', icon: 'W', description: 'Accumulation/Distribution' },
  { type: 'elliott', label: 'Elliott Waves', color: '#ec4899', icon: '\u223F', description: 'Wave count overlay' },
  { type: 'gann', label: 'Gann Angles', color: '#14b8a6', icon: '\u2220', description: 'Gann angle lines' },
];

export default function Sidebar({ overlays, onToggle, signals }: SidebarProps) {
  const activeSignals = signals?.signals || [];

  return (
    <div className="flex flex-col h-full">
      {/* Overlay Toggles */}
      <div className="p-3">
        <div className="flex items-center gap-2 mb-3">
          <svg className="w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6.429 9.75L2.25 12l4.179 2.25m0-4.5l5.571 3 5.571-3m-11.142 0L2.25 7.5 12 2.25l9.75 5.25-4.179 2.25m0 0L12 12.75 6.43 9.75m11.14 0l4.179 2.25L12 17.25 2.25 12l4.179-2.25" />
          </svg>
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Overlays</h3>
        </div>

        <div className="space-y-0.5">
          {OVERLAY_CONFIG.map(config => (
            <OverlayToggle
              key={config.type}
              type={config.type}
              label={config.label}
              color={config.color}
              icon={config.icon}
              active={overlays[config.type]}
              onToggle={() => onToggle(config.type)}
            />
          ))}
        </div>
      </div>

      {/* Signal Legend */}
      <div className="p-3 border-t border-panel-border">
        <div className="flex items-center gap-2 mb-3">
          <svg className="w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.568 3H5.25A2.25 2.25 0 003 5.25v4.318c0 .597.237 1.17.659 1.591l9.581 9.581c.699.699 1.78.872 2.607.33a18.095 18.095 0 005.223-5.223c.542-.827.369-1.908-.33-2.607L11.16 3.66A2.25 2.25 0 009.568 3z" />
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 6h.008v.008H6V6z" />
          </svg>
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Legend</h3>
        </div>

        <div className="space-y-1.5">
          {OVERLAY_CONFIG.map(config => (
            <div key={config.type} className="flex items-center gap-2">
              <div
                className="w-2.5 h-2.5 rounded-full shrink-0"
                style={{ backgroundColor: config.color, opacity: overlays[config.type] ? 1 : 0.3 }}
              />
              <span className={`text-[10px] ${overlays[config.type] ? 'text-slate-400' : 'text-slate-600'}`}>
                {config.description}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Active Signals Summary */}
      {activeSignals.length > 0 && (
        <div className="p-3 border-t border-panel-border">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Active Today</h3>
          <div className="space-y-1.5">
            {activeSignals.map((sig: any, i: number) => (
              <div
                key={i}
                className="px-2 py-1.5 rounded text-[10px] border"
                style={{
                  borderColor: `${sig.color}40`,
                  backgroundColor: `${sig.color}10`,
                }}
              >
                <div className="font-medium" style={{ color: sig.color }}>{sig.label}</div>
                <div className="text-slate-500 mt-0.5 leading-tight">{sig.description}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
