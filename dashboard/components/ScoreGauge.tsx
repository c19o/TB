'use client';

import type { UnifiedScore } from '@/types';

interface ScoreGaugeProps {
  score: UnifiedScore | null;
  compact?: boolean;
}

export default function ScoreGauge({ score, compact }: ScoreGaugeProps) {
  const value = score?.score ?? 0;

  // Map -10..+10 to 0..180 degrees
  const angle = ((value + 10) / 20) * 180;

  if (compact) {
    return (
      <div className="flex items-center gap-3">
        {/* Mini gauge */}
        <div className="relative w-16 h-9 overflow-hidden">
          <svg viewBox="0 0 100 55" className="w-full h-full">
            {/* Background arc */}
            <path
              d="M 5 50 A 45 45 0 0 1 95 50"
              fill="none"
              stroke="rgba(50, 50, 80, 0.3)"
              strokeWidth="6"
              strokeLinecap="round"
            />
            {/* Colored arc segments */}
            <path
              d="M 5 50 A 45 45 0 0 1 27.5 12"
              fill="none"
              stroke="#ef4444"
              strokeWidth="6"
              strokeLinecap="round"
              opacity="0.4"
            />
            <path
              d="M 27.5 12 A 45 45 0 0 1 50 5"
              fill="none"
              stroke="#f59e0b"
              strokeWidth="6"
              strokeLinecap="round"
              opacity="0.4"
            />
            <path
              d="M 50 5 A 45 45 0 0 1 72.5 12"
              fill="none"
              stroke="#22c55e"
              strokeWidth="6"
              strokeLinecap="round"
              opacity="0.4"
            />
            <path
              d="M 72.5 12 A 45 45 0 0 1 95 50"
              fill="none"
              stroke="#3b82f6"
              strokeWidth="6"
              strokeLinecap="round"
              opacity="0.4"
            />
            {/* Needle */}
            <line
              x1="50"
              y1="50"
              x2={50 + 35 * Math.cos((180 - angle) * Math.PI / 180)}
              y2={50 - 35 * Math.sin((180 - angle) * Math.PI / 180)}
              stroke="white"
              strokeWidth="2"
              strokeLinecap="round"
              className="transition-all duration-1000"
            />
            {/* Center dot */}
            <circle cx="50" cy="50" r="3" fill="white" />
          </svg>
        </div>

        {/* Score text */}
        <div className="flex flex-col items-center">
          <span className="text-[9px] text-slate-500 uppercase tracking-wider leading-none">Score</span>
          <span className={`text-lg font-bold num-display leading-none ${
            value >= 3 ? 'text-green-400' :
            value >= 0 ? 'text-slate-300' :
            value >= -3 ? 'text-yellow-400' :
            'text-red-400'
          }`}>
            {value > 0 ? '+' : ''}{value.toFixed(1)}
          </span>
        </div>
      </div>
    );
  }

  // Full gauge (not currently used but available)
  return (
    <div className="glass-panel p-4">
      <div className="relative w-32 h-20 mx-auto">
        <svg viewBox="0 0 100 55" className="w-full h-full">
          <path d="M 5 50 A 45 45 0 0 1 95 50" fill="none" stroke="rgba(50,50,80,0.3)" strokeWidth="8" strokeLinecap="round" />
          <line
            x1="50" y1="50"
            x2={50 + 40 * Math.cos((180 - angle) * Math.PI / 180)}
            y2={50 - 40 * Math.sin((180 - angle) * Math.PI / 180)}
            stroke="white" strokeWidth="2.5" strokeLinecap="round"
            className="transition-all duration-1000"
          />
          <circle cx="50" cy="50" r="4" fill="white" />
        </svg>
      </div>
      <div className="text-center mt-2">
        <span className="text-2xl font-bold num-display text-white">
          {value > 0 ? '+' : ''}{value.toFixed(1)}
        </span>
        <span className="text-xs text-slate-500 block">Unified Score</span>
      </div>
    </div>
  );
}
