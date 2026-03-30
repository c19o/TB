'use client';
import { API } from '@/lib/api';

import { useEffect, useState } from 'react';

interface SignalPanelProps {
  score: any | null;
  signals: any;
}

function ComponentBar({ label, value, max = 10 }: { label: string; value: number; max?: number }) {
  const percentage = Math.min(100, Math.abs(value) / max * 100);
  const isPositive = value >= 0;
  const color = isPositive ? '#22c55e' : '#ef4444';

  return (
    <div className="mb-2">
      <div className="flex items-center justify-between mb-0.5">
        <span className="text-[10px] text-slate-500">{label}</span>
        <span className="text-[10px] num-display" style={{ color }}>
          {value > 0 ? '+' : ''}{typeof value === 'number' ? value.toFixed(1) : value}
        </span>
      </div>
      <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
        <div className="flex h-full">
          {!isPositive && (
            <div className="ml-auto h-full rounded-full" style={{ width: `${percentage}%`, backgroundColor: color }} />
          )}
          {isPositive && (
            <div className="h-full rounded-full" style={{ width: `${percentage}%`, backgroundColor: color }} />
          )}
        </div>
      </div>
    </div>
  );
}

function SignalItem({ signal }: { signal: any }) {
  const dirColor = signal.direction > 0 ? 'text-green-400' : signal.direction < 0 ? 'text-red-400' : 'text-slate-500';
  const arrow = signal.direction > 0 ? '+' : signal.direction < 0 ? '-' : '~';

  return (
    <div className="flex items-start gap-1.5 py-0.5">
      <span className={`text-[9px] font-mono ${dirColor} shrink-0 mt-0.5`}>{arrow}</span>
      <div className="min-w-0">
        <span className="text-[10px] text-slate-400">{signal.name}</span>
        <span className="text-[9px] text-slate-600 ml-1">{signal.detail}</span>
      </div>
      <span className={`text-[9px] num-display shrink-0 ${dirColor}`}>
        {signal.weight > 0 ? '+' : ''}{signal.weight?.toFixed(1)}
      </span>
    </div>
  );
}

export default function SignalPanel({ score, signals }: SignalPanelProps) {
  const [manipulation, setManipulation] = useState<any>(null);

  useEffect(() => {
    const today = new Date().toISOString().split('T')[0];
    fetch(`${API}/api/manipulation?date=${today}`)
      .then(r => r.json())
      .then(data => {
        if (!data.error) setManipulation(data);
      })
      .catch(console.error);
  }, []);

  if (!score) {
    return (
      <div className="p-3">
        <div className="text-xs text-slate-600 text-center py-4">Loading signals...</div>
      </div>
    );
  }

  const convergenceStrength = Math.abs(score.score);
  const convergenceLabel = convergenceStrength >= 7 ? 'STRONG' : convergenceStrength >= 4 ? 'MODERATE' : 'WEAK';

  // Group signals by direction
  const bearishSignals = (score.signals || []).filter((s: any) => s.direction < 0);
  const bullishSignals = (score.signals || []).filter((s: any) => s.direction > 0);
  const neutralSignals = (score.signals || []).filter((s: any) => s.direction === 0);

  return (
    <div className="p-3">
      {/* Section header */}
      <div className="flex items-center gap-2 mb-3">
        <svg className="w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
        </svg>
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Signal Convergence</h3>
      </div>

      {/* Convergence score */}
      <div className="glass-panel p-3 mb-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider">Convergence</span>
          <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
            convergenceLabel === 'STRONG' ? 'bg-bullish/10 text-bullish' :
            convergenceLabel === 'MODERATE' ? 'bg-warning/10 text-warning' :
            'bg-slate-700/50 text-slate-500'
          }`}>
            {convergenceLabel}
          </span>
        </div>

        {/* Score display */}
        <div className="flex items-baseline gap-1 mb-1">
          <span className={`text-2xl font-bold num-display ${
            score.score >= 0 ? 'text-green-400' : 'text-red-400'
          }`}>
            {score.score > 0 ? '+' : ''}{typeof score.score === 'number' ? score.score.toFixed(2) : score.score}
          </span>
          <span className="text-xs text-slate-600">/ 10</span>
        </div>

        {/* Confidence + Action */}
        <div className="flex items-center gap-3 mb-3">
          {score.confidence != null && (
            <span className="text-[10px] text-slate-500">
              Confidence: <span className="text-slate-300 num-display">{score.confidence}%</span>
            </span>
          )}
          {score.action && (
            <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
              score.action === 'STRONG_BUY' ? 'bg-green-500/15 text-green-400' :
              score.action === 'BUY' ? 'bg-green-500/10 text-green-500' :
              score.action === 'STRONG_SELL' ? 'bg-red-500/15 text-red-400' :
              score.action === 'SELL' ? 'bg-red-500/10 text-red-500' :
              'bg-slate-700/50 text-slate-400'
            }`}>
              {score.action}
            </span>
          )}
        </div>

        {/* Component bars */}
        {score.components && (
          <>
            <ComponentBar label="Numerology" value={score.components.numerology || 0} />
            <ComponentBar label="Technical" value={score.components.technical || 0} />
            <ComponentBar label="Tweets" value={score.components.tweets || 0} />
            <ComponentBar label="Manipulation" value={score.components.manipulation || 0} />
          </>
        )}
      </div>

      {/* Signal counts */}
      <div className="glass-panel p-3 mb-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider">Signals</span>
          <span className="text-[10px] text-slate-500 num-display">
            {score.signal_count || (bearishSignals.length + bullishSignals.length + neutralSignals.length)} total
          </span>
        </div>

        <div className="flex items-center gap-3 text-[10px] mb-2">
          <span className="text-green-500">{score.bullish_count || bullishSignals.length} bullish</span>
          <span className="text-red-500">{score.bearish_count || bearishSignals.length} bearish</span>
          <span className="text-slate-500">{score.neutral_count || neutralSignals.length} neutral</span>
        </div>

        {/* Top signals list (show top 5) */}
        {score.signals && score.signals.length > 0 && (
          <div className="max-h-[150px] overflow-y-auto">
            {[...score.signals]
              .sort((a: any, b: any) => Math.abs(b.weight) - Math.abs(a.weight))
              .slice(0, 8)
              .map((sig: any, i: number) => (
                <SignalItem key={i} signal={sig} />
              ))
            }
          </div>
        )}
      </div>

      {/* Threat level + Manipulation */}
      <div className="glass-panel p-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] text-slate-500 uppercase tracking-wider">Threat Level</span>
          <div className="flex items-center gap-1.5">
            <div className={`w-2 h-2 rounded-full ${
              score.threat_level === 'CRITICAL' ? 'bg-red-500 animate-pulse' :
              score.threat_level === 'HIGH' ? 'bg-orange-500' :
              score.threat_level === 'MODERATE' ? 'bg-yellow-500' :
              'bg-green-500'
            }`} />
            <span className={`text-xs font-bold ${
              score.threat_level === 'CRITICAL' ? 'text-red-500' :
              score.threat_level === 'HIGH' ? 'text-orange-500' :
              score.threat_level === 'MODERATE' ? 'text-yellow-500' :
              'text-green-500'
            }`}>
              {score.threat_level || 'LOW'}
            </span>
          </div>
        </div>

        {/* Manipulation data */}
        {manipulation && (
          <div className="mt-2 pt-2 border-t border-panel-border">
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] text-slate-600">Manipulation Scan</span>
              <span className={`text-[10px] font-bold ${
                manipulation.threat_level === 'CRITICAL' ? 'text-red-500' :
                manipulation.threat_level === 'HIGH' ? 'text-orange-500' :
                manipulation.threat_level === 'MODERATE' ? 'text-yellow-500' :
                'text-green-500'
              }`}>
                {manipulation.threat_level}
              </span>
            </div>
            {manipulation.layers && (
              <div className="text-[9px] text-slate-600">
                {Object.keys(manipulation.layers).map(layer => (
                  <div key={layer} className="flex items-center justify-between py-0.5">
                    <span className="text-slate-500">{layer.replace(/_/g, ' ')}</span>
                    <span className="text-slate-400">active</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* TA info */}
        {score.ta_bias && (
          <div className="mt-2 pt-2 border-t border-panel-border">
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] text-slate-600">TA Bias</span>
              <span className={`text-[10px] font-bold ${
                score.ta_bias === 'bullish' ? 'text-green-500' :
                score.ta_bias === 'bearish' ? 'text-red-500' :
                'text-slate-400'
              }`}>
                {score.ta_bias.toUpperCase()}
              </span>
            </div>
            {score.wyckoff_phase && (
              <div className="flex items-center justify-between py-0.5">
                <span className="text-[9px] text-slate-600">Wyckoff</span>
                <span className="text-[9px] text-slate-400">{score.wyckoff_phase}</span>
              </div>
            )}
            {score.elliott_wave && (
              <div className="flex items-center justify-between py-0.5">
                <span className="text-[9px] text-slate-600">Elliott</span>
                <span className="text-[9px] text-slate-400">{score.elliott_wave}</span>
              </div>
            )}
            {score.gann_direction && (
              <div className="flex items-center justify-between py-0.5">
                <span className="text-[9px] text-slate-600">Gann</span>
                <span className="text-[9px] text-slate-400">{score.gann_direction}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
