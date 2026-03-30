'use client';

import { useState, useEffect } from 'react';
import { X, RotateCcw, Trash2, Check } from 'lucide-react';
import { INDICATOR_CATALOG, type IndicatorDef } from './IndicatorSelector';

export interface IndicatorConfig {
  id: string;
  params: Record<string, number | string>;
  color: string;
  lineWidth: number;
  opacity: number;
  source: string;
}

interface IndicatorSettingsProps {
  indicatorId: string | null;
  config: IndicatorConfig | null;
  onApply: (config: IndicatorConfig) => void;
  onRemove: (id: string) => void;
  onClose: () => void;
}

const SOURCE_OPTIONS = [
  { value: 'close', label: 'Close' },
  { value: 'open', label: 'Open' },
  { value: 'high', label: 'High' },
  { value: 'low', label: 'Low' },
  { value: 'hl2', label: 'HL/2' },
  { value: 'hlc3', label: 'HLC/3' },
];

export default function IndicatorSettings({
  indicatorId,
  config,
  onApply,
  onRemove,
  onClose,
}: IndicatorSettingsProps) {
  const [localConfig, setLocalConfig] = useState<IndicatorConfig | null>(null);

  const def = indicatorId
    ? INDICATOR_CATALOG.find(i => i.id === indicatorId) || null
    : null;

  useEffect(() => {
    if (config) {
      setLocalConfig({ ...config });
    } else if (def) {
      setLocalConfig({
        id: def.id,
        params: { ...def.defaultParams },
        color: def.color,
        lineWidth: 2,
        opacity: 100,
        source: 'close',
      });
    }
  }, [config, def]);

  if (!indicatorId || !localConfig || !def) return null;

  const handleParamChange = (key: string, value: number) => {
    setLocalConfig(prev => prev ? {
      ...prev,
      params: { ...prev.params, [key]: value },
    } : null);
  };

  const handleReset = () => {
    setLocalConfig({
      id: def.id,
      params: { ...def.defaultParams },
      color: def.color,
      lineWidth: 2,
      opacity: 100,
      source: 'close',
    });
  };

  const paramEntries = Object.entries(localConfig.params).filter(
    ([_, v]) => typeof v === 'number'
  );

  return (
    <div
      className="absolute z-[90] top-2 right-2 w-[300px] rounded-xl border shadow-2xl overflow-hidden"
      style={{
        background: 'rgba(15, 15, 25, 0.95)',
        borderColor: 'rgba(50, 50, 80, 0.4)',
        backdropFilter: 'blur(20px)',
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b" style={{ borderColor: 'rgba(50, 50, 80, 0.3)' }}>
        <div className="flex items-center gap-2">
          <div
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: localConfig.color }}
          />
          <span className="text-sm font-semibold text-slate-200">{def.name}</span>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded hover:bg-white/5 text-slate-500 hover:text-slate-300 transition-colors"
        >
          <X size={16} />
        </button>
      </div>

      {/* Body */}
      <div className="px-4 py-3 space-y-4">
        {/* Period / numeric params */}
        {paramEntries.map(([key, value]) => (
          <div key={key}>
            <label className="flex items-center justify-between mb-1.5">
              <span className="text-xs text-slate-400 capitalize">{key}</span>
              <span className="text-xs text-slate-300 font-mono">{value as number}</span>
            </label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min={1}
                max={key === 'stddev' || key === 'multiplier' ? 10 : 500}
                step={key === 'stddev' || key === 'multiplier' ? 0.5 : 1}
                value={value as number}
                onChange={e => handleParamChange(key, parseFloat(e.target.value))}
                className="flex-1 h-1.5 rounded-full appearance-none cursor-pointer"
                style={{
                  background: `linear-gradient(to right, ${localConfig.color} 0%, ${localConfig.color} ${
                    ((value as number) / (key === 'stddev' || key === 'multiplier' ? 10 : 500)) * 100
                  }%, rgba(50,50,80,0.4) ${
                    ((value as number) / (key === 'stddev' || key === 'multiplier' ? 10 : 500)) * 100
                  }%, rgba(50,50,80,0.4) 100%)`,
                }}
              />
              <input
                type="number"
                value={value as number}
                onChange={e => handleParamChange(key, parseFloat(e.target.value) || 1)}
                className="w-16 px-2 py-1 text-xs font-mono text-slate-200 rounded border outline-none"
                style={{
                  background: 'rgba(10, 10, 20, 0.6)',
                  borderColor: 'rgba(50, 50, 80, 0.3)',
                }}
              />
            </div>
          </div>
        ))}

        {/* Color */}
        <div>
          <label className="text-xs text-slate-400 mb-1.5 block">Color</label>
          <div className="flex items-center gap-2">
            <input
              type="color"
              value={localConfig.color}
              onChange={e => setLocalConfig(prev => prev ? { ...prev, color: e.target.value } : null)}
              className="w-8 h-8 rounded cursor-pointer border-0 bg-transparent"
            />
            <input
              type="text"
              value={localConfig.color}
              onChange={e => setLocalConfig(prev => prev ? { ...prev, color: e.target.value } : null)}
              className="flex-1 px-2 py-1.5 text-xs font-mono text-slate-200 rounded border outline-none"
              style={{
                background: 'rgba(10, 10, 20, 0.6)',
                borderColor: 'rgba(50, 50, 80, 0.3)',
              }}
            />
          </div>
        </div>

        {/* Line width */}
        <div>
          <label className="flex items-center justify-between mb-1.5">
            <span className="text-xs text-slate-400">Line Width</span>
            <span className="text-xs text-slate-300 font-mono">{localConfig.lineWidth}px</span>
          </label>
          <div className="flex gap-1.5">
            {[1, 2, 3, 4, 5].map(w => (
              <button
                key={w}
                onClick={() => setLocalConfig(prev => prev ? { ...prev, lineWidth: w } : null)}
                className={`flex-1 py-1.5 rounded text-xs font-mono transition-all ${
                  localConfig.lineWidth === w
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/40'
                    : 'text-slate-500 hover:text-slate-300 border border-transparent hover:bg-white/5'
                }`}
                style={{ borderColor: localConfig.lineWidth === w ? 'rgba(59,130,246,0.4)' : 'rgba(50,50,80,0.2)' }}
              >
                {w}
              </button>
            ))}
          </div>
        </div>

        {/* Opacity */}
        <div>
          <label className="flex items-center justify-between mb-1.5">
            <span className="text-xs text-slate-400">Opacity</span>
            <span className="text-xs text-slate-300 font-mono">{localConfig.opacity}%</span>
          </label>
          <input
            type="range"
            min={10}
            max={100}
            value={localConfig.opacity}
            onChange={e => setLocalConfig(prev => prev ? { ...prev, opacity: parseInt(e.target.value) } : null)}
            className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, ${localConfig.color} 0%, ${localConfig.color} ${localConfig.opacity}%, rgba(50,50,80,0.4) ${localConfig.opacity}%, rgba(50,50,80,0.4) 100%)`,
            }}
          />
        </div>

        {/* Source */}
        <div>
          <label className="text-xs text-slate-400 mb-1.5 block">Source</label>
          <div className="grid grid-cols-3 gap-1.5">
            {SOURCE_OPTIONS.map(opt => (
              <button
                key={opt.value}
                onClick={() => setLocalConfig(prev => prev ? { ...prev, source: opt.value } : null)}
                className={`py-1.5 rounded text-xs font-mono transition-all ${
                  localConfig.source === opt.value
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/40'
                    : 'text-slate-500 hover:text-slate-300 border hover:bg-white/5'
                }`}
                style={{ borderColor: localConfig.source === opt.value ? 'rgba(59,130,246,0.4)' : 'rgba(50,50,80,0.2)' }}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="flex items-center gap-2 px-4 py-3 border-t" style={{ borderColor: 'rgba(50, 50, 80, 0.3)' }}>
        <button
          onClick={() => { onRemove(indicatorId); onClose(); }}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-red-400 hover:bg-red-500/10 transition-colors"
        >
          <Trash2 size={13} />
          Remove
        </button>
        <div className="flex-1" />
        <button
          onClick={handleReset}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:bg-white/5 transition-colors"
        >
          <RotateCcw size={13} />
          Reset
        </button>
        <button
          onClick={() => { onApply(localConfig); onClose(); }}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 transition-colors"
        >
          <Check size={13} />
          Apply
        </button>
      </div>
    </div>
  );
}
