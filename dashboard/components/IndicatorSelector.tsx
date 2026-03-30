'use client';

import { useState, useMemo } from 'react';
import {
  X, Search, TrendingUp, Activity, BarChart3, Waves,
  ArrowUpDown, Gauge, Volume2, Target, Zap, Layers,
  LineChart, ArrowUp,
} from 'lucide-react';

export interface IndicatorDef {
  id: string;
  name: string;
  shortName: string;
  description: string;
  category: 'trend' | 'oscillator' | 'volume' | 'volatility' | 'esoteric';
  type: 'overlay' | 'oscillator' | 'volume';
  icon: React.ReactNode;
  defaultParams: Record<string, number | string>;
  color: string;
  /** KLineChart built-in indicator name */
  klineName?: string;
}

export const INDICATOR_CATALOG: IndicatorDef[] = [
  {
    id: 'ema_9', name: 'EMA 9', shortName: 'EMA', description: 'Exponential Moving Average (fast)',
    category: 'trend', type: 'overlay', icon: <TrendingUp size={18} />,
    defaultParams: { period: 9 }, color: '#f59e0b', klineName: 'EMA',
  },
  {
    id: 'ema_21', name: 'EMA 21', shortName: 'EMA', description: 'Exponential Moving Average (medium)',
    category: 'trend', type: 'overlay', icon: <TrendingUp size={18} />,
    defaultParams: { period: 21 }, color: '#3b82f6', klineName: 'EMA',
  },
  {
    id: 'ema_50', name: 'EMA 50', shortName: 'EMA', description: 'Exponential Moving Average (50)',
    category: 'trend', type: 'overlay', icon: <TrendingUp size={18} />,
    defaultParams: { period: 50 }, color: '#22c55e', klineName: 'EMA',
  },
  {
    id: 'ema_200', name: 'EMA 200', shortName: 'EMA', description: 'Exponential Moving Average (200)',
    category: 'trend', type: 'overlay', icon: <TrendingUp size={18} />,
    defaultParams: { period: 200 }, color: '#ef4444', klineName: 'EMA',
  },
  {
    id: 'sma_50', name: 'SMA 50', shortName: 'SMA', description: 'Simple Moving Average (50)',
    category: 'trend', type: 'overlay', icon: <LineChart size={18} />,
    defaultParams: { period: 50 }, color: '#06b6d4', klineName: 'SMA',
  },
  {
    id: 'sma_200', name: 'SMA 200', shortName: 'SMA', description: 'Simple Moving Average (200)',
    category: 'trend', type: 'overlay', icon: <LineChart size={18} />,
    defaultParams: { period: 200 }, color: '#ec4899', klineName: 'SMA',
  },
  {
    id: 'rsi', name: 'RSI (14)', shortName: 'RSI', description: 'Relative Strength Index — overbought/oversold',
    category: 'oscillator', type: 'oscillator', icon: <Activity size={18} />,
    defaultParams: { period: 14 }, color: '#a78bfa', klineName: 'RSI',
  },
  {
    id: 'macd', name: 'MACD', shortName: 'MACD', description: 'Moving Average Convergence Divergence',
    category: 'oscillator', type: 'oscillator', icon: <BarChart3 size={18} />,
    defaultParams: { fast: 12, slow: 26, signal: 9 }, color: '#3b82f6', klineName: 'MACD',
  },
  {
    id: 'bollinger', name: 'Bollinger Bands', shortName: 'BB', description: 'Volatility bands around SMA',
    category: 'volatility', type: 'overlay', icon: <Waves size={18} />,
    defaultParams: { period: 20, stddev: 2 }, color: '#8b5cf6', klineName: 'BOLL',
  },
  {
    id: 'stochastic', name: 'Stochastic', shortName: 'STOCH', description: 'Stochastic Oscillator %K/%D',
    category: 'oscillator', type: 'oscillator', icon: <ArrowUpDown size={18} />,
    defaultParams: { k: 14, d: 3, smooth: 3 }, color: '#f59e0b', klineName: 'KDJ',
  },
  {
    id: 'atr', name: 'ATR (14)', shortName: 'ATR', description: 'Average True Range — volatility measure',
    category: 'volatility', type: 'oscillator', icon: <Gauge size={18} />,
    defaultParams: { period: 14 }, color: '#14b8a6', klineName: 'ATR',
  },
  {
    id: 'obv', name: 'OBV', shortName: 'OBV', description: 'On-Balance Volume — cumulative volume flow',
    category: 'volume', type: 'volume', icon: <Volume2 size={18} />,
    defaultParams: {}, color: '#06b6d4', klineName: 'OBV',
  },
  {
    id: 'vwap', name: 'VWAP', shortName: 'VWAP', description: 'Volume Weighted Average Price',
    category: 'volume', type: 'overlay', icon: <Target size={18} />,
    defaultParams: {}, color: '#f97316', klineName: 'AVP',
  },
  {
    id: 'cci', name: 'CCI (20)', shortName: 'CCI', description: 'Commodity Channel Index',
    category: 'oscillator', type: 'oscillator', icon: <Zap size={18} />,
    defaultParams: { period: 20 }, color: '#ec4899', klineName: 'CCI',
  },
  {
    id: 'supertrend', name: 'Supertrend', shortName: 'ST', description: 'Trend-following overlay with ATR bands',
    category: 'trend', type: 'overlay', icon: <ArrowUp size={18} />,
    defaultParams: { period: 10, multiplier: 3 }, color: '#22c55e', klineName: 'SAR',
  },
  {
    id: 'ichimoku', name: 'Ichimoku Cloud', shortName: 'ICHI', description: 'Full Ichimoku Kinko Hyo system',
    category: 'esoteric', type: 'overlay', icon: <Layers size={18} />,
    defaultParams: { tenkan: 9, kijun: 26, senkou: 52 }, color: '#8b5cf6', klineName: 'MA',
  },
];

const CATEGORIES = [
  { key: 'all', label: 'All' },
  { key: 'trend', label: 'Trend' },
  { key: 'oscillator', label: 'Oscillators' },
  { key: 'volume', label: 'Volume' },
  { key: 'volatility', label: 'Volatility' },
  { key: 'esoteric', label: 'Esoteric' },
];

interface IndicatorSelectorProps {
  isOpen: boolean;
  onClose: () => void;
  activeIndicators: string[];
  onToggleIndicator: (id: string) => void;
}

export default function IndicatorSelector({
  isOpen,
  onClose,
  activeIndicators,
  onToggleIndicator,
}: IndicatorSelectorProps) {
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState('all');

  const filtered = useMemo(() => {
    return INDICATOR_CATALOG.filter(ind => {
      const matchSearch = search === '' ||
        ind.name.toLowerCase().includes(search.toLowerCase()) ||
        ind.shortName.toLowerCase().includes(search.toLowerCase()) ||
        ind.description.toLowerCase().includes(search.toLowerCase());
      const matchCat = category === 'all' || ind.category === category;
      return matchSearch && matchCat;
    });
  }, [search, category]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div
        className="relative w-[700px] max-w-[90vw] max-h-[80vh] flex flex-col rounded-xl border shadow-2xl overflow-hidden"
        style={{
          background: 'rgba(15, 15, 25, 0.95)',
          borderColor: 'rgba(50, 50, 80, 0.4)',
          backdropFilter: 'blur(20px)',
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b" style={{ borderColor: 'rgba(50, 50, 80, 0.3)' }}>
          <div className="flex items-center gap-3">
            <BarChart3 size={20} className="text-blue-400" />
            <h2 className="text-lg font-semibold text-slate-200">Indicators</h2>
            <span className="text-xs text-slate-500 font-mono">{activeIndicators.length} active</span>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-white/5 text-slate-400 hover:text-slate-200 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Search */}
        <div className="px-5 py-3 border-b" style={{ borderColor: 'rgba(50, 50, 80, 0.3)' }}>
          <div className="relative">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search indicators..."
              className="w-full pl-10 pr-4 py-2.5 rounded-lg text-sm text-slate-200 placeholder-slate-600 font-mono outline-none transition-colors"
              style={{
                background: 'rgba(10, 10, 20, 0.6)',
                border: '1px solid rgba(50, 50, 80, 0.3)',
              }}
              autoFocus
            />
          </div>
        </div>

        {/* Category tabs */}
        <div className="flex gap-1 px-5 py-2 border-b" style={{ borderColor: 'rgba(50, 50, 80, 0.3)' }}>
          {CATEGORIES.map(cat => (
            <button
              key={cat.key}
              onClick={() => setCategory(cat.key)}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-all ${
                category === cat.key
                  ? 'bg-blue-500/20 text-blue-400 border border-blue-500/40'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-white/5 border border-transparent'
              }`}
            >
              {cat.label}
            </button>
          ))}
        </div>

        {/* Grid */}
        <div className="flex-1 overflow-y-auto p-5">
          <div className="grid grid-cols-2 gap-3">
            {filtered.map(ind => {
              const isActive = activeIndicators.includes(ind.id);
              return (
                <button
                  key={ind.id}
                  onClick={() => onToggleIndicator(ind.id)}
                  className={`flex items-start gap-3 p-3.5 rounded-lg border text-left transition-all group ${
                    isActive
                      ? 'border-blue-500/50 bg-blue-500/10'
                      : 'border-transparent hover:border-slate-700/50 hover:bg-white/[0.03]'
                  }`}
                  style={{
                    background: isActive ? 'rgba(59, 130, 246, 0.08)' : undefined,
                    borderColor: isActive ? 'rgba(59, 130, 246, 0.4)' : 'rgba(50, 50, 80, 0.2)',
                  }}
                >
                  <div
                    className={`flex items-center justify-center w-9 h-9 rounded-lg shrink-0 transition-colors ${
                      isActive ? 'bg-blue-500/20' : 'bg-white/[0.04] group-hover:bg-white/[0.06]'
                    }`}
                    style={{ color: isActive ? ind.color : '#64748b' }}
                  >
                    {ind.icon}
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <span className={`text-sm font-medium ${isActive ? 'text-slate-100' : 'text-slate-300'}`}>
                        {ind.name}
                      </span>
                      {isActive && (
                        <span className="flex w-2 h-2 rounded-full bg-blue-400" />
                      )}
                    </div>
                    <p className="text-[11px] text-slate-500 mt-0.5 leading-tight">
                      {ind.description}
                    </p>
                    <span
                      className="inline-block mt-1 text-[10px] font-mono px-1.5 py-0.5 rounded"
                      style={{
                        background: 'rgba(50, 50, 80, 0.3)',
                        color: ind.type === 'overlay' ? '#22c55e' : ind.type === 'oscillator' ? '#a78bfa' : '#06b6d4',
                      }}
                    >
                      {ind.type}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>

          {filtered.length === 0 && (
            <div className="text-center py-12 text-slate-600 text-sm">
              No indicators found matching &quot;{search}&quot;
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
