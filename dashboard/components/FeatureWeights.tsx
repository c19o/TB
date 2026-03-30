'use client';

import { API } from '@/lib/api';
import { useState, useEffect, useMemo } from 'react';
import { X, Search, ChevronDown, ChevronRight, Layers } from 'lucide-react';

interface FeaturesResponse {
  tf: string;
  total: number;
  categories: Record<string, string[]>;
}

const CATEGORY_COLORS: Record<string, { bg: string; text: string; border: string; dot: string }> = {
  Technical:   { bg: 'bg-blue-500/10',    text: 'text-blue-400',    border: 'border-blue-500/30',    dot: 'bg-blue-400' },
  Astrology:   { bg: 'bg-purple-500/10',  text: 'text-purple-400',  border: 'border-purple-500/30',  dot: 'bg-purple-400' },
  Numerology:  { bg: 'bg-amber-500/10',   text: 'text-amber-400',   border: 'border-amber-500/30',   dot: 'bg-amber-400' },
  Gematria:    { bg: 'bg-pink-500/10',    text: 'text-pink-400',    border: 'border-pink-500/30',    dot: 'bg-pink-400' },
  Tweets:      { bg: 'bg-cyan-500/10',    text: 'text-cyan-400',    border: 'border-cyan-500/30',    dot: 'bg-cyan-400' },
  News:        { bg: 'bg-emerald-500/10', text: 'text-emerald-400', border: 'border-emerald-500/30', dot: 'bg-emerald-400' },
  Sports:      { bg: 'bg-orange-500/10',  text: 'text-orange-400',  border: 'border-orange-500/30',  dot: 'bg-orange-400' },
  'On-chain':  { bg: 'bg-teal-500/10',    text: 'text-teal-400',    border: 'border-teal-500/30',    dot: 'bg-teal-400' },
  Macro:       { bg: 'bg-indigo-500/10',  text: 'text-indigo-400',  border: 'border-indigo-500/30',  dot: 'bg-indigo-400' },
  Regime:      { bg: 'bg-red-500/10',     text: 'text-red-400',     border: 'border-red-500/30',     dot: 'bg-red-400' },
  KNN:         { bg: 'bg-violet-500/10',  text: 'text-violet-400',  border: 'border-violet-500/30',  dot: 'bg-violet-400' },
  Cross:       { bg: 'bg-fuchsia-500/10', text: 'text-fuchsia-400', border: 'border-fuchsia-500/30', dot: 'bg-fuchsia-400' },
  Other:       { bg: 'bg-slate-500/10',   text: 'text-slate-400',   border: 'border-slate-500/30',   dot: 'bg-slate-400' },
};

const DEFAULT_COLORS = { bg: 'bg-slate-500/10', text: 'text-slate-400', border: 'border-slate-500/30', dot: 'bg-slate-400' };

function getCategoryColors(name: string) {
  return CATEGORY_COLORS[name] || DEFAULT_COLORS;
}

interface FeatureWeightsProps {
  isOpen: boolean;
  onClose: () => void;
  timeframe: string;
}

export default function FeatureWeights({ isOpen, onClose, timeframe }: FeatureWeightsProps) {
  const [data, setData] = useState<FeaturesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [search, setSearch] = useState('');

  useEffect(() => {
    if (!isOpen) return;
    setLoading(true);
    setError(null);
    fetch(`${API}/api/features?tf=${timeframe}`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((d: FeaturesResponse) => {
        setData(d);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, [isOpen, timeframe]);

  const filtered = useMemo(() => {
    if (!data?.categories) return {};
    if (!search) return data.categories;
    const q = search.toLowerCase();
    const result: Record<string, string[]> = {};
    for (const [cat, features] of Object.entries(data.categories)) {
      const matches = features.filter(f => f.toLowerCase().includes(q));
      if (matches.length > 0 || cat.toLowerCase().includes(q)) {
        result[cat] = matches.length > 0 ? matches : features;
      }
    }
    return result;
  }, [data, search]);

  const filteredTotal = useMemo(() => {
    return Object.values(filtered).reduce((sum, arr) => sum + arr.length, 0);
  }, [filtered]);

  const toggleCategory = (cat: string) => {
    setExpanded(prev => ({ ...prev, [cat]: !prev[cat] }));
  };

  const expandAll = () => {
    if (!data?.categories) return;
    const all: Record<string, boolean> = {};
    Object.keys(data.categories).forEach(k => { all[k] = true; });
    setExpanded(all);
  };

  const collapseAll = () => setExpanded({});

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      {/* Modal */}
      <div
        className="relative w-[560px] max-w-[90vw] max-h-[85vh] flex flex-col rounded-xl border shadow-2xl overflow-hidden"
        style={{
          background: 'rgba(15, 15, 25, 0.95)',
          borderColor: 'rgba(50, 50, 80, 0.4)',
          backdropFilter: 'blur(20px)',
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b" style={{ borderColor: 'rgba(50, 50, 80, 0.3)' }}>
          <div className="flex items-center gap-3">
            <Layers size={20} className="text-purple-400" />
            <h2 className="text-lg font-semibold text-slate-200">ML Feature Weights</h2>
            <span className="text-xs text-slate-500 font-mono px-2 py-0.5 rounded" style={{ background: 'rgba(50, 50, 80, 0.4)' }}>
              {timeframe}
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-white/5 text-slate-400 hover:text-slate-200 transition-colors"
          >
            <X size={18} />
          </button>
        </div>

        {/* Total count + search */}
        <div className="px-5 py-3 border-b" style={{ borderColor: 'rgba(50, 50, 80, 0.3)' }}>
          {data && (
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                <span className="text-sm font-medium text-slate-200 num-display">
                  {data.total} features active
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={expandAll}
                  className="text-[10px] text-slate-500 hover:text-slate-300 px-2 py-0.5 rounded hover:bg-white/5 transition-colors"
                >
                  Expand all
                </button>
                <button
                  onClick={collapseAll}
                  className="text-[10px] text-slate-500 hover:text-slate-300 px-2 py-0.5 rounded hover:bg-white/5 transition-colors"
                >
                  Collapse all
                </button>
              </div>
            </div>
          )}
          <div className="relative">
            <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              value={search}
              onChange={e => setSearch(e.target.value)}
              placeholder="Search features..."
              className="w-full pl-10 pr-4 py-2.5 rounded-lg text-sm text-slate-200 placeholder-slate-600 font-mono outline-none transition-colors"
              style={{
                background: 'rgba(10, 10, 20, 0.6)',
                border: '1px solid rgba(50, 50, 80, 0.3)',
              }}
            />
          </div>
          {search && (
            <div className="mt-2 text-[11px] text-slate-500">
              {filteredTotal} feature{filteredTotal !== 1 ? 's' : ''} matching &quot;{search}&quot;
            </div>
          )}
        </div>

        {/* Categories list */}
        <div className="flex-1 overflow-y-auto px-5 py-3">
          {loading && (
            <div className="flex items-center justify-center py-12">
              <div className="w-5 h-5 border-2 border-purple-400/30 border-t-purple-400 rounded-full animate-spin" />
              <span className="ml-3 text-sm text-slate-500">Loading features...</span>
            </div>
          )}

          {error && (
            <div className="text-center py-12">
              <span className="text-sm text-red-400">Failed to load features: {error}</span>
            </div>
          )}

          {!loading && !error && data && (
            <div className="space-y-1.5">
              {Object.entries(filtered)
                .sort(([, a], [, b]) => b.length - a.length)
                .map(([category, features]) => {
                  const colors = getCategoryColors(category);
                  const isExpanded = expanded[category] || false;

                  return (
                    <div key={category} className="rounded-lg overflow-hidden" style={{ border: '1px solid rgba(50, 50, 80, 0.2)' }}>
                      {/* Category header */}
                      <button
                        onClick={() => toggleCategory(category)}
                        className="w-full flex items-center gap-3 px-3 py-2.5 text-left hover:bg-white/[0.02] transition-colors"
                      >
                        {isExpanded
                          ? <ChevronDown size={14} className="text-slate-500 shrink-0" />
                          : <ChevronRight size={14} className="text-slate-500 shrink-0" />
                        }
                        <div className={`w-2.5 h-2.5 rounded-full shrink-0 ${colors.dot}`} />
                        <span className={`text-sm font-medium ${colors.text}`}>
                          {category}
                        </span>
                        <span
                          className={`ml-auto text-[11px] font-mono font-bold px-2 py-0.5 rounded-full ${colors.bg} ${colors.text} ${colors.border} border`}
                        >
                          {features.length}
                        </span>
                      </button>

                      {/* Expanded feature list */}
                      {isExpanded && (
                        <div className="px-3 pb-3 pt-1" style={{ background: 'rgba(0, 0, 0, 0.15)' }}>
                          <div className="grid grid-cols-2 gap-x-4 gap-y-0.5">
                            {features.map(feature => (
                              <div
                                key={feature}
                                className="flex items-center gap-2 py-1 group"
                              >
                                <div className={`w-1 h-1 rounded-full shrink-0 ${colors.dot} opacity-40`} />
                                <span className="text-[11px] text-slate-400 font-mono truncate group-hover:text-slate-200 transition-colors">
                                  {feature}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
            </div>
          )}

          {!loading && !error && data && Object.keys(filtered).length === 0 && (
            <div className="text-center py-12 text-slate-600 text-sm">
              No features found matching &quot;{search}&quot;
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
