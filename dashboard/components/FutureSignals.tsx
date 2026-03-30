'use client';
import { useState, useEffect } from 'react';
import { API } from '@/lib/api';

interface SignalEvent {
  date: string;
  name: string;
  category: string;
  type: 'bullish' | 'bearish' | 'volatility';
  description: string;
  window: string;
}

interface FutureSignalsProps {
  onClose: () => void;
}

const CATEGORIES = [
  { key: 'all', label: 'All' },
  { key: 'economic', label: 'Economic' },
  { key: 'crypto', label: 'Crypto' },
  { key: 'astronomical', label: 'Astro' },
  { key: 'astrological', label: 'Planets' },
  { key: 'lunar', label: 'Lunar' },
  { key: 'hebrew', label: 'Hebrew' },
  { key: 'cultural', label: 'Cultural' },
  { key: 'seasonal', label: 'Seasonal' },
  { key: 'political', label: 'Political' },
  { key: 'tax', label: 'Tax' },
  { key: 'esoteric', label: 'Esoteric' },
];

const DAY_OPTIONS = [7, 14, 30, 60, 90];

const CATEGORY_COLORS: Record<string, string> = {
  economic: '#f59e0b',
  crypto: '#22c55e',
  astronomical: '#8b5cf6',
  astrological: '#8b5cf6',
  lunar: '#64748b',
  hebrew: '#6366f1',
  cultural: '#3b82f6',
  seasonal: '#22c55e',
  political: '#f59e0b',
  tax: '#ef4444',
  esoteric: '#8b5cf6',
};

const TYPE_COLORS: Record<string, string> = {
  bullish: '#22c55e',
  bearish: '#ef4444',
  volatility: '#f59e0b',
};

const CATEGORY_ICONS: Record<string, string> = {
  economic: '\u{1F4CA}',
  crypto: '\u{20BF}',
  astronomical: '\u{1F30C}',
  astrological: '\u{2649}',
  lunar: '\u{1F319}',
  hebrew: '\u{2721}',
  cultural: '\u{1F30F}',
  seasonal: '\u{1F343}',
  political: '\u{1F3DB}',
  tax: '\u{1F4B8}',
  esoteric: '\u{1F52E}',
};

function formatDate(dateStr: string): string {
  const d = new Date(dateStr + 'T00:00:00');
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  const diff = Math.round((d.getTime() - today.getTime()) / 86400000);

  const weekday = d.toLocaleDateString('en-US', { weekday: 'short' });
  const month = d.toLocaleDateString('en-US', { month: 'short' });
  const day = d.getDate();

  if (diff === 0) return `Today — ${weekday} ${month} ${day}`;
  if (diff === 1) return `Tomorrow — ${weekday} ${month} ${day}`;
  return `${weekday} ${month} ${day} (${diff}d)`;
}

export default function FutureSignals({ onClose }: FutureSignalsProps) {
  const [events, setEvents] = useState<SignalEvent[]>([]);
  const [days, setDays] = useState(30);
  const [category, setCategory] = useState('all');
  const [expandedEvent, setExpandedEvent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    const url = `${API}/api/future-signals?days=${days}`;
    fetch(url)
      .then(r => r.json())
      .then(data => {
        setEvents(data.events || []);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch future signals:', err);
        setLoading(false);
      });
  }, [days]);

  // Filter by category
  const filtered = category === 'all'
    ? events
    : events.filter(e => e.category === category);

  // Group by date
  const grouped: Record<string, SignalEvent[]> = {};
  for (const e of filtered) {
    if (!grouped[e.date]) grouped[e.date] = [];
    grouped[e.date].push(e);
  }
  const sortedDates = Object.keys(grouped).sort();

  // Count by category for badges
  const categoryCounts: Record<string, number> = {};
  for (const e of events) {
    categoryCounts[e.category] = (categoryCounts[e.category] || 0) + 1;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" onClick={onClose}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <div
        className="relative bg-[#12121f] border border-slate-700/50 rounded-xl shadow-2xl w-[700px] max-h-[80vh] overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-slate-700/50">
          <div className="flex items-center gap-3">
            <span className="text-lg">&#x1F52E;</span>
            <h2 className="text-sm font-semibold text-slate-200 tracking-wide uppercase">
              Upcoming Signals
            </h2>
            <span className="text-xs text-slate-500">
              {filtered.length} events in {sortedDates.length} days
            </span>
          </div>
          <div className="flex items-center gap-3">
            {/* Days selector */}
            <select
              value={days}
              onChange={e => setDays(Number(e.target.value))}
              className="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-xs text-slate-300 cursor-pointer focus:outline-none focus:border-slate-500"
            >
              {DAY_OPTIONS.map(d => (
                <option key={d} value={d}>{d} days</option>
              ))}
            </select>
            {/* Close button */}
            <button
              onClick={onClose}
              className="text-slate-500 hover:text-slate-300 text-xl font-bold w-8 h-8 flex items-center justify-center rounded hover:bg-slate-700/50"
            >
              &#x2715;
            </button>
          </div>
        </div>

        {/* Category filter tabs */}
        <div className="flex items-center gap-1 px-5 py-2 border-b border-slate-700/30 overflow-x-auto">
          {CATEGORIES.map(cat => {
            const count = cat.key === 'all' ? events.length : (categoryCounts[cat.key] || 0);
            const isActive = category === cat.key;
            if (cat.key !== 'all' && count === 0) return null;
            return (
              <button
                key={cat.key}
                onClick={() => setCategory(cat.key)}
                className={`flex items-center gap-1 px-2 py-1 text-xs rounded-md font-medium transition-all whitespace-nowrap ${
                  isActive
                    ? 'bg-slate-600/30 text-slate-200 border border-slate-500/40'
                    : 'text-slate-500 hover:text-slate-300 hover:bg-white/5 border border-transparent'
                }`}
              >
                <span>{cat.label}</span>
                {count > 0 && (
                  <span className={`px-1 py-0 text-[9px] font-bold rounded-full ${
                    isActive ? 'bg-slate-500/30 text-slate-300' : 'bg-slate-700/50 text-slate-500'
                  }`}>
                    {count}
                  </span>
                )}
              </button>
            );
          })}
        </div>

        {/* Events timeline */}
        <div className="overflow-y-auto max-h-[60vh] px-5 py-3">
          {loading ? (
            <div className="flex items-center justify-center py-12 text-slate-500 text-sm">
              Loading signals...
            </div>
          ) : sortedDates.length === 0 ? (
            <div className="flex items-center justify-center py-12 text-slate-500 text-sm">
              No signals in this period
            </div>
          ) : (
            sortedDates.map(date => (
              <div key={date} className="mb-4">
                {/* Date header */}
                <div className="flex items-center gap-2 mb-2">
                  <div className="text-xs font-semibold text-slate-400 tracking-wide">
                    {formatDate(date)}
                  </div>
                  <div className="flex-1 h-px bg-slate-700/50" />
                  <span className="text-[10px] text-slate-600">
                    {grouped[date].length} signal{grouped[date].length > 1 ? 's' : ''}
                  </span>
                </div>

                {/* Events for this date */}
                <div className="space-y-1 ml-2">
                  {grouped[date].map((evt, i) => {
                    const eventKey = `${date}-${evt.name}-${i}`;
                    const isExpanded = expandedEvent === eventKey;
                    const catColor = CATEGORY_COLORS[evt.category] || '#64748b';
                    const typeColor = TYPE_COLORS[evt.type] || '#f59e0b';
                    const icon = CATEGORY_ICONS[evt.category] || '\u{2728}';

                    return (
                      <div key={eventKey}>
                        <button
                          onClick={() => setExpandedEvent(isExpanded ? null : eventKey)}
                          className="w-full flex items-center gap-2 px-3 py-1.5 rounded-lg hover:bg-white/[0.03] transition-all text-left group"
                        >
                          <span className="text-sm w-5 text-center shrink-0">{icon}</span>
                          <span className="text-xs font-medium text-slate-300 flex-1">
                            {evt.name}
                          </span>
                          <span
                            className="text-[9px] px-1.5 py-0.5 rounded font-medium uppercase tracking-wider"
                            style={{ color: catColor, backgroundColor: `${catColor}15`, border: `1px solid ${catColor}30` }}
                          >
                            {evt.category}
                          </span>
                          <span
                            className="text-[9px] px-1.5 py-0.5 rounded font-medium"
                            style={{ color: typeColor }}
                          >
                            {evt.type === 'bullish' ? '\u25B2' : evt.type === 'bearish' ? '\u25BC' : '\u25C6'}{' '}
                            {evt.type}
                          </span>
                          <span className="text-[10px] text-slate-600 opacity-0 group-hover:opacity-100 transition-opacity">
                            {isExpanded ? '\u25B2' : '\u25BC'}
                          </span>
                        </button>

                        {isExpanded && (
                          <div className="ml-10 mr-3 mb-2 px-3 py-2 rounded bg-slate-800/50 border border-slate-700/30">
                            <p className="text-xs text-slate-400 leading-relaxed">
                              {evt.description}
                            </p>
                            <p className="text-[10px] text-slate-600 mt-1">
                              Window: {evt.window}
                            </p>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
