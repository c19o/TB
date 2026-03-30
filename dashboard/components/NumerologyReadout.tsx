'use client';
import { API } from '@/lib/api';

import { useEffect, useState } from 'react';

interface NumerologyData {
  day_of_year: number;
  date_reduction: number;
  days_remaining: number;
  moon_phase: string;
  zodiac_sign: string;
  planetary_hour: string;
  is_caution_number: boolean;
  is_pump_number: boolean;
  master_number: boolean;
}

interface EngineSignal {
  name: string;
  weight: number;
  direction: number;
  detail: string;
}

const MOON_ICONS: Record<string, string> = {
  'New Moon': '\u{1F311}',
  'Waxing Crescent': '\u{1F312}',
  'First Quarter': '\u{1F313}',
  'Waxing Gibbous': '\u{1F314}',
  'Full Moon': '\u{1F315}',
  'Waning Gibbous': '\u{1F316}',
  'Last Quarter': '\u{1F317}',
  'Waning Crescent': '\u{1F318}',
};

const ZODIAC_ICONS: Record<string, string> = {
  'Aries': '\u2648',
  'Taurus': '\u2649',
  'Gemini': '\u264A',
  'Cancer': '\u264B',
  'Leo': '\u264C',
  'Virgo': '\u264D',
  'Libra': '\u264E',
  'Scorpio': '\u264F',
  'Sagittarius': '\u2650',
  'Capricorn': '\u2651',
  'Aquarius': '\u2652',
  'Pisces': '\u2653',
};

const PLANET_ICONS: Record<string, string> = {
  'Sun': '\u2609',
  'Moon': '\u263D',
  'Mercury': '\u263F',
  'Venus': '\u2640',
  'Mars': '\u2642',
  'Jupiter': '\u2643',
  'Saturn': '\u2644',
};

export default function NumerologyReadout() {
  const [data, setData] = useState<NumerologyData | null>(null);
  const [engineSignals, setEngineSignals] = useState<EngineSignal[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const today = new Date().toISOString().split('T')[0];

    // Fetch from signals endpoint for basic numerology data
    const signalsPromise = fetch(`${API}/api/signals?date=${today}`)
      .then(r => r.json())
      .then(result => {
        if (result.numerology) {
          setData(result.numerology);
        }
      })
      .catch(console.error);

    // Fetch from score endpoint to get the full engine signals (numerology-related)
    const scorePromise = fetch(`${API}/api/score?date=${today}`)
      .then(r => r.json())
      .then(result => {
        if (result.signals) {
          // Filter numerology/esoteric-related signals
          const numSignals = result.signals.filter((s: EngineSignal) => {
            const name = s.name || '';
            return (
              name.includes('date_reduction') ||
              name.includes('day_of_year') ||
              name.includes('moon') ||
              name.includes('zodiac') ||
              name.includes('planetary') ||
              name.includes('master') ||
              name.includes('caution') ||
              name.includes('pump') ||
              name.includes('fibonacci') ||
              name.includes('angel') ||
              name.includes('clock') ||
              name.includes('mirror') ||
              name.includes('digit') ||
              name.includes('energy') ||
              name.includes('ritual') ||
              name.includes('inversion') ||
              name.includes('gematria') ||
              name.includes('numerology')
            );
          });
          setEngineSignals(numSignals);
        }
      })
      .catch(console.error);

    Promise.all([signalsPromise, scorePromise]).finally(() => setLoading(false));
  }, []);

  if (loading || !data) {
    return (
      <div className="p-3">
        <div className="text-xs text-slate-600 text-center py-2">Loading...</div>
      </div>
    );
  }

  return (
    <div className="p-3">
      <div className="flex items-center gap-2 mb-3">
        <span className="text-sm text-esoteric">{ZODIAC_ICONS[data.zodiac_sign] || ''}</span>
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Numerology</h3>
      </div>

      <div className="space-y-2">
        {/* Day of Year */}
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-slate-600">Day of Year</span>
          <span className="text-xs num-display text-white font-semibold">{data.day_of_year}</span>
        </div>

        {/* Date Reduction */}
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-slate-600">Reduction</span>
          <span className={`text-xs num-display font-bold px-1.5 py-0.5 rounded ${
            data.master_number ? 'bg-esoteric/15 text-esoteric' :
            data.is_caution_number ? 'bg-bearish/15 text-bearish' :
            data.is_pump_number ? 'bg-bullish/15 text-bullish' :
            'text-white'
          }`}>
            {data.date_reduction}
            {data.master_number && ' M'}
          </span>
        </div>

        {/* Days Remaining */}
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-slate-600">Days Left</span>
          <span className="text-xs num-display text-slate-400">{data.days_remaining}</span>
        </div>

        {/* Divider */}
        <div className="border-t border-panel-border my-1" />

        {/* Moon Phase */}
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-slate-600">Moon</span>
          <span className="text-xs text-slate-300 flex items-center gap-1">
            <span>{MOON_ICONS[data.moon_phase] || ''}</span>
            <span className="text-[10px]">{data.moon_phase}</span>
          </span>
        </div>

        {/* Zodiac */}
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-slate-600">Zodiac</span>
          <span className="text-xs text-slate-300 flex items-center gap-1">
            <span>{ZODIAC_ICONS[data.zodiac_sign] || ''}</span>
            <span className="text-[10px]">{data.zodiac_sign}</span>
          </span>
        </div>

        {/* Planetary Hour */}
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-slate-600">Planet</span>
          <span className="text-xs text-slate-300 flex items-center gap-1">
            <span>{PLANET_ICONS[data.planetary_hour] || ''}</span>
            <span className="text-[10px]">{data.planetary_hour}</span>
          </span>
        </div>

        {/* Caution / Pump / Master flags */}
        {(data.is_caution_number || data.is_pump_number || data.master_number) && (
          <>
            <div className="border-t border-panel-border my-1" />
            <div className="flex gap-1 flex-wrap">
              {data.is_caution_number && (
                <span className="px-1.5 py-0.5 text-[9px] rounded bg-bearish/10 text-bearish border border-bearish/20 font-bold">
                  CAUTION DAY
                </span>
              )}
              {data.is_pump_number && (
                <span className="px-1.5 py-0.5 text-[9px] rounded bg-green-500/10 text-green-500 border border-green-500/20 font-bold">
                  PUMP DAY
                </span>
              )}
              {data.master_number && (
                <span className="px-1.5 py-0.5 text-[9px] rounded bg-esoteric/10 text-esoteric border border-esoteric/20 font-bold">
                  MASTER
                </span>
              )}
            </div>
          </>
        )}

        {/* Engine signals from the 42-signal Python engine */}
        {engineSignals.length > 0 && (
          <>
            <div className="border-t border-panel-border my-1" />
            <div className="text-[9px] text-slate-600 uppercase tracking-wider mb-1">Engine Signals ({engineSignals.length})</div>
            <div className="max-h-[120px] overflow-y-auto space-y-0.5">
              {engineSignals
                .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
                .map((sig, i) => {
                  const dirColor = sig.direction > 0
                    ? 'text-green-500'
                    : sig.direction < 0
                    ? 'text-red-500'
                    : 'text-slate-500';

                  return (
                    <div key={i} className="flex items-start justify-between gap-1">
                      <span className="text-[9px] text-slate-500 truncate">{sig.name.replace(/_/g, ' ')}</span>
                      <span className={`text-[9px] num-display shrink-0 ${dirColor}`}>
                        {sig.weight > 0 ? '+' : ''}{sig.weight.toFixed(1)}
                      </span>
                    </div>
                  );
                })
              }
            </div>
          </>
        )}
      </div>
    </div>
  );
}
