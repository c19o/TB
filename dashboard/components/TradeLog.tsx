'use client';
import { API } from '@/lib/api';

import { useState, useEffect } from 'react';

interface Trade {
  id: number;
  tf: string;
  direction: string;
  confidence: number;
  entry_price: number;
  entry_time: string;
  exit_price: number | null;
  exit_time: string | null;
  pnl: number | null;
  exit_reason: string | null;
  regime: string;
  status: string;
}

export default function TradeLog() {
  const [trades, setTrades] = useState<Trade[]>([]);

  useEffect(() => {
    const fetchTrades = () => {
      fetch(`${API}/api/ml-trades?limit=30`)
        .then(r => r.json())
        .then(d => {
          const all = [...(d.open_trades || []), ...(d.closed_trades || [])];
          all.sort((a: Trade, b: Trade) => {
            const ta = a.exit_time || a.entry_time || '';
            const tb = b.exit_time || b.entry_time || '';
            return tb.localeCompare(ta);
          });
          setTrades(all.slice(0, 30));
        })
        .catch(() => {});
    };
    fetchTrades();
    const interval = setInterval(fetchTrades, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center gap-2 px-3 py-1.5 border-b border-panel-border shrink-0">
        <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">Trade Log</span>
        <span className="text-[9px] text-slate-600">{trades.length} recent</span>
      </div>
      <div className="flex-1 overflow-y-auto">
        <table className="w-full text-[10px]">
          <thead className="sticky top-0 bg-[#0f1117]">
            <tr className="text-slate-500">
              <th className="text-left px-2 py-1 font-medium">Time</th>
              <th className="text-left px-1 py-1 font-medium">TF</th>
              <th className="text-left px-1 py-1 font-medium">Dir</th>
              <th className="text-right px-1 py-1 font-medium">Entry</th>
              <th className="text-right px-1 py-1 font-medium">Exit</th>
              <th className="text-right px-1 py-1 font-medium">PnL</th>
              <th className="text-center px-1 py-1 font-medium">Exit</th>
              <th className="text-left px-1 py-1 font-medium">Regime</th>
              <th className="text-right px-2 py-1 font-medium">Conf</th>
            </tr>
          </thead>
          <tbody>
            {trades.map(t => (
              <tr key={t.id} className={`border-t border-slate-800/50 hover:bg-slate-800/30 ${t.status === 'open' ? 'bg-blue-500/5' : ''}`}>
                <td className="px-2 py-0.5 text-slate-400 font-mono">
                  {(t.exit_time || t.entry_time || '').split('T')[1]?.slice(0, 5) || '—'}
                </td>
                <td className="px-1 py-0.5">
                  <span className="px-1 py-0.5 rounded bg-slate-700/50 font-mono font-bold text-slate-300 text-[9px]">
                    {t.tf?.toUpperCase()}
                  </span>
                </td>
                <td className="px-1 py-0.5">
                  <span className={`text-[9px] font-bold ${t.direction === 'LONG' ? 'text-green-400' : 'text-red-400'}`}>
                    {t.direction}
                  </span>
                </td>
                <td className="px-1 py-0.5 text-right font-mono text-slate-300">
                  ${t.entry_price?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </td>
                <td className="px-1 py-0.5 text-right font-mono text-slate-400">
                  {t.exit_price ? `$${t.exit_price.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : '—'}
                </td>
                <td className={`px-1 py-0.5 text-right font-mono font-bold ${
                  t.pnl === null ? 'text-blue-400' : t.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {t.pnl !== null ? `${t.pnl >= 0 ? '+' : ''}$${t.pnl.toFixed(2)}` : 'OPEN'}
                </td>
                <td className="px-1 py-0.5 text-center">
                  {t.exit_reason ? (
                    <span className={`px-1 py-0.5 rounded text-[8px] font-bold ${
                      t.exit_reason === 'TP' ? 'bg-green-500/20 text-green-400' :
                      t.exit_reason === 'SL' ? 'bg-red-500/20 text-red-400' :
                      'bg-yellow-500/20 text-yellow-400'
                    }`}>{t.exit_reason}</span>
                  ) : t.status === 'open' ? (
                    <span className="px-1 py-0.5 rounded text-[8px] font-bold bg-blue-500/20 text-blue-400 animate-pulse">LIVE</span>
                  ) : '—'}
                </td>
                <td className="px-1 py-0.5 text-slate-500 text-[9px]">{t.regime}</td>
                <td className="px-2 py-0.5 text-right font-mono text-slate-400">
                  {t.confidence ? `${(t.confidence * 100).toFixed(0)}%` : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
