'use client';
import { API } from '@/lib/api';

import { useState, useEffect } from 'react';
import TradeDetailModal from './TradeDetailModal';

interface Trade {
  id: number;
  tf: string;
  direction: string;
  confidence: number;
  entry_price: number;
  entry_time: string;
  exit_price: number | null;
  exit_time: string | null;
  stop_price: number;
  tp_price: number;
  pnl: number | null;
  pnl_pct: number | null;
  bars_held: number | null;
  exit_reason: string | null;
  regime: string;
  leverage: number;
  risk_pct: number;
  features_json: string | null;
  status: string;
}

interface Account {
  balance: number;
  peak_balance: number;
  total_trades: number;
  wins: number;
  losses: number;
  max_dd: number;
  roi: number;
  win_rate: number;
  mode: string;
}

interface MLData {
  account: Account;
  open_trades: Trade[];
  closed_trades: Trade[];
  today: { trades: number; wins: number };
}

export default function MLTrading({ btcPrice }: { btcPrice?: number }) {
  const [data, setData] = useState<MLData | null>(null);
  const [tab, setTab] = useState<'live' | 'history' | 'stats'>('live');
  const [selectedTrade, setSelectedTrade] = useState<Trade | null>(null);
  const [traderStatus, setTraderStatus] = useState<'running' | 'stopped' | 'loading'>('loading');
  const [toggling, setToggling] = useState(false);

  // Check trader status
  useEffect(() => {
    const checkStatus = () => {
      fetch(`${API}/api/trader/status?mode=ml`)
        .then(r => r.json())
        .then(d => setTraderStatus(d.running ? 'running' : 'stopped'))
        .catch(() => setTraderStatus('stopped'));
    };
    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  const toggleTrader = async () => {
    setToggling(true);
    try {
      const action = traderStatus === 'running' ? 'stop' : 'start';
      const res = await fetch(`${API}/api/trader/${action}?mode=ml`, { method: 'POST' });
      if (res.ok) {
        setTraderStatus(action === 'start' ? 'running' : 'stopped');
      }
    } catch {}
    setToggling(false);
  };

  useEffect(() => {
    const fetchData = () => {
      fetch(`${API}/api/ml-trades?limit=50`)
        .then(r => r.json())
        .then(d => setData(d))
        .catch(() => {});
    };
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  if (!data) return <div className="p-3 text-xs text-slate-500">Loading ML trades...</div>;

  const { account, open_trades, closed_trades, today } = data;

  // Compute unrealized PnL from open trades
  const unrealizedTotal = open_trades.reduce((sum: number, t: Trade) => {
    if (btcPrice && t.entry_price) {
      const isLong = t.direction === 'LONG';
      const pnlPct = (btcPrice - t.entry_price) / t.entry_price * (isLong ? 1 : -1) * t.leverage;
      return sum + account.balance * (t.risk_pct / 100) * pnlPct;
    }
    return sum;
  }, 0);
  const effectiveBalance = account.balance + unrealizedTotal;

  return (
    <div className="flex flex-col h-full text-xs">
      {/* Header with mode + balance (includes unrealized PnL like Bitget) */}
      <div className="p-2 border-b border-panel-border">
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-1.5">
            <div className={`w-2 h-2 rounded-full ${traderStatus === 'running' ? 'bg-green-500 animate-pulse' : 'bg-slate-600'}`} />
            <span className="font-bold text-slate-200 uppercase text-[10px] tracking-wider">
              ML TRADER
            </span>
            <span className={`text-[9px] px-1 py-0.5 rounded font-medium ${
              traderStatus === 'running' ? 'bg-green-500/15 text-green-400' : 'bg-slate-700/50 text-slate-500'
            }`}>
              {traderStatus === 'loading' ? '...' : traderStatus === 'running' ? 'LIVE' : 'OFF'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={toggleTrader}
              disabled={toggling || traderStatus === 'loading'}
              className={`px-2 py-1 rounded text-[9px] font-bold transition-all disabled:opacity-50 ${
                traderStatus === 'running'
                  ? 'bg-red-500/15 text-red-400 border border-red-500/30 hover:bg-red-500/25'
                  : 'bg-green-500/15 text-green-400 border border-green-500/30 hover:bg-green-500/25'
              }`}
            >
              {toggling ? '...' : traderStatus === 'running' ? 'DISABLE' : 'ENABLE'}
            </button>
          </div>
        </div>
        <div className="flex items-center justify-between mb-1">
          <div className="flex items-center gap-1.5" />
          <div className="text-right">
            <span className="font-mono font-bold text-sm text-slate-100">
              ${effectiveBalance.toFixed(2)}
            </span>
            {unrealizedTotal !== 0 && (
              <div className={`font-mono text-[9px] ${unrealizedTotal >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {unrealizedTotal >= 0 ? '+' : ''}{unrealizedTotal.toFixed(2)} unrealized
              </div>
            )}
          </div>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-4 gap-1 text-[10px]">
          <div className="text-center">
            <div className={`font-mono font-bold ${account.roi >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {account.roi >= 0 ? '+' : ''}{account.roi.toFixed(1)}%
            </div>
            <div className="text-slate-500">ROI</div>
          </div>
          <div className="text-center">
            <div className="font-mono font-bold text-slate-200">{account.win_rate.toFixed(0)}%</div>
            <div className="text-slate-500">WR</div>
          </div>
          <div className="text-center">
            <div className="font-mono font-bold text-slate-200">{today.trades}</div>
            <div className="text-slate-500">Today</div>
          </div>
          <div className="text-center">
            <div className="font-mono font-bold text-red-400">{account.max_dd.toFixed(1)}%</div>
            <div className="text-slate-500">DD</div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-panel-border">
        {(['live', 'history', 'stats'] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`flex-1 py-1.5 text-[10px] font-semibold uppercase tracking-wider transition-colors
              ${tab === t ? 'text-blue-400 border-b-2 border-blue-400' : 'text-slate-500 hover:text-slate-300'}`}
          >
            {t === 'live' ? `Live (${open_trades.length})` : t === 'history' ? `History (${account.total_trades})` : 'Stats'}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {tab === 'live' && (
          <div className="p-1.5 space-y-1">
            {open_trades.length === 0 ? (
              <div className="text-center text-slate-600 py-4">No open positions</div>
            ) : (
              open_trades.map(trade => (
                <TradeCard key={trade.id} trade={trade} btcPrice={btcPrice} isOpen onDetailClick={setSelectedTrade} />
              ))
            )}
          </div>
        )}

        {tab === 'history' && (
          <div className="p-1.5 space-y-0.5">
            {closed_trades.map(trade => (
              <TradeCard key={trade.id} trade={trade} onDetailClick={setSelectedTrade} />
            ))}
          </div>
        )}

        {tab === 'stats' && (
          <div className="p-3 space-y-2 text-[11px]">
            <StatRow label="Balance" value={`$${account.balance.toFixed(2)}`} />
            <StatRow label="Peak" value={`$${account.peak_balance.toFixed(2)}`} />
            <StatRow label="Total Trades" value={account.total_trades.toString()} />
            <StatRow label="Wins / Losses" value={`${account.wins} / ${account.losses}`} />
            <StatRow label="Win Rate" value={`${account.win_rate.toFixed(1)}%`} color={account.win_rate > 60 ? 'text-green-400' : undefined} />
            <StatRow label="Max Drawdown" value={`${account.max_dd.toFixed(1)}%`} color="text-red-400" />
            <StatRow label="Today's Trades" value={`${today.trades} (${today.wins}W)`} />
            <StatRow label="Profit Factor" value={account.wins > 0 ? (account.wins / Math.max(account.losses, 1)).toFixed(2) + 'x' : '—'} />
          </div>
        )}
      </div>
      {/* Trade Detail Modal */}
      {selectedTrade && (
        <TradeDetailModal trade={selectedTrade} onClose={() => setSelectedTrade(null)} />
      )}
    </div>
  );
}

function TradeCard({ trade, btcPrice, isOpen, onDetailClick }: { trade: Trade; btcPrice?: number; isOpen?: boolean; onDetailClick?: (t: Trade) => void }) {
  const [expanded, setExpanded] = useState(false);
  const isLong = trade.direction === 'LONG';
  const unrealizedPnl = isOpen && btcPrice && trade.entry_price
    ? (btcPrice - trade.entry_price) / trade.entry_price * (isLong ? 1 : -1) * trade.leverage * 100
    : null;

  let features: Record<string, number> = {};
  try { if (trade.features_json) features = JSON.parse(trade.features_json); } catch {}

  return (
    <div
      className={`rounded border cursor-pointer transition-colors ${
        isOpen
          ? 'border-blue-500/30 bg-blue-500/5 hover:bg-blue-500/10'
          : trade.pnl && trade.pnl > 0
            ? 'border-green-500/20 bg-green-500/5 hover:bg-green-500/10'
            : 'border-red-500/20 bg-red-500/5 hover:bg-red-500/10'
      }`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="p-1.5">
        {/* Row 1: Direction, TF, Leverage, Entry Price */}
        <div className="flex items-center gap-1.5">
          {/* Direction text */}
          <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold ${isLong ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
            {isLong ? 'LONG' : 'SHORT'}
          </span>

          {/* TF badge */}
          <span className="px-1 py-0.5 rounded bg-slate-700/50 text-[9px] font-mono font-bold text-slate-300">
            {trade.tf.toUpperCase()}
          </span>

          {/* Leverage */}
          <span className="text-[9px] text-yellow-400 font-mono font-bold">
            {trade.leverage?.toFixed(0)}x
          </span>

          {/* Position size + capital risked */}
          <span className="text-[9px] text-slate-500 font-mono">
            ${(100 * (trade.risk_pct / 100) * trade.leverage).toFixed(0)} pos
          </span>
          <span className="text-[9px] text-orange-400/70 font-mono">
            ${(100 * (trade.risk_pct / 100)).toFixed(2)} risk
          </span>

          <span className="flex-1" />

          {/* Entry price */}
          <span className="font-mono text-slate-300 text-[10px]">
            ${trade.entry_price?.toLocaleString(undefined, { maximumFractionDigits: 0 })}
          </span>
        </div>

        {/* Row 2: PnL $, PnL %, Exit reason */}
        <div className="flex items-center gap-1.5 mt-1">
          {/* Profit/Loss arrow */}
          {trade.pnl !== null ? (
            <span className={`text-xs ${trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {trade.pnl >= 0 ? '▲' : '▼'}
            </span>
          ) : isOpen && unrealizedPnl !== null ? (
            <span className={`text-xs ${unrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {unrealizedPnl >= 0 ? '▲' : '▼'}
            </span>
          ) : null}

          {/* PnL $ and % */}
          {isOpen && unrealizedPnl !== null ? (
            <span className={`font-mono font-bold text-[10px] ${unrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {unrealizedPnl >= 0 ? '+' : ''}{unrealizedPnl.toFixed(2)}%
            </span>
          ) : trade.pnl !== null ? (
            <>
              <span className={`font-mono font-bold text-[10px] ${trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ${trade.pnl >= 0 ? '+' : ''}{trade.pnl.toFixed(2)}
              </span>
              {trade.pnl_pct !== null && (
                <span className={`font-mono text-[9px] ${trade.pnl >= 0 ? 'text-green-400/70' : 'text-red-400/70'}`}>
                  ({trade.pnl_pct >= 0 ? '+' : ''}{trade.pnl_pct.toFixed(2)}%)
                </span>
              )}
            </>
          ) : (
            <span className="text-[9px] text-blue-400 animate-pulse">WAITING...</span>
          )}

          <span className="flex-1" />

          {/* Risk/Position size */}
          <span className="text-[8px] text-slate-500 font-mono">
            {trade.risk_pct?.toFixed(1)}% risk
          </span>

          {/* Exit reason badge */}
          {trade.exit_reason && (
            <span className={`px-1 py-0.5 rounded text-[8px] font-bold ${
              trade.exit_reason === 'TP' ? 'bg-green-500/20 text-green-400' :
              trade.exit_reason === 'SL' ? 'bg-red-500/20 text-red-400' :
              'bg-yellow-500/20 text-yellow-400'
            }`}>
              {trade.exit_reason}
            </span>
          )}
          {isOpen && (
            <span className="px-1 py-0.5 rounded text-[8px] font-bold bg-blue-500/20 text-blue-400 animate-pulse">LIVE</span>
          )}
        </div>
      </div>

      {/* Expanded details */}
      {expanded && (
        <div className="px-2 pb-2 text-[9px] text-slate-400 space-y-1 border-t border-slate-700/50 mt-1 pt-1">
          <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
            <span>Confidence: <b className="text-slate-200">{(trade.confidence * 100).toFixed(1)}%</b></span>
            <span>Regime: <b className="text-slate-200">{trade.regime}</b></span>
            <span>Leverage: <b className="text-slate-200">{trade.leverage?.toFixed(0)}x</b></span>
            <span>Risk: <b className="text-slate-200">{trade.risk_pct?.toFixed(1)}%</b></span>
            <span>SL: <b className="text-red-400">${trade.stop_price?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</b></span>
            <span>TP: <b className="text-green-400">${trade.tp_price?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</b></span>
            {trade.bars_held != null && <span>Held: <b className="text-slate-200">{trade.bars_held} bars</b></span>}
            <span>Entry: <b className="text-slate-200">{trade.entry_time?.split('T')[1]?.slice(0, 5) || '—'}</b></span>
          </div>

          {/* View Details button */}
          <button
            onClick={(e) => { e.stopPropagation(); onDetailClick?.(trade); }}
            className="w-full mt-1 py-1 rounded bg-blue-500/10 border border-blue-500/30 text-blue-400 text-[9px] font-semibold hover:bg-blue-500/20 transition-colors"
          >
            View Full Trade Reasoning
          </button>
        </div>
      )}
    </div>
  );
}

function StatRow({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-slate-500">{label}</span>
      <span className={`font-mono font-bold ${color || 'text-slate-200'}`}>{value}</span>
    </div>
  );
}
