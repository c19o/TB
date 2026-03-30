'use client';
import { API } from '@/lib/api';

import { useState, useEffect, useCallback } from 'react';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface AccountInfo {
  balance: number;
  peak_balance: number;
  total_trades: number;
  wins: number;
  losses: number;
  pnl: number;
  roi: number;
  win_rate: number;
  profit_factor: number;
  max_dd: number;
  current_dd: number;
}

interface Position {
  id: number;
  signal_name: string;
  direction: string;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  trailing_stop: number | null;
  leverage: number;
  risk_amount: number;
  position_size: number;
  timeframe: string;
  category: string;
  status: string;
  exit_price: number | null;
  pnl: number | null;
  result: string | null;
  opened_at: string;
  closed_at: string | null;
}

interface SignalFired {
  id: number;
  timestamp: string;
  candle_time: string;
  timeframe: string;
  signal_name: string;
  direction: string;
  strength: number;
  category: string;
  action: string;
  btc_price: number;
}

interface EquityPoint {
  timestamp: string;
  balance: number;
}

interface PaperTradingData {
  account: AccountInfo;
  open_positions: Position[];
  recent_trades: Position[];
  equity_curve: EquityPoint[];
}

interface SignalsData {
  signals: SignalFired[];
  summary: {
    total: number;
    traded: number;
    filtered: number;
    by_action: Record<string, number>;
  };
}

interface AnxAccount {
  balance: number;
  initial_balance: number;
  pnl: number;
  roi: number;
  peak_balance: number;
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  profit_factor: number;
  avg_hold_days: number;
  in_position: boolean;
  entry_price: number;
  entry_date: string | null;
  stop_loss: number;
  position_size: number;
  directionality: number;
  green_wave: boolean;
  updated_at: string;
}

interface AnxTrade {
  id: number;
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  pnl: number;
  pct_return: number;
  hold_days: number;
  reason: string;
  balance_after: number;
}

interface AnxData {
  account: AnxAccount;
  trades: AnxTrade[];
}

// ---------------------------------------------------------------------------
// Tab type
// ---------------------------------------------------------------------------
type Tab = 'overview' | 'positions' | 'history' | 'signals' | 'anx';

// ---------------------------------------------------------------------------
// Mini sparkline SVG
// ---------------------------------------------------------------------------
function Sparkline({ data, width = 200, height = 40 }: { data: number[]; width?: number; height?: number }) {
  if (data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;

  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  }).join(' ');

  const lastVal = data[data.length - 1];
  const firstVal = data[0];
  const color = lastVal >= firstVal ? '#22c55e' : '#ef4444';

  return (
    <svg width={width} height={height} className="shrink-0">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function PaperTrading({ btcPrice }: { btcPrice?: number }) {
  const [data, setData] = useState<PaperTradingData | null>(null);
  const [signalsData, setSignalsData] = useState<SignalsData | null>(null);
  const [anxData, setAnxData] = useState<AnxData | null>(null);
  const [tab, setTab] = useState<Tab>('overview');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [traderStatus, setTraderStatus] = useState<'running' | 'stopped' | 'loading'>('loading');
  const [toggling, setToggling] = useState(false);

  // Check trader status
  useEffect(() => {
    const checkStatus = () => {
      fetch(`${API}/api/trader/status?mode=paper`)
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
      const res = await fetch(`${API}/api/trader/${action}?mode=paper`, { method: 'POST' });
      if (res.ok) {
        setTraderStatus(action === 'start' ? 'running' : 'stopped');
      }
    } catch {}
    setToggling(false);
  };

  const fetchData = useCallback(async () => {
    try {
      const [paperRes, sigRes, anxRes] = await Promise.all([
        fetch(`${API}/api/paper-trading`),
        fetch(`${API}/api/paper-trading/signals?hours=24`),
        fetch(`${API}/api/paper-trading/anx`),
      ]);

      if (paperRes.ok) {
        const d = await paperRes.json();
        if (d.error) {
          setError(d.error);
        } else {
          setData(d);
          setError(null);
        }
      } else {
        const errData = await paperRes.json().catch(() => ({}));
        setError(errData.error || `HTTP ${paperRes.status}`);
      }

      if (sigRes.ok) {
        const s = await sigRes.json();
        if (!s.error) setSignalsData(s);
      }

      if (anxRes.ok) {
        const a = await anxRes.json();
        if (!a.error) setAnxData(a);
      }
    } catch (e: any) {
      setError(e.message || 'Failed to connect');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 15_000);
    return () => clearInterval(interval);
  }, [fetchData]);

  if (loading) {
    return (
      <div className="p-4 text-center text-slate-500 text-sm">
        Loading paper trading data...
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4">
        <div className="text-xs text-orange-400 bg-orange-500/10 border border-orange-500/20 rounded p-3">
          Paper Trader: {error}
        </div>
      </div>
    );
  }

  if (!data) return null;

  const { account: rawAccount, open_positions = [], recent_trades = [], equity_curve = [] } = data;

  // Default account values when no trades exist yet
  const account = {
    balance: 100, peak_balance: 100, total_trades: 0, wins: 0, losses: 0,
    pnl: 0, roi: 0, win_rate: 0, profit_factor: 0, max_dd: 0, current_dd: 0,
    ...rawAccount,
  };

  // Compute unrealized P&L for open positions
  const unrealizedPnl = open_positions.reduce((sum, pos) => {
    if (!btcPrice) return sum;
    if (pos.direction === 'long') {
      return sum + ((btcPrice - pos.entry_price) / pos.entry_price) * pos.position_size;
    } else {
      return sum + ((pos.entry_price - btcPrice) / pos.entry_price) * pos.position_size;
    }
  }, 0);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-3 py-2 border-b border-panel-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${traderStatus === 'running' ? 'bg-yellow-500 animate-pulse' : 'bg-slate-600'}`} />
            <span className="text-xs font-semibold text-slate-300 tracking-wider">
              PAPER TRADER
            </span>
            <span className={`text-[9px] px-1 py-0.5 rounded font-medium ${
              traderStatus === 'running' ? 'bg-yellow-500/15 text-yellow-400' : 'bg-slate-700/50 text-slate-500'
            }`}>
              {traderStatus === 'loading' ? '...' : traderStatus === 'running' ? 'ACTIVE' : 'OFF'}
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
        <div className="flex items-center justify-between mt-1">
          <div />
          <span className={`num-display text-lg font-bold ${account.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            ${account.balance.toFixed(2)}
          </span>
        </div>

        {/* Quick stats row */}
        <div className="flex items-center justify-between mt-1.5 text-[10px]">
          <span className={`num-display ${account.pnl >= 0 ? 'text-green-500' : 'text-red-500'}`}>
            P&L: ${account.pnl >= 0 ? '+' : ''}{account.pnl.toFixed(2)} ({account.roi >= 0 ? '+' : ''}{account.roi.toFixed(1)}%)
          </span>
          <span className="text-slate-500 num-display">
            {account.total_trades} trades | {account.win_rate}% WR | PF {account.profit_factor}
          </span>
        </div>

        {/* Equity sparkline */}
        {equity_curve.length > 2 && (
          <div className="mt-1.5 flex justify-center">
            <Sparkline data={equity_curve.map(p => p.balance)} width={190} height={28} />
          </div>
        )}
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-panel-border shrink-0">
        {(['overview', 'positions', 'history', 'signals', 'anx'] as Tab[]).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`flex-1 px-2 py-1.5 text-[10px] font-medium uppercase tracking-wider transition-all ${
              tab === t
                ? 'text-blue-400 border-b-2 border-blue-400 bg-blue-500/5'
                : 'text-slate-600 hover:text-slate-400'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-2">
        {tab === 'overview' && (
          <OverviewTab account={account} openCount={open_positions.length} unrealizedPnl={unrealizedPnl} />
        )}
        {tab === 'positions' && (
          <PositionsTab positions={open_positions} btcPrice={btcPrice} />
        )}
        {tab === 'history' && (
          <HistoryTab trades={recent_trades} />
        )}
        {tab === 'signals' && (
          <SignalsTab data={signalsData} />
        )}
        {tab === 'anx' && (
          <AnxTab data={anxData} btcPrice={btcPrice} />
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Overview Tab
// ---------------------------------------------------------------------------
function OverviewTab({ account, openCount, unrealizedPnl }: {
  account: AccountInfo;
  openCount: number;
  unrealizedPnl: number;
}) {
  const stats = [
    { label: 'Balance', value: `$${account.balance.toFixed(2)}`, color: account.pnl >= 0 ? 'text-green-400' : 'text-red-400' },
    { label: 'Peak', value: `$${account.peak_balance.toFixed(2)}`, color: 'text-slate-300' },
    { label: 'P&L', value: `$${account.pnl >= 0 ? '+' : ''}${account.pnl.toFixed(2)}`, color: account.pnl >= 0 ? 'text-green-400' : 'text-red-400' },
    { label: 'ROI', value: `${account.roi >= 0 ? '+' : ''}${account.roi.toFixed(1)}%`, color: account.roi >= 0 ? 'text-green-400' : 'text-red-400' },
    { label: 'Win Rate', value: `${account.win_rate}%`, color: account.win_rate >= 52 ? 'text-green-400' : account.win_rate >= 48 ? 'text-yellow-400' : 'text-red-400' },
    { label: 'Profit Factor', value: account.profit_factor.toFixed(2), color: account.profit_factor >= 1.5 ? 'text-green-400' : account.profit_factor >= 1 ? 'text-yellow-400' : 'text-red-400' },
    { label: 'Total Trades', value: account.total_trades.toString(), color: 'text-slate-300' },
    { label: 'Wins/Losses', value: `${account.wins}/${account.losses}`, color: 'text-slate-300' },
    { label: 'Max DD', value: `${account.max_dd.toFixed(1)}%`, color: account.max_dd > 15 ? 'text-red-400' : account.max_dd > 10 ? 'text-yellow-400' : 'text-green-400' },
    { label: 'Open Positions', value: openCount.toString(), color: 'text-blue-400' },
    { label: 'Unrealized P&L', value: `$${unrealizedPnl >= 0 ? '+' : ''}${unrealizedPnl.toFixed(2)}`, color: unrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400' },
  ];

  return (
    <div className="grid grid-cols-2 gap-1">
      {stats.map(s => (
        <div key={s.label} className="flex justify-between items-center px-2 py-1 rounded bg-white/[0.02]">
          <span className="text-[10px] text-slate-500">{s.label}</span>
          <span className={`text-[11px] num-display font-medium ${s.color}`}>{s.value}</span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Positions Tab
// ---------------------------------------------------------------------------
function PositionsTab({ positions, btcPrice }: { positions: Position[]; btcPrice?: number }) {
  if (positions.length === 0) {
    return <div className="text-center text-slate-600 text-xs py-4">No open positions</div>;
  }

  return (
    <div className="space-y-1.5">
      {positions.map(pos => {
        let pnl = 0;
        if (btcPrice) {
          if (pos.direction === 'long') {
            pnl = ((btcPrice - pos.entry_price) / pos.entry_price) * pos.position_size;
          } else {
            pnl = ((pos.entry_price - btcPrice) / pos.entry_price) * pos.position_size;
          }
        }
        const isWinning = pnl > 0;

        return (
          <div
            key={pos.id}
            className={`p-2 rounded border text-[10px] ${
              isWinning
                ? 'border-green-500/20 bg-green-500/5'
                : 'border-red-500/20 bg-red-500/5'
            }`}
          >
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-1.5">
                <span className={`px-1 py-0.5 rounded text-[9px] font-bold ${
                  pos.direction === 'long' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                }`}>
                  {pos.direction.toUpperCase()}
                </span>
                <span className="text-slate-300 font-medium">{pos.signal_name}</span>
              </div>
              <span className={`num-display font-bold ${isWinning ? 'text-green-400' : 'text-red-400'}`}>
                ${pnl >= 0 ? '+' : ''}{pnl.toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between mt-1 text-slate-500">
              <span>Entry: ${pos.entry_price.toFixed(2)}</span>
              <span>SL: ${pos.stop_loss.toFixed(2)}</span>
              <span>TP: ${pos.take_profit.toFixed(2)}</span>
              <span>{pos.leverage}x</span>
            </div>
            <div className="flex justify-between mt-0.5 text-slate-600">
              <span>{pos.timeframe} | {pos.category}</span>
              <span>Risk: ${pos.risk_amount.toFixed(2)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// History Tab
// ---------------------------------------------------------------------------
function HistoryTab({ trades }: { trades: Position[] }) {
  if (trades.length === 0) {
    return <div className="text-center text-slate-600 text-xs py-4">No trade history yet</div>;
  }

  return (
    <div className="space-y-1">
      {trades.map(t => (
        <div
          key={t.id}
          className={`flex items-center justify-between px-2 py-1 rounded text-[10px] ${
            (t.pnl ?? 0) > 0 ? 'bg-green-500/5' : 'bg-red-500/5'
          }`}
        >
          <div className="flex items-center gap-1.5 min-w-0">
            <span className={`shrink-0 px-1 py-0.5 rounded text-[9px] font-bold ${
              t.direction === 'long' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
            }`}>
              {t.direction === 'long' ? 'L' : 'S'}
            </span>
            <span className="text-slate-400 truncate" title={t.signal_name}>
              {t.signal_name}
            </span>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-slate-600">{t.result}</span>
            <span className={`num-display font-medium ${(t.pnl ?? 0) > 0 ? 'text-green-400' : 'text-red-400'}`}>
              ${(t.pnl ?? 0) >= 0 ? '+' : ''}{(t.pnl ?? 0).toFixed(2)}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Signals Tab
// ---------------------------------------------------------------------------
function SignalsTab({ data }: { data: SignalsData | null }) {
  if (!data || !data.signals || data.signals.length === 0 || !data.summary) {
    return <div className="text-center text-slate-600 text-xs py-4">No signals in last 24h</div>;
  }

  const actionColors: Record<string, string> = {
    traded: 'text-green-400 bg-green-500/10',
    filtered_ema: 'text-yellow-400 bg-yellow-500/10',
    filtered_trend: 'text-orange-400 bg-orange-500/10',
    filtered_voc: 'text-purple-400 bg-purple-500/10',
    filtered_cooldown: 'text-blue-400 bg-blue-500/10',
    filtered_equity_curve: 'text-cyan-400 bg-cyan-500/10',
    filtered_daily_loss_cap: 'text-red-400 bg-red-500/10',
  };

  return (
    <div>
      {/* Summary */}
      <div className="flex gap-2 mb-2 text-[10px]">
        <span className="px-1.5 py-0.5 rounded bg-green-500/10 text-green-400">
          Traded: {data.summary.traded}
        </span>
        <span className="px-1.5 py-0.5 rounded bg-red-500/10 text-red-400">
          Filtered: {data.summary.filtered}
        </span>
      </div>

      {/* Signal list */}
      <div className="space-y-1">
        {data.signals.slice(0, 50).map(s => {
          const colorClass = actionColors[s.action] || 'text-slate-400 bg-slate-500/10';
          return (
            <div key={s.id} className="flex items-center justify-between px-2 py-1 rounded bg-white/[0.02] text-[10px]">
              <div className="flex items-center gap-1.5 min-w-0">
                <span className={`shrink-0 px-1 py-0.5 rounded text-[9px] font-medium ${colorClass}`}>
                  {s.action.replace('filtered_', '').toUpperCase()}
                </span>
                <span className={`shrink-0 px-1 py-0.5 rounded text-[9px] font-bold ${
                  s.direction === 'long' ? 'text-green-400' : s.direction === 'short' ? 'text-red-400' : 'text-slate-400'
                }`}>
                  {s.direction === 'long' ? 'L' : s.direction === 'short' ? 'S' : '-'}
                </span>
                <span className="text-slate-400 truncate" title={s.signal_name}>
                  {s.signal_name}
                </span>
              </div>
              <div className="flex items-center gap-1.5 shrink-0 text-slate-600">
                <span>{s.timeframe}</span>
                <span>str:{s.strength}</span>
                <span className="num-display">${s.btc_price?.toFixed(0)}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// AlphanumetriX Tab
// ---------------------------------------------------------------------------
function AnxTab({ data, btcPrice }: { data: AnxData | null; btcPrice?: number }) {
  if (!data) {
    return <div className="text-center text-slate-600 text-xs py-4">AlphanumetriX not initialized</div>;
  }

  const { account: rawAcct2, trades = [] } = data;
  const account = {
    balance: 100, pnl: 0, roi: 0, win_rate: 0, profit_factor: 0, max_dd: 0,
    in_position: false, entry_price: 0, position_size: 0, stop_loss: 0,
    directionality: 0, total_trades: 0, wins: 0, losses: 0, peak_balance: 100,
    ...rawAcct2,
  };
  const unrealizedPnl = account.in_position && btcPrice
    ? ((btcPrice - (account.entry_price || 1)) / (account.entry_price || 1)) * (account.position_size || 0)
    : 0;

  // Directionality bar visualization (-13 to +13)
  const dirPct = (((account.directionality || 0) + 13) / 26) * 100;

  return (
    <div className="space-y-2">
      {/* Wave State */}
      <div className={`p-2 rounded border text-[10px] ${
        account.green_wave
          ? 'border-green-500/30 bg-green-500/10'
          : 'border-red-500/30 bg-red-500/10'
      }`}>
        <div className="flex justify-between items-center">
          <span className="font-bold text-slate-300">WAVE STATE</span>
          <span className={`px-2 py-0.5 rounded font-bold text-[11px] ${
            account.green_wave ? 'bg-green-500/30 text-green-400' : 'bg-red-500/30 text-red-400'
          }`}>
            {account.green_wave ? 'GREEN WAVE' : 'NO WAVE'}
          </span>
        </div>
        <div className="mt-1.5">
          <div className="flex justify-between text-[9px] text-slate-500 mb-0.5">
            <span>-13 (Bear)</span>
            <span>Dir: {account.directionality}</span>
            <span>+13 (Bull)</span>
          </div>
          <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all ${
                account.directionality > 0 ? 'bg-green-500' : account.directionality < 0 ? 'bg-red-500' : 'bg-yellow-500'
              }`}
              style={{ width: `${dirPct}%` }}
            />
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-1">
        {[
          { label: 'Balance', value: `$${account.balance.toFixed(2)}`, color: account.pnl >= 0 ? 'text-green-400' : 'text-red-400' },
          { label: 'P&L', value: `$${account.pnl >= 0 ? '+' : ''}${account.pnl.toFixed(2)}`, color: account.pnl >= 0 ? 'text-green-400' : 'text-red-400' },
          { label: 'ROI', value: `${account.roi >= 0 ? '+' : ''}${account.roi.toFixed(1)}%`, color: account.roi >= 0 ? 'text-green-400' : 'text-red-400' },
          { label: 'Win Rate', value: `${account.win_rate}%`, color: account.win_rate >= 52 ? 'text-green-400' : 'text-yellow-400' },
          { label: 'Trades', value: `${account.wins}W / ${account.losses}L`, color: 'text-slate-300' },
          { label: 'PF', value: account.profit_factor.toFixed(2), color: account.profit_factor >= 1.5 ? 'text-green-400' : 'text-yellow-400' },
          { label: 'Avg Hold', value: `${account.avg_hold_days}d`, color: 'text-slate-300' },
          { label: 'Leverage', value: '2x', color: 'text-blue-400' },
        ].map(s => (
          <div key={s.label} className="flex justify-between items-center px-2 py-1 rounded bg-white/[0.02]">
            <span className="text-[10px] text-slate-500">{s.label}</span>
            <span className={`text-[11px] num-display font-medium ${s.color}`}>{s.value}</span>
          </div>
        ))}
      </div>

      {/* Open Position */}
      {account.in_position && (
        <div className={`p-2 rounded border text-[10px] ${
          unrealizedPnl >= 0 ? 'border-green-500/20 bg-green-500/5' : 'border-red-500/20 bg-red-500/5'
        }`}>
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-1.5">
              <span className="px-1 py-0.5 rounded text-[9px] font-bold bg-green-500/20 text-green-400">LONG</span>
              <span className="text-slate-300 font-medium">ANX Green Wave</span>
            </div>
            <span className={`num-display font-bold ${unrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ${unrealizedPnl >= 0 ? '+' : ''}{unrealizedPnl.toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between mt-1 text-slate-500">
            <span>Entry: ${account.entry_price.toFixed(2)}</span>
            <span>SL: ${account.stop_loss.toFixed(2)}</span>
            <span>2x lev</span>
          </div>
          {account.entry_date && (
            <div className="mt-0.5 text-slate-600">
              Since: {new Date(account.entry_date).toLocaleDateString()}
            </div>
          )}
        </div>
      )}

      {/* Trade History */}
      {trades.length > 0 && (
        <div>
          <div className="text-[10px] text-slate-500 font-medium mb-1">TRADE HISTORY</div>
          <div className="space-y-1">
            {trades.map(t => (
              <div
                key={t.id}
                className={`flex items-center justify-between px-2 py-1 rounded text-[10px] ${
                  t.pnl > 0 ? 'bg-green-500/5' : 'bg-red-500/5'
                }`}
              >
                <div className="flex items-center gap-1.5 min-w-0">
                  <span className="text-slate-400 truncate">
                    {t.entry_date ? new Date(t.entry_date).toLocaleDateString() : '?'}
                  </span>
                  <span className="text-slate-600">{t.hold_days}d</span>
                  <span className="text-slate-600">{t.reason}</span>
                </div>
                <span className={`num-display font-medium shrink-0 ${t.pnl > 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${t.pnl >= 0 ? '+' : ''}{t.pnl.toFixed(2)} ({t.pct_return >= 0 ? '+' : ''}{t.pct_return.toFixed(1)}%)
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
