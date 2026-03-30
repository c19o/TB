'use client';

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

const FEATURE_LABELS: Record<string, string> = {
  h1_return: '1H Momentum',
  h4_return: '4H Momentum',
  d_return: 'Daily Return',
  close_vs_sma_5: 'Price vs SMA5',
  close_vs_sma_100: 'Price vs SMA100',
  close_position: 'Close Position in Range',
  stoch_k: 'Stochastic %K',
  rsi_14: 'RSI 14',
  rsi_7: 'RSI 7 (fast)',
  h1_rsi14: '1H RSI',
  h1_bb_pctb: '1H Bollinger %B',
  bb_pctb: 'Bollinger %B',
  macd_histogram: 'MACD Histogram',
  adx_14: 'ADX Trend Strength',
  volume_ratio: 'Volume Ratio',
  atr_14: 'ATR Volatility',
  consec_red: 'Consecutive Red',
  consec_green: 'Consecutive Green',
  news_count: 'News Articles',
  fear_greed: 'Fear & Greed',
  arabic_lot_increase: 'Arabic Lot of Increase',
  arabic_lot_catastrophe: 'Arabic Lot of Catastrophe',
  arabic_lot_treachery: 'Arabic Lot of Treachery',
  date_dr: 'Date Digital Root',
  golden_ratio_dist: 'Golden Ratio Distance',
  w_rsi14: 'Weekly RSI',
  vedic_nakshatra: 'Vedic Nakshatra',
  above_sma100: 'Above SMA100',
};

function interpretValue(key: string, val: number): { text: string; color: string } {
  if (key.includes('return') || key.includes('momentum')) {
    return { text: `${val > 0 ? '+' : ''}${val.toFixed(4)}% (${val > 0 ? 'BULLISH' : 'BEARISH'})`, color: val > 0 ? 'text-green-400' : 'text-red-400' };
  }
  if (key === 'stoch_k' || key === 'rsi_14' || key === 'rsi_7' || key === 'h1_rsi14') {
    const zone = val > 70 ? 'OVERBOUGHT' : val < 30 ? 'OVERSOLD' : 'neutral';
    const c = val > 70 ? 'text-red-400' : val < 30 ? 'text-green-400' : 'text-slate-300';
    return { text: `${val.toFixed(1)} (${zone})`, color: c };
  }
  if (key.includes('bb_pctb')) {
    const zone = val > 0.8 ? 'NEAR UPPER' : val < 0.2 ? 'NEAR LOWER' : 'mid-band';
    return { text: `${val.toFixed(2)} (${zone})`, color: 'text-slate-300' };
  }
  if (key === 'adx_14') {
    return { text: `${val.toFixed(1)} (${val > 25 ? 'STRONG' : 'WEAK'} trend)`, color: val > 25 ? 'text-yellow-400' : 'text-slate-500' };
  }
  if (key === 'date_dr') {
    const meaning = [6, 9].includes(Math.round(val)) ? 'CAUTION' : [3, 7].includes(Math.round(val)) ? 'PUMP' : 'neutral';
    const c = meaning === 'CAUTION' ? 'text-red-400' : meaning === 'PUMP' ? 'text-green-400' : 'text-slate-300';
    return { text: `${Math.round(val)} (${meaning})`, color: c };
  }
  if (key === 'fear_greed') {
    const zone = val < 20 ? 'EXTREME FEAR' : val < 40 ? 'FEAR' : val > 80 ? 'EXTREME GREED' : val > 60 ? 'GREED' : 'NEUTRAL';
    return { text: `${val.toFixed(0)} (${zone})`, color: 'text-slate-300' };
  }
  if (key.includes('close_vs_sma')) {
    return { text: `${val > 0 ? '+' : ''}${val.toFixed(4)} (${val > 0 ? 'ABOVE' : 'BELOW'})`, color: val > 0 ? 'text-green-400' : 'text-red-400' };
  }
  return { text: val.toFixed(4), color: 'text-slate-300' };
}

export default function TradeDetailModal({ trade, onClose }: { trade: Trade; onClose: () => void }) {
  let features: Record<string, number> = {};
  try { if (trade.features_json) features = JSON.parse(trade.features_json); } catch {}

  const isLong = trade.direction === 'LONG';
  const isProfitable = trade.pnl !== null && trade.pnl > 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" onClick={onClose}>
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
      <div
        className="relative bg-[#12121f] border border-slate-700/50 rounded-xl shadow-2xl w-[600px] max-h-[80vh] overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-700/50">
          <div className="flex items-center gap-3">
            <span className={`px-2 py-1 rounded font-bold text-sm ${isLong ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
              {trade.direction}
            </span>
            <span className="px-2 py-1 rounded bg-slate-700/50 font-mono font-bold text-sm text-slate-300">
              {trade.tf.toUpperCase()}
            </span>
            <span className="text-yellow-400 font-mono font-bold">{trade.leverage?.toFixed(0)}x</span>
            {trade.status === 'open' && (
              <span className="px-2 py-1 rounded bg-blue-500/20 text-blue-400 text-xs font-bold animate-pulse">LIVE</span>
            )}
            {isProfitable && <span className="text-green-400 text-lg">&#x2713;</span>}
            {trade.pnl !== null && !isProfitable && <span className="text-red-400 text-lg">&#x2717;</span>}
          </div>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-300 text-xl font-bold w-8 h-8 flex items-center justify-center rounded hover:bg-slate-700/50">
            &#x2715;
          </button>
        </div>

        {/* Trade Info */}
        <div className="p-4 border-b border-slate-700/50">
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <div className="text-slate-500 text-xs mb-1">Entry</div>
              <div className="font-mono font-bold text-slate-200">${trade.entry_price?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
              <div className="text-slate-500 text-[10px]">{trade.entry_time?.replace('T', ' ').slice(0, 19)}</div>
            </div>
            <div>
              <div className="text-slate-500 text-xs mb-1">Exit</div>
              <div className="font-mono font-bold text-slate-200">
                {trade.exit_price ? `$${trade.exit_price.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : '—'}
              </div>
              <div className="text-slate-500 text-[10px]">{trade.exit_time?.replace('T', ' ').slice(0, 19) || 'Open'}</div>
            </div>
            <div>
              <div className="text-slate-500 text-xs mb-1">PnL</div>
              <div className={`font-mono font-bold ${trade.pnl !== null ? (trade.pnl >= 0 ? 'text-green-400' : 'text-red-400') : 'text-blue-400'}`}>
                {trade.pnl !== null ? `${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(2)}` : 'OPEN'}
              </div>
              {trade.pnl_pct !== null && (
                <div className={`text-[10px] ${trade.pnl_pct >= 0 ? 'text-green-400/70' : 'text-red-400/70'}`}>
                  {trade.pnl_pct >= 0 ? '+' : ''}{trade.pnl_pct.toFixed(2)}%
                </div>
              )}
            </div>
          </div>

          <div className="grid grid-cols-4 gap-3 mt-3 text-xs">
            <div className="text-center p-2 rounded bg-slate-800/50">
              <div className="text-slate-500 mb-0.5">SL</div>
              <div className="font-mono text-red-400">${trade.stop_price?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
            </div>
            <div className="text-center p-2 rounded bg-slate-800/50">
              <div className="text-slate-500 mb-0.5">TP</div>
              <div className="font-mono text-green-400">${trade.tp_price?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
            </div>
            <div className="text-center p-2 rounded bg-slate-800/50">
              <div className="text-slate-500 mb-0.5">Confidence</div>
              <div className="font-mono text-blue-400">{(trade.confidence * 100).toFixed(1)}%</div>
            </div>
            <div className="text-center p-2 rounded bg-slate-800/50">
              <div className="text-slate-500 mb-0.5">Regime</div>
              <div className="font-mono text-slate-200">{trade.regime}</div>
            </div>
          </div>

          {trade.exit_reason && (
            <div className="mt-2 flex items-center gap-2">
              <span className="text-slate-500 text-xs">Exit:</span>
              <span className={`px-2 py-0.5 rounded text-xs font-bold ${
                trade.exit_reason === 'TP' ? 'bg-green-500/20 text-green-400' :
                trade.exit_reason === 'SL' ? 'bg-red-500/20 text-red-400' :
                'bg-yellow-500/20 text-yellow-400'
              }`}>{trade.exit_reason}</span>
              {trade.bars_held !== null && <span className="text-slate-500 text-xs">after {trade.bars_held} bars ({trade.risk_pct?.toFixed(1)}% risk)</span>}
            </div>
          )}
        </div>

        {/* Feature Reasoning */}
        <div className="p-4 overflow-y-auto max-h-[40vh]">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Trade Reasoning (ML Features)</h3>
          {Object.keys(features).length > 0 ? (
            <div className="space-y-1">
              {Object.entries(features).map(([key, val]) => {
                const label = FEATURE_LABELS[key] || key.replace(/_/g, ' ');
                const { text, color } = interpretValue(key, val);
                return (
                  <div key={key} className="flex justify-between items-center py-0.5 border-b border-slate-800/30">
                    <span className="text-slate-400 text-xs">{label}</span>
                    <span className={`font-mono text-xs ${color}`}>{text}</span>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-slate-600 text-sm">No feature data available</div>
          )}
        </div>
      </div>
    </div>
  );
}
