'use client';
import { API } from '@/lib/api';

import { useEffect, useState } from 'react';

interface CoinListProps {
  activeSymbol: string;
  onSelect: (symbol: string) => void;
}

interface CoinData {
  symbol: string;
  name: string;
  shortName: string;
  price: number | null;
  change24h: number;
  sparkline: number[];
}

function MiniSparkline({ data, positive }: { data: number[]; positive: boolean }) {
  if (data.length < 2) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const h = 20;
  const w = 50;

  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w;
    const y = h - ((v - min) / range) * h;
    return `${x},${y}`;
  }).join(' ');

  const color = positive ? '#22c55e' : '#ef4444';

  return (
    <svg width={w} height={h} className="opacity-60">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function CoinList({ activeSymbol, onSelect }: CoinListProps) {
  const [coins, setCoins] = useState<CoinData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API}/api/coins`)
      .then(r => r.json())
      .then(data => {
        if (data.coins) {
          setCoins(data.coins);
        }
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to fetch coins:', err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="p-2">
        <div className="flex items-center gap-2 px-2 py-2">
          <svg className="w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v12m-3-2.818l.879.659c1.171.879 3.07.879 4.242 0 1.172-.879 1.172-2.303 0-3.182C13.536 12.219 12.768 12 12 12c-.725 0-1.45-.22-2.003-.659-1.106-.879-1.106-2.303 0-3.182s2.9-.879 4.006 0l.415.33M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Markets</h3>
        </div>
        <div className="text-xs text-slate-600 text-center py-4">Loading...</div>
      </div>
    );
  }

  return (
    <div className="p-2">
      <div className="flex items-center gap-2 px-2 py-2">
        <svg className="w-4 h-4 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v12m-3-2.818l.879.659c1.171.879 3.07.879 4.242 0 1.172-.879 1.172-2.303 0-3.182C13.536 12.219 12.768 12 12 12c-.725 0-1.45-.22-2.003-.659-1.106-.879-1.106-2.303 0-3.182s2.9-.879 4.006 0l.415.33M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Markets</h3>
      </div>

      <div className="space-y-0.5">
        {coins.map(coin => {
          const isActive = activeSymbol === coin.symbol;
          const hasData = coin.price !== null;

          return (
            <button
              key={coin.symbol}
              onClick={() => onSelect(coin.symbol)}
              className={`coin-item w-full flex items-center justify-between px-2 py-2 rounded text-left ${
                isActive ? 'active' : ''
              } ${!hasData ? 'opacity-40' : ''}`}
            >
              <div className="flex items-center gap-2 min-w-0">
                {/* Coin icon placeholder */}
                <div className={`w-6 h-6 rounded-full flex items-center justify-center text-[9px] font-bold shrink-0 ${
                  isActive ? 'bg-bullish/20 text-bullish' : 'bg-slate-800 text-slate-500'
                }`}>
                  {coin.shortName.slice(0, 2)}
                </div>

                <div className="min-w-0">
                  <div className={`text-xs font-medium truncate ${isActive ? 'text-white' : 'text-slate-400'}`}>
                    {coin.shortName}
                  </div>
                  <div className="text-[9px] text-slate-600 truncate">{coin.name}</div>
                </div>
              </div>

              <div className="flex items-center gap-2 shrink-0">
                {hasData && (
                  <>
                    {coin.sparkline.length > 1 && (
                      <MiniSparkline data={coin.sparkline} positive={coin.change24h >= 0} />
                    )}
                    <div className="text-right">
                      <div className="text-[10px] num-display text-slate-400">
                        ${coin.price! >= 1000
                          ? Math.round(coin.price!).toLocaleString()
                          : coin.price! >= 1
                          ? coin.price!.toFixed(2)
                          : coin.price!.toFixed(4)
                        }
                      </div>
                      <div className={`text-[9px] num-display ${coin.change24h >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                        {coin.change24h >= 0 ? '+' : ''}{coin.change24h.toFixed(1)}%
                      </div>
                    </div>
                  </>
                )}
                {!hasData && (
                  <span className="text-[9px] text-slate-600">No data</span>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
