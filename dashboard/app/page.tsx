'use client';
import { API } from '@/lib/api';

import { useState, useEffect, useCallback, useRef } from 'react';
import { BarChart3, X, Settings, Layers } from 'lucide-react';
import Chart from '@/components/Chart';
import Sidebar from '@/components/Sidebar';
import CoinList from '@/components/CoinList';
import SignalPanel from '@/components/SignalPanel';
import ScoreGauge from '@/components/ScoreGauge';
import NumerologyReadout from '@/components/NumerologyReadout';
import IndicatorSelector, { INDICATOR_CATALOG } from '@/components/IndicatorSelector';
import IndicatorSettings from '@/components/IndicatorSettings';
import DrawingToolbar from '@/components/DrawingToolbar';
import type { IndicatorConfig } from '@/components/IndicatorSettings';
import MLTrading from '@/components/MLTrading';
import PaperTrading from '@/components/PaperTrading';
import FeatureWeights from '@/components/FeatureWeights';
import FutureSignals from '@/components/FutureSignals';
import TradeLog from '@/components/TradeLog';
import type { DrawingTool } from '@/components/DrawingToolbar';
import type { OverlayType, Timeframe, UnifiedScore } from '@/types';

const OVERLAY_DEFAULTS: Record<OverlayType, boolean> = {
  lunar: true,
  caution: true,
  pump: true,
  ritual: true,
  tweet: false,
  wyckoff: false,
  elliott: false,
  gann: false,
};

export default function Dashboard() {
  const [symbol, setSymbol] = useState('BTC/USDT');
  const [timeframe, setTimeframe] = useState<Timeframe>('1d');
  const [overlays, setOverlays] = useState(OVERLAY_DEFAULTS);
  const [score, setScore] = useState<UnifiedScore | null>(null);
  const [btcPrice, setBtcPrice] = useState<{ price: number; change24h: number } | null>(null);
  const [signals, setSignals] = useState<any>(null);

  // Indicator state
  const [showIndicatorModal, setShowIndicatorModal] = useState(false);
  const [activeIndicators, setActiveIndicators] = useState<string[]>([]);
  const [indicatorConfigs, setIndicatorConfigs] = useState<Record<string, IndicatorConfig>>({});
  const [settingsIndicator, setSettingsIndicator] = useState<string | null>(null);

  // Prediction overlay state
  const [showPredictions, setShowPredictions] = useState(false);

  // Feature weights modal
  const [showFeatureWeights, setShowFeatureWeights] = useState(false);

  // Drawing state
  const [drawingTool, setDrawingTool] = useState<DrawingTool>(null);
  // Matrix modal (replaces left sidebar)
  const [showMatrixModal, setShowMatrixModal] = useState(false);
  // Future signals modal
  const [showFutureSignals, setShowFutureSignals] = useState(false);
  // Trading view mode toggle
  const [tradingView, setTradingView] = useState<'ml' | 'paper'>('ml');
  const chartContainerRef = useRef<HTMLDivElement>(null);

  const today = new Date().toISOString().split('T')[0];

  // Fetch unified score
  useEffect(() => {
    fetch(`${API}/api/score?date=${today}`)
      .then(r => r.json())
      .then(data => setScore(data))
      .catch(console.error);
  }, [today]);

  // Fetch signals for today
  useEffect(() => {
    fetch(`${API}/api/signals?date=${today}`)
      .then(r => r.json())
      .then(data => setSignals(data))
      .catch(console.error);
  }, [today]);

  // Fetch live BTC price directly from Bitget client-side every 5 seconds
  useEffect(() => {
    let prevClose = 0;
    // Get prev close from DB for 24h change
    fetch(`${API}/api/tick?symbol=BTC/USDT&timeframe=1d`)
      .then(r => r.json())
      .then(d => { if (d.close) prevClose = d.close; })
      .catch(() => {});

    const fetchBtcPrice = () => {
      fetch('https://api.bitget.com/api/v2/mix/market/ticker?productType=USDT-FUTURES&symbol=BTCUSDT')
        .then(r => r.json())
        .then(resp => {
          const data = resp.data?.[0] || resp.data || {};
          const price = parseFloat(data.lastPr || data.last || '0');
          if (price > 0) {
            const change24h = prevClose > 0 ? ((price - prevClose) / prevClose) * 100 : 0;
            setBtcPrice({ price, change24h: Math.round(change24h * 100) / 100 });
          }
        })
        .catch(() => {
          // Fallback to server API
          fetch(`${API}/api/btc-price`)
            .then(r => r.json())
            .then(data => {
              if (data.price != null) setBtcPrice({ price: data.price, change24h: data.change24h ?? 0 });
            })
            .catch(() => {});
        });
    };

    fetchBtcPrice();
    const interval = setInterval(fetchBtcPrice, 5_000);
    return () => clearInterval(interval);
  }, []);

  const toggleOverlay = useCallback((type: OverlayType) => {
    setOverlays(prev => ({ ...prev, [type]: !prev[type] }));
  }, []);

  const toggleIndicator = useCallback((id: string) => {
    setActiveIndicators(prev =>
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    );
  }, []);

  const removeIndicator = useCallback((id: string) => {
    setActiveIndicators(prev => prev.filter(i => i !== id));
    setIndicatorConfigs(prev => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
  }, []);

  const applyIndicatorConfig = useCallback((config: IndicatorConfig) => {
    setIndicatorConfigs(prev => ({ ...prev, [config.id]: config }));
  }, []);

  const handleDeleteAllDrawings = useCallback(() => {
    // Access the delete function exposed by Chart via its container div
    const el = chartContainerRef.current?.querySelector('.w-full.h-full');
    if (el && (el as any).__deleteAllDrawings) {
      (el as any).__deleteAllDrawings();
    }
    setDrawingTool(null);
  }, []);

  const threatClass = score
    ? `threat-${score.threat_level.toLowerCase()}`
    : 'threat-low';

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      {/* ===== TOP BAR ===== */}
      <header className="glass-panel-flat border-b border-panel-border flex items-center justify-between px-4 py-2 h-14 shrink-0 z-50">
        <div className="flex items-center gap-4">
          {/* Logo */}
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded bg-gradient-to-br from-esoteric to-bullish flex items-center justify-center text-white font-bold text-sm">
              22
            </div>
            <span className="font-semibold text-sm tracking-wider text-slate-300 hidden sm:block">
              SAVAGE22 TERMINAL
            </span>
          </div>

          {/* BTC Price */}
          <div className="flex items-center gap-2 ml-4">
            <span className="text-xs text-slate-500">BTC</span>
            <span className="num-display text-lg font-semibold text-white">
              {btcPrice ? `$${btcPrice.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '---'}
            </span>
            {btcPrice && (
              <span className={`num-display text-xs font-medium ${btcPrice.change24h >= 0 ? 'ticker-positive' : 'ticker-negative'}`}>
                {btcPrice.change24h >= 0 ? '+' : ''}{btcPrice.change24h.toFixed(2)}%
              </span>
            )}
          </div>
        </div>

        {/* Center: Warnings (score meter removed — redundant with Signal Convergence) */}
        <div className="flex items-center gap-6">

          {/* Warnings */}
          <div className="flex items-center gap-3">
            {score?.inversion_warning && (
              <div className="flex items-center gap-1 px-2 py-1 rounded bg-warning/10 border border-warning/30">
                <svg className="w-3.5 h-3.5 text-warning" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                <span className="text-xs font-medium text-warning">INVERSION</span>
              </div>
            )}
            {score?.phase_shift && (
              <div className="flex items-center gap-1 px-2 py-1 rounded bg-esoteric/10 border border-esoteric/30">
                <svg className="w-3.5 h-3.5 text-esoteric" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span className="text-xs font-medium text-esoteric">PHASE SHIFT</span>
              </div>
            )}
            <div className={`flex items-center gap-1 px-2 py-1 rounded ${
              score?.threat_level === 'CRITICAL' ? 'bg-bearish/10 border border-bearish/30' :
              score?.threat_level === 'HIGH' ? 'bg-orange-500/10 border border-orange-500/30' :
              score?.threat_level === 'MODERATE' ? 'bg-warning/10 border border-warning/30' :
              'bg-green-500/10 border border-green-500/30'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                score?.threat_level === 'CRITICAL' ? 'bg-bearish animate-pulse' :
                score?.threat_level === 'HIGH' ? 'bg-orange-500' :
                score?.threat_level === 'MODERATE' ? 'bg-warning' :
                'bg-green-500'
              }`} />
              <span className={`text-xs font-bold ${threatClass}`}>
                {score?.threat_level || 'LOW'}
              </span>
            </div>
          </div>
        </div>

        {/* Right: Date */}
        <div className="text-xs text-slate-500 num-display">
          {today}
        </div>
      </header>

      {/* ===== MAIN CONTENT ===== */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left sidebar removed — content moved to Matrix modal (triggered by button in toolbar) */}

        {/* Matrix Modal (replaces left sidebar) */}
        {showMatrixModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center" onClick={() => setShowMatrixModal(false)}>
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" />
            <div className="relative bg-[#12121f] border border-slate-700/50 rounded-xl shadow-2xl w-[400px] max-h-[80vh] overflow-hidden" onClick={e => e.stopPropagation()}>
              <div className="flex items-center justify-between p-3 border-b border-slate-700/50">
                <span className="font-bold text-slate-200 text-sm">Matrix Decoder</span>
                <button onClick={() => setShowMatrixModal(false)} className="text-slate-500 hover:text-slate-300 text-lg font-bold w-7 h-7 flex items-center justify-center rounded hover:bg-slate-700/50">&#x2715;</button>
              </div>
              <div className="overflow-y-auto max-h-[70vh]">
                <Sidebar overlays={overlays} onToggle={toggleOverlay} signals={signals} />
                <div className="border-t border-slate-700/50">
                  <NumerologyReadout />
                </div>
                <div className="border-t border-slate-700/50">
                  <SignalPanel score={score} signals={signals} />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Center: Chart + Bottom */}
        <main className="flex-1 flex flex-col overflow-hidden">
          {/* Timeframe + Time Range + Indicators bar */}
          <div className="flex items-center gap-2 px-3 py-2 glass-panel-flat border-b border-panel-border shrink-0">
            {/* Matrix decoder button */}
            <button
              onClick={() => setShowMatrixModal(true)}
              className="flex items-center gap-1 px-2 py-1 text-xs rounded-md font-medium transition-all mr-2 bg-esoteric/10 text-esoteric border border-esoteric/30 hover:bg-esoteric/20"
              title="Matrix Decoder: Numerology, Signals, Overlays"
            >
              <span className="text-sm">22</span>
              <span>Matrix</span>
            </button>

            <div className="flex items-center gap-1 mr-4">
              <span className="text-xs text-slate-500 mr-1">TF</span>
              {(['1m', '5m', '15m', '1h', '4h', '1d', '1w'] as Timeframe[]).map(tf => (
                <button
                  key={tf}
                  onClick={() => setTimeframe(tf)}
                  className={`px-2 py-0.5 text-xs rounded font-mono transition-all ${
                    timeframe === tf
                      ? 'bg-bullish/20 text-bullish border border-bullish/40'
                      : 'text-slate-500 hover:text-slate-300 hover:bg-white/5 border border-transparent'
                  }`}
                >
                  {tf}
                </button>
              ))}
            </div>

            <div className="w-px h-5 bg-panel-border ml-2" />

            {/* Indicators button */}
            <button
              onClick={() => setShowIndicatorModal(true)}
              className={`flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-md font-medium transition-all ml-2 ${
                activeIndicators.length > 0
                  ? 'bg-blue-500/15 text-blue-400 border border-blue-500/30'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-white/5 border border-transparent'
              }`}
            >
              <BarChart3 size={14} />
              <span>Indicators</span>
              {activeIndicators.length > 0 && (
                <span className="ml-0.5 px-1.5 py-0 text-[10px] font-bold rounded-full bg-blue-500/20 text-blue-300">
                  {activeIndicators.length}
                </span>
              )}
            </button>

            {/* Predictions toggle */}
            <button
              onClick={() => setShowPredictions(p => !p)}
              className={`flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-md font-medium transition-all ml-1 ${
                showPredictions
                  ? 'bg-purple-500/15 text-purple-400 border border-purple-500/30'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-white/5 border border-transparent'
              }`}
              title="Show ML price predictions as ghost candles"
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
              <span>Predictions</span>
            </button>

            {/* ML Features button */}
            <button
              onClick={() => setShowFeatureWeights(true)}
              className="flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-md font-medium transition-all ml-1 text-slate-500 hover:text-slate-300 hover:bg-white/5 border border-transparent"
              title="ML feature weights by category"
            >
              <Layers size={14} />
              <span>Features</span>
            </button>

            {/* Future Signals button */}
            <button
              onClick={() => setShowFutureSignals(true)}
              className={`flex items-center gap-1.5 px-2.5 py-1 text-xs rounded-md font-medium transition-all ml-1 ${
                showFutureSignals
                  ? 'bg-esoteric/15 text-esoteric border border-esoteric/30'
                  : 'text-slate-500 hover:text-slate-300 hover:bg-white/5 border border-transparent'
              }`}
              title="Upcoming signals: FOMC, eclipses, retrogrades, holidays..."
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              <span>Signals</span>
            </button>

            {/* Active indicator pills */}
            <div className="flex items-center gap-1 ml-1 overflow-x-auto max-w-[300px]">
              {activeIndicators.map(id => {
                const def = INDICATOR_CATALOG.find(c => c.id === id);
                if (!def) return null;
                const config = indicatorConfigs[id];
                return (
                  <div
                    key={id}
                    className="flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono shrink-0 group cursor-pointer"
                    style={{
                      background: 'rgba(50, 50, 80, 0.2)',
                      border: `1px solid ${config?.color || def.color}40`,
                    }}
                  >
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: config?.color || def.color }}
                    />
                    <span className="text-slate-300">{def.shortName}</span>
                    <button
                      onClick={(e) => { e.stopPropagation(); setSettingsIndicator(id); }}
                      className="opacity-0 group-hover:opacity-100 transition-opacity text-slate-500 hover:text-slate-300"
                    >
                      <Settings size={10} />
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); removeIndicator(id); }}
                      className="opacity-0 group-hover:opacity-100 transition-opacity text-slate-500 hover:text-red-400"
                    >
                      <X size={10} />
                    </button>
                  </div>
                );
              })}
            </div>

            <div className="flex-1" />
            <span className="text-xs text-slate-600 num-display">{symbol}</span>
          </div>

          {/* Chart area with drawing toolbar */}
          <div className="flex-1 overflow-hidden relative min-h-0" ref={chartContainerRef} style={{ minHeight: '300px' }}>
            {/* Drawing toolbar on left */}
            <DrawingToolbar
              activeTool={drawingTool}
              onSelectTool={setDrawingTool}
              onDeleteAll={handleDeleteAllDrawings}
            />

            {/* Indicator settings panel */}
            {settingsIndicator && (
              <IndicatorSettings
                indicatorId={settingsIndicator}
                config={indicatorConfigs[settingsIndicator] || null}
                onApply={applyIndicatorConfig}
                onRemove={removeIndicator}
                onClose={() => setSettingsIndicator(null)}
              />
            )}

            <Chart
              key={`chart-${symbol}-${timeframe}`}
              symbol={symbol}
              timeframe={timeframe}
              overlays={overlays}
              showPredictions={showPredictions}
              activeIndicators={activeIndicators}
              indicatorConfigs={indicatorConfigs}
              drawingTool={drawingTool}
              onDrawingComplete={() => setDrawingTool(null)}
            />
          </div>

          {/* Bottom: Trade Log */}
          <div className="h-[160px] shrink-0 glass-panel-flat border-t border-panel-border overflow-hidden">
            <TradeLog />
          </div>
        </main>

        {/* Right Sidebar — Trading View Toggle + Markets */}
        <aside className="w-[280px] shrink-0 glass-panel-flat border-l border-panel-border flex flex-col overflow-hidden">
          {/* View Mode Toggle */}
          <div className="px-2 py-1.5 border-b border-panel-border shrink-0">
            <div className="flex items-center bg-slate-800/60 rounded-lg p-0.5">
              <button
                onClick={() => setTradingView('ml')}
                className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-[10px] font-semibold tracking-wider transition-all ${
                  tradingView === 'ml'
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30 shadow-sm shadow-blue-500/10'
                    : 'text-slate-500 hover:text-slate-400 border border-transparent'
                }`}
              >
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                ML TRADING
              </button>
              <button
                onClick={() => setTradingView('paper')}
                className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-[10px] font-semibold tracking-wider transition-all ${
                  tradingView === 'paper'
                    ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30 shadow-sm shadow-yellow-500/10'
                    : 'text-slate-500 hover:text-slate-400 border border-transparent'
                }`}
              >
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
                PAPER
              </button>
            </div>
          </div>

          {/* Trading View Content */}
          <div className="h-[55%] overflow-y-auto border-b border-panel-border">
            {tradingView === 'ml' ? (
              <MLTrading btcPrice={btcPrice?.price} />
            ) : (
              <PaperTrading btcPrice={btcPrice?.price} />
            )}
          </div>
          <div className="flex-1 overflow-y-auto">
            <CoinList
              activeSymbol={symbol}
              onSelect={(s) => setSymbol(s)}
            />
          </div>
        </aside>
      </div>

      {/* ===== Indicator Selector Modal ===== */}
      <IndicatorSelector
        isOpen={showIndicatorModal}
        onClose={() => setShowIndicatorModal(false)}
        activeIndicators={activeIndicators}
        onToggleIndicator={toggleIndicator}
      />

      {/* ===== Feature Weights Modal ===== */}
      <FeatureWeights
        isOpen={showFeatureWeights}
        onClose={() => setShowFeatureWeights(false)}
        timeframe={timeframe}
      />

      {/* ===== Future Signals Modal ===== */}
      {showFutureSignals && (
        <FutureSignals onClose={() => setShowFutureSignals(false)} />
      )}
    </div>
  );
}
