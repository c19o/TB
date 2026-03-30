'use client';
import { API } from '@/lib/api';

import { useEffect, useRef, useState, useCallback } from 'react';
import type { OverlayType, Timeframe } from '@/types';
import type { IndicatorConfig } from './IndicatorSettings';
import type { DrawingTool } from './DrawingToolbar';
import { INDICATOR_CATALOG } from './IndicatorSelector';

interface PredictionData {
  timestamp: string;
  timeframe: string;
  direction: 'LONG' | 'SHORT';
  confidence: number;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
}

interface GhostCandle {
  x: number;
  y: number;
  width: number;
  height: number;
  wickTop: number;
  wickBottom: number;
  wickX: number;
  color: string;
}

interface PredictionOverlay {
  ghostCandles: GhostCandle[];
  entryY: number;
  slY: number;
  tpY: number;
  confidenceText: string;
  confidenceX: number;
  confidenceY: number;
  direction: 'LONG' | 'SHORT';
  chartWidth: number;
  startX: number;
}

interface ChartProps {
  symbol: string;
  timeframe: Timeframe;
  overlays: Record<OverlayType, boolean>;
  showPredictions?: boolean;
  activeIndicators?: string[];
  indicatorConfigs?: Record<string, IndicatorConfig>;
  drawingTool?: DrawingTool;
  onDrawingComplete?: () => void;
}

const OVERLAY_COLORS: Record<string, string> = {
  caution: '#ef4444',
  pump: '#22c55e',
  lunar: '#a78bfa',
  ritual: '#8b5cf6',
  tweet: '#06b6d4',
  wyckoff: '#f59e0b',
  elliott: '#ec4899',
  gann: '#14b8a6',
};

// Map our drawing tool names to KLineChart overlay names
const DRAWING_TOOL_TO_OVERLAY: Record<string, string> = {
  trendline: 'segment',
  hline: 'horizontalStraightLine',
  vline: 'verticalStraightLine',
  fib: 'fibonacciLine',
  rectangle: 'rect',
  circle: 'circle',
  arrow: 'arrow',
  measure: 'segment',
  text: 'simpleAnnotation',
  freehand: 'segment',
};

// Map our indicator IDs to KLineChart indicator names
function mapIndicatorToKLine(id: string): { name: string; calcParams?: number[]; paneId?: string } | null {
  const base = id.split('_')[0];
  const suffix = id.split('_')[1];

  switch (base) {
    case 'ema': return { name: 'EMA', calcParams: [parseInt(suffix) || 9], paneId: 'candle_pane' };
    case 'sma': return { name: 'SMA', calcParams: [parseInt(suffix) || 50], paneId: 'candle_pane' };
    case 'rsi': return { name: 'RSI', calcParams: [14] };
    case 'macd': return { name: 'MACD', calcParams: [12, 26, 9] };
    case 'bollinger': return { name: 'BOLL', calcParams: [20, 2], paneId: 'candle_pane' };
    case 'stochastic': return { name: 'KDJ', calcParams: [14, 3, 3] };
    case 'atr': return { name: 'ATR', calcParams: [14] };
    case 'obv': return { name: 'OBV' };
    case 'cci': return { name: 'CCI', calcParams: [20] };
    case 'vwap': return { name: 'AVP', paneId: 'candle_pane' };
    case 'supertrend': return { name: 'SAR', paneId: 'candle_pane' };
    case 'ichimoku': return { name: 'MA', calcParams: [9, 26, 52], paneId: 'candle_pane' };
    default: return null;
  }
}

interface TooltipData {
  x: number;
  y: number;
  signals: string[];
  date: string;
  direction: string;
  types: string[];
}

interface MarkerInfo {
  timestamp: number;
  value: number;
  text: string;
  position: string;
  types: string[];
  details: Array<{ type: string; label: string; direction: string; color: string }>;
}

export default function Chart({
  symbol,
  timeframe,
  overlays,
  showPredictions = false,
  activeIndicators = [],
  indicatorConfigs = {},
  drawingTool = null,
  onDrawingComplete,
}: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const markerDataRef = useRef<MarkerInfo[]>([]);
  const hoverTimerRef = useRef<NodeJS.Timeout | null>(null);
  const activeIndicatorPanesRef = useRef<Map<string, string>>(new Map());
  const overlayIdsRef = useRef<string[]>([]);
  const drawingToolRef = useRef<DrawingTool>(drawingTool);
  const candleDataRef = useRef<any[]>([]);
  const klinechartsModuleRef = useRef<any>(null);
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);
  const [tradeMarkers, setTradeMarkers] = useState<Array<{
    id: number; entryX: number; entryY: number; exitX: number | null; exitY: number | null;
    slY: number | null; tpY: number | null; slX2: number; tpX2: number;
    color: string; label: string; isOpen: boolean; pnl: number | null; direction: string;
  }>>([]);
  const [signalMarkers, setSignalMarkers] = useState<Array<{ key: string; x: number; y: number; color: string; label: string; above: boolean }>>([]);
  const [cursorStyle, setCursorStyle] = useState('default');
  const [measureInfo, setMeasureInfo] = useState<{ pct: string; diff: string; color: string } | null>(null);
  const [staleWarning, setStaleWarning] = useState<string | null>(null);
  const measureStartRef = useRef<{ timestamp: number; value: number } | null>(null);
  const [predictionData, setPredictionData] = useState<PredictionData | null>(null);
  const [predictionOverlay, setPredictionOverlay] = useState<PredictionOverlay | null>(null);

  // Fetch prediction data every 30s when predictions are enabled
  useEffect(() => {
    if (!showPredictions) {
      setPredictionData(null);
      setPredictionOverlay(null);
      return;
    }

    const fetchPrediction = () => {
      fetch(`${API}/api/prediction`)
        .then(r => { if (!r.ok) throw new Error(`${r.status}`); return r.json(); })
        .then((data: PredictionData) => {
          if (data && data.entry_price && data.direction) {
            setPredictionData(data);
          } else {
            setPredictionData(null);
          }
        })
        .catch(() => setPredictionData(null));
    };

    fetchPrediction();
    const interval = setInterval(fetchPrediction, 30000);
    return () => clearInterval(interval);
  }, [showPredictions]);

  // Compute ghost candle positions when prediction data or chart state changes
  useEffect(() => {
    if (!showPredictions || !predictionData) {
      setPredictionOverlay(null);
      return;
    }

    // Only show if prediction timeframe matches current chart timeframe
    if (predictionData.timeframe !== timeframe) {
      setPredictionOverlay(null);
      return;
    }

    const chart = chartRef.current;
    const container = containerRef.current;
    if (!chart || !container) return;

    let cancelled = false;

    const computeOverlay = () => {
      if (cancelled || !chartRef.current) return;
      const dataList = candleDataRef.current;
      const visibleRange = chart.getVisibleRange?.();
      if (!visibleRange || !dataList || dataList.length === 0) return;

      const rect = container.getBoundingClientRect();
      const chartWidth = rect.width - 60;
      const chartHeight = rect.height - 30;

      const visFrom = visibleRange.from || 0;
      const visTo = visibleRange.to || dataList.length;
      const visCount = visTo - visFrom;
      if (visCount <= 0) return;

      // Price range from visible candles
      let minPrice = Infinity, maxPrice = -Infinity;
      for (let i = visFrom; i < Math.min(visTo, dataList.length); i++) {
        const c = dataList[i];
        if (c.low < minPrice) minPrice = c.low;
        if (c.high > maxPrice) maxPrice = c.high;
      }
      // Extend price range to include prediction levels
      minPrice = Math.min(minPrice, predictionData.stop_loss, predictionData.take_profit, predictionData.entry_price);
      maxPrice = Math.max(maxPrice, predictionData.stop_loss, predictionData.take_profit, predictionData.entry_price);
      const priceRange = maxPrice - minPrice;
      if (priceRange <= 0) return;

      const priceToY = (price: number): number => 10 + ((maxPrice - price) / priceRange) * (chartHeight - 20);

      // Last candle position
      const lastIdx = dataList.length - 1;
      const lastCandle = dataList[lastIdx];
      const barWidth = chartWidth / visCount;
      const lastBarX = 40 + ((lastIdx - visFrom) / visCount) * chartWidth;

      // Generate 4 ghost candles projecting from last real candle
      const numGhost = 4;
      const isLong = predictionData.direction === 'LONG';
      const startPrice = lastCandle.close;
      const targetPrice = predictionData.take_profit;
      const priceStep = (targetPrice - startPrice) / numGhost;
      const color = isLong ? 'rgba(34, 197, 94, 0.4)' : 'rgba(239, 68, 68, 0.4)';

      const ghostCandles: GhostCandle[] = [];
      for (let i = 0; i < numGhost; i++) {
        const open = startPrice + priceStep * i;
        const close = startPrice + priceStep * (i + 1);
        // Add some wick variation
        const wickExtend = Math.abs(priceStep) * 0.3;
        const high = Math.max(open, close) + wickExtend;
        const low = Math.min(open, close) - wickExtend;

        const candleX = lastBarX + barWidth * (i + 1);
        const bodyTop = priceToY(Math.max(open, close));
        const bodyBottom = priceToY(Math.min(open, close));
        const bodyHeight = Math.max(bodyBottom - bodyTop, 2);

        ghostCandles.push({
          x: candleX - barWidth * 0.3,
          y: bodyTop,
          width: barWidth * 0.6,
          height: bodyHeight,
          wickTop: priceToY(high),
          wickBottom: priceToY(low),
          wickX: candleX,
          color,
        });
      }

      const entryY = priceToY(predictionData.entry_price);
      const slY = priceToY(predictionData.stop_loss);
      const tpY = priceToY(predictionData.take_profit);

      // Position confidence text near the middle ghost candle
      const midGhost = ghostCandles[Math.floor(numGhost / 2)];
      const confidenceX = midGhost ? midGhost.x + midGhost.width + 4 : lastBarX + barWidth * 3;
      const confidenceY = midGhost ? midGhost.y - 16 : entryY - 20;

      setPredictionOverlay({
        ghostCandles,
        entryY,
        slY,
        tpY,
        confidenceText: `${(predictionData.confidence * 100).toFixed(0)}%`,
        confidenceX,
        confidenceY,
        direction: predictionData.direction,
        chartWidth: chartWidth + 40,
        startX: lastBarX,
      });
    };

    // Compute on mount and periodically (positions shift on scroll/zoom)
    const timer = setTimeout(computeOverlay, 500);
    const interval = setInterval(computeOverlay, 4000);
    return () => { cancelled = true; clearTimeout(timer); clearInterval(interval); };
  }, [showPredictions, predictionData, timeframe]);

  // Keep drawing tool ref synced
  useEffect(() => {
    drawingToolRef.current = drawingTool;
    setCursorStyle(drawingTool && drawingTool !== 'select' && drawingTool !== 'delete' ? 'crosshair' : 'default');
    if (drawingTool && drawingTool !== 'select' && drawingTool !== 'delete' && drawingTool !== 'measure') {
      const chart = chartRef.current;
      if (chart) {
        const overlayName = DRAWING_TOOL_TO_OVERLAY[drawingTool];
        if (overlayName) {
          chart.createOverlay(overlayName);
        }
      }
    }
    // Reset measure tool state
    if (drawingTool !== 'measure') {
      measureStartRef.current = null;
      setMeasureInfo(null);
    }
  }, [drawingTool]);

  // Delete all drawings
  const deleteAllDrawings = useCallback(() => {
    const chart = chartRef.current;
    if (!chart) return;
    for (const id of overlayIdsRef.current) {
      try { chart.removeOverlay(id); } catch (_) {}
    }
    overlayIdsRef.current = [];
    // Also remove all overlays generically
    try { chart.removeOverlay(); } catch (_) {}
  }, []);

  useEffect(() => {
    if (containerRef.current) {
      (containerRef.current as any).__deleteAllDrawings = deleteAllDrawings;
    }
  }, [deleteAllDrawings]);

  // Initialize chart — delay until container has real dimensions
  // Re-runs on symbol OR timeframe change (dispose+reinit is the only reliable
  // way to switch periods in klinecharts v10-beta1; setPeriod doesn't visually refresh)
  useEffect(() => {
    if (!containerRef.current) return;

    let disposed = false;
    setStaleWarning(null); // clear stale warning on reinit

    const initChart = async () => {
      const klinecharts = klinechartsModuleRef.current || await import('klinecharts');
      klinechartsModuleRef.current = klinecharts; // cache for fast re-init on TF change
      const { init, dispose } = klinecharts;

      if (disposed || !containerRef.current) return;

      // Wait for container to have real dimensions
      const waitForSize = () => new Promise<void>((resolve) => {
        const check = () => {
          if (!containerRef.current) return;
          const rect = containerRef.current.getBoundingClientRect();
          if (rect.width > 50 && rect.height > 50) {
            resolve();
          } else {
            requestAnimationFrame(check);
          }
        };
        check();
      });
      await waitForSize();
      if (disposed || !containerRef.current) return;

      containerRef.current.innerHTML = '';

      const chart = init(containerRef.current, {
        styles: {
          grid: {
            show: true,
            horizontal: {
              show: true,
              size: 1,
              color: 'rgba(50, 50, 80, 0.12)',
              style: 'dashed' as any,
            },
            vertical: {
              show: true,
              size: 1,
              color: 'rgba(50, 50, 80, 0.12)',
              style: 'dashed' as any,
            },
          },
          candle: {
            type: 'candle_solid' as any,
            priceMark: {
              show: true,
              high: { show: true, color: '#64748b', textSize: 10, textFamily: "'JetBrains Mono', monospace" },
              low: { show: true, color: '#64748b', textSize: 10, textFamily: "'JetBrains Mono', monospace" },
              last: {
                show: true,
                upColor: '#22c55e',
                downColor: '#ef4444',
                noChangeColor: '#64748b',
                line: { show: true, style: 'dashed' as any, size: 1 },
                text: {
                  show: true,
                  size: 11,
                  family: "'JetBrains Mono', monospace",
                  paddingLeft: 4,
                  paddingRight: 4,
                  paddingTop: 2,
                  paddingBottom: 2,
                  borderRadius: 2,
                },
              },
            },
            bar: {
              upColor: '#22c55e',
              downColor: '#ef4444',
              noChangeColor: '#64748b',
              upBorderColor: '#22c55e',
              downBorderColor: '#ef4444',
              noChangeBorderColor: '#64748b',
              upWickColor: '#22c55e',
              downWickColor: '#ef4444',
              noChangeWickColor: '#64748b',
            },
            tooltip: {
              showRule: 'always' as any,
              showType: 'standard' as any,
              title: {
                show: true,
                size: 11,
                family: "'JetBrains Mono', monospace",
                color: '#64748b',
                weight: 'normal',
                marginLeft: 8,
                marginTop: 4,
                marginRight: 8,
                marginBottom: 0,
              },
              legend: {
                size: 11,
                family: "'JetBrains Mono', monospace",
                color: '#64748b',
                weight: 'normal',
                marginLeft: 8,
                marginTop: 2,
                marginRight: 8,
                marginBottom: 0,
              },
            },
          },
          indicator: {
            lastValueMark: { show: false },
            tooltip: {
              showRule: 'always' as any,
              showType: 'standard' as any,
              title: {
                show: true,
                size: 10,
                family: "'JetBrains Mono', monospace",
                color: '#64748b',
                weight: 'normal',
                marginLeft: 8,
                marginTop: 4,
                marginRight: 8,
                marginBottom: 0,
              },
              legend: {
                size: 10,
                family: "'JetBrains Mono', monospace",
                color: '#64748b',
                weight: 'normal',
                marginLeft: 8,
                marginTop: 2,
                marginRight: 8,
                marginBottom: 0,
              },
            },
          },
          xAxis: {
            show: true,
            size: 'auto' as any,
            axisLine: { show: true, color: 'rgba(50, 50, 80, 0.3)', size: 1 },
            tickLine: { show: true, size: 1, length: 3, color: 'rgba(50, 50, 80, 0.3)' },
            tickText: {
              show: true,
              color: '#64748b',
              size: 11,
              family: "'JetBrains Mono', monospace",
            },
          },
          yAxis: {
            show: true,
            size: 'auto' as any,
            axisLine: { show: true, color: 'rgba(50, 50, 80, 0.3)', size: 1 },
            tickLine: { show: true, size: 1, length: 3, color: 'rgba(50, 50, 80, 0.3)' },
            tickText: {
              show: true,
              color: '#64748b',
              size: 11,
              family: "'JetBrains Mono', monospace",
            },
          },
          separator: {
            size: 1,
            color: 'rgba(50, 50, 80, 0.3)',
          },
          crosshair: {
            show: true,
            horizontal: {
              show: true,
              line: { show: true, style: 'dashed' as any, size: 1, color: 'rgba(59, 130, 246, 0.6)' },
              text: {
                show: true,
                style: 'fill' as any,
                size: 11,
                family: "'JetBrains Mono', monospace",
                weight: 'normal',
                color: '#ffffff',
                backgroundColor: '#1a1a2e',
                borderStyle: 'solid' as any,
                borderDashedValue: [2],
                borderColor: 'rgba(59, 130, 246, 0.6)',
                borderSize: 1,
                borderRadius: 2,
                paddingLeft: 4,
                paddingRight: 4,
                paddingTop: 2,
                paddingBottom: 2,
              },
            },
            vertical: {
              show: true,
              line: { show: true, style: 'dashed' as any, size: 1, color: 'rgba(59, 130, 246, 0.6)' },
              text: {
                show: true,
                style: 'fill' as any,
                size: 11,
                family: "'JetBrains Mono', monospace",
                weight: 'normal',
                color: '#ffffff',
                backgroundColor: '#1a1a2e',
                borderStyle: 'solid' as any,
                borderDashedValue: [2],
                borderColor: 'rgba(59, 130, 246, 0.6)',
                borderSize: 1,
                borderRadius: 2,
                paddingLeft: 4,
                paddingRight: 4,
                paddingTop: 2,
                paddingBottom: 2,
              },
            },
          },
          overlay: {
            point: { color: '#3b82f6', borderColor: '#3b82f6', borderSize: 1, radius: 4, activeColor: '#60a5fa', activeBorderColor: '#60a5fa', activeBorderSize: 1, activeRadius: 6 },
            line: { style: 'solid' as any, color: '#3b82f6', size: 2, dashedValue: [2], smooth: false },
            rect: { style: 'fill' as any, color: 'rgba(59, 130, 246, 0.15)', borderColor: '#3b82f6', borderSize: 1, borderStyle: 'solid' as any, borderDashedValue: [2], borderRadius: 0 },
            text: { style: 'fill' as any, color: '#e2e8f0', size: 12, family: "'JetBrains Mono', monospace", weight: 'normal', borderStyle: 'solid' as any, borderDashedValue: [2], borderSize: 0, borderColor: 'transparent', borderRadius: 0, backgroundColor: 'transparent', paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0 },
          },
        },
      });

      if (!chart) return;

      // Trade markers rendered as HTML overlays (see tradeMarkers state)

      // Add volume indicator on the main pane with reduced height
      // Volume in separate sub-pane (not candle_pane — that pulls Y-axis to 0)
      chart.createIndicator('VOL', false, { height: 60 });

      // Fit 20% more candles by reducing bar width
      try { chart.setBarSpace(6); } catch(e) {} // default is ~8, 6 = 25% more candles

      chartRef.current = chart;
      activeIndicatorPanesRef.current.clear();
      overlayIdsRef.current = [];

      // Force resize after init
      setTimeout(() => { if (chart) chart.resize(); }, 100);

      // v10 beta1: getBars receives { type, from, to, count, callback }
      // type: 'initial' | 'backward' | 'forward'
      // For backward: fetch older data before earliest candle
      // callback(data, more) — more=true means keep requesting on scroll
      // Set symbol and period BEFORE data loader (per klinecharts v10 best practice)
      chart.setSymbol({ symbol: symbol, pricePrecision: 2, volumePrecision: 4 });
      // Set period based on current timeframe
      const periodMap: Record<string, { span: number; type: string }> = {
        '1m': { span: 1, type: 'minute' }, '5m': { span: 5, type: 'minute' },
        '15m': { span: 15, type: 'minute' }, '1h': { span: 1, type: 'hour' },
        '4h': { span: 4, type: 'hour' }, '1d': { span: 1, type: 'day' }, '1w': { span: 1, type: 'week' },
      };
      timeframeRef.current = timeframe; // ensure ref matches prop for this init
      chart.setPeriod((periodMap[timeframe] || { span: 1, type: 'day' }) as any);

      chart.setDataLoader({
        getBars: (params: any) => {
          // Derive TF from params.period (klinecharts passes current period)
          // This is the correct v10 approach — don't rely on React refs
          const period = params.period || {};
          const periodToTf = (p: any): string => {
            if (!p || !p.type) return timeframeRef.current; // fallback
            if (p.type === 'minute' && p.span === 1) return '1m';
            if (p.type === 'minute' && p.span === 5) return '5m';
            if (p.type === 'minute' && p.span === 15) return '15m';
            if (p.type === 'hour' && p.span === 1) return '1h';
            if (p.type === 'hour' && p.span === 4) return '4h';
            if (p.type === 'day') return '1d';
            if (p.type === 'week') return '1w';
            return timeframeRef.current;
          };
          const tf = periodToTf(period);
          const sym = params.symbol?.symbol || symbolRef.current;

          const tfMs: Record<string, number> = {
            '1m': 60000, '5m': 300000, '15m': 900000, '1h': 3600000,
            '4h': 14400000, '1d': 86400000, '1w': 604800000,
          };
          const barMs = tfMs[tf] || 86400000;
          const initialCount: Record<string, number> = {
            '1m': 2000, '5m': 2000, '15m': 5000, '1h': 5000,
            '4h': 5000, '1d': 5000, '1w': 1000,
          };
          const batchSize = 5000;

          const type = params.type || 'init';
          console.log('[Chart] getBars called, type:', type, 'tf:', tf, 'period:', JSON.stringify(period));

          let fromTs: number;
          let toTs: number;

          if (type === 'backward') {
            // Load older candles before the earliest currently loaded
            const earliest = candleDataRef.current.length > 0
              ? candleDataRef.current[0].timestamp
              : Date.now();
            toTs = earliest - 1; // just before earliest
            fromTs = toTs - batchSize * barMs;
          } else if (type === 'forward') {
            // Load newer candles after the latest
            const latest = candleDataRef.current.length > 0
              ? candleDataRef.current[candleDataRef.current.length - 1].timestamp
              : Date.now();
            fromTs = latest + 1;
            toTs = Date.now();
          } else {
            // Initial load: get ALL available history
            fromTs = 0;  // epoch start = get everything
            toTs = Date.now();
          }

          // Convert to seconds for API
          const fromSec = Math.floor(fromTs / 1000);
          const toSec = Math.floor(toTs / 1000);

          // Bitget granularity map for fallback when DB is empty
          const bitgetGran: Record<string, string> = {
            '1m': '1m', '5m': '5m', '15m': '15m', '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W',
          };

          const fetchLimit = (type === 'backward' || type === 'forward') ? batchSize : 0;  // 0 = unlimited for initial load
          fetch(`${API}/api/candles?symbol=${encodeURIComponent(sym)}&timeframe=${tf}&from=${fromSec}&to=${toSec}&limit=${fetchLimit}`)
            .then(r => { if (!r.ok || !r.headers.get('content-type')?.includes('json')) throw new Error(`candles: ${r.status}`); return r.json(); })
            .then(data => {
              if (!data.candles || data.candles.length === 0) {
                // DB has no data for this TF — fall back to Bitget API for initial load
                if (type === 'initial' || type === 'init') {
                  // Fallback to Bitget via our proxy (avoids CORS)
                  fetch(`${API}/api/bitget-candles?timeframe=${tf}&limit=200`)
                    .then(r2 => r2.json())
                    .then(resp => {
                      const raw = resp?.candles || resp?.data || resp;
                      if (!Array.isArray(raw) || raw.length === 0) { params.callback([], false); return; }
                      const klineData = raw.map((c: any) => ({
                        timestamp: c.timestamp || parseInt(c[0]),
                        open: c.open || parseFloat(c[1]),
                        high: c.high || parseFloat(c[2]),
                        low: c.low || parseFloat(c[3]),
                        close: c.close || parseFloat(c[4]),
                        volume: c.volume || parseFloat(c[5]) || 0,
                      })).sort((a: any, b: any) => a.timestamp - b.timestamp);
                      candleDataRef.current = klineData;
                      params.callback(klineData, false);
                      setStaleWarning(null); // fresh Bitget data, no staleness
                      console.log('[Chart] DB empty, loaded', klineData.length, 'candles from Bitget for', tf);
                    })
                    .catch(() => params.callback([], false));
                } else {
                  params.callback([], false);
                }
                return;
              }
              const seen = new Set<number>();
              const klineData = data.candles
                .filter((c: any) => { if (seen.has(c.time)) return false; seen.add(c.time); return true; })
                .sort((a: any, b: any) => a.time - b.time)
                .map((c: any) => ({
                  timestamp: c.time * 1000,
                  open: Number(c.open), high: Number(c.high),
                  low: Number(c.low), close: Number(c.close),
                  volume: Number(c.volume),
                }));

              if (type === 'backward') {
                // Prepend to existing data ref
                candleDataRef.current = [...klineData, ...candleDataRef.current];
              } else if (type === 'forward') {
                candleDataRef.current = [...candleDataRef.current, ...klineData];
              } else {
                candleDataRef.current = klineData;
              }

              // more=true if we got a full batch (likely more data exists)
              const hasMore = klineData.length >= batchSize * 0.8;
              params.callback(klineData, hasMore);
              console.log('[Chart] Loaded', klineData.length, 'candles (type:', type, ', more:', hasMore, ')');

              // Staleness check for 1m/5m — warn if last candle is >2x the TF interval old
              if ((type === 'init' || type === 'initial') && (tf === '1m' || tf === '5m') && klineData.length > 0) {
                const lastTs = klineData[klineData.length - 1].timestamp;
                const ageMs = Date.now() - lastTs;
                const staleThreshold = barMs * 2;
                if (ageMs > staleThreshold) {
                  const ageMins = Math.round(ageMs / 60000);
                  console.warn(`[Chart] ${tf} data is stale: last candle is ${ageMins}m old (threshold: ${Math.round(staleThreshold / 60000)}m)`);
                  setStaleWarning(`${tf} data is ${ageMins}m behind — refetching from Bitget`);
                  // Auto-refetch from Bitget to fill the gap
                  fetch(`${API}/api/bitget-candles?timeframe=${tf}&limit=200`)
                    .then(r2 => r2.json())
                    .then(resp => {
                      const raw = resp?.candles || resp?.data || resp;
                      if (!Array.isArray(raw) || raw.length === 0 || !chartRef.current) return;
                      const sorted = [...raw].sort((a: any, b: any) => (a.timestamp || parseInt(a[0])) - (b.timestamp || parseInt(b[0])));
                      for (const c of sorted) {
                        try {
                          chartRef.current.updateData({
                            timestamp: c.timestamp || parseInt(c[0]),
                            open: c.open || parseFloat(c[1]),
                            high: c.high || parseFloat(c[2]),
                            low: c.low || parseFloat(c[3]),
                            close: c.close || parseFloat(c[4]),
                            volume: c.volume || parseFloat(c[5]) || 0,
                          });
                        } catch (_) {}
                      }
                      setStaleWarning(null);
                      console.log('[Chart] Staleness resolved: backfilled', sorted.length, 'candles from Bitget');
                    })
                    .catch(() => { setTimeout(() => setStaleWarning(null), 5000); });
                } else {
                  setStaleWarning(null);
                }
              }
            })
            .catch(e => { console.error('[Chart] fetch error:', e); params.callback([], false); });
        },
      });
      // Symbol + period already set BEFORE data loader (lines above)

      // Backfill + realtime: fetch 200 candles from Bitget to fill gap between stale DB and now
      const BITGET_GRAN: Record<string, string> = {
        '1m': '1m', '5m': '5m', '15m': '15m', '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W',
      };
      const backfillFromBitget = () => {
        if (disposed || !chartRef.current) return;
        const tf = timeframeRef.current;
        const gran = BITGET_GRAN[tf] || '1D';
        fetch(`${API}/api/bitget-candles?timeframe=${tf}&limit=200`)
          .then(r => r.json())
          .then(resp => {
            // Proxy returns {candles: [{timestamp, open, high, low, close, volume}]}
            const raw = resp?.candles || resp?.data || resp;
            if (!Array.isArray(raw) || raw.length === 0 || !chartRef.current) return;
            const sorted = [...raw].sort((a: any, b: any) => (a.timestamp || parseInt(a[0])) - (b.timestamp || parseInt(b[0])));
            for (const c of sorted) {
              try {
                chartRef.current.updateData({
                  timestamp: c.timestamp || parseInt(c[0]),
                  open: c.open || parseFloat(c[1]),
                  high: c.high || parseFloat(c[2]),
                  low: c.low || parseFloat(c[3]),
                  close: c.close || parseFloat(c[4]),
                  volume: c.volume || parseFloat(c[5]) || 0,
                });
              } catch (_) {}
            }
            // Update candle data ref with Bitget candles
            const bitgetCandles = sorted.map((c: any) => ({
              timestamp: c.timestamp || parseInt(c[0]),
              open: c.open || parseFloat(c[1]),
              high: c.high || parseFloat(c[2]),
              low: c.low || parseFloat(c[3]),
              close: c.close || parseFloat(c[4]),
              volume: c.volume || parseFloat(c[5]) || 0,
            }));
            const existingTs = new Set(candleDataRef.current.map(c => c.timestamp));
            const newCandles = bitgetCandles.filter(c => !existingTs.has(c.timestamp));
            if (newCandles.length > 0) {
              candleDataRef.current = [...candleDataRef.current, ...newCandles].sort((a, b) => a.timestamp - b.timestamp);
            }
            console.log('[Chart] Backfilled', sorted.length, 'candles from Bitget');
          })
          .catch(() => {});
      };
      // Backfill 2 seconds after init (let DB data load first)
      setTimeout(backfillFromBitget, 2000);

      // Real-time tick updates every 3 seconds
      const realtimeInterval = setInterval(() => {
        if (disposed || !chartRef.current) return;
        const tf = timeframeRef.current;
        const gran = BITGET_GRAN[tf] || '1D';
        fetch(`https://api.bitget.com/api/v2/mix/market/candles?productType=USDT-FUTURES&symbol=BTCUSDT&granularity=${gran}&limit=2`)
          .then(r => r.json())
          .then(resp => {
            const raw = resp?.data || resp;
            if (!Array.isArray(raw) || raw.length === 0 || !chartRef.current) return;
            const c = raw[raw.length - 1];
            try {
              chartRef.current.updateData({
                timestamp: parseInt(c[0]),
                open: parseFloat(c[1]),
                high: parseFloat(c[2]),
                low: parseFloat(c[3]),
                close: parseFloat(c[4]),
                volume: parseFloat(c[5]) || 0,
              });
            } catch (_) {}
          })
          .catch(() => {});
      }, 3000);

      // Resize observer
      const resizeObserver = new ResizeObserver(entries => {
        if (entries.length > 0 && chart) {
          chart.resize();
        }
      });
      resizeObserver.observe(containerRef.current!);

      return () => {
        clearInterval(realtimeInterval);
        resizeObserver.disconnect();
        dispose(containerRef.current!);
      };
    };

    const cleanup = initChart();
    return () => {
      disposed = true;
      chartRef.current = null; // clear ref immediately so nothing uses the disposed chart
      candleDataRef.current = [];
      activeIndicatorPanesRef.current.clear();
      overlayIdsRef.current = [];
      cleanup?.then(fn => fn?.());
      if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
    };
  }, [symbol, timeframe]); // Re-create on symbol OR timeframe change (dispose+reinit for v10-beta1)

  // Store current timeframe/symbol in refs so getBars can access without re-creating chart
  const timeframeRef = useRef(timeframe);
  const symbolRef = useRef(symbol);
  useEffect(() => { symbolRef.current = symbol; }, [symbol]);

  // TF change is now handled by the main init effect (dispose+reinit approach)
  // Just keep ref in sync for any code that reads it outside the effect
  useEffect(() => { timeframeRef.current = timeframe; }, [timeframe]);

  // TF bar size in ms — used for timestamp matching
  const getTfBarMs = (tf: string): number => {
    const map: Record<string, number> = {
      '1m': 60000, '5m': 300000, '15m': 900000, '1h': 3600000,
      '4h': 14400000, '1d': 86400000, '1w': 604800000,
    };
    return map[tf] || 86400000;
  };

  // Trade + signal markers as HTML overlays — bypasses broken klinecharts v10-beta1 overlay API
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) { setTradeMarkers([]); return; }
    let cancelled = false;

    const updateMarkers = () => {
      if (cancelled || !chartRef.current) return;
      const chart = chartRef.current;
      const container = containerRef.current;
      if (!container) return;

      const tf = timeframeRef.current;
      const barMs = getTfBarMs(tf);
      // Tolerance: 2x bar size for matching trades to candles
      const tolerance = Math.max(barMs * 2, 3600000); // at least 1 hour for low TFs

      fetch(`${API}/api/ml-trades?limit=50`)
        .then(r => r.json())
        .then(data => {
          if (cancelled || !chartRef.current) return;
          const trades = data.all_trades || [];
          const rect = container.getBoundingClientRect();
          const chartWidth = rect.width - 60; // subtract y-axis
          const chartHeight = rect.height - 30; // subtract x-axis
          const markers: typeof tradeMarkers = [];

          // Get visible range
          const visibleRange = chart.getVisibleRange?.();
          const dataList = candleDataRef.current;
          if (!visibleRange || !dataList || dataList.length === 0) return;

          const visFrom = visibleRange.from || 0;
          const visTo = visibleRange.to || dataList.length;
          const visCount = visTo - visFrom;
          if (visCount <= 0) return;

          // Get price range from visible candles (with padding for SL/TP lines)
          let minPrice = Infinity, maxPrice = -Infinity;
          for (let i = visFrom; i < Math.min(visTo, dataList.length); i++) {
            const c = dataList[i];
            if (c.low < minPrice) minPrice = c.low;
            if (c.high > maxPrice) maxPrice = c.high;
          }
          if (minPrice === Infinity) return;
          const priceRange = maxPrice - minPrice;
          if (priceRange <= 0) return;

          // Helper: find candle index closest to timestamp
          const findIdx = (ts: number): number => {
            let best = -1, bestDiff = Infinity;
            for (let i = 0; i < dataList.length; i++) {
              const diff = Math.abs(dataList[i].timestamp - ts);
              if (diff < bestDiff) { bestDiff = diff; best = i; }
            }
            return bestDiff <= tolerance ? best : -1;
          };

          // Helper: timestamp to X pixel
          const tsToX = (idx: number): number => 40 + ((idx - visFrom) / visCount) * chartWidth;
          // Helper: price to Y pixel
          const priceToY = (price: number): number => 10 + ((maxPrice - price) / priceRange) * (chartHeight - 20);

          trades.forEach((trade: any) => {
            if (!trade.entry_time || !trade.entry_price) return;
            const entryTs = new Date(trade.entry_time).getTime();
            const isLong = trade.direction === 'LONG';
            const isOpen = trade.status === 'open';

            const entryIdx = findIdx(entryTs);
            if (entryIdx < 0) return; // not in loaded data at all

            // Even if entry is off-screen, we might want SL/TP lines visible
            const entryVisible = entryIdx >= visFrom && entryIdx < visTo;

            const entryX = tsToX(entryIdx);
            const entryY = priceToY(trade.entry_price);

            // Exit position
            let exitX: number | null = null;
            let exitY: number | null = null;
            if (trade.exit_time && trade.exit_price) {
              const exitTs = new Date(trade.exit_time).getTime();
              const exitIdx = findIdx(exitTs);
              if (exitIdx >= 0 && exitIdx >= visFrom && exitIdx < visTo) {
                exitX = tsToX(exitIdx);
                exitY = priceToY(trade.exit_price);
              }
            }

            // SL/TP lines extend from entry to exit (or current rightmost visible candle)
            const lineEndIdx = trade.exit_time
              ? findIdx(new Date(trade.exit_time).getTime())
              : Math.min(visTo - 1, dataList.length - 1);
            const lineEndX = lineEndIdx >= 0 ? tsToX(Math.max(lineEndIdx, entryIdx)) : tsToX(visTo - 1);

            const slY = trade.stop_price ? priceToY(trade.stop_price) : null;
            const tpY = trade.tp_price ? priceToY(trade.tp_price) : null;

            // Only add if any part is visible
            if (entryVisible || (exitX !== null)) {
              markers.push({
                id: trade.id,
                entryX, entryY,
                exitX, exitY,
                slY, tpY,
                slX2: lineEndX, tpX2: lineEndX,
                color: isLong ? '#22c55e' : '#ef4444',
                label: isLong ? 'LONG' : 'SHORT',
                isOpen,
                pnl: trade.pnl ?? null,
                direction: trade.direction,
              });
            }
          });

          setTradeMarkers(markers);

          // Fetch signal overlays using same geometry context
          const activeOvs = Object.entries(overlays).filter(([_, v]) => v).map(([k]) => k);
          const isLowTf = ['1m', '5m'].includes(tf);
          if (isLowTf) {
            setSignalMarkers([]); // Daily signals meaningless on 1m/5m
          } else if (activeOvs.length > 0) {
            // Fetch all signals in the loaded data range
            const earliestTs = dataList.length > 0 ? Math.floor(dataList[0].timestamp / 1000) : 0;
            const latestTs = dataList.length > 0 ? Math.floor(dataList[dataList.length - 1].timestamp / 1000) + 86400 : Math.floor(Date.now() / 1000);

            fetch(`${API}/api/overlays?type=all&from=${earliestTs}&to=${latestTs}`)
              .then(r2 => r2.json())
              .then(odata => {
                if (cancelled || !odata.points) return;
                const filtered = odata.points.filter((p: any) => overlays[p.type as OverlayType]);
                const sMarks: Array<{ key: string; x: number; y: number; color: string; label: string; above: boolean }> = [];

                filtered.forEach((p: any) => {
                  const pTs = p.time * 1000;
                  // Use TF-aware tolerance for signal matching
                  const signalTolerance = barMs;
                  let sidx = -1;
                  let bestDiff = Infinity;
                  for (let i = 0; i < dataList.length; i++) {
                    const diff = Math.abs(dataList[i].timestamp - pTs);
                    if (diff < bestDiff) { bestDiff = diff; sidx = i; }
                  }
                  if (bestDiff > signalTolerance) return; // no matching candle within tolerance
                  if (sidx < visFrom || sidx >= visTo) return;
                  const candle = dataList[sidx];
                  const isBearish = p.direction === 'bearish';
                  const price = isBearish ? candle.high : candle.low;
                  const sx = tsToX(sidx);
                  const sy = priceToY(price);
                  if (sx > 40 && sx < chartWidth + 40 && sy > 0 && sy < chartHeight) {
                    sMarks.push({
                      key: `${p.time}-${p.type}`,
                      x: sx, y: isBearish ? sy - 14 : sy + 4,
                      color: OVERLAY_COLORS[p.type] || '#fff',
                      label: p.label?.substring(0, 8) || p.type,
                      above: isBearish,
                    });
                  }
                });
                setSignalMarkers(sMarks);
              })
              .catch(() => {});
          } else {
            setSignalMarkers([]); // no overlays active or low TF
          }
        })
        .catch(() => {});
    };

    // Update markers periodically (positions change on scroll/zoom)
    // Retry at 1s until chart has data AND visible range is valid, then settle to 4s
    let retryCount = 0;
    const retryUntilReady = () => {
      if (cancelled) return;
      updateMarkers();
      retryCount++;
      const chart = chartRef.current;
      const vr = chart?.getVisibleRange?.();
      const hasData = candleDataRef.current.length > 0;
      const hasRange = vr && (vr.to - vr.from) > 0;
      if ((!hasData || !hasRange) && retryCount < 20) {
        setTimeout(retryUntilReady, 1000);
      }
    };
    const timer = setTimeout(retryUntilReady, 1500);
    const interval = setInterval(updateMarkers, 4000);
    return () => { cancelled = true; clearTimeout(timer); clearInterval(interval); };
  }, [timeframe, overlays]);

  // Indicator management
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;

    // Remove indicators that are no longer active
    const toRemove: string[] = [];
    activeIndicatorPanesRef.current.forEach((paneId, id) => {
      if (!activeIndicators.includes(id)) {
        const mapping = mapIndicatorToKLine(id);
        if (mapping) {
          try {
            chart.removeIndicator(paneId, mapping.name);
          } catch (_) {}
        }
        toRemove.push(id);
      }
    });
    toRemove.forEach(id => activeIndicatorPanesRef.current.delete(id));

    // Add new indicators
    for (const id of activeIndicators) {
      if (activeIndicatorPanesRef.current.has(id)) continue;

      const mapping = mapIndicatorToKLine(id);
      if (!mapping) continue;

      const config = indicatorConfigs[id];
      const catalogDef = INDICATOR_CATALOG.find(c => c.id === id);

      try {
        // For overlay indicators, put on candle_pane; for oscillators, create new pane
        const isMainPane = mapping.paneId === 'candle_pane';

        const paneId = chart.createIndicator(
          {
            name: mapping.name,
            calcParams: mapping.calcParams,
            styles: config?.color ? {
              lines: [
                { color: config.color, size: config.lineWidth || 2 },
                { color: config.color + '80', size: config.lineWidth || 2 },
                { color: config.color + '40', size: config.lineWidth || 2 },
              ],
            } : catalogDef?.color ? {
              lines: [
                { color: catalogDef.color, size: 2 },
                { color: catalogDef.color + '80', size: 2 },
                { color: catalogDef.color + '40', size: 2 },
              ],
            } : undefined,
          },
          isMainPane,
          isMainPane ? { id: 'candle_pane' } : undefined
        );

        activeIndicatorPanesRef.current.set(id, paneId || (isMainPane ? 'candle_pane' : 'unknown'));
      } catch (e) {
        console.error(`Failed to add indicator ${id}:`, e);
      }
    }
  }, [activeIndicators, indicatorConfigs, symbol, timeframe]);

  // Mouse move handler for marker tooltips + shift+click measure tool
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleMouseMove = (e: MouseEvent) => {
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const chart = chartRef.current;
      if (!chart) return;

      // v10: convertFromPixel removed — use getVisibleRange + candleData to estimate timestamp
      const visibleRange = chart.getVisibleRange?.();
      const dataList = candleDataRef.current;
      if (!visibleRange || !dataList || dataList.length === 0) {
        if (hoverTimerRef.current) { clearTimeout(hoverTimerRef.current); hoverTimerRef.current = null; }
        setTooltip(null);
        return;
      }

      // Estimate which candle the mouse is over based on X position
      const chartWidth = rect.width - 50; // subtract Y-axis width
      const drawingToolbarWidth = 40;
      const adjustedX = x - drawingToolbarWidth;
      const visibleCount = (visibleRange.to || 0) - (visibleRange.from || 0);
      if (visibleCount <= 0 || adjustedX < 0) { setTooltip(null); return; }
      const barIndex = Math.floor((adjustedX / chartWidth) * visibleCount) + (visibleRange.from || 0);
      const candle = dataList[Math.min(Math.max(barIndex, 0), dataList.length - 1)];
      if (!candle) { setTooltip(null); return; }

      const candleTs = candle.timestamp;
      const tolerance = timeframeRef.current === '1m' ? 60000 : timeframeRef.current === '5m' ? 300000 :
                        timeframeRef.current === '15m' ? 900000 : timeframeRef.current === '1h' ? 3600000 :
                        timeframeRef.current === '4h' ? 14400000 : 86400000;

      let matched: MarkerInfo | undefined;
      for (const m of markerDataRef.current) {
        if (Math.abs(m.timestamp - candleTs) <= tolerance) {
          matched = m;
          break;
        }
      }

      // Also check signal overlay markers for this candle
      const nearbySignals: string[] = [];
      const nearbyTypes: string[] = [];
      for (const sm of signalMarkers) {
        // Check if signal marker is close to mouse position (within 20px)
        if (Math.abs(sm.x - x) < 20 && Math.abs(sm.y - y) < 30) {
          nearbySignals.push(sm.label);
          nearbyTypes.push(sm.label.split(' ')[0] || 'signal');
        }
      }

      if (matched || nearbySignals.length > 0) {
        if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
        hoverTimerRef.current = setTimeout(() => {
          const tradeSignals = matched ? matched.text.split(' | ') : [];
          const allSignals = [...tradeSignals, ...nearbySignals];
          const dateStr = matched ? new Date(matched.timestamp).toUTCString() : new Date(candleTs).toUTCString();
          setTooltip({
            x, y, signals: allSignals,
            date: dateStr,
            direction: matched ? (matched.position === 'aboveBar' ? 'bearish' : 'bullish') : (nearbySignals.some(s => s.includes('🔴') || s.includes('⚠') || s.includes('💀')) ? 'bearish' : 'bullish'),
            types: matched ? matched.types : nearbyTypes,
          });
        }, 300);
      } else {
        if (hoverTimerRef.current) { clearTimeout(hoverTimerRef.current); hoverTimerRef.current = null; }
        setTooltip(null);
      }

      // Measure tool: show info while dragging
      if (measureStartRef.current && e.shiftKey && candle) {
        const startVal = measureStartRef.current.value;
        const curVal = candle.close;
        if (curVal) {
          const diff = curVal - startVal;
          const pct = ((diff / startVal) * 100).toFixed(2);
          setMeasureInfo({
            pct: `${diff >= 0 ? '+' : ''}${pct}%`,
            diff: `${diff >= 0 ? '+' : ''}${diff.toFixed(2)}`,
            color: diff >= 0 ? '#22c55e' : '#ef4444',
          });
        }
      }
    };

    // Shift+click for measure tool
    const handleClick = (e: MouseEvent) => {
      if (!e.shiftKey) return;
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const chart = chartRef.current;
      if (!chart) return;

      const point = chart.convertFromPixel?.({ x, y }, { paneId: 'candle_pane' });
      if (!point) return;

      if (!measureStartRef.current) {
        measureStartRef.current = { timestamp: point.timestamp, value: point.value };
        setMeasureInfo(null);
      } else {
        // Second click: finalize measure, create a segment overlay
        const start = measureStartRef.current;
        const end = { timestamp: point.timestamp, value: point.value };
        const diff = end.value - start.value;
        const pct = ((diff / start.value) * 100).toFixed(2);

        try {
          chart.createOverlay({
            name: 'segment',
            points: [
              { timestamp: start.timestamp, value: start.value },
              { timestamp: end.timestamp, value: end.value },
            ],
            styles: {
              line: {
                style: 'dashed' as any,
                color: diff >= 0 ? '#22c55e' : '#ef4444',
                size: 1,
              },
            },
          });
        } catch (_) {}

        measureStartRef.current = null;
        setMeasureInfo({
          pct: `${diff >= 0 ? '+' : ''}${pct}%`,
          diff: `${diff >= 0 ? '+' : ''}${diff.toFixed(2)}`,
          color: diff >= 0 ? '#22c55e' : '#ef4444',
        });
        // Clear measure info after 3s
        setTimeout(() => setMeasureInfo(null), 3000);
      }
    };

    container.addEventListener('mousemove', handleMouseMove);
    container.addEventListener('click', handleClick);

    return () => {
      container.removeEventListener('mousemove', handleMouseMove);
      container.removeEventListener('click', handleClick);
    };
  }, [timeframe]);

  return (
    <div className="relative w-full h-full" style={{ minHeight: '500px', cursor: cursorStyle }}>
      <div ref={containerRef} className="w-full h-full" />

      {/* Trade markers — TradingView-style: entry arrow + SL/TP lines + exit bar */}
      {tradeMarkers.map(m => (
        <div key={m.id} className="absolute inset-0 z-30 pointer-events-none overflow-hidden">
          {/* SL line (red dashed) from entry to exit/current */}
          {m.slY !== null && (
            <div
              className="absolute"
              style={{
                left: Math.min(m.entryX, m.slX2),
                top: m.slY,
                width: Math.abs(m.slX2 - m.entryX) + 1,
                height: 0,
                borderTop: '1px dashed #ef4444',
                opacity: 0.7,
              }}
            />
          )}
          {/* TP line (green dashed) from entry to exit/current */}
          {m.tpY !== null && (
            <div
              className="absolute"
              style={{
                left: Math.min(m.entryX, m.tpX2),
                top: m.tpY,
                width: Math.abs(m.tpX2 - m.entryX) + 1,
                height: 0,
                borderTop: '1px dashed #22c55e',
                opacity: 0.7,
              }}
            />
          )}
          {/* SL label */}
          {m.slY !== null && (
            <div
              className="absolute text-[8px] font-mono font-bold"
              style={{ left: m.slX2 + 2, top: (m.slY ?? 0) - 6, color: '#ef4444' }}
            >
              SL
            </div>
          )}
          {/* TP label */}
          {m.tpY !== null && (
            <div
              className="absolute text-[8px] font-mono font-bold"
              style={{ left: m.tpX2 + 2, top: (m.tpY ?? 0) - 6, color: '#22c55e' }}
            >
              TP
            </div>
          )}
          {/* Entry arrow */}
          <div
            className="absolute flex flex-col items-center"
            style={{ left: m.entryX - 10, top: m.direction === 'LONG' ? m.entryY : m.entryY - 20 }}
          >
            <div
              className="font-mono font-bold text-[10px] leading-none text-white px-1 rounded"
              style={{
                backgroundColor: m.color + 'cc',
                boxShadow: `0 0 8px ${m.color}60`,
                border: m.isOpen ? '1px solid #fff' : `1px solid ${m.color}`,
              }}
            >
              {m.direction === 'LONG' ? '\u25B2' : '\u25BC'} {m.label}
            </div>
          </div>
          {/* Exit bar (gold vertical stripe) */}
          {m.exitX !== null && m.exitY !== null && (
            <>
              <div
                className="absolute"
                style={{
                  left: m.exitX - 2,
                  top: 0,
                  width: 4,
                  height: '100%',
                  backgroundColor: 'rgba(234, 179, 8, 0.25)',
                }}
              />
              <div
                className="absolute font-mono font-bold text-[9px] px-1 rounded"
                style={{
                  left: m.exitX - 14,
                  top: m.exitY - 8,
                  color: m.pnl !== null && m.pnl >= 0 ? '#22c55e' : '#ef4444',
                  backgroundColor: 'rgba(10, 10, 20, 0.85)',
                  border: `1px solid ${m.pnl !== null && m.pnl >= 0 ? '#22c55e' : '#ef4444'}40`,
                }}
              >
                {m.pnl !== null ? `${m.pnl >= 0 ? '+' : ''}${m.pnl.toFixed(2)}%` : 'EXIT'}
              </div>
            </>
          )}
        </div>
      ))}

      {/* Signal overlay markers as HTML */}
      {signalMarkers.map(m => (
        <div
          key={m.key}
          className="absolute z-20 pointer-events-none"
          style={{ left: m.x - 4, top: m.y }}
        >
          <div
            className="w-2 h-2 rounded-sm"
            style={{ backgroundColor: m.color, boxShadow: `0 0 4px ${m.color}60` }}
            title={m.label}
          />
        </div>
      ))}

      {/* Prediction ghost candles overlay */}
      {predictionOverlay && showPredictions && (
        <div className="absolute inset-0 z-25 pointer-events-none overflow-hidden">
          {/* Ghost candle bodies + wicks */}
          {predictionOverlay.ghostCandles.map((gc, i) => (
            <div key={`ghost-${i}`}>
              {/* Wick */}
              <div
                className="absolute"
                style={{
                  left: gc.wickX - 0.5,
                  top: gc.wickTop,
                  width: 1,
                  height: gc.wickBottom - gc.wickTop,
                  backgroundColor: gc.color,
                }}
              />
              {/* Body */}
              <div
                className="absolute rounded-[1px]"
                style={{
                  left: gc.x,
                  top: gc.y,
                  width: gc.width,
                  height: gc.height,
                  backgroundColor: gc.color,
                  border: `1px solid ${predictionOverlay.direction === 'LONG' ? 'rgba(34, 197, 94, 0.6)' : 'rgba(239, 68, 68, 0.6)'}`,
                }}
              />
            </div>
          ))}

          {/* Entry price line (white dashed) */}
          <div
            className="absolute"
            style={{
              left: predictionOverlay.startX,
              top: predictionOverlay.entryY,
              width: predictionOverlay.chartWidth - predictionOverlay.startX,
              height: 0,
              borderTop: '1px dashed rgba(255, 255, 255, 0.5)',
            }}
          />
          <div
            className="absolute text-[9px] font-mono font-medium"
            style={{ right: 4, top: predictionOverlay.entryY - 12, color: 'rgba(255, 255, 255, 0.7)' }}
          >
            ENTRY
          </div>

          {/* Stop loss line (red dashed) */}
          <div
            className="absolute"
            style={{
              left: predictionOverlay.startX,
              top: predictionOverlay.slY,
              width: predictionOverlay.chartWidth - predictionOverlay.startX,
              height: 0,
              borderTop: '1px dashed rgba(239, 68, 68, 0.6)',
            }}
          />
          <div
            className="absolute text-[9px] font-mono font-medium"
            style={{ right: 4, top: predictionOverlay.slY - 12, color: '#ef4444' }}
          >
            SL
          </div>

          {/* Take profit line (green dashed) */}
          <div
            className="absolute"
            style={{
              left: predictionOverlay.startX,
              top: predictionOverlay.tpY,
              width: predictionOverlay.chartWidth - predictionOverlay.startX,
              height: 0,
              borderTop: '1px dashed rgba(34, 197, 94, 0.6)',
            }}
          />
          <div
            className="absolute text-[9px] font-mono font-medium"
            style={{ right: 4, top: predictionOverlay.tpY - 12, color: '#22c55e' }}
          >
            TP
          </div>

          {/* Confidence label */}
          <div
            className="absolute text-[11px] font-mono font-bold px-1.5 py-0.5 rounded"
            style={{
              left: predictionOverlay.confidenceX,
              top: predictionOverlay.confidenceY,
              color: predictionOverlay.direction === 'LONG' ? '#22c55e' : '#ef4444',
              backgroundColor: 'rgba(10, 10, 20, 0.8)',
              border: `1px solid ${predictionOverlay.direction === 'LONG' ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)'}`,
            }}
          >
            {predictionOverlay.direction} {predictionOverlay.confidenceText}
          </div>
        </div>
      )}

      {/* Stale data warning for 1m/5m */}
      {staleWarning && (
        <div
          className="absolute top-2 left-2 z-40 px-3 py-1.5 rounded-lg text-xs font-medium text-amber-400 font-mono"
          style={{
            background: 'rgba(245, 158, 11, 0.1)',
            border: '1px solid rgba(245, 158, 11, 0.3)',
          }}
        >
          {staleWarning}
        </div>
      )}

      {/* Drawing tool active indicator */}
      {drawingTool && drawingTool !== 'select' && drawingTool !== 'delete' && (
        <div
          className="absolute top-2 left-1/2 -translate-x-1/2 z-40 px-3 py-1.5 rounded-lg text-xs font-medium text-blue-400 font-mono"
          style={{
            background: 'rgba(59, 130, 246, 0.1)',
            border: '1px solid rgba(59, 130, 246, 0.3)',
          }}
        >
          {drawingTool === 'measure'
            ? 'Shift+Click two points to measure'
            : 'Drawing mode active \u00b7 ESC to cancel'}
        </div>
      )}

      {/* Measure info display */}
      {measureInfo && (
        <div
          className="absolute top-2 right-2 z-40 px-3 py-1.5 rounded-lg text-xs font-bold font-mono"
          style={{
            background: 'rgba(10, 10, 20, 0.9)',
            border: `1px solid ${measureInfo.color}40`,
            color: measureInfo.color,
          }}
        >
          {measureInfo.pct} ({measureInfo.diff})
        </div>
      )}

      {/* Marker Hover Tooltip */}
      {tooltip && (
        <div
          className="absolute z-50 pointer-events-none animate-in fade-in duration-200"
          style={{
            left: Math.min(tooltip.x + 12, (containerRef.current?.offsetWidth || 800) - 320),
            top: Math.max(tooltip.y - 20, 10),
            maxWidth: '320px',
          }}
        >
          <div className="rounded-lg border shadow-xl backdrop-blur-md"
            style={{
              background: 'rgba(10, 10, 20, 0.95)',
              borderColor: tooltip.direction === 'bearish' ? 'rgba(239, 68, 68, 0.4)' : 'rgba(34, 197, 94, 0.4)',
              boxShadow: tooltip.direction === 'bearish'
                ? '0 0 20px rgba(239, 68, 68, 0.15)'
                : '0 0 20px rgba(34, 197, 94, 0.15)',
            }}
          >
            {/* Header */}
            <div className="px-3 py-2 border-b" style={{ borderColor: 'rgba(50, 50, 80, 0.3)' }}>
              <div className="flex items-center gap-2">
                <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                  tooltip.direction === 'bearish'
                    ? 'bg-red-500/20 text-red-400'
                    : 'bg-green-500/20 text-green-400'
                }`}>
                  {tooltip.direction === 'bearish' ? 'BEARISH' : 'BULLISH'}
                </span>
                <span className="text-[10px] text-slate-500 font-mono">
                  {tooltip.date.replace('GMT', 'UTC')}
                </span>
              </div>
            </div>

            {/* Signals */}
            <div className="px-3 py-2 space-y-1.5">
              {tooltip.signals.map((signal, i) => {
                let color = 'text-slate-300';
                if (signal.includes('caution') || signal.includes('CAUTION')) { color = 'text-red-400'; }
                else if (signal.includes('pump') || signal.includes('PUMP')) { color = 'text-green-400'; }
                else if (signal.includes('lunar') || signal.includes('LUNAR')) { color = 'text-yellow-400'; }
                else if (signal.includes('ritual') || signal.includes('RITUAL')) { color = 'text-purple-400'; }

                return (
                  <div key={i} className={`flex items-start gap-2 text-xs ${color}`}>
                    <span className="flex-shrink-0 mt-0.5 w-1.5 h-1.5 rounded-full bg-current" />
                    <span className="font-mono leading-tight">{signal}</span>
                  </div>
                );
              })}
            </div>

            {/* Footer */}
            <div className="px-3 py-1.5 border-t text-[10px] text-slate-500 font-mono"
              style={{ borderColor: 'rgba(50, 50, 80, 0.3)' }}
            >
              {tooltip.signals.length} signal{tooltip.signals.length > 1 ? 's' : ''} on this candle
              {tooltip.types.length > 0 && (
                <span className="ml-2">
                  [{Array.from(new Set(tooltip.types)).join(', ')}]
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
