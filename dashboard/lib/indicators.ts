// ── Pure TypeScript indicator calculation engine ──
// All functions accept arrays of numbers and return arrays of numbers (or objects).

export interface OHLCV {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// ── Helpers ──

function getSource(candles: OHLCV[], source: string): number[] {
  switch (source) {
    case 'open': return candles.map(c => c.open);
    case 'high': return candles.map(c => c.high);
    case 'low': return candles.map(c => c.low);
    case 'close': return candles.map(c => c.close);
    case 'hl2': return candles.map(c => (c.high + c.low) / 2);
    case 'hlc3': return candles.map(c => (c.high + c.low + c.close) / 3);
    case 'ohlc4': return candles.map(c => (c.open + c.high + c.low + c.close) / 4);
    default: return candles.map(c => c.close);
  }
}

// ── SMA ──

export function sma(values: number[], period: number): (number | null)[] {
  const result: (number | null)[] = [];
  for (let i = 0; i < values.length; i++) {
    if (i < period - 1) {
      result.push(null);
    } else {
      let sum = 0;
      for (let j = i - period + 1; j <= i; j++) {
        sum += values[j];
      }
      result.push(sum / period);
    }
  }
  return result;
}

// ── EMA ──

export function ema(values: number[], period: number): (number | null)[] {
  const result: (number | null)[] = [];
  const k = 2 / (period + 1);

  // Use SMA for the first value
  let prevEma: number | null = null;
  for (let i = 0; i < values.length; i++) {
    if (i < period - 1) {
      result.push(null);
    } else if (i === period - 1) {
      // First EMA = SMA of first `period` values
      let sum = 0;
      for (let j = 0; j < period; j++) sum += values[j];
      prevEma = sum / period;
      result.push(prevEma);
    } else {
      prevEma = values[i] * k + prevEma! * (1 - k);
      result.push(prevEma);
    }
  }
  return result;
}

// ── RSI ──

export function rsi(values: number[], period: number = 14): (number | null)[] {
  const result: (number | null)[] = [];
  if (values.length < period + 1) {
    return values.map(() => null);
  }

  const gains: number[] = [];
  const losses: number[] = [];

  for (let i = 1; i < values.length; i++) {
    const change = values[i] - values[i - 1];
    gains.push(change > 0 ? change : 0);
    losses.push(change < 0 ? -change : 0);
  }

  // First average
  let avgGain = 0;
  let avgLoss = 0;
  for (let i = 0; i < period; i++) {
    avgGain += gains[i];
    avgLoss += losses[i];
  }
  avgGain /= period;
  avgLoss /= period;

  // Pad nulls for indices 0..period
  for (let i = 0; i <= period; i++) {
    result.push(null);
  }

  // First RSI value at index=period
  if (avgLoss === 0) {
    result[period] = 100;
  } else {
    const rs = avgGain / avgLoss;
    result[period] = 100 - 100 / (1 + rs);
  }

  // Subsequent values using smoothed method
  for (let i = period; i < gains.length; i++) {
    avgGain = (avgGain * (period - 1) + gains[i]) / period;
    avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
    if (avgLoss === 0) {
      result.push(100);
    } else {
      const rs = avgGain / avgLoss;
      result.push(100 - 100 / (1 + rs));
    }
  }

  return result;
}

// ── MACD ──

export interface MACDResult {
  line: (number | null)[];
  signal: (number | null)[];
  histogram: (number | null)[];
}

export function macd(
  values: number[],
  fastPeriod: number = 12,
  slowPeriod: number = 26,
  signalPeriod: number = 9
): MACDResult {
  const fastEma = ema(values, fastPeriod);
  const slowEma = ema(values, slowPeriod);

  const macdLine: (number | null)[] = [];
  for (let i = 0; i < values.length; i++) {
    if (fastEma[i] !== null && slowEma[i] !== null) {
      macdLine.push(fastEma[i]! - slowEma[i]!);
    } else {
      macdLine.push(null);
    }
  }

  // Signal line = EMA of MACD line (only non-null values)
  const nonNullStart = macdLine.findIndex(v => v !== null);
  const macdValues = macdLine.slice(nonNullStart).map(v => v!);
  const signalEma = ema(macdValues, signalPeriod);

  const signalLine: (number | null)[] = new Array(nonNullStart).fill(null);
  for (const v of signalEma) signalLine.push(v);

  const histogram: (number | null)[] = [];
  for (let i = 0; i < values.length; i++) {
    if (macdLine[i] !== null && signalLine[i] !== null) {
      histogram.push(macdLine[i]! - signalLine[i]!);
    } else {
      histogram.push(null);
    }
  }

  return { line: macdLine, signal: signalLine, histogram };
}

// ── Bollinger Bands ──

export interface BollingerResult {
  upper: (number | null)[];
  middle: (number | null)[];
  lower: (number | null)[];
}

export function bollinger(
  values: number[],
  period: number = 20,
  stddev: number = 2
): BollingerResult {
  const middle = sma(values, period);
  const upper: (number | null)[] = [];
  const lower: (number | null)[] = [];

  for (let i = 0; i < values.length; i++) {
    if (middle[i] === null) {
      upper.push(null);
      lower.push(null);
    } else {
      let sumSq = 0;
      for (let j = i - period + 1; j <= i; j++) {
        const diff = values[j] - middle[i]!;
        sumSq += diff * diff;
      }
      const sd = Math.sqrt(sumSq / period);
      upper.push(middle[i]! + stddev * sd);
      lower.push(middle[i]! - stddev * sd);
    }
  }

  return { upper, middle, lower };
}

// ── Stochastic ──

export interface StochasticResult {
  k: (number | null)[];
  d: (number | null)[];
}

export function stochastic(
  candles: OHLCV[],
  kPeriod: number = 14,
  dPeriod: number = 3,
  smooth: number = 3
): StochasticResult {
  const rawK: (number | null)[] = [];

  for (let i = 0; i < candles.length; i++) {
    if (i < kPeriod - 1) {
      rawK.push(null);
    } else {
      let highest = -Infinity;
      let lowest = Infinity;
      for (let j = i - kPeriod + 1; j <= i; j++) {
        if (candles[j].high > highest) highest = candles[j].high;
        if (candles[j].low < lowest) lowest = candles[j].low;
      }
      const range = highest - lowest;
      rawK.push(range === 0 ? 50 : ((candles[i].close - lowest) / range) * 100);
    }
  }

  // Smooth %K with SMA
  const nonNullK = rawK.filter(v => v !== null) as number[];
  const smoothedK = sma(nonNullK, smooth);
  const kResult: (number | null)[] = new Array(kPeriod - 1).fill(null);
  for (const v of smoothedK) kResult.push(v);

  // %D = SMA of smoothed %K
  const nonNullSmoothedK = kResult.filter(v => v !== null) as number[];
  const dValues = sma(nonNullSmoothedK, dPeriod);
  const dStart = kResult.findIndex(v => v !== null);
  const dResult: (number | null)[] = new Array(dStart).fill(null);
  for (const v of dValues) dResult.push(v);

  // Pad to same length
  while (kResult.length < candles.length) kResult.push(null);
  while (dResult.length < candles.length) dResult.push(null);

  return {
    k: kResult.slice(0, candles.length),
    d: dResult.slice(0, candles.length),
  };
}

// ── ATR ──

export function atr(candles: OHLCV[], period: number = 14): (number | null)[] {
  const trueRanges: number[] = [];

  for (let i = 0; i < candles.length; i++) {
    if (i === 0) {
      trueRanges.push(candles[i].high - candles[i].low);
    } else {
      const hl = candles[i].high - candles[i].low;
      const hc = Math.abs(candles[i].high - candles[i - 1].close);
      const lc = Math.abs(candles[i].low - candles[i - 1].close);
      trueRanges.push(Math.max(hl, hc, lc));
    }
  }

  // Use Wilder's smoothing (RMA)
  const result: (number | null)[] = [];
  for (let i = 0; i < candles.length; i++) {
    if (i < period - 1) {
      result.push(null);
    } else if (i === period - 1) {
      let sum = 0;
      for (let j = 0; j < period; j++) sum += trueRanges[j];
      result.push(sum / period);
    } else {
      const prev = result[i - 1]!;
      result.push((prev * (period - 1) + trueRanges[i]) / period);
    }
  }

  return result;
}

// ── OBV ──

export function obv(candles: OHLCV[]): number[] {
  const result: number[] = [0];
  for (let i = 1; i < candles.length; i++) {
    if (candles[i].close > candles[i - 1].close) {
      result.push(result[i - 1] + candles[i].volume);
    } else if (candles[i].close < candles[i - 1].close) {
      result.push(result[i - 1] - candles[i].volume);
    } else {
      result.push(result[i - 1]);
    }
  }
  return result;
}

// ── VWAP ──

export function vwap(candles: OHLCV[]): (number | null)[] {
  const result: (number | null)[] = [];
  let cumVolPrice = 0;
  let cumVol = 0;

  for (let i = 0; i < candles.length; i++) {
    const typicalPrice = (candles[i].high + candles[i].low + candles[i].close) / 3;
    cumVolPrice += typicalPrice * candles[i].volume;
    cumVol += candles[i].volume;
    result.push(cumVol === 0 ? null : cumVolPrice / cumVol);
  }

  return result;
}

// ── CCI ──

export function cci(candles: OHLCV[], period: number = 20): (number | null)[] {
  const tp = candles.map(c => (c.high + c.low + c.close) / 3);
  const tpSma = sma(tp, period);
  const result: (number | null)[] = [];

  for (let i = 0; i < candles.length; i++) {
    if (tpSma[i] === null) {
      result.push(null);
    } else {
      // Mean deviation
      let sumDev = 0;
      for (let j = i - period + 1; j <= i; j++) {
        sumDev += Math.abs(tp[j] - tpSma[i]!);
      }
      const meanDev = sumDev / period;
      result.push(meanDev === 0 ? 0 : (tp[i] - tpSma[i]!) / (0.015 * meanDev));
    }
  }

  return result;
}

// ── Supertrend ──

export interface SupertrendResult {
  line: (number | null)[];
  direction: (number | null)[]; // 1 = up (bullish), -1 = down (bearish)
}

export function supertrend(
  candles: OHLCV[],
  period: number = 10,
  multiplier: number = 3
): SupertrendResult {
  const atrValues = atr(candles, period);
  const line: (number | null)[] = [];
  const direction: (number | null)[] = [];

  const upperBand: number[] = [];
  const lowerBand: number[] = [];
  let prevDir = 1;
  let prevUpper = 0;
  let prevLower = 0;

  for (let i = 0; i < candles.length; i++) {
    if (atrValues[i] === null) {
      line.push(null);
      direction.push(null);
      upperBand.push(0);
      lowerBand.push(0);
      continue;
    }

    const hl2 = (candles[i].high + candles[i].low) / 2;
    let upper = hl2 + multiplier * atrValues[i]!;
    let lower = hl2 - multiplier * atrValues[i]!;

    // Adjust bands
    if (i > 0 && prevLower !== 0) {
      lower = lower > prevLower || candles[i - 1].close < prevLower ? lower : prevLower;
    }
    if (i > 0 && prevUpper !== 0) {
      upper = upper < prevUpper || candles[i - 1].close > prevUpper ? upper : prevUpper;
    }

    upperBand.push(upper);
    lowerBand.push(lower);

    let dir: number;
    if (i === period - 1) {
      dir = candles[i].close > upper ? 1 : -1;
    } else {
      if (prevDir === 1) {
        dir = candles[i].close < lower ? -1 : 1;
      } else {
        dir = candles[i].close > upper ? 1 : -1;
      }
    }

    line.push(dir === 1 ? lower : upper);
    direction.push(dir);
    prevDir = dir;
    prevUpper = upper;
    prevLower = lower;
  }

  return { line, direction };
}

// ── Ichimoku ──

export interface IchimokuResult {
  tenkan: (number | null)[];
  kijun: (number | null)[];
  senkouA: (number | null)[];
  senkouB: (number | null)[];
  chikou: (number | null)[];
}

function donchianMid(candles: OHLCV[], period: number, index: number): number | null {
  if (index < period - 1) return null;
  let highest = -Infinity;
  let lowest = Infinity;
  for (let j = index - period + 1; j <= index; j++) {
    if (candles[j].high > highest) highest = candles[j].high;
    if (candles[j].low < lowest) lowest = candles[j].low;
  }
  return (highest + lowest) / 2;
}

export function ichimoku(
  candles: OHLCV[],
  tenkanPeriod: number = 9,
  kijunPeriod: number = 26,
  senkouBPeriod: number = 52
): IchimokuResult {
  const tenkan: (number | null)[] = [];
  const kijun: (number | null)[] = [];
  const senkouA: (number | null)[] = [];
  const senkouB: (number | null)[] = [];
  const chikou: (number | null)[] = [];

  for (let i = 0; i < candles.length; i++) {
    tenkan.push(donchianMid(candles, tenkanPeriod, i));
    kijun.push(donchianMid(candles, kijunPeriod, i));
  }

  // Senkou A = (tenkan + kijun) / 2, shifted forward by kijunPeriod
  // Senkou B = donchian(senkouBPeriod), shifted forward by kijunPeriod
  // For display, we store these offset in the array (future values appended)
  for (let i = 0; i < candles.length + kijunPeriod; i++) {
    const srcIdx = i - kijunPeriod;
    if (srcIdx < 0 || srcIdx >= candles.length) {
      senkouA.push(null);
      senkouB.push(null);
    } else {
      const t = tenkan[srcIdx];
      const k = kijun[srcIdx];
      senkouA.push(t !== null && k !== null ? (t + k) / 2 : null);
      senkouB.push(donchianMid(candles, senkouBPeriod, srcIdx));
    }
  }

  // Chikou = close shifted back by kijunPeriod
  for (let i = 0; i < candles.length; i++) {
    const srcIdx = i + kijunPeriod;
    chikou.push(srcIdx < candles.length ? candles[srcIdx].close : null);
  }

  return { tenkan, kijun, senkouA, senkouB, chikou };
}

// ── Master calculator ──

export interface IndicatorRequest {
  name: string;
  params?: Record<string, number | string>;
}

export interface IndicatorOutput {
  name: string;
  type: 'overlay' | 'oscillator' | 'volume';
  series: Record<string, (number | null)[]>;
  times: number[];
}

export function parseIndicator(raw: string): IndicatorRequest {
  // Examples: ema_50, sma_200, rsi, macd, bollinger, stochastic_14_3_3
  const parts = raw.toLowerCase().split('_');
  const name = parts[0];
  const params: Record<string, number | string> = {};

  switch (name) {
    case 'ema':
    case 'sma':
      if (parts[1]) params.period = parseInt(parts[1]);
      break;
    case 'rsi':
      if (parts[1]) params.period = parseInt(parts[1]);
      break;
    case 'macd':
      if (parts[1]) params.fast = parseInt(parts[1]);
      if (parts[2]) params.slow = parseInt(parts[2]);
      if (parts[3]) params.signal = parseInt(parts[3]);
      break;
    case 'bollinger':
      if (parts[1]) params.period = parseInt(parts[1]);
      if (parts[2]) params.stddev = parseFloat(parts[2]);
      break;
    case 'stochastic':
      if (parts[1]) params.k = parseInt(parts[1]);
      if (parts[2]) params.d = parseInt(parts[2]);
      if (parts[3]) params.smooth = parseInt(parts[3]);
      break;
    case 'atr':
      if (parts[1]) params.period = parseInt(parts[1]);
      break;
    case 'cci':
      if (parts[1]) params.period = parseInt(parts[1]);
      break;
    case 'supertrend':
      if (parts[1]) params.period = parseInt(parts[1]);
      if (parts[2]) params.multiplier = parseFloat(parts[2]);
      break;
    case 'ichimoku':
      if (parts[1]) params.tenkan = parseInt(parts[1]);
      if (parts[2]) params.kijun = parseInt(parts[2]);
      if (parts[3]) params.senkou = parseInt(parts[3]);
      break;
  }

  return { name, params };
}

export function calculateIndicator(
  candles: OHLCV[],
  req: IndicatorRequest,
  source: string = 'close'
): IndicatorOutput {
  const times = candles.map(c => c.time);
  const src = getSource(candles, source);

  switch (req.name) {
    case 'ema': {
      const period = (req.params?.period as number) || 20;
      return {
        name: `EMA ${period}`,
        type: 'overlay',
        series: { value: ema(src, period) },
        times,
      };
    }
    case 'sma': {
      const period = (req.params?.period as number) || 20;
      return {
        name: `SMA ${period}`,
        type: 'overlay',
        series: { value: sma(src, period) },
        times,
      };
    }
    case 'rsi': {
      const period = (req.params?.period as number) || 14;
      return {
        name: `RSI ${period}`,
        type: 'oscillator',
        series: { value: rsi(src, period) },
        times,
      };
    }
    case 'macd': {
      const fast = (req.params?.fast as number) || 12;
      const slow = (req.params?.slow as number) || 26;
      const sig = (req.params?.signal as number) || 9;
      const result = macd(src, fast, slow, sig);
      return {
        name: `MACD ${fast},${slow},${sig}`,
        type: 'oscillator',
        series: { line: result.line, signal: result.signal, histogram: result.histogram },
        times,
      };
    }
    case 'bollinger': {
      const period = (req.params?.period as number) || 20;
      const sd = (req.params?.stddev as number) || 2;
      const result = bollinger(src, period, sd);
      return {
        name: `BB ${period},${sd}`,
        type: 'overlay',
        series: { upper: result.upper, middle: result.middle, lower: result.lower },
        times,
      };
    }
    case 'stochastic': {
      const k = (req.params?.k as number) || 14;
      const d = (req.params?.d as number) || 3;
      const smooth = (req.params?.smooth as number) || 3;
      const result = stochastic(candles, k, d, smooth);
      return {
        name: `Stoch ${k},${d},${smooth}`,
        type: 'oscillator',
        series: { k: result.k, d: result.d },
        times,
      };
    }
    case 'atr': {
      const period = (req.params?.period as number) || 14;
      return {
        name: `ATR ${period}`,
        type: 'oscillator',
        series: { value: atr(candles, period) },
        times,
      };
    }
    case 'obv': {
      return {
        name: 'OBV',
        type: 'volume',
        series: { value: obv(candles) },
        times,
      };
    }
    case 'vwap': {
      return {
        name: 'VWAP',
        type: 'overlay',
        series: { value: vwap(candles) },
        times,
      };
    }
    case 'cci': {
      const period = (req.params?.period as number) || 20;
      return {
        name: `CCI ${period}`,
        type: 'oscillator',
        series: { value: cci(candles, period) },
        times,
      };
    }
    case 'supertrend': {
      const period = (req.params?.period as number) || 10;
      const mult = (req.params?.multiplier as number) || 3;
      const result = supertrend(candles, period, mult);
      return {
        name: `ST ${period},${mult}`,
        type: 'overlay',
        series: { line: result.line, direction: result.direction },
        times,
      };
    }
    case 'ichimoku': {
      const tenkanP = (req.params?.tenkan as number) || 9;
      const kijunP = (req.params?.kijun as number) || 26;
      const senkouP = (req.params?.senkou as number) || 52;
      const result = ichimoku(candles, tenkanP, kijunP, senkouP);
      return {
        name: `Ichimoku ${tenkanP},${kijunP},${senkouP}`,
        type: 'overlay',
        series: {
          tenkan: result.tenkan,
          kijun: result.kijun,
          senkouA: result.senkouA,
          senkouB: result.senkouB,
          chikou: result.chikou,
        },
        times,
      };
    }
    default:
      return {
        name: req.name,
        type: 'overlay',
        series: {},
        times,
      };
  }
}
