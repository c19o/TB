export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Signal {
  type: 'lunar' | 'caution' | 'pump' | 'ritual' | 'tweet' | 'wyckoff' | 'elliott' | 'gann';
  date: string;
  timestamp: number;
  label: string;
  description: string;
  direction: 'bullish' | 'bearish' | 'neutral';
  strength: number;
  color: string;
}

export interface Tweet {
  tweet_id: string;
  user_handle: string;
  user_name: string;
  created_at: string;
  ts_unix: number;
  full_text: string;
  retweet_count: number;
  favorite_count: number;
  reply_count: number;
  is_retweet: number;
  date_gematria: string | null;
  day_of_year: number;
  gematria_hits?: string[];
  misdirection_flag?: boolean;
}

export interface UnifiedScore {
  score: number;
  confidence: number;
  action: string;
  components: {
    numerology: number;
    technical: number;
    tweets: number;
    manipulation: number;
  };
  threat_level: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
  inversion_warning: boolean;
  inversion_setup: boolean;
  phase_shift: boolean;
  volatility_alert: boolean;
  signal_count: number;
  bearish_count: number;
  bullish_count: number;
  neutral_count: number;
  signals: Array<{
    name: string;
    weight: number;
    direction: number;
    detail: string;
  }>;
  ta_bias?: string;
  ta_confidence?: number;
  wyckoff_phase?: string;
  elliott_wave?: string;
  gann_direction?: string;
  date: string;
}

export interface NumerologyData {
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

export interface OverlayPoint {
  time: number;
  type: string;
  label: string;
  color: string;
  direction?: 'bullish' | 'bearish' | 'neutral';
}

export type OverlayType = 'lunar' | 'caution' | 'pump' | 'ritual' | 'tweet' | 'wyckoff' | 'elliott' | 'gann';

export type Timeframe = '1m' | '5m' | '15m' | '1h' | '4h' | '1d' | '1w';

