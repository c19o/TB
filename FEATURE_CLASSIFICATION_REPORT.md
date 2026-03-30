# Feature Classification Report
Generated: 2026-03-18 22:47:06

## Timeframe: 1H

### Model Performance
- Direction classifier accuracy: 0.6898 (194 trees)
- Volatility regressor R2: 0.2621 (58 trees)
- Total features analyzed: 373

### Category Summary
| Category | Count | Pct |
|----------|-------|-----|
| DIRECTIONAL | 12 | 3.2% |
| VOLATILITY | 157 | 42.1% |
| DUAL | 90 | 24.1% |
| NOISE | 114 | 30.6% |

### Category Changes (vs previous run)
| Feature | Old | New |
|---------|-----|-----|
| ema_5 | NOISE | VOLATILITY |
| sma_10 | NOISE | VOLATILITY |
| sma_100 | NOISE | VOLATILITY |
| sma_200 | VOLATILITY | NOISE |
| ema_200 | DUAL | NOISE |
| above_sma200 | NOISE | DUAL |
| rsi_7 | DIRECTIONAL | VOLATILITY |
| rsi_14_os | NOISE | DUAL |
| bb_upper_20 | VOLATILITY | NOISE |
| bb_lower_20 | VOLATILITY | DUAL |
| bb_pctb_20 | DIRECTIONAL | VOLATILITY |
| bb_squeeze_20 | NOISE | DUAL |
| macd_histogram | DUAL | VOLATILITY |
| stoch_k_14 | VOLATILITY | DUAL |
| stoch_d_14 | DIRECTIONAL | VOLATILITY |
| ichimoku_kijun | NOISE | DUAL |
| ichimoku_senkou_b | DUAL | VOLATILITY |
| ichimoku_tk_cross | DUAL | NOISE |
| sar_bullish | NOISE | DUAL |
| supertrend | VOLATILITY | DUAL |
| pivot_s1 | VOLATILITY | DUAL |
| pivot_s2 | VOLATILITY | DUAL |
| obv_trend | NOISE | DUAL |
| vwap_24 | DUAL | VOLATILITY |
| body_pct | DIRECTIONAL | VOLATILITY |
| upper_wick | VOLATILITY | DIRECTIONAL |
| bear_engulfing | NOISE | DUAL |
| consec_red | DIRECTIONAL | DUAL |
| adx_14 | DUAL | VOLATILITY |
| cci_20 | DUAL | VOLATILITY |
| williams_r_14 | VOLATILITY | DUAL |
| mfi_14 | DIRECTIONAL | DUAL |
| cmf_20 | DIRECTIONAL | VOLATILITY |
| keltner_upper | NOISE | DUAL |
| keltner_lower | DUAL | NOISE |
| donchian_upper | VOLATILITY | DUAL |
| donchian_lower | DUAL | VOLATILITY |
| elliott_zigzag | DUAL | NOISE |
| consensio_score | NOISE | DUAL |
| h4_trend | NOISE | DUAL |
| h4_macd | VOLATILITY | DUAL |
| w_trend | NOISE | DUAL |
| w_bb_pctb | DUAL | VOLATILITY |
| hour_sin | DUAL | VOLATILITY |
| is_hour_0 | VOLATILITY | NOISE |
| is_hour_4 | VOLATILITY | DUAL |
| is_hour_8 | VOLATILITY | DUAL |
| is_hour_12 | NOISE | DUAL |
| is_hour_16 | DIRECTIONAL | DUAL |
| is_asia_session | DIRECTIONAL | DUAL |

### Top 20 Directional Features
| Feature | MI_dir | MI_vol | Ratio | Gain_A | Gain_B |
|---------|--------|--------|-------|--------|--------|
| is_hour_20 | 0.00813 | 0.00000 | 999.00 | 14.1 | 5.9 |
| upper_wick | 0.00588 | 0.00000 | 999.00 | 13.1 | 12.7 |
| price_dr | 0.00311 | 0.00000 | 999.00 | 0.0 | 11.5 |
| vedic_guna_encoded | 0.00174 | 0.00000 | 999.00 | 10.1 | 0.0 |
| west_digital_root | 0.00001 | 0.00000 | 999.00 | 7.6 | 5.4 |
| digital_root_price | 0.00204 | 0.00000 | 999.00 | 8.5 | 11.3 |
| rsi_14_lag8 | 0.00027 | 0.00000 | 999.00 | 9.4 | 11.0 |
| bb_pctb_20_lag48 | 0.00598 | 0.00000 | 999.00 | 9.6 | 14.0 |
| bb_pctb_20_lag8 | 0.00198 | 0.00000 | 999.00 | 9.8 | 12.7 |
| rsi_14_lag48 | 0.00411 | 0.00000 | 999.00 | 10.0 | 26.5 |
| is_73 | 0.00663 | 0.00009 | 72.30 | 0.0 | 6.7 |
| day_27 | 0.00655 | 0.00021 | 31.32 | 9.3 | 0.0 |
| sport_score_dr_mode | 0.00787 | 0.00106 | 7.42 | 10.0 | 0.0 |
| bazi_btc_friendly | 0.01162 | 0.00157 | 7.40 | 9.3 | 3.2 |
| macd_histogram_lag24 | 0.00729 | 0.00248 | 2.93 | 8.9 | 4.5 |
| knn_direction | 0.00486 | 0.00223 | 2.18 | 8.0 | 8.4 |
| bear_engulfing | 0.00535 | 0.00264 | 2.02 | 0.0 | 6.0 |
| consec_green | 0.00694 | 0.00454 | 1.53 | 22.6 | 6.6 |
| is_hour_12 | 0.00072 | 0.00055 | 1.31 | 15.1 | 7.5 |
| obv_trend | 0.00580 | 0.00475 | 1.22 | 0.0 | 27.7 |

### Top 20 Volatility Features
| Feature | MI_dir | MI_vol | Ratio | Gain_A | Gain_B |
|---------|--------|--------|-------|--------|--------|
| sma_100 | 0.00000 | 0.03075 | 0.00 | 9.7 | 9.1 |
| sma_50 | 0.00000 | 0.03243 | 0.00 | 7.6 | 8.9 |
| sma_20_slope | 0.00000 | 0.03678 | 0.00 | 9.0 | 11.1 |
| ema_100 | 0.00000 | 0.03688 | 0.00 | 0.0 | 6.5 |
| sma_200_slope | 0.00000 | 0.03989 | 0.00 | 8.2 | 6.1 |
| rsi_14_os | 0.00000 | 0.00314 | 0.00 | 12.2 | 0.0 |
| bb_pctb_20 | 0.00000 | 0.01493 | 0.00 | 23.8 | 10.1 |
| bb_lower_20 | 0.00000 | 0.03395 | 0.00 | 12.1 | 3.9 |
| volatility_ratio | 0.00000 | 0.00262 | 0.00 | 9.8 | 7.5 |
| volatility_10bar | 0.00000 | 0.08070 | 0.00 | 12.9 | 6.2 |
| atr_14_pct | 0.00000 | 0.09120 | 0.00 | 10.7 | 136.7 |
| macd_signal | 0.00000 | 0.02199 | 0.00 | 9.4 | 6.3 |
| ichimoku_below_cloud | 0.00000 | 0.00212 | 0.00 | 23.0 | 0.0 |
| ichimoku_senkou_b | 0.00000 | 0.03004 | 0.00 | 5.9 | 6.3 |
| cci_20 | 0.00000 | 0.01485 | 0.00 | 10.8 | 10.1 |
| keltner_upper | 0.00000 | 0.03509 | 0.00 | 14.2 | 4.8 |
| obv_sma_20 | 0.00000 | 0.04719 | 0.00 | 9.1 | 7.5 |
| donchian_upper | 0.00000 | 0.03603 | 0.00 | 11.3 | 0.0 |
| donchian_lower | 0.00000 | 0.03524 | 0.00 | 10.6 | 6.9 |
| wyckoff_phase | 0.00000 | 0.01269 | 0.00 | 7.6 | 6.2 |

### Full Feature Table
| Feature | Category | MI_dir | MI_vol | Ratio | Gain_A | Gain_B |
|---------|----------|--------|--------|-------|--------|--------|
| upper_wick | DIRECTIONAL | 0.00588 | 0.00000 | 999.00 | 13.1 | 12.7 |
| is_hour_20 | DIRECTIONAL | 0.00813 | 0.00000 | 999.00 | 14.1 | 5.9 |
| vedic_guna_encoded | DIRECTIONAL | 0.00174 | 0.00000 | 999.00 | 10.1 | 0.0 |
| rsi_14_lag8 | DIRECTIONAL | 0.00027 | 0.00000 | 999.00 | 9.4 | 11.0 |
| rsi_14_lag48 | DIRECTIONAL | 0.00411 | 0.00000 | 999.00 | 10.0 | 26.5 |
| bb_pctb_20_lag8 | DIRECTIONAL | 0.00198 | 0.00000 | 999.00 | 9.8 | 12.7 |
| bb_pctb_20_lag48 | DIRECTIONAL | 0.00598 | 0.00000 | 999.00 | 9.6 | 14.0 |
| day_27 | DIRECTIONAL | 0.00655 | 0.00021 | 31.32 | 9.3 | 0.0 |
| sport_score_dr_mode | DIRECTIONAL | 0.00787 | 0.00106 | 7.42 | 10.0 | 0.0 |
| bazi_btc_friendly | DIRECTIONAL | 0.01162 | 0.00157 | 7.40 | 9.3 | 3.2 |
| macd_histogram_lag24 | DIRECTIONAL | 0.00729 | 0.00248 | 2.93 | 8.9 | 4.5 |
| consec_green | DIRECTIONAL | 0.00694 | 0.00454 | 1.53 | 22.6 | 6.6 |
| digital_root_price | DUAL | 0.00204 | 0.00000 | 999.00 | 8.5 | 11.3 |
| west_digital_root | DUAL | 0.00001 | 0.00000 | 999.00 | 7.6 | 5.4 |
| price_dr | DUAL | 0.00311 | 0.00000 | 999.00 | 0.0 | 11.5 |
| is_73 | DUAL | 0.00663 | 0.00009 | 72.30 | 0.0 | 6.7 |
| knn_direction | DUAL | 0.00486 | 0.00223 | 2.18 | 8.0 | 8.4 |
| bear_engulfing | DUAL | 0.00535 | 0.00264 | 2.02 | 0.0 | 6.0 |
| is_hour_12 | DUAL | 0.00072 | 0.00055 | 1.31 | 15.1 | 7.5 |
| obv_trend | DUAL | 0.00580 | 0.00475 | 1.22 | 0.0 | 27.7 |
| is_hour_16 | DUAL | 0.00276 | 0.00237 | 1.17 | 11.8 | 8.6 |
| dow_cos | DUAL | 0.00583 | 0.00516 | 1.13 | 9.0 | 5.1 |
| ichimoku_above_cloud | DUAL | 0.00000 | 0.00000 | 1.00 | 19.3 | 0.0 |
| mfi_14 | DUAL | 0.00000 | 0.00000 | 1.00 | 8.1 | 5.5 |
| is_hour_6 | DUAL | 0.00000 | 0.00000 | 1.00 | 18.8 | 0.0 |
| is_asia_session | DUAL | 0.00000 | 0.00000 | 1.00 | 9.7 | 8.8 |
| is_ny_session | DUAL | 0.00000 | 0.00000 | 1.00 | 11.7 | 0.0 |
| day_of_month | DUAL | 0.00000 | 0.00000 | 1.00 | 8.2 | 7.8 |
| pump_date | DUAL | 0.00000 | 0.00000 | 1.00 | 9.3 | 0.0 |
| price_dr_6 | DUAL | 0.00000 | 0.00000 | 1.00 | 12.4 | 0.0 |
| vedic_nakshatra | DUAL | 0.00000 | 0.00000 | 1.00 | 8.6 | 0.0 |
| bazi_day_branch | DUAL | 0.00000 | 0.00000 | 1.00 | 7.9 | 36.8 |
| bazi_day_clash_branch | DUAL | 0.00000 | 0.00000 | 1.00 | 9.1 | 0.0 |
| arabic_lot_increase_moon_conj | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 5.6 |
| macro_gold_roc5d | DUAL | 0.00000 | 0.00000 | 1.00 | 8.5 | 20.5 |
| macro_hyg_roc5d | DUAL | 0.00000 | 0.00000 | 1.00 | 8.2 | 10.5 |
| bazi_branch_idx | DUAL | 0.00000 | 0.00000 | 1.00 | 8.2 | 11.5 |
| tzolkin_sign_idx | DUAL | 0.00000 | 0.00000 | 1.00 | 9.1 | 6.0 |
| tzolkin_kin | DUAL | 0.00000 | 0.00000 | 1.00 | 7.6 | 8.1 |
| above_sma200 | DUAL | 0.00801 | 0.00806 | 0.99 | 19.5 | 0.0 |
| consec_red | DUAL | 0.00386 | 0.00435 | 0.89 | 36.0 | 8.1 |
| macd_histogram_lag8 | DUAL | 0.00589 | 0.00696 | 0.85 | 8.7 | 4.2 |
| is_london_session | DUAL | 0.00274 | 0.00327 | 0.84 | 8.8 | 1.5 |
| h4_bb_pctb | DUAL | 0.00413 | 0.00496 | 0.83 | 20.4 | 14.3 |
| stoch_k_14 | DUAL | 0.00433 | 0.00531 | 0.81 | 23.5 | 8.1 |
| williams_r_14 | DUAL | 0.00424 | 0.00536 | 0.79 | 21.2 | 5.4 |
| bb_pctb_20_lag4 | DUAL | 0.00255 | 0.00482 | 0.53 | 12.2 | 2.7 |
| consensio_score | DUAL | 0.00273 | 0.00522 | 0.52 | 13.6 | 4.1 |
| bb_squeeze_20 | DUAL | 0.00279 | 0.00636 | 0.44 | 10.2 | 0.0 |
| vedic_nature_encoded | DUAL | 0.00140 | 0.00329 | 0.42 | 8.6 | 0.0 |
| macro_ibit | DUAL | 0.00605 | 0.01428 | 0.42 | 10.1 | 0.0 |
| w_trend | DUAL | 0.00309 | 0.00809 | 0.38 | 12.5 | 0.0 |
| consec_red_x_bb_os | DUAL | 0.00099 | 0.00271 | 0.37 | 12.8 | 3.9 |
| h4_trend | DUAL | 0.00271 | 0.00783 | 0.35 | 8.6 | 0.0 |
| rsi_14_lag24 | DUAL | 0.00362 | 0.01310 | 0.28 | 8.6 | 5.1 |
| bull_engulfing | DUAL | 0.00039 | 0.00148 | 0.26 | 11.3 | 0.0 |
| seasonal_vol_direction | DUAL | 0.00262 | 0.01473 | 0.18 | 9.6 | 3.7 |
| day_of_week | DUAL | 0.00180 | 0.01411 | 0.13 | 12.4 | 3.4 |
| current_dd_depth | DUAL | 0.00319 | 0.02748 | 0.12 | 9.7 | 4.3 |
| news_count_today | DUAL | 0.00210 | 0.02499 | 0.08 | 10.4 | 1.6 |
| macro_ibit_roc20d | DUAL | 0.00052 | 0.00831 | 0.06 | 9.5 | 0.0 |
| sw_bars_since_severe | DUAL | 0.00092 | 0.01637 | 0.06 | 18.4 | 0.0 |
| onchain_difficulty | DUAL | 0.00032 | 0.04406 | 0.01 | 10.0 | 2.2 |
| rsi_14_os | DUAL | 0.00000 | 0.00314 | 0.00 | 12.2 | 0.0 |
| bb_lower_20 | DUAL | 0.00000 | 0.03395 | 0.00 | 12.1 | 3.9 |
| ichimoku_kijun | DUAL | 0.00000 | 0.03846 | 0.00 | 9.1 | 0.0 |
| ichimoku_below_cloud | DUAL | 0.00000 | 0.00212 | 0.00 | 23.0 | 0.0 |
| sar_bullish | DUAL | 0.00000 | 0.00136 | 0.00 | 10.5 | 0.0 |
| supertrend | DUAL | 0.00000 | 0.03705 | 0.00 | 9.5 | 2.5 |
| supertrend_bullish | DUAL | 0.00000 | 0.01278 | 0.00 | 12.1 | 0.0 |
| pivot_s1 | DUAL | 0.00000 | 0.03461 | 0.00 | 9.3 | 0.0 |
| pivot_s2 | DUAL | 0.00000 | 0.03536 | 0.00 | 9.0 | 5.1 |
| keltner_upper | DUAL | 0.00000 | 0.03509 | 0.00 | 14.2 | 4.8 |
| donchian_upper | DUAL | 0.00000 | 0.03603 | 0.00 | 11.3 | 0.0 |
| h4_macd | DUAL | 0.00000 | 0.03126 | 0.00 | 9.0 | 4.7 |
| is_hour_4 | DUAL | 0.00000 | 0.00219 | 0.00 | 12.4 | 0.0 |
| is_hour_8 | DUAL | 0.00000 | 0.00098 | 0.00 | 9.8 | 0.0 |
| is_london_ny_overlap | DUAL | 0.00000 | 0.00277 | 0.00 | 9.6 | 1.0 |
| dow_sin | DUAL | 0.00000 | 0.01421 | 0.00 | 9.4 | 3.4 |
| price_contains_93 | DUAL | 0.00000 | 0.00198 | 0.00 | 12.6 | 0.0 |
| tweets_today | DUAL | 0.00000 | 0.01492 | 0.00 | 9.3 | 4.2 |
| news_count_1h | DUAL | 0.00000 | 0.00394 | 0.00 | 11.0 | 0.0 |
| macro_spx_roc20d | DUAL | 0.00000 | 0.00293 | 0.00 | 8.6 | 4.4 |
| macro_us10y_roc20d | DUAL | 0.00000 | 0.01431 | 0.00 | 8.6 | 5.1 |
| macro_russell_roc20d | DUAL | 0.00000 | 0.01885 | 0.00 | 8.5 | 3.1 |
| macro_oil_roc20d | DUAL | 0.00000 | 0.00058 | 0.00 | 8.9 | 4.0 |
| macro_silver_roc20d | DUAL | 0.00000 | 0.01842 | 0.00 | 9.0 | 2.9 |
| macro_mstr | DUAL | 0.00000 | 0.03302 | 0.00 | 8.9 | 3.3 |
| macro_mstr_roc20d | DUAL | 0.00000 | 0.02281 | 0.00 | 8.5 | 3.8 |
| macro_coin | DUAL | 0.00000 | 0.03112 | 0.00 | 10.9 | 3.9 |
| macro_coin_roc20d | DUAL | 0.00000 | 0.01730 | 0.00 | 9.6 | 4.2 |
| macro_tlt | DUAL | 0.00000 | 0.04161 | 0.00 | 9.7 | 4.4 |
| onchain_n_transactions | DUAL | 0.00000 | 0.02506 | 0.00 | 9.0 | 0.0 |
| golden_ratio_ext | DUAL | 0.00000 | 0.03136 | 0.00 | 9.4 | 0.0 |
| fib_21_from_low | DUAL | 0.00000 | 0.03146 | 0.00 | 9.5 | 4.1 |
| bb_pctb_20_lag24 | DUAL | 0.00000 | 0.00530 | 0.00 | 8.9 | 3.8 |
| fear_greed_lag120 | DUAL | 0.00000 | 0.02558 | 0.00 | 9.2 | 4.9 |
| bazi_stem_idx | DUAL | 0.00000 | 0.00311 | 0.00 | 9.6 | 3.2 |
| sw_sunspot_number | DUAL | 0.00000 | 0.00903 | 0.00 | 9.8 | 4.6 |
| schumann_133d_sin | DUAL | 0.00000 | 0.00327 | 0.00 | 9.0 | 4.8 |
| schumann_783d_sin | DUAL | 0.00000 | 0.00349 | 0.00 | 10.0 | 4.7 |
| esoteric_vol_score | DUAL | 0.00000 | 0.01227 | 0.00 | 10.5 | 2.4 |
| doji | NOISE | 0.00195 | 0.00000 | 999.00 | 0.0 | 3.0 |
| hammer | NOISE | 0.00483 | 0.00000 | 999.00 | 6.8 | 0.0 |
| is_quarter_end | NOISE | 0.00100 | 0.00000 | 999.00 | 0.0 | 0.0 |
| is_monday | NOISE | 0.00414 | 0.00000 | 999.00 | 0.0 | 0.0 |
| day_13 | NOISE | 0.00098 | 0.00000 | 999.00 | 0.0 | 0.0 |
| is_223 | NOISE | 0.00151 | 0.00000 | 999.00 | 0.0 | 0.0 |
| is_322 | NOISE | 0.00335 | 0.00000 | 999.00 | 0.0 | 0.0 |
| is_19 | NOISE | 0.00459 | 0.00000 | 999.00 | 0.0 | 0.0 |
| date_dr | NOISE | 0.00486 | 0.00000 | 999.00 | 8.3 | 0.0 |
| price_contains_322 | NOISE | 0.00210 | 0.00000 | 999.00 | 0.0 | 0.0 |
| price_contains_113 | NOISE | 0.00117 | 0.00000 | 999.00 | 0.0 | 0.0 |
| digital_root_genesis | NOISE | 0.00134 | 0.00000 | 999.00 | 7.7 | 0.0 |
| mayan_sign_idx | NOISE | 0.00161 | 0.00000 | 999.00 | 7.5 | 0.0 |
| sephirah | NOISE | 0.00375 | 0.00000 | 999.00 | 7.9 | 0.0 |
| shemitah_year | NOISE | 0.00274 | 0.00000 | 999.00 | 0.0 | 0.0 |
| headline_gem_ord_mean | NOISE | 0.00176 | 0.00000 | 999.00 | 0.0 | 0.0 |
| headline_caps_any | NOISE | 0.00141 | 0.00000 | 999.00 | 0.0 | 0.0 |
| news_article_count | NOISE | 0.00282 | 0.00000 | 999.00 | 0.0 | 0.0 |
| bazi_element_idx | NOISE | 0.00451 | 0.00000 | 999.00 | 6.2 | 0.0 |
| sw_kp_is_storm | NOISE | 0.00260 | 0.00000 | 999.00 | 0.0 | 0.0 |
| sw_post_storm_7d | NOISE | 0.00312 | 0.00000 | 999.00 | 0.0 | 0.0 |
| headline_caution_any | NOISE | 0.00580 | 0.00016 | 37.36 | 0.0 | 0.0 |
| headline_gem_dr_mode | NOISE | 0.00907 | 0.00025 | 36.42 | 0.0 | 0.0 |
| btc_213 | NOISE | 0.00381 | 0.00030 | 12.90 | 0.0 | 0.0 |
| supertrend_flip | NOISE | 0.00270 | 0.00039 | 6.86 | 0.0 | 0.0 |
| ichimoku_in_cloud | NOISE | 0.00695 | 0.00107 | 6.51 | 0.0 | 0.0 |
| is_hour_0 | NOISE | 0.00454 | 0.00087 | 5.21 | 0.0 | 0.0 |
| day_21 | NOISE | 0.00099 | 0.00020 | 4.88 | 0.0 | 0.0 |
| vortex_369 | NOISE | 0.00633 | 0.00138 | 4.59 | 8.5 | 0.0 |
| shooting_star | NOISE | 0.00564 | 0.00152 | 3.71 | 0.0 | 3.0 |
| sw_severe_decay | NOISE | 0.00138 | 0.00048 | 2.90 | 0.0 | 0.0 |
| near_fib_13 | NOISE | 0.00271 | 0.00097 | 2.79 | 0.0 | 0.0 |
| gold_tweet_this_1h | NOISE | 0.00149 | 0.00060 | 2.50 | 0.0 | 0.0 |
| sar_flip | NOISE | 0.00601 | 0.00245 | 2.45 | 0.0 | 0.0 |
| gold_tweet_today | NOISE | 0.00329 | 0.00161 | 2.04 | 0.0 | 0.0 |
| vedic_key_nakshatra | NOISE | 0.00730 | 0.00387 | 1.88 | 0.0 | 0.0 |
| west_soft_aspects | NOISE | 0.00476 | 0.00275 | 1.73 | 7.9 | 4.7 |
| mayan_tone_1 | NOISE | 0.00353 | 0.00275 | 1.28 | 0.0 | 0.0 |
| cross_date_price_dr_match | NOISE | 0.00158 | 0.00158 | 1.00 | 6.3 | 0.0 |
| golden_cross | NOISE | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| macd_cross_up | NOISE | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| is_month_end | NOISE | 0.00000 | 0.00000 | 1.00 | 5.3 | 0.0 |
| is_37 | NOISE | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| is_17 | NOISE | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| west_moon_mansion | NOISE | 0.00000 | 0.00000 | 1.00 | 7.7 | 4.0 |
| bazi_day_element_idx | NOISE | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| mayan_tone_13 | NOISE | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| arabic_lot_commerce_moon_conj | NOISE | 0.00000 | 0.00000 | 1.00 | 6.3 | 0.0 |
| arabic_lot_treachery_moon_conj | NOISE | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| tweets_this_1h | NOISE | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| price_is_master | NOISE | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| headline_sentiment_mean | NOISE | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| horse_races_today | NOISE | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| equinox_pre_post | NOISE | 0.00052 | 0.00053 | 0.97 | 7.9 | 0.0 |
| sport_overtime_count | NOISE | 0.00142 | 0.00197 | 0.72 | 0.0 | 0.0 |
| rsi_14_ob | NOISE | 0.00211 | 0.00313 | 0.67 | 0.0 | 0.0 |
| eclipse_window | NOISE | 0.00063 | 0.00099 | 0.63 | 6.1 | 0.0 |
| west_hard_aspects | NOISE | 0.00436 | 0.00706 | 0.62 | 5.8 | 0.0 |
| gtrends_interest_high | NOISE | 0.00399 | 0.00662 | 0.60 | 0.0 | 0.0 |
| chakra_heart_161d_sin | NOISE | 0.00353 | 0.00823 | 0.43 | 8.5 | 4.5 |
| jubilee_proximity | NOISE | 0.00851 | 0.02133 | 0.40 | 0.0 | 0.0 |
| sw_bars_since_storm | NOISE | 0.00591 | 0.02347 | 0.25 | 7.2 | 0.0 |
| month_cos | NOISE | 0.00308 | 0.01255 | 0.25 | 8.4 | 0.0 |
| sport_winner_gem_dr_mode | NOISE | 0.00095 | 0.00400 | 0.24 | 0.0 | 1.5 |
| funding_rate_neg | NOISE | 0.00126 | 0.00561 | 0.22 | 0.0 | 0.0 |
| onchain_miners_revenue | NOISE | 0.00356 | 0.02290 | 0.16 | 8.3 | 2.7 |
| sma_20 | NOISE | 0.00604 | 0.04068 | 0.15 | 0.0 | 0.0 |
| num_bullish_signals | NOISE | 0.00104 | 0.00813 | 0.13 | 8.3 | 4.4 |
| ema_200 | NOISE | 0.00382 | 0.03226 | 0.12 | 4.6 | 0.0 |
| ema_10 | NOISE | 0.00375 | 0.03857 | 0.10 | 0.0 | 0.0 |
| dd_from_ath | NOISE | 0.00247 | 0.02750 | 0.09 | 7.0 | 2.6 |
| fear_greed_lag240 | NOISE | 0.00043 | 0.00737 | 0.06 | 7.8 | 4.9 |
| d_trend | NOISE | 0.00034 | 0.00624 | 0.06 | 0.0 | 0.0 |
| sma_200 | NOISE | 0.00140 | 0.03047 | 0.05 | 8.1 | 0.0 |
| sma_5 | NOISE | 0.00019 | 0.03792 | 0.01 | 6.3 | 0.0 |
| ema_20 | NOISE | 0.00000 | 0.03948 | 0.00 | 0.0 | 0.0 |
| ema_50 | NOISE | 0.00000 | 0.03489 | 0.00 | 0.0 | 0.0 |
| death_cross | NOISE | 0.00000 | 0.00046 | 0.00 | 0.0 | 0.0 |
| bb_upper_20 | NOISE | 0.00000 | 0.04047 | 0.00 | 5.3 | 0.0 |
| macd_cross_down | NOISE | 0.00000 | 0.00074 | 0.00 | 6.1 | 0.0 |
| ichimoku_tk_cross | NOISE | 0.00000 | 0.00043 | 0.00 | 0.0 | 0.0 |
| keltner_lower | NOISE | 0.00000 | 0.03231 | 0.00 | 8.3 | 3.1 |
| elliott_zigzag | NOISE | 0.00000 | 0.00189 | 0.00 | 0.0 | 0.0 |
| consensio_green_wave | NOISE | 0.00000 | 0.00012 | 0.00 | 0.0 | 0.0 |
| consensio_red_wave | NOISE | 0.00000 | 0.00183 | 0.00 | 0.0 | 0.0 |
| is_friday | NOISE | 0.00000 | 0.00329 | 0.00 | 0.0 | 0.0 |
| month | NOISE | 0.00000 | 0.00523 | 0.00 | 6.2 | 0.0 |
| day_6 | NOISE | 0.00000 | 0.00125 | 0.00 | 0.0 | 0.0 |
| is_113 | NOISE | 0.00000 | 0.00077 | 0.00 | 0.0 | 0.0 |
| is_93 | NOISE | 0.00000 | 0.00006 | 0.00 | 0.0 | 0.0 |
| is_39 | NOISE | 0.00000 | 0.00002 | 0.00 | 0.0 | 0.0 |
| west_planetary_strength | NOISE | 0.00000 | 0.01255 | 0.00 | 7.7 | 4.1 |
| mayan_tone_9 | NOISE | 0.00000 | 0.00198 | 0.00 | 0.0 | 0.0 |
| arabic_lot_increase | NOISE | 0.00000 | 0.01826 | 0.00 | 8.0 | 2.9 |
| arabic_lot_catastrophe_moon_conj | NOISE | 0.00000 | 0.00034 | 0.00 | 0.0 | 0.0 |
| fg_extreme_fear | NOISE | 0.00000 | 0.01103 | 0.00 | 0.0 | 0.0 |
| fg_extreme_greed | NOISE | 0.00000 | 0.00578 | 0.00 | 0.0 | 0.0 |
| red_tweet_this_1h | NOISE | 0.00000 | 0.00008 | 0.00 | 0.0 | 0.0 |
| red_tweet_today | NOISE | 0.00000 | 0.00123 | 0.00 | 0.0 | 0.0 |
| misdirection | NOISE | 0.00000 | 0.00169 | 0.00 | 0.0 | 0.0 |
| news_sentiment_1h | NOISE | 0.00000 | 0.00265 | 0.00 | 0.0 | 0.0 |
| caution_gematria_1h | NOISE | 0.00000 | 0.00270 | 0.00 | 0.0 | 0.0 |
| macro_ibit_roc5d | NOISE | 0.00000 | 0.01215 | 0.00 | 0.0 | 0.0 |
| btc_vix_corr | NOISE | 0.00000 | 0.01307 | 0.00 | 8.0 | 3.3 |
| onchain_hash_rate_capitulation | NOISE | 0.00000 | 0.00313 | 0.00 | 0.0 | 0.0 |
| funding_rate_high | NOISE | 0.00000 | 0.00115 | 0.00 | 0.0 | 0.0 |
| near_fib_21 | NOISE | 0.00000 | 0.00150 | 0.00 | 0.0 | 0.0 |
| price_contains_213 | NOISE | 0.00000 | 0.00093 | 0.00 | 0.0 | 0.0 |
| sport_games_today | NOISE | 0.00000 | 0.00532 | 0.00 | 7.6 | 0.0 |
| horse_winner_gem_dr_mode | NOISE | 0.00000 | 0.00140 | 0.00 | 0.0 | 0.0 |
| cross_moon_x_news_caution | NOISE | 0.00000 | 0.00023 | 0.00 | 0.0 | 0.0 |
| sw_kp_is_severe | NOISE | 0.00000 | 0.00014 | 0.00 | 0.0 | 0.0 |
| sw_storm_decay | NOISE | 0.00000 | 0.00082 | 0.00 | 8.5 | 0.0 |
| date_palindrome | NOISE | 0.00000 | 0.00010 | 0.00 | 0.0 | 0.0 |
| hour_cos | VOLATILITY | 0.00628 | 0.00942 | 0.67 | 12.4 | 6.0 |
| schumann_143d_sin | VOLATILITY | 0.00257 | 0.00403 | 0.64 | 8.2 | 5.2 |
| gann_sq9_distance | VOLATILITY | 0.00437 | 0.00729 | 0.60 | 8.5 | 9.7 |
| rsi_21 | VOLATILITY | 0.00180 | 0.00311 | 0.58 | 18.9 | 30.9 |
| doy_cos | VOLATILITY | 0.00246 | 0.00484 | 0.51 | 9.1 | 10.7 |
| fg_vs_price_div | VOLATILITY | 0.00298 | 0.00602 | 0.49 | 8.9 | 10.0 |
| cmf_20 | VOLATILITY | 0.00234 | 0.00494 | 0.47 | 9.9 | 16.0 |
| bb_pctb_20_lag1 | VOLATILITY | 0.00081 | 0.00181 | 0.45 | 10.1 | 8.5 |
| rsi_14 | VOLATILITY | 0.00332 | 0.00797 | 0.42 | 18.4 | 9.7 |
| body_pct | VOLATILITY | 0.00073 | 0.00188 | 0.39 | 10.3 | 8.9 |
| knn_confidence | VOLATILITY | 0.00119 | 0.00317 | 0.37 | 7.7 | 5.9 |
| equinox_proximity | VOLATILITY | 0.00269 | 0.00881 | 0.31 | 7.9 | 6.9 |
| pivot_r2 | VOLATILITY | 0.01057 | 0.03589 | 0.29 | 6.1 | 11.1 |
| macd_line | VOLATILITY | 0.00423 | 0.01527 | 0.28 | 9.0 | 5.3 |
| macd_histogram | VOLATILITY | 0.00380 | 0.01390 | 0.27 | 10.0 | 7.1 |
| h4_return | VOLATILITY | 0.01763 | 0.07468 | 0.24 | 54.4 | 28.7 |
| mayan_tone | VOLATILITY | 0.00211 | 0.01009 | 0.21 | 7.9 | 22.3 |
| tzolkin_tone | VOLATILITY | 0.00214 | 0.01038 | 0.21 | 7.7 | 10.1 |
| rsi_14_lag1 | VOLATILITY | 0.00124 | 0.00637 | 0.19 | 11.1 | 7.6 |
| knn_avg_return | VOLATILITY | 0.00070 | 0.00378 | 0.18 | 8.8 | 8.0 |
| w_rsi14 | VOLATILITY | 0.00478 | 0.02743 | 0.17 | 9.8 | 10.8 |
| macd_histogram_lag48 | VOLATILITY | 0.00263 | 0.01692 | 0.16 | 9.2 | 8.0 |
| obv | VOLATILITY | 0.00578 | 0.04205 | 0.14 | 7.8 | 21.3 |
| cycle_confluence_score | VOLATILITY | 0.00083 | 0.00649 | 0.13 | 8.4 | 31.0 |
| return_8bar | VOLATILITY | 0.00400 | 0.03398 | 0.12 | 14.1 | 15.1 |
| return_4bar | VOLATILITY | 0.00328 | 0.03274 | 0.10 | 57.7 | 17.9 |
| gann_sq9_level | VOLATILITY | 0.00342 | 0.03436 | 0.10 | 0.0 | 5.9 |
| gtrends_interest | VOLATILITY | 0.00272 | 0.03060 | 0.09 | 7.5 | 8.6 |
| return_24bar | VOLATILITY | 0.00224 | 0.02522 | 0.09 | 14.5 | 13.9 |
| h4_vol_ratio | VOLATILITY | 0.00267 | 0.03315 | 0.08 | 11.0 | 25.2 |
| bb_width_20 | VOLATILITY | 0.00504 | 0.06389 | 0.08 | 12.8 | 9.1 |
| adx_14 | VOLATILITY | 0.00118 | 0.01542 | 0.08 | 9.6 | 18.6 |
| return_48bar | VOLATILITY | 0.00259 | 0.03798 | 0.07 | 12.0 | 8.8 |
| rsi_7 | VOLATILITY | 0.00096 | 0.01542 | 0.06 | 32.7 | 12.5 |
| atr_14 | VOLATILITY | 0.00186 | 0.03127 | 0.06 | 8.6 | 6.4 |
| ichimoku_tenkan | VOLATILITY | 0.00210 | 0.03826 | 0.05 | 0.0 | 71.8 |
| bazi_day_stem | VOLATILITY | 0.00015 | 0.00290 | 0.05 | 0.0 | 28.2 |
| news_sentiment_today | VOLATILITY | 0.00058 | 0.01168 | 0.05 | 9.1 | 8.2 |
| ema_5 | VOLATILITY | 0.00182 | 0.03831 | 0.05 | 0.0 | 7.0 |
| sma_50_slope | VOLATILITY | 0.00145 | 0.03267 | 0.04 | 15.2 | 7.9 |
| sma_10 | VOLATILITY | 0.00149 | 0.03688 | 0.04 | 8.0 | 27.0 |
| h4_ema50_dist | VOLATILITY | 0.00173 | 0.05138 | 0.03 | 9.3 | 36.2 |
| knn_pattern_std | VOLATILITY | 0.00227 | 0.07060 | 0.03 | 10.1 | 11.1 |
| vwap_24 | VOLATILITY | 0.00115 | 0.03647 | 0.03 | 8.2 | 12.9 |
| sw_solar_cycle_phase | VOLATILITY | 0.00029 | 0.00952 | 0.03 | 0.0 | 12.6 |
| sw_solar_flux_f107 | VOLATILITY | 0.00051 | 0.02034 | 0.02 | 8.9 | 6.3 |
| return_1bar | VOLATILITY | 0.00080 | 0.04883 | 0.02 | 26.6 | 11.4 |
| ema50_declining | VOLATILITY | 0.00016 | 0.01216 | 0.01 | 0.0 | 20.9 |
| stoch_d_14 | VOLATILITY | 0.00011 | 0.00924 | 0.01 | 10.6 | 9.9 |
| d_rsi14 | VOLATILITY | 0.00008 | 0.01978 | 0.00 | 9.1 | 8.0 |
| volatility_24bar | VOLATILITY | 0.00008 | 0.07743 | 0.00 | 10.5 | 9.8 |
| sma_50 | VOLATILITY | 0.00000 | 0.03243 | 0.00 | 7.6 | 8.9 |
| sma_100 | VOLATILITY | 0.00000 | 0.03075 | 0.00 | 9.7 | 9.1 |
| ema_100 | VOLATILITY | 0.00000 | 0.03688 | 0.00 | 0.0 | 6.5 |
| sma_20_slope | VOLATILITY | 0.00000 | 0.03678 | 0.00 | 9.0 | 11.1 |
| sma_200_slope | VOLATILITY | 0.00000 | 0.03989 | 0.00 | 8.2 | 6.1 |
| bb_pctb_20 | VOLATILITY | 0.00000 | 0.01493 | 0.00 | 23.8 | 10.1 |
| macd_signal | VOLATILITY | 0.00000 | 0.02199 | 0.00 | 9.4 | 6.3 |
| atr_14_pct | VOLATILITY | 0.00000 | 0.09120 | 0.00 | 10.7 | 136.7 |
| volatility_10bar | VOLATILITY | 0.00000 | 0.08070 | 0.00 | 12.9 | 6.2 |
| volatility_ratio | VOLATILITY | 0.00000 | 0.00262 | 0.00 | 9.8 | 7.5 |
| ichimoku_senkou_a | VOLATILITY | 0.00000 | 0.03460 | 0.00 | 4.8 | 18.6 |
| ichimoku_senkou_b | VOLATILITY | 0.00000 | 0.03004 | 0.00 | 5.9 | 6.3 |
| sar_value | VOLATILITY | 0.00000 | 0.03178 | 0.00 | 5.8 | 11.4 |
| pivot | VOLATILITY | 0.00000 | 0.03679 | 0.00 | 0.0 | 19.9 |
| pivot_r1 | VOLATILITY | 0.00000 | 0.03873 | 0.00 | 9.6 | 23.7 |
| obv_sma_20 | VOLATILITY | 0.00000 | 0.04719 | 0.00 | 9.1 | 7.5 |
| cci_20 | VOLATILITY | 0.00000 | 0.01485 | 0.00 | 10.8 | 10.1 |
| donchian_lower | VOLATILITY | 0.00000 | 0.03524 | 0.00 | 10.6 | 6.9 |
| wyckoff_phase | VOLATILITY | 0.00000 | 0.01269 | 0.00 | 7.6 | 6.2 |
| h4_rsi14 | VOLATILITY | 0.00000 | 0.01686 | 0.00 | 11.7 | 10.2 |
| h4_volatility | VOLATILITY | 0.00000 | 0.07560 | 0.00 | 8.9 | 22.8 |
| h4_atr_pct | VOLATILITY | 0.00000 | 0.08925 | 0.00 | 10.1 | 66.0 |
| d_ema50_dist | VOLATILITY | 0.00000 | 0.03828 | 0.00 | 10.0 | 10.6 |
| d_bb_pctb | VOLATILITY | 0.00000 | 0.01321 | 0.00 | 11.1 | 13.3 |
| d_macd | VOLATILITY | 0.00000 | 0.02721 | 0.00 | 8.6 | 7.4 |
| d_return | VOLATILITY | 0.00000 | 0.02857 | 0.00 | 23.0 | 7.4 |
| d_volatility | VOLATILITY | 0.00000 | 0.04641 | 0.00 | 9.3 | 7.1 |
| w_bb_pctb | VOLATILITY | 0.00000 | 0.02867 | 0.00 | 6.2 | 11.6 |
| hour_sin | VOLATILITY | 0.00000 | 0.01418 | 0.00 | 11.5 | 5.7 |
| month_sin | VOLATILITY | 0.00000 | 0.00404 | 0.00 | 5.8 | 8.2 |
| doy_sin | VOLATILITY | 0.00000 | 0.00802 | 0.00 | 7.3 | 10.0 |
| is_weekend | VOLATILITY | 0.00000 | 0.01010 | 0.00 | 0.0 | 7.2 |
| day_of_year | VOLATILITY | 0.00000 | 0.01020 | 0.00 | 10.2 | 9.3 |
| west_moon_phase | VOLATILITY | 0.00000 | 0.01572 | 0.00 | 7.6 | 12.9 |
| psi | VOLATILITY | 0.00000 | 0.01283 | 0.00 | 7.3 | 8.2 |
| lunar_phase_sin | VOLATILITY | 0.00000 | 0.00799 | 0.00 | 8.3 | 9.1 |
| lunar_phase_cos | VOLATILITY | 0.00000 | 0.00597 | 0.00 | 8.4 | 6.1 |
| vedic_tithi | VOLATILITY | 0.00000 | 0.01064 | 0.00 | 7.7 | 17.0 |
| vedic_yoga | VOLATILITY | 0.00000 | 0.00547 | 0.00 | 8.4 | 7.1 |
| arabic_lot_commerce | VOLATILITY | 0.00000 | 0.00078 | 0.00 | 8.4 | 7.8 |
| arabic_lot_catastrophe | VOLATILITY | 0.00000 | 0.01674 | 0.00 | 10.5 | 5.6 |
| arabic_lot_treachery | VOLATILITY | 0.00000 | 0.00653 | 0.00 | 9.4 | 7.9 |
| fear_greed | VOLATILITY | 0.00000 | 0.02223 | 0.00 | 11.4 | 11.6 |
| fg_roc | VOLATILITY | 0.00000 | 0.02142 | 0.00 | 8.7 | 6.2 |
| macro_dxy | VOLATILITY | 0.00000 | 0.03895 | 0.00 | 8.9 | 6.2 |
| macro_dxy_roc5d | VOLATILITY | 0.00000 | 0.00875 | 0.00 | 8.2 | 7.8 |
| macro_dxy_roc20d | VOLATILITY | 0.00000 | 0.00697 | 0.00 | 9.5 | 13.6 |
| macro_gold | VOLATILITY | 0.00000 | 0.03662 | 0.00 | 9.1 | 10.2 |
| macro_gold_roc20d | VOLATILITY | 0.00000 | 0.01640 | 0.00 | 8.9 | 6.5 |
| macro_spx | VOLATILITY | 0.00000 | 0.03415 | 0.00 | 9.1 | 6.9 |
| macro_spx_roc5d | VOLATILITY | 0.00000 | 0.00105 | 0.00 | 8.8 | 7.6 |
| macro_vix | VOLATILITY | 0.00000 | 0.02105 | 0.00 | 8.7 | 6.9 |
| macro_vix_roc5d | VOLATILITY | 0.00000 | 0.00145 | 0.00 | 8.5 | 7.6 |
| macro_vix_roc20d | VOLATILITY | 0.00000 | 0.01049 | 0.00 | 8.0 | 9.4 |
| macro_us10y | VOLATILITY | 0.00000 | 0.03426 | 0.00 | 9.5 | 5.5 |
| macro_us10y_roc5d | VOLATILITY | 0.00000 | 0.01047 | 0.00 | 9.4 | 12.3 |
| macro_nasdaq | VOLATILITY | 0.00000 | 0.02549 | 0.00 | 10.0 | 9.3 |
| macro_nasdaq_roc5d | VOLATILITY | 0.00000 | 0.00415 | 0.00 | 8.9 | 10.9 |
| macro_nasdaq_roc20d | VOLATILITY | 0.00000 | 0.02542 | 0.00 | 8.7 | 7.3 |
| macro_russell | VOLATILITY | 0.00000 | 0.01624 | 0.00 | 9.4 | 7.3 |
| macro_russell_roc5d | VOLATILITY | 0.00000 | 0.01530 | 0.00 | 7.3 | 11.0 |
| macro_oil | VOLATILITY | 0.00000 | 0.02081 | 0.00 | 8.8 | 7.3 |
| macro_oil_roc5d | VOLATILITY | 0.00000 | 0.00216 | 0.00 | 8.0 | 8.0 |
| macro_silver | VOLATILITY | 0.00000 | 0.03402 | 0.00 | 10.9 | 5.6 |
| macro_silver_roc5d | VOLATILITY | 0.00000 | 0.00449 | 0.00 | 9.3 | 9.4 |
| macro_mstr_roc5d | VOLATILITY | 0.00000 | 0.01008 | 0.00 | 7.6 | 6.3 |
| macro_coin_roc5d | VOLATILITY | 0.00000 | 0.00906 | 0.00 | 7.0 | 10.6 |
| macro_hyg | VOLATILITY | 0.00000 | 0.05023 | 0.00 | 9.8 | 10.3 |
| macro_hyg_roc20d | VOLATILITY | 0.00000 | 0.00946 | 0.00 | 8.8 | 5.8 |
| macro_tlt_roc5d | VOLATILITY | 0.00000 | 0.00150 | 0.00 | 8.5 | 10.8 |
| macro_tlt_roc20d | VOLATILITY | 0.00000 | 0.00705 | 0.00 | 8.3 | 9.5 |
| btc_dxy_corr | VOLATILITY | 0.00000 | 0.01104 | 0.00 | 8.2 | 5.3 |
| btc_gold_corr | VOLATILITY | 0.00000 | 0.00626 | 0.00 | 8.3 | 5.8 |
| btc_spx_corr | VOLATILITY | 0.00000 | 0.00728 | 0.00 | 8.7 | 7.2 |
| onchain_hash_rate | VOLATILITY | 0.00000 | 0.04207 | 0.00 | 10.7 | 5.9 |
| onchain_mempool_size | VOLATILITY | 0.00000 | 0.01675 | 0.00 | 9.2 | 8.7 |
| onchain_hash_rate_roc | VOLATILITY | 0.00000 | 0.00464 | 0.00 | 8.5 | 17.6 |
| funding_rate | VOLATILITY | 0.00000 | 0.01980 | 0.00 | 8.8 | 5.6 |
| golden_ratio_dist | VOLATILITY | 0.00000 | 0.05377 | 0.00 | 10.4 | 10.8 |
| fib_13_from_high | VOLATILITY | 0.00000 | 0.03837 | 0.00 | 8.9 | 5.9 |
| rsi_14_lag4 | VOLATILITY | 0.00000 | 0.00520 | 0.00 | 11.2 | 6.3 |
| macd_histogram_lag1 | VOLATILITY | 0.00000 | 0.00933 | 0.00 | 9.6 | 6.4 |
| macd_histogram_lag4 | VOLATILITY | 0.00000 | 0.01684 | 0.00 | 10.6 | 6.3 |
| fear_greed_lag24 | VOLATILITY | 0.00000 | 0.02672 | 0.00 | 9.7 | 13.2 |
| fear_greed_lag72 | VOLATILITY | 0.00000 | 0.02658 | 0.00 | 9.0 | 5.4 |
| num_bearish_signals | VOLATILITY | 0.00000 | 0.00328 | 0.00 | 11.6 | 14.6 |
| signal_agreement | VOLATILITY | 0.00000 | 0.01149 | 0.00 | 8.5 | 6.4 |
| rsi_x_bbpctb | VOLATILITY | 0.00000 | 0.00582 | 0.00 | 15.5 | 16.7 |
| fg_x_moon_phase | VOLATILITY | 0.00000 | 0.00620 | 0.00 | 9.7 | 26.3 |
| rsi_bullish_div | VOLATILITY | 0.00000 | 0.00332 | 0.00 | 0.0 | 6.0 |
| ema50_rising | VOLATILITY | 0.00000 | 0.00586 | 0.00 | 0.0 | 7.0 |
| ema50_slope | VOLATILITY | 0.00000 | 0.03864 | 0.00 | 12.2 | 6.9 |
| sw_kp_index | VOLATILITY | 0.00000 | 0.00582 | 0.00 | 8.4 | 17.6 |
| sw_kp_delta_3d | VOLATILITY | 0.00000 | 0.01040 | 0.00 | 8.7 | 7.2 |
| sw_kp_x_moon_phase | VOLATILITY | 0.00000 | 0.00370 | 0.00 | 8.3 | 9.4 |
| schumann_133d_cos | VOLATILITY | 0.00000 | 0.00718 | 0.00 | 8.5 | 13.7 |
| schumann_143d_cos | VOLATILITY | 0.00000 | 0.00678 | 0.00 | 8.1 | 11.1 |
| schumann_783d_cos | VOLATILITY | 0.00000 | 0.00917 | 0.00 | 9.9 | 5.7 |
| chakra_heart_161d_cos | VOLATILITY | 0.00000 | 0.00240 | 0.00 | 8.3 | 10.1 |
| jupiter_365d_sin | VOLATILITY | 0.00000 | 0.00457 | 0.00 | 7.0 | 14.5 |
| jupiter_365d_cos | VOLATILITY | 0.00000 | 0.01142 | 0.00 | 9.3 | 17.4 |
| mercury_1216d_sin | VOLATILITY | 0.00000 | 0.01220 | 0.00 | 9.4 | 9.2 |
| mercury_1216d_cos | VOLATILITY | 0.00000 | 0.01308 | 0.00 | 7.6 | 5.8 |
| vol_regime_transition | VOLATILITY | 0.00000 | 0.01005 | 0.00 | 9.1 | 10.0 |
| vol_directional_asymmetry | VOLATILITY | 0.00000 | 0.00735 | 0.00 | 9.1 | 15.3 |
| knn_best_match_dist | VOLATILITY | 0.00000 | 0.00767 | 0.00 | 7.7 | 7.6 |

## Timeframe: 4H

### Model Performance
- Direction classifier accuracy: 0.6485 (36 trees)
- Volatility regressor R2: 0.1790 (52 trees)
- Total features analyzed: 510

### Category Summary
| Category | Count | Pct |
|----------|-------|-----|
| DIRECTIONAL | 87 | 17.1% |
| VOLATILITY | 187 | 36.7% |
| DUAL | 236 | 46.3% |
| NOISE | 0 | 0.0% |

### Category Changes (vs previous run)
| Feature | Old | New |
|---------|-----|-----|
| sma_5 | NOISE | DUAL |
| ema_5 | NOISE | DUAL |
| sma_10 | NOISE | DUAL |
| ema_10 | NOISE | DUAL |
| sma_20 | NOISE | VOLATILITY |
| ema_20 | NOISE | VOLATILITY |
| ema_50 | NOISE | VOLATILITY |
| sma_100 | NOISE | DUAL |
| golden_cross | NOISE | DIRECTIONAL |
| death_cross | NOISE | DUAL |
| above_sma200 | NOISE | DUAL |
| rsi_14_ob | NOISE | DUAL |
| rsi_14_os | NOISE | DUAL |
| bb_squeeze_20 | NOISE | DIRECTIONAL |
| macd_cross_up | NOISE | DUAL |
| macd_cross_down | NOISE | DIRECTIONAL |
| ichimoku_tenkan | VOLATILITY | DUAL |
| ichimoku_kijun | NOISE | DUAL |
| ichimoku_senkou_b | DUAL | VOLATILITY |
| ichimoku_in_cloud | NOISE | DUAL |
| sar_bullish | NOISE | DUAL |
| sar_flip | NOISE | DIRECTIONAL |
| supertrend_flip | NOISE | DIRECTIONAL |
| pivot | VOLATILITY | DUAL |
| pivot_r2 | VOLATILITY | DUAL |
| obv_trend | NOISE | DUAL |
| vwap_6 | VOLATILITY | DUAL |
| doji | NOISE | DUAL |
| hammer | NOISE | DUAL |
| shooting_star | NOISE | DUAL |
| bear_engulfing | NOISE | DUAL |
| keltner_upper | NOISE | VOLATILITY |
| consensio_score | NOISE | DUAL |
| consensio_green_wave | NOISE | DUAL |
| consensio_red_wave | NOISE | DUAL |
| golden_ratio_ext | DUAL | VOLATILITY |
| fib_21_from_low | NOISE | DUAL |
| fib_13_from_high | VOLATILITY | DUAL |
| near_fib_13 | NOISE | DUAL |
| consec_red_x_bb_os | NOISE | DIRECTIONAL |
| rsi_bullish_div | NOISE | DUAL |
| num_bullish_signals | NOISE | VOLATILITY |
| month_sin | DUAL | DIRECTIONAL |
| month_cos | NOISE | DUAL |
| is_hour_12 | NOISE | DUAL |
| month | DUAL | VOLATILITY |
| is_month_end | NOISE | DUAL |
| is_quarter_end | NOISE | DIRECTIONAL |
| is_monday | NOISE | DIRECTIONAL |
| is_friday | DUAL | DIRECTIONAL |

### Top 20 Directional Features
| Feature | MI_dir | MI_vol | Ratio | Gain_A | Gain_B |
|---------|--------|--------|-------|--------|--------|
| date_palindrome | 0.00401 | 0.00000 | 2054.21 | 0.0 | 0.0 |
| body_pct | 0.00190 | 0.00000 | 999.00 | 9.8 | 15.7 |
| stoch_d_14 | 0.00250 | 0.00000 | 999.00 | 12.4 | 18.2 |
| cross_tweet_gem_price_dr_match | 0.00099 | 0.00000 | 999.00 | 0.0 | 0.0 |
| eclipse_decay_slow | 0.00913 | 0.00000 | 999.00 | 8.3 | 3.3 |
| bb_pctb_20_lag42 | 0.00122 | 0.00000 | 999.00 | 9.2 | 16.7 |
| cross_price_tweet_dr_match | 0.00692 | 0.00000 | 999.00 | 0.0 | 0.0 |
| consec_green | 0.00091 | 0.00000 | 999.00 | 17.1 | 15.0 |
| tweet_astro_nakshatra_nature | 0.00942 | 0.00000 | 999.00 | 0.0 | 0.0 |
| tweet_astro_nakshatra_guna | 0.00252 | 0.00000 | 999.00 | 0.0 | 0.0 |
| d_trend | 0.00179 | 0.00000 | 999.00 | 0.0 | 0.0 |
| sw_kp_is_severe | 0.00043 | 0.00000 | 999.00 | 0.0 | 0.0 |
| sw_storm_decay | 0.00636 | 0.00000 | 999.00 | 0.0 | 7.7 |
| gem_match_date_news | 0.00274 | 0.00000 | 999.00 | 0.0 | 0.0 |
| mfi_14 | 0.00306 | 0.00000 | 999.00 | 9.9 | 11.3 |
| cmf_20 | 0.00185 | 0.00000 | 999.00 | 6.7 | 26.9 |
| gem_pump_convergence | 0.00285 | 0.00000 | 999.00 | 0.0 | 0.0 |
| consec_red | 0.00117 | 0.00000 | 999.00 | 17.3 | 6.5 |
| tweet_astro_is_new_moon | 0.00161 | 0.00000 | 999.00 | 0.0 | 0.0 |
| west_hard_aspects | 0.00366 | 0.00000 | 999.00 | 0.0 | 31.4 |

### Top 20 Volatility Features
| Feature | MI_dir | MI_vol | Ratio | Gain_A | Gain_B |
|---------|--------|--------|-------|--------|--------|
| ema_200 | 0.00000 | 0.02527 | 0.00 | 0.0 | 0.0 |
| cross_fg_greed_x_funding | 0.00000 | 0.00787 | 0.00 | 0.0 | 16.6 |
| cross_nakshatra_x_moon | 0.00000 | 0.00793 | 0.00 | 10.5 | 13.9 |
| cross_shmita_x_kp_storm | 0.00000 | 0.00002 | 0.00 | 0.0 | 0.0 |
| cross_gold_tweet_x_funding | 0.00000 | 0.00002 | 0.00 | 0.0 | 0.0 |
| cross_caps_tweet_x_vol | 0.00000 | 0.00408 | 0.00 | 0.0 | 0.0 |
| cross_fg_fear_x_eclipse | 0.00000 | 0.00328 | 0.00 | 0.0 | 0.0 |
| return_1bar | 0.00000 | 0.02622 | 0.00 | 20.1 | 10.4 |
| rsi_21 | 0.00000 | 0.00655 | 0.00 | 15.6 | 12.2 |
| rsi_14_ob | 0.00000 | 0.00130 | 0.00 | 0.0 | 0.0 |
| bb_upper_20 | 0.00000 | 0.02783 | 0.00 | 5.2 | 16.7 |
| bb_lower_20 | 0.00000 | 0.02128 | 0.00 | 1.5 | 12.6 |
| gold_tweet_decay_fast | 0.00000 | 0.00305 | 0.00 | 0.0 | 0.0 |
| gold_tweet_decay_slow | 0.00000 | 0.00195 | 0.00 | 0.0 | 0.0 |
| bars_since_red_tweet | 0.00000 | 0.00496 | 0.00 | 0.0 | 0.0 |
| ichimoku_senkou_a | 0.00000 | 0.02737 | 0.00 | 10.4 | 15.1 |
| red_tweet_decay_fast | 0.00000 | 0.00354 | 0.00 | 0.0 | 0.0 |
| sma_50_slope | 0.00000 | 0.03456 | 0.00 | 10.0 | 11.2 |
| death_cross | 0.00000 | 0.00001 | 0.00 | 0.0 | 0.0 |
| obv | 0.00000 | 0.04461 | 0.00 | 10.2 | 20.3 |

### Full Feature Table
| Feature | Category | MI_dir | MI_vol | Ratio | Gain_A | Gain_B |
|---------|----------|--------|--------|-------|--------|--------|
| date_palindrome | DIRECTIONAL | 0.00401 | 0.00000 | 2054.21 | 0.0 | 0.0 |
| golden_cross | DIRECTIONAL | 0.00546 | 0.00000 | 999.00 | 0.0 | 0.0 |
| rsi_7 | DIRECTIONAL | 0.00579 | 0.00000 | 999.00 | 12.8 | 31.2 |
| bb_pctb_20 | DIRECTIONAL | 0.00454 | 0.00000 | 999.00 | 6.9 | 24.5 |
| macd_cross_down | DIRECTIONAL | 0.00320 | 0.00000 | 999.00 | 0.0 | 0.0 |
| stoch_d_14 | DIRECTIONAL | 0.00250 | 0.00000 | 999.00 | 12.4 | 18.2 |
| body_pct | DIRECTIONAL | 0.00190 | 0.00000 | 999.00 | 9.8 | 15.7 |
| consec_green | DIRECTIONAL | 0.00091 | 0.00000 | 999.00 | 17.1 | 15.0 |
| consec_red | DIRECTIONAL | 0.00117 | 0.00000 | 999.00 | 17.3 | 6.5 |
| mfi_14 | DIRECTIONAL | 0.00306 | 0.00000 | 999.00 | 9.9 | 11.3 |
| cmf_20 | DIRECTIONAL | 0.00185 | 0.00000 | 999.00 | 6.7 | 26.9 |
| bb_pctb_20_lag42 | DIRECTIONAL | 0.00122 | 0.00000 | 999.00 | 9.2 | 16.7 |
| month_sin | DIRECTIONAL | 0.00657 | 0.00000 | 999.00 | 0.0 | 10.4 |
| is_hour_16 | DIRECTIONAL | 0.00092 | 0.00000 | 999.00 | 17.9 | 17.9 |
| is_london_ny_overlap | DIRECTIONAL | 0.00067 | 0.00000 | 999.00 | 11.6 | 33.1 |
| is_quarter_end | DIRECTIONAL | 0.00874 | 0.00000 | 999.00 | 0.0 | 0.0 |
| is_friday | DIRECTIONAL | 0.00216 | 0.00000 | 999.00 | 0.0 | 0.0 |
| is_93 | DIRECTIONAL | 0.00078 | 0.00000 | 999.00 | 0.0 | 0.0 |
| is_223 | DIRECTIONAL | 0.00338 | 0.00000 | 999.00 | 0.0 | 0.0 |
| is_37 | DIRECTIONAL | 0.00299 | 0.00000 | 999.00 | 0.0 | 0.0 |
| is_17 | DIRECTIONAL | 0.00486 | 0.00000 | 999.00 | 0.0 | 0.0 |
| is_19 | DIRECTIONAL | 0.00175 | 0.00000 | 999.00 | 0.0 | 0.0 |
| price_contains_322 | DIRECTIONAL | 0.00336 | 0.00000 | 999.00 | 0.0 | 0.0 |
| price_contains_213 | DIRECTIONAL | 0.00400 | 0.00000 | 999.00 | 0.0 | 0.0 |
| west_moon_mansion | DIRECTIONAL | 0.00260 | 0.00000 | 999.00 | 7.9 | 6.6 |
| west_hard_aspects | DIRECTIONAL | 0.00366 | 0.00000 | 999.00 | 0.0 | 31.4 |
| west_digital_root | DIRECTIONAL | 0.00260 | 0.00000 | 999.00 | 0.0 | 12.6 |
| vedic_tithi | DIRECTIONAL | 0.00194 | 0.00000 | 999.00 | 9.3 | 22.0 |
| vedic_yoga | DIRECTIONAL | 0.00269 | 0.00000 | 999.00 | 8.7 | 5.9 |
| mayan_tone | DIRECTIONAL | 0.00079 | 0.00000 | 999.00 | 6.4 | 0.0 |
| mayan_tone_13 | DIRECTIONAL | 0.00655 | 0.00000 | 999.00 | 0.0 | 0.0 |
| arabic_lot_increase_moon_conj | DIRECTIONAL | 0.00108 | 0.00000 | 999.00 | 0.0 | 0.0 |
| tweets_today | DIRECTIONAL | 0.00246 | 0.00000 | 999.00 | 0.0 | 0.0 |
| news_gem_reverse_mean | DIRECTIONAL | 0.00204 | 0.00000 | 999.00 | 9.4 | 14.7 |
| news_gem_dr_rev_mode | DIRECTIONAL | 0.00242 | 0.00000 | 999.00 | 0.0 | 4.4 |
| headline_sentiment_mean | DIRECTIONAL | 0.00269 | 0.00000 | 999.00 | 8.7 | 0.0 |
| headline_caps_any | DIRECTIONAL | 0.00794 | 0.00000 | 999.00 | 0.0 | 8.1 |
| sport_winner_gem_dr_mode | DIRECTIONAL | 0.00023 | 0.00000 | 999.00 | 0.0 | 0.0 |
| cross_price_tweet_dr_match | DIRECTIONAL | 0.00692 | 0.00000 | 999.00 | 0.0 | 0.0 |
| gem_match_date_news | DIRECTIONAL | 0.00274 | 0.00000 | 999.00 | 0.0 | 0.0 |
| gem_pump_convergence | DIRECTIONAL | 0.00285 | 0.00000 | 999.00 | 0.0 | 0.0 |
| tweet_astro_is_new_moon | DIRECTIONAL | 0.00161 | 0.00000 | 999.00 | 0.0 | 0.0 |
| tweet_astro_nakshatra_nature | DIRECTIONAL | 0.00942 | 0.00000 | 999.00 | 0.0 | 0.0 |
| tweet_astro_nakshatra_guna | DIRECTIONAL | 0.00252 | 0.00000 | 999.00 | 0.0 | 0.0 |
| d_trend | DIRECTIONAL | 0.00179 | 0.00000 | 999.00 | 0.0 | 0.0 |
| sw_kp_is_severe | DIRECTIONAL | 0.00043 | 0.00000 | 999.00 | 0.0 | 0.0 |
| sw_storm_decay | DIRECTIONAL | 0.00636 | 0.00000 | 999.00 | 0.0 | 7.7 |
| cross_tweet_gem_price_dr_match | DIRECTIONAL | 0.00099 | 0.00000 | 999.00 | 0.0 | 0.0 |
| eclipse_decay_slow | DIRECTIONAL | 0.00913 | 0.00000 | 999.00 | 8.3 | 3.3 |
| kp_storm_decay_fast | DIRECTIONAL | 0.00066 | 0.00000 | 999.00 | 0.0 | 0.0 |
| is_hour_20 | DIRECTIONAL | 0.00563 | 0.00003 | 195.16 | 33.7 | 8.6 |
| cross_moon_x_gold_tweet | DIRECTIONAL | 0.00294 | 0.00002 | 117.92 | 0.0 | 0.0 |
| cross_kp_storm_x_moon | DIRECTIONAL | 0.00310 | 0.00007 | 41.56 | 0.0 | 0.0 |
| sw_kp_is_storm | DIRECTIONAL | 0.00237 | 0.00006 | 36.77 | 0.0 | 0.0 |
| bars_since_caps_tweet | DIRECTIONAL | 0.00755 | 0.00023 | 32.61 | 0.0 | 0.0 |
| digital_root_price | DIRECTIONAL | 0.01099 | 0.00049 | 22.59 | 8.9 | 10.6 |
| cross_consec_green_x_caps | DIRECTIONAL | 0.00170 | 0.00008 | 22.04 | 0.0 | 0.0 |
| is_asia_session | DIRECTIONAL | 0.00029 | 0.00002 | 13.50 | 12.1 | 18.4 |
| gold_tweet_this_4h | DIRECTIONAL | 0.00329 | 0.00032 | 10.45 | 0.0 | 0.0 |
| sar_flip | DIRECTIONAL | 0.00158 | 0.00023 | 6.97 | 0.0 | 0.0 |
| consec_red_x_bb_os | DIRECTIONAL | 0.00661 | 0.00119 | 5.54 | 0.0 | 0.0 |
| supertrend_flip | DIRECTIONAL | 0.00080 | 0.00016 | 4.93 | 0.0 | 0.0 |
| tweet_caps_any | DIRECTIONAL | 0.00461 | 0.00103 | 4.49 | 0.0 | 0.0 |
| news_astro_is_new_moon | DIRECTIONAL | 0.00198 | 0.00048 | 4.12 | 0.0 | 0.0 |
| news_gem_dr_jew_mode | DIRECTIONAL | 0.00438 | 0.00107 | 4.08 | 10.6 | 5.6 |
| price_is_master | DIRECTIONAL | 0.00626 | 0.00156 | 4.02 | 0.0 | 0.0 |
| tweet_gem_dr_rev_mode | DIRECTIONAL | 0.00938 | 0.00248 | 3.79 | 0.0 | 0.0 |
| red_tweet_today | DIRECTIONAL | 0.00648 | 0.00186 | 3.48 | 0.0 | 0.0 |
| cross_eclipse_x_consec_green | DIRECTIONAL | 0.00253 | 0.00073 | 3.48 | 0.0 | 0.0 |
| btc_213 | DIRECTIONAL | 0.00090 | 0.00026 | 3.44 | 0.0 | 0.0 |
| knn_confidence | DIRECTIONAL | 0.00478 | 0.00140 | 3.42 | 7.7 | 10.2 |
| gem_match_sport_news | DIRECTIONAL | 0.00384 | 0.00118 | 3.26 | 0.0 | 0.0 |
| macd_histogram_lag42 | DIRECTIONAL | 0.00642 | 0.00205 | 3.13 | 9.6 | 10.2 |
| gem_match_decay_fast | DIRECTIONAL | 0.00813 | 0.00291 | 2.80 | 0.0 | 9.8 |
| btc_dxy_corr | DIRECTIONAL | 0.00315 | 0.00125 | 2.52 | 10.7 | 10.2 |
| is_monday | DIRECTIONAL | 0.00232 | 0.00094 | 2.47 | 0.0 | 0.0 |
| cross_date_price_dr_match | DIRECTIONAL | 0.00664 | 0.00295 | 2.25 | 0.0 | 0.0 |
| gem_match_date_tweet | DIRECTIONAL | 0.00257 | 0.00118 | 2.18 | 0.0 | 0.0 |
| tweet_excl_max | DIRECTIONAL | 0.00346 | 0.00163 | 2.12 | 0.0 | 0.0 |
| bb_squeeze_20 | DIRECTIONAL | 0.00425 | 0.00210 | 2.02 | 0.0 | 7.4 |
| bazi_day_clash_branch | DIRECTIONAL | 0.00449 | 0.00242 | 1.85 | 8.9 | 22.3 |
| sport_score_diff_dr_mode | DIRECTIONAL | 0.00293 | 0.00159 | 1.85 | 0.0 | 0.0 |
| bars_since_gold_tweet | DIRECTIONAL | 0.00398 | 0.00228 | 1.75 | 0.0 | 0.0 |
| signal_agreement | DIRECTIONAL | 0.00561 | 0.00328 | 1.71 | 14.0 | 8.9 |
| cross_tweet_news_gem_match | DIRECTIONAL | 0.00234 | 0.00146 | 1.61 | 0.0 | 0.0 |
| tweet_gem_jewish_mean | DIRECTIONAL | 0.00312 | 0.00196 | 1.59 | 7.6 | 0.0 |
| tweet_sentiment_mean | DIRECTIONAL | 0.00797 | 0.00504 | 1.58 | 0.0 | 0.0 |
| rsi_x_bbpctb | DUAL | 0.00750 | 0.00510 | 1.47 | 11.0 | 21.0 |
| elliott_zigzag | DUAL | 0.00578 | 0.00406 | 1.42 | 0.0 | 11.5 |
| tweet_gem_btc_energy | DUAL | 0.00223 | 0.00160 | 1.39 | 0.0 | 0.0 |
| headline_gem_ord_mean | DUAL | 0.00488 | 0.00360 | 1.36 | 0.0 | 4.0 |
| sport_loser_gem_english_mean | DUAL | 0.00719 | 0.00531 | 1.35 | 0.0 | 0.0 |
| adx_14 | DUAL | 0.01090 | 0.00834 | 1.31 | 8.6 | 13.9 |
| date_dr | DUAL | 0.00947 | 0.00725 | 1.31 | 8.7 | 21.6 |
| hour_sin | DUAL | 0.00123 | 0.00098 | 1.26 | 12.0 | 13.1 |
| bb_pctb_20_lag24 | DUAL | 0.00842 | 0.00682 | 1.23 | 9.2 | 12.5 |
| cross_shmita_x_bear | DUAL | 0.00390 | 0.00320 | 1.22 | 0.0 | 0.0 |
| news_gem_reduction_mean | DUAL | 0.00513 | 0.00448 | 1.14 | 0.0 | 8.2 |
| shemitah_year | DUAL | 0.00160 | 0.00148 | 1.08 | 0.0 | 0.0 |
| onchain_hash_rate_capitulation | DUAL | 0.00346 | 0.00327 | 1.06 | 0.0 | 0.0 |
| tweet_astro_nakshatra | DUAL | 0.00452 | 0.00428 | 1.06 | 10.9 | 0.0 |
| new_moon_decay_slow | DUAL | 0.00148 | 0.00146 | 1.01 | 0.0 | 0.0 |
| macd_cross_up | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| ichimoku_in_cloud | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| sar_bullish | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| supertrend_bullish | DUAL | 0.00000 | 0.00000 | 1.00 | 9.7 | 38.5 |
| obv_trend | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| doji | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| shooting_star | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| bull_engulfing | DUAL | 0.00000 | 0.00000 | 1.00 | 10.9 | 0.0 |
| bear_engulfing | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 8.2 |
| bb_pctb_20_lag1 | DUAL | 0.00000 | 0.00000 | 1.00 | 7.5 | 20.6 |
| bb_pctb_20_lag6 | DUAL | 0.00000 | 0.00000 | 1.00 | 11.8 | 15.2 |
| is_hour_12 | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 18.7 |
| is_ny_session | DUAL | 0.00000 | 0.00000 | 1.00 | 8.2 | 0.0 |
| is_month_end | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| pump_date | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| price_dr_6 | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| west_moon_phase | DUAL | 0.00000 | 0.00000 | 1.00 | 5.9 | 12.5 |
| digital_root_genesis | DUAL | 0.00000 | 0.00000 | 1.00 | 13.1 | 21.2 |
| lunar_phase_sin | DUAL | 0.00000 | 0.00000 | 1.00 | 9.1 | 14.8 |
| vedic_nakshatra | DUAL | 0.00000 | 0.00000 | 1.00 | 7.5 | 8.0 |
| bazi_day_element_idx | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| bazi_btc_friendly | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| mayan_tone_1 | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 8.7 |
| arabic_lot_catastrophe_moon_conj | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| tweet_gem_caution | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| news_gem_dr_eng_mode | DUAL | 0.00000 | 0.00000 | 1.00 | 11.7 | 9.0 |
| news_gem_caution | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| news_gem_pump | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| news_gem_btc_energy | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| headline_caution_any | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| macro_tlt_roc5d | DUAL | 0.00000 | 0.00000 | 1.00 | 8.2 | 3.2 |
| macro_tlt_roc20d | DUAL | 0.00000 | 0.00000 | 1.00 | 8.3 | 12.4 |
| btc_spx_corr | DUAL | 0.00000 | 0.00000 | 1.00 | 8.1 | 15.8 |
| gem_match_price_tweet | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| gem_match_horse_tweet | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| gem_match_price_sport | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| gem_caution_convergence | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| news_astro_nakshatra | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| news_astro_planetary_hour_idx | DUAL | 0.00000 | 0.00000 | 1.00 | 7.6 | 5.3 |
| sw_kp_delta_3d | DUAL | 0.00000 | 0.00000 | 1.00 | 7.2 | 18.2 |
| sw_post_storm_7d | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| jupiter_365d_sin | DUAL | 0.00000 | 0.00000 | 1.00 | 7.3 | 7.4 |
| equinox_pre_post | DUAL | 0.00000 | 0.00000 | 1.00 | 9.6 | 13.7 |
| cross_moon_x_news_caution | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| cross_nakshatra_x_red_tweet | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| cross_news_caution_x_moon | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| cross_gold_tweet_x_consec_green | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| cross_red_tweet_x_bear | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| cross_day13_x_red_tweet | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| cross_schumann_peak_x_kp_storm | DUAL | 0.00000 | 0.00000 | 1.00 | 0.0 | 0.0 |
| eclipse_decay_fast | DUAL | 0.00000 | 0.00000 | 1.00 | 10.3 | 5.8 |
| knn_direction | DUAL | 0.00000 | 0.00000 | 1.00 | 7.6 | 8.5 |
| knn_avg_return | DUAL | 0.00000 | 0.00000 | 1.00 | 8.1 | 15.8 |
| bars_since_eclipse | DUAL | 0.00589 | 0.00591 | 1.00 | 8.6 | 14.4 |
| rsi_14_lag24 | DUAL | 0.00340 | 0.00344 | 0.99 | 11.2 | 10.2 |
| num_bearish_signals | DUAL | 0.00378 | 0.00388 | 0.98 | 7.5 | 13.2 |
| tweet_gem_dr_mode | DUAL | 0.00258 | 0.00288 | 0.90 | 0.0 | 0.0 |
| macd_histogram | DUAL | 0.00309 | 0.00358 | 0.86 | 8.1 | 17.4 |
| tweet_user_gem_reverse_mean | DUAL | 0.00338 | 0.00418 | 0.81 | 0.0 | 13.5 |
| rsi_14_lag42 | DUAL | 0.00361 | 0.00453 | 0.80 | 7.1 | 11.6 |
| tweet_gem_reduction_mean | DUAL | 0.00468 | 0.00596 | 0.79 | 0.0 | 7.8 |
| tweet_gem_english_mean | DUAL | 0.00438 | 0.00567 | 0.77 | 0.0 | 0.0 |
| tweet_user_gem_ordinal_mean | DUAL | 0.00232 | 0.00301 | 0.77 | 0.0 | 15.3 |
| tweet_astro_moon_phase_day | DUAL | 0.00548 | 0.00709 | 0.77 | 0.0 | 7.7 |
| doy_cos | DUAL | 0.00205 | 0.00268 | 0.76 | 7.4 | 4.3 |
| cci_20 | DUAL | 0.00894 | 0.01181 | 0.76 | 7.4 | 18.5 |
| tweet_gem_dr_sat_mode | DUAL | 0.00132 | 0.00184 | 0.72 | 0.0 | 0.0 |
| btc_gold_corr | DUAL | 0.00550 | 0.00765 | 0.72 | 0.7 | 9.1 |
| sport_loser_gem_ordinal_mean | DUAL | 0.00444 | 0.00645 | 0.69 | 7.8 | 13.6 |
| funding_rate_high | DUAL | 0.00094 | 0.00144 | 0.65 | 0.0 | 0.0 |
| gem_triple_match | DUAL | 0.00055 | 0.00084 | 0.65 | 0.0 | 0.0 |
| sport_venue_gem_reverse_mean | DUAL | 0.00401 | 0.00634 | 0.63 | 0.0 | 0.0 |
| news_caution_decay_fast | DUAL | 0.00628 | 0.01010 | 0.62 | 11.4 | 5.0 |
| above_sma200 | DUAL | 0.00277 | 0.00471 | 0.59 | 5.0 | 0.0 |
| sport_loser_gem_dr_mode | DUAL | 0.00224 | 0.00402 | 0.56 | 0.0 | 0.0 |
| tweets_this_4h | DUAL | 0.00333 | 0.00609 | 0.55 | 0.0 | 0.0 |
| sport_venue_gem_dr_mode | DUAL | 0.00401 | 0.00738 | 0.54 | 0.0 | 0.0 |
| tweet_gem_reverse_mean | DUAL | 0.00354 | 0.00759 | 0.47 | 0.0 | 0.0 |
| day_6 | DUAL | 0.00060 | 0.00150 | 0.40 | 0.0 | 0.0 |
| news_gem_english_mean | DUAL | 0.00099 | 0.00251 | 0.39 | 0.0 | 0.0 |
| tweet_gem_dr_jew_mode | DUAL | 0.00273 | 0.00723 | 0.38 | 9.9 | 0.0 |
| news_gem_jewish_mean | DUAL | 0.00458 | 0.01263 | 0.36 | 9.3 | 0.0 |
| ichimoku_below_cloud | DUAL | 0.00093 | 0.00260 | 0.36 | 0.0 | 0.0 |
| sw_solar_cycle_phase | DUAL | 0.00543 | 0.01623 | 0.33 | 0.0 | 0.0 |
| sport_venue_gem_satanic_mean | DUAL | 0.00200 | 0.00601 | 0.33 | 0.0 | 0.0 |
| ichimoku_above_cloud | DUAL | 0.00107 | 0.00337 | 0.32 | 11.0 | 0.0 |
| w_trend | DUAL | 0.00063 | 0.00227 | 0.28 | 0.0 | 0.0 |
| cross_fg_fear_x_moon | DUAL | 0.00188 | 0.00847 | 0.22 | 0.0 | 0.0 |
| cross_moon_x_tweet_caution | DUAL | 0.00025 | 0.00112 | 0.22 | 0.0 | 0.0 |
| consensio_red_wave | DUAL | 0.00155 | 0.00718 | 0.22 | 0.0 | 0.0 |
| w_macd | DUAL | 0.00455 | 0.02844 | 0.16 | 10.7 | 0.0 |
| rsi_14_os | DUAL | 0.00061 | 0.00413 | 0.15 | 0.0 | 0.0 |
| w_rsi14 | DUAL | 0.00239 | 0.01875 | 0.13 | 8.3 | 0.0 |
| fg_extreme_fear | DUAL | 0.00116 | 0.01061 | 0.11 | 0.0 | 0.0 |
| news_gem_dr_ord_mode | DUAL | 0.00055 | 0.00569 | 0.10 | 0.0 | 0.0 |
| pivot_r2 | DUAL | 0.00230 | 0.02411 | 0.10 | 0.0 | 0.0 |
| tweet_user_gem_reduction_mean | DUAL | 0.00052 | 0.00579 | 0.09 | 0.0 | 0.0 |
| ema_5 | DUAL | 0.00255 | 0.03627 | 0.07 | 0.0 | 0.0 |
| w_ema50_dist | DUAL | 0.00248 | 0.03564 | 0.07 | 7.2 | 3.9 |
| is_weekend | DUAL | 0.00125 | 0.01808 | 0.07 | 0.0 | 0.0 |
| vwap_6 | DUAL | 0.00204 | 0.03408 | 0.06 | 0.0 | 0.0 |
| macro_hyg | DUAL | 0.00159 | 0.03720 | 0.04 | 9.9 | 0.0 |
| dd_from_ath | DUAL | 0.00093 | 0.02717 | 0.03 | 7.3 | 0.0 |
| consensio_score | DUAL | 0.00035 | 0.01433 | 0.02 | 9.0 | 0.0 |
| fib_13_from_high | DUAL | 0.00043 | 0.03778 | 0.01 | 0.0 | 4.7 |
| onchain_miners_revenue | DUAL | 0.00010 | 0.02188 | 0.00 | 8.4 | 0.0 |
| sma_5 | DUAL | 0.00000 | 0.03511 | 0.00 | 0.0 | 0.0 |
| sma_10 | DUAL | 0.00000 | 0.03219 | 0.00 | 0.0 | 0.0 |
| ema_10 | DUAL | 0.00000 | 0.04014 | 0.00 | 0.0 | 0.0 |
| sma_100 | DUAL | 0.00000 | 0.02167 | 0.00 | 0.0 | 0.0 |
| ema_200 | DUAL | 0.00000 | 0.02527 | 0.00 | 0.0 | 0.0 |
| death_cross | DUAL | 0.00000 | 0.00001 | 0.00 | 0.0 | 0.0 |
| rsi_14_ob | DUAL | 0.00000 | 0.00130 | 0.00 | 0.0 | 0.0 |
| ichimoku_tenkan | DUAL | 0.00000 | 0.03565 | 0.00 | 6.3 | 0.0 |
| ichimoku_kijun | DUAL | 0.00000 | 0.03436 | 0.00 | 9.4 | 0.0 |
| ichimoku_tk_cross | DUAL | 0.00000 | 0.00074 | 0.00 | 0.0 | 0.0 |
| pivot | DUAL | 0.00000 | 0.03460 | 0.00 | 0.0 | 0.0 |
| hammer | DUAL | 0.00000 | 0.00307 | 0.00 | 0.0 | 0.0 |
| keltner_lower | DUAL | 0.00000 | 0.02669 | 0.00 | 0.0 | 0.0 |
| donchian_lower | DUAL | 0.00000 | 0.02640 | 0.00 | 6.2 | 0.0 |
| consensio_green_wave | DUAL | 0.00000 | 0.00058 | 0.00 | 0.0 | 0.0 |
| fib_21_from_low | DUAL | 0.00000 | 0.02473 | 0.00 | 6.9 | 0.0 |
| near_fib_13 | DUAL | 0.00000 | 0.00068 | 0.00 | 0.0 | 0.0 |
| rsi_bullish_div | DUAL | 0.00000 | 0.00321 | 0.00 | 0.0 | 0.0 |
| month_cos | DUAL | 0.00000 | 0.00136 | 0.00 | 0.0 | 0.0 |
| day_13 | DUAL | 0.00000 | 0.00155 | 0.00 | 0.0 | 0.0 |
| day_21 | DUAL | 0.00000 | 0.00005 | 0.00 | 0.0 | 0.0 |
| day_27 | DUAL | 0.00000 | 0.00050 | 0.00 | 0.0 | 0.0 |
| is_113 | DUAL | 0.00000 | 0.00042 | 0.00 | 0.0 | 0.0 |
| is_39 | DUAL | 0.00000 | 0.00008 | 0.00 | 0.0 | 0.0 |
| is_322 | DUAL | 0.00000 | 0.00042 | 0.00 | 0.0 | 0.0 |
| is_73 | DUAL | 0.00000 | 0.00035 | 0.00 | 0.0 | 0.0 |
| price_contains_113 | DUAL | 0.00000 | 0.00091 | 0.00 | 0.0 | 0.0 |
| price_contains_93 | DUAL | 0.00000 | 0.00174 | 0.00 | 0.0 | 0.0 |
| vortex_369 | DUAL | 0.00000 | 0.00133 | 0.00 | 7.8 | 5.1 |
| sephirah | DUAL | 0.00000 | 0.00084 | 0.00 | 0.0 | 0.0 |
| jubilee_proximity | DUAL | 0.00000 | 0.03542 | 0.00 | 0.0 | 0.0 |
| vedic_key_nakshatra | DUAL | 0.00000 | 0.00007 | 0.00 | 0.0 | 3.8 |
| mayan_tone_9 | DUAL | 0.00000 | 0.00172 | 0.00 | 0.0 | 0.0 |
| arabic_lot_commerce_moon_conj | DUAL | 0.00000 | 0.00220 | 0.00 | 0.0 | 0.0 |
| arabic_lot_treachery_moon_conj | DUAL | 0.00000 | 0.00055 | 0.00 | 0.0 | 0.0 |
| red_tweet_this_4h | DUAL | 0.00000 | 0.00065 | 0.00 | 0.0 | 0.0 |
| gold_tweet_today | DUAL | 0.00000 | 0.00021 | 0.00 | 0.0 | 0.0 |
| misdirection | DUAL | 0.00000 | 0.00077 | 0.00 | 0.0 | 0.0 |
| tweet_gem_dr_ord_mode | DUAL | 0.00000 | 0.00328 | 0.00 | 0.0 | 0.0 |
| tweet_gem_dr_eng_mode | DUAL | 0.00000 | 0.00157 | 0.00 | 0.0 | 0.0 |
| tweet_user_gem_english_mean | DUAL | 0.00000 | 0.00305 | 0.00 | 0.0 | 0.0 |
| tweet_user_gem_jewish_mean | DUAL | 0.00000 | 0.00423 | 0.00 | 0.0 | 0.0 |
| tweet_user_gem_satanic_mean | DUAL | 0.00000 | 0.00582 | 0.00 | 0.0 | 0.0 |
| tweet_user_gem_dr_mode | DUAL | 0.00000 | 0.00423 | 0.00 | 0.0 | 0.0 |
| tweet_gem_pump | DUAL | 0.00000 | 0.00036 | 0.00 | 0.0 | 0.0 |
| tweet_gem_ord_mean | DUAL | 0.00000 | 0.00464 | 0.00 | 0.0 | 0.0 |
| user_gem_dr_mode | DUAL | 0.00000 | 0.00439 | 0.00 | 0.0 | 0.0 |
| caution_gematria_4h | DUAL | 0.00000 | 0.00363 | 0.00 | 0.0 | 0.0 |
| news_article_count | DUAL | 0.00000 | 0.00264 | 0.00 | 0.0 | 0.0 |
| sport_winner_gem_ordinal_mean | DUAL | 0.00000 | 0.00352 | 0.00 | 0.0 | 0.0 |
| sport_winner_gem_reduction_mean | DUAL | 0.00000 | 0.00700 | 0.00 | 0.0 | 0.0 |
| sport_winner_gem_english_mean | DUAL | 0.00000 | 0.00230 | 0.00 | 0.0 | 0.0 |
| sport_winner_gem_jewish_mean | DUAL | 0.00000 | 0.00163 | 0.00 | 8.0 | 0.0 |
| sport_winner_gem_satanic_mean | DUAL | 0.00000 | 0.00958 | 0.00 | 0.0 | 0.0 |
| sport_score_dr_mode | DUAL | 0.00000 | 0.00128 | 0.00 | 0.0 | 0.0 |
| sport_overtime_count | DUAL | 0.00000 | 0.00181 | 0.00 | 0.0 | 0.0 |
| sport_games_today | DUAL | 0.00000 | 0.00315 | 0.00 | 0.0 | 0.0 |
| sport_loser_gem_reverse_mean | DUAL | 0.00000 | 0.00451 | 0.00 | 0.0 | 0.0 |
| sport_loser_gem_reduction_mean | DUAL | 0.00000 | 0.00696 | 0.00 | 0.0 | 0.0 |
| sport_loser_gem_jewish_mean | DUAL | 0.00000 | 0.00881 | 0.00 | 0.0 | 0.7 |
| sport_loser_gem_satanic_mean | DUAL | 0.00000 | 0.00893 | 0.00 | 0.0 | 0.0 |
| sport_venue_gem_english_mean | DUAL | 0.00000 | 0.00798 | 0.00 | 0.0 | 0.0 |
| sport_venue_gem_jewish_mean | DUAL | 0.00000 | 0.00495 | 0.00 | 0.0 | 0.0 |
| sport_score_total_dr_mode | DUAL | 0.00000 | 0.00353 | 0.00 | 0.0 | 0.0 |
| macro_nasdaq | DUAL | 0.00000 | 0.02797 | 0.00 | 0.0 | 4.8 |
| macro_oil | DUAL | 0.00000 | 0.02401 | 0.00 | 8.2 | 4.7 |
| macro_coin | DUAL | 0.00000 | 0.02686 | 0.00 | 8.5 | 0.0 |
| macro_coin_roc5d | DUAL | 0.00000 | 0.01841 | 0.00 | 9.5 | 0.0 |
| macro_coin_roc20d | DUAL | 0.00000 | 0.01062 | 0.00 | 7.3 | 1.4 |
| macro_ibit | DUAL | 0.00000 | 0.00595 | 0.00 | 0.0 | 0.0 |
| macro_ibit_roc20d | DUAL | 0.00000 | 0.01194 | 0.00 | 0.0 | 0.0 |
| fg_extreme_greed | DUAL | 0.00000 | 0.00586 | 0.00 | 0.0 | 0.0 |
| fear_greed_lag6 | DUAL | 0.00000 | 0.03885 | 0.00 | 13.2 | 0.0 |
| fear_greed_lag18 | DUAL | 0.00000 | 0.03819 | 0.00 | 6.9 | 0.0 |
| funding_rate_neg | DUAL | 0.00000 | 0.00327 | 0.00 | 0.0 | 0.0 |
| gem_match_tweet_news | DUAL | 0.00000 | 0.00146 | 0.00 | 0.0 | 0.0 |
| gem_match_date_sport | DUAL | 0.00000 | 0.00202 | 0.00 | 0.0 | 0.0 |
| gem_match_tweet_sport | DUAL | 0.00000 | 0.00023 | 0.00 | 0.0 | 0.0 |
| gem_match_price_news | DUAL | 0.00000 | 0.00109 | 0.00 | 0.0 | 0.0 |
| tweet_astro_lunar_sin | DUAL | 0.00000 | 0.00278 | 0.00 | 0.0 | 0.0 |
| tweet_astro_is_full_moon | DUAL | 0.00000 | 0.00015 | 0.00 | 0.0 | 0.0 |
| tweet_astro_key_nakshatra | DUAL | 0.00000 | 0.00237 | 0.00 | 0.0 | 0.0 |
| tweet_astro_planetary_hour_idx | DUAL | 0.00000 | 0.00800 | 0.00 | 8.1 | 0.0 |
| news_astro_lunar_cos | DUAL | 0.00000 | 0.01549 | 0.00 | 14.1 | 0.0 |
| news_astro_is_full_moon | DUAL | 0.00000 | 0.00272 | 0.00 | 0.0 | 0.0 |
| news_astro_nakshatra_nature | DUAL | 0.00000 | 0.00300 | 0.00 | 0.0 | 0.0 |
| news_astro_nakshatra_guna | DUAL | 0.00000 | 0.00038 | 0.00 | 0.0 | 0.0 |
| news_astro_key_nakshatra | DUAL | 0.00000 | 0.00427 | 0.00 | 0.0 | 0.0 |
| d_macd | DUAL | 0.00000 | 0.01884 | 0.00 | 11.8 | 0.0 |
| w_bb_pctb | DUAL | 0.00000 | 0.01703 | 0.00 | 8.0 | 0.0 |
| ema50_declining | DUAL | 0.00000 | 0.01177 | 0.00 | 0.0 | 0.0 |
| sw_bars_since_storm | DUAL | 0.00000 | 0.03314 | 0.00 | 0.0 | 0.0 |
| sw_bars_since_severe | DUAL | 0.00000 | 0.00834 | 0.00 | 0.0 | 0.0 |
| sw_severe_decay | DUAL | 0.00000 | 0.00040 | 0.00 | 0.0 | 0.0 |
| eclipse_window | DUAL | 0.00000 | 0.00361 | 0.00 | 0.0 | 0.0 |
| cycle_confluence_score | DUAL | 0.00000 | 0.00416 | 0.00 | 9.8 | 0.0 |
| cross_kp_storm_x_funding | DUAL | 0.00000 | 0.00031 | 0.00 | 0.0 | 0.0 |
| cross_eclipse_x_funding | DUAL | 0.00000 | 0.00552 | 0.00 | 0.0 | 0.0 |
| cross_schumann_peak_x_funding | DUAL | 0.00000 | 0.00669 | 0.00 | 0.0 | 0.0 |
| cross_eclipse_x_moon | DUAL | 0.00000 | 0.00467 | 0.00 | 0.0 | 0.0 |
| cross_shmita_x_kp_storm | DUAL | 0.00000 | 0.00002 | 0.00 | 0.0 | 0.0 |
| cross_gold_tweet_x_funding | DUAL | 0.00000 | 0.00002 | 0.00 | 0.0 | 0.0 |
| cross_caps_tweet_x_vol | DUAL | 0.00000 | 0.00408 | 0.00 | 0.0 | 0.0 |
| cross_fg_fear_x_eclipse | DUAL | 0.00000 | 0.00328 | 0.00 | 0.0 | 0.0 |
| full_moon_decay_fast | DUAL | 0.00000 | 0.00194 | 0.00 | 11.1 | 0.0 |
| full_moon_decay_slow | DUAL | 0.00000 | 0.00046 | 0.00 | 9.2 | 3.6 |
| bars_since_new_moon | DUAL | 0.00000 | 0.00147 | 0.00 | 0.0 | 0.0 |
| new_moon_decay_fast | DUAL | 0.00000 | 0.00232 | 0.00 | 0.0 | 0.0 |
| gold_tweet_decay_fast | DUAL | 0.00000 | 0.00305 | 0.00 | 0.0 | 0.0 |
| gold_tweet_decay_slow | DUAL | 0.00000 | 0.00195 | 0.00 | 0.0 | 0.0 |
| bars_since_red_tweet | DUAL | 0.00000 | 0.00496 | 0.00 | 0.0 | 0.0 |
| red_tweet_decay_fast | DUAL | 0.00000 | 0.00354 | 0.00 | 0.0 | 0.0 |
| red_tweet_decay_slow | DUAL | 0.00000 | 0.00853 | 0.00 | 0.0 | 0.0 |
| caps_tweet_decay_fast | DUAL | 0.00000 | 0.00037 | 0.00 | 0.0 | 0.0 |
| bars_since_kp_storm | DUAL | 0.00000 | 0.03314 | 0.00 | 0.0 | 0.0 |
| sport_winner_gem_reverse_mean | VOLATILITY | 0.00322 | 0.00510 | 0.63 | 0.0 | 7.9 |
| rsi_14 | VOLATILITY | 0.00601 | 0.01024 | 0.59 | 10.4 | 20.3 |
| fg_vs_price_div | VOLATILITY | 0.00355 | 0.00619 | 0.57 | 9.2 | 18.4 |
| bazi_day_branch | VOLATILITY | 0.00119 | 0.00221 | 0.54 | 8.6 | 28.2 |
| news_gem_satanic_mean | VOLATILITY | 0.00330 | 0.00644 | 0.51 | 2.9 | 8.9 |
| hour_cos | VOLATILITY | 0.00158 | 0.00368 | 0.43 | 11.7 | 24.8 |
| is_london_session | VOLATILITY | 0.00110 | 0.00269 | 0.41 | 11.4 | 19.5 |
| macd_histogram_lag12 | VOLATILITY | 0.00278 | 0.00681 | 0.41 | 8.4 | 10.1 |
| sport_venue_gem_ordinal_mean | VOLATILITY | 0.00339 | 0.00837 | 0.41 | 0.0 | 10.7 |
| news_astro_lunar_sin | VOLATILITY | 0.00297 | 0.00735 | 0.40 | 11.9 | 10.8 |
| num_bullish_signals | VOLATILITY | 0.00472 | 0.01214 | 0.39 | 8.8 | 9.4 |
| bars_since_gem_match | VOLATILITY | 0.00170 | 0.00468 | 0.36 | 9.7 | 9.0 |
| chakra_heart_161d_cos | VOLATILITY | 0.00448 | 0.01256 | 0.36 | 7.3 | 23.4 |
| is_hour_8 | VOLATILITY | 0.00126 | 0.00435 | 0.29 | 10.9 | 15.5 |
| vol_regime_transition | VOLATILITY | 0.00014 | 0.00047 | 0.29 | 8.9 | 13.5 |
| return_6bar | VOLATILITY | 0.00848 | 0.03194 | 0.27 | 25.3 | 19.1 |
| rsi_14_lag12 | VOLATILITY | 0.00079 | 0.00313 | 0.25 | 8.5 | 13.0 |
| news_count_today | VOLATILITY | 0.00588 | 0.02820 | 0.21 | 10.1 | 8.0 |
| bars_since_news_caution | VOLATILITY | 0.00230 | 0.01230 | 0.19 | 9.0 | 7.7 |
| macd_histogram_lag6 | VOLATILITY | 0.00058 | 0.00311 | 0.19 | 9.1 | 15.3 |
| wyckoff_phase | VOLATILITY | 0.00102 | 0.00565 | 0.18 | 12.9 | 10.3 |
| gann_sq9_level | VOLATILITY | 0.00680 | 0.03793 | 0.18 | 10.4 | 7.3 |
| jupiter_365d_cos | VOLATILITY | 0.00191 | 0.01118 | 0.17 | 2.3 | 19.3 |
| upper_wick | VOLATILITY | 0.00117 | 0.00771 | 0.15 | 12.1 | 13.7 |
| west_soft_aspects | VOLATILITY | 0.00006 | 0.00043 | 0.15 | 0.0 | 13.4 |
| fear_greed_lag30 | VOLATILITY | 0.00233 | 0.01646 | 0.14 | 7.6 | 21.2 |
| day_of_week | VOLATILITY | 0.00334 | 0.02425 | 0.14 | 8.4 | 13.3 |
| schumann_143d_sin | VOLATILITY | 0.00180 | 0.01369 | 0.13 | 7.6 | 12.9 |
| pivot_s2 | VOLATILITY | 0.00337 | 0.02667 | 0.13 | 0.0 | 9.9 |
| pivot_s1 | VOLATILITY | 0.00400 | 0.03283 | 0.12 | 0.0 | 9.4 |
| sma_200_slope | VOLATILITY | 0.00412 | 0.03386 | 0.12 | 11.0 | 8.3 |
| bb_width_20 | VOLATILITY | 0.00701 | 0.05905 | 0.12 | 10.3 | 9.5 |
| volatility_6bar | VOLATILITY | 0.00541 | 0.05425 | 0.10 | 10.7 | 10.4 |
| bars_since_full_moon | VOLATILITY | 0.00157 | 0.01591 | 0.10 | 9.3 | 8.1 |
| macd_signal | VOLATILITY | 0.00196 | 0.02233 | 0.09 | 8.2 | 12.2 |
| schumann_143d_cos | VOLATILITY | 0.00064 | 0.00738 | 0.09 | 8.3 | 9.3 |
| macro_ibit_roc5d | VOLATILITY | 0.00074 | 0.00976 | 0.08 | 0.0 | 14.2 |
| macro_mstr | VOLATILITY | 0.00257 | 0.03405 | 0.08 | 10.5 | 16.1 |
| sar_value | VOLATILITY | 0.00107 | 0.01526 | 0.07 | 6.7 | 12.6 |
| w_vol_ratio | VOLATILITY | 0.00269 | 0.04498 | 0.06 | 8.7 | 7.6 |
| mayan_sign_idx | VOLATILITY | 0.00049 | 0.00938 | 0.05 | 7.9 | 9.9 |
| supertrend | VOLATILITY | 0.00115 | 0.02500 | 0.05 | 0.0 | 14.7 |
| onchain_n_transactions | VOLATILITY | 0.00138 | 0.03301 | 0.04 | 8.7 | 25.4 |
| current_dd_depth | VOLATILITY | 0.00114 | 0.02721 | 0.04 | 7.6 | 11.5 |
| knn_best_match_dist | VOLATILITY | 0.00052 | 0.01274 | 0.04 | 8.1 | 14.1 |
| bb_pctb_20_lag12 | VOLATILITY | 0.00010 | 0.00333 | 0.03 | 8.1 | 12.6 |
| ichimoku_senkou_b | VOLATILITY | 0.00085 | 0.03217 | 0.03 | 10.3 | 6.4 |
| ema50_slope | VOLATILITY | 0.00097 | 0.04149 | 0.02 | 6.6 | 10.5 |
| btc_vix_corr | VOLATILITY | 0.00009 | 0.00461 | 0.02 | 6.7 | 11.8 |
| macro_tlt | VOLATILITY | 0.00040 | 0.03218 | 0.01 | 10.2 | 11.2 |
| ema_20 | VOLATILITY | 0.00037 | 0.03611 | 0.01 | 0.0 | 19.5 |
| gtrends_interest | VOLATILITY | 0.00041 | 0.04112 | 0.01 | 0.0 | 25.2 |
| news_gem_dr_sat_mode | VOLATILITY | 0.00010 | 0.01091 | 0.01 | 10.6 | 8.4 |
| d_return | VOLATILITY | 0.00038 | 0.05879 | 0.01 | 29.3 | 38.2 |
| sw_solar_flux_f107 | VOLATILITY | 0.00010 | 0.01844 | 0.01 | 9.3 | 7.9 |
| rsi_14_lag1 | VOLATILITY | 0.00004 | 0.01215 | 0.00 | 9.6 | 11.2 |
| sma_50 | VOLATILITY | 0.00006 | 0.02439 | 0.00 | 0.0 | 9.4 |
| news_sentiment_today | VOLATILITY | 0.00002 | 0.01504 | 0.00 | 11.1 | 7.8 |
| sma_20 | VOLATILITY | 0.00000 | 0.02699 | 0.00 | 0.0 | 9.5 |
| ema_50 | VOLATILITY | 0.00000 | 0.03242 | 0.00 | 12.3 | 12.9 |
| ema_100 | VOLATILITY | 0.00000 | 0.02020 | 0.00 | 4.9 | 14.1 |
| sma_200 | VOLATILITY | 0.00000 | 0.03063 | 0.00 | 6.2 | 6.9 |
| sma_20_slope | VOLATILITY | 0.00000 | 0.03388 | 0.00 | 10.4 | 6.5 |
| sma_50_slope | VOLATILITY | 0.00000 | 0.03456 | 0.00 | 10.0 | 11.2 |
| rsi_21 | VOLATILITY | 0.00000 | 0.00655 | 0.00 | 15.6 | 12.2 |
| bb_upper_20 | VOLATILITY | 0.00000 | 0.02783 | 0.00 | 5.2 | 16.7 |
| bb_lower_20 | VOLATILITY | 0.00000 | 0.02128 | 0.00 | 1.5 | 12.6 |
| macd_line | VOLATILITY | 0.00000 | 0.02301 | 0.00 | 10.2 | 11.8 |
| stoch_k_14 | VOLATILITY | 0.00000 | 0.01134 | 0.00 | 15.7 | 23.6 |
| atr_14 | VOLATILITY | 0.00000 | 0.02060 | 0.00 | 10.5 | 14.0 |
| atr_14_pct | VOLATILITY | 0.00000 | 0.06844 | 0.00 | 11.0 | 213.0 |
| return_1bar | VOLATILITY | 0.00000 | 0.02622 | 0.00 | 20.1 | 10.4 |
| return_12bar | VOLATILITY | 0.00000 | 0.02836 | 0.00 | 10.0 | 37.8 |
| return_42bar | VOLATILITY | 0.00000 | 0.02342 | 0.00 | 9.1 | 23.5 |
| volatility_42bar | VOLATILITY | 0.00000 | 0.06211 | 0.00 | 9.0 | 41.3 |
| volatility_ratio | VOLATILITY | 0.00000 | 0.01445 | 0.00 | 9.3 | 11.2 |
| ichimoku_senkou_a | VOLATILITY | 0.00000 | 0.02737 | 0.00 | 10.4 | 15.1 |
| pivot_r1 | VOLATILITY | 0.00000 | 0.02890 | 0.00 | 6.2 | 9.8 |
| obv | VOLATILITY | 0.00000 | 0.04461 | 0.00 | 10.2 | 20.3 |
| obv_sma_20 | VOLATILITY | 0.00000 | 0.03699 | 0.00 | 9.1 | 16.8 |
| williams_r_14 | VOLATILITY | 0.00000 | 0.01132 | 0.00 | 11.1 | 22.0 |
| keltner_upper | VOLATILITY | 0.00000 | 0.03262 | 0.00 | 0.0 | 21.9 |
| donchian_upper | VOLATILITY | 0.00000 | 0.03011 | 0.00 | 0.0 | 17.7 |
| gann_sq9_distance | VOLATILITY | 0.00000 | 0.00948 | 0.00 | 7.8 | 11.6 |
| golden_ratio_ext | VOLATILITY | 0.00000 | 0.02327 | 0.00 | 7.8 | 11.6 |
| golden_ratio_dist | VOLATILITY | 0.00000 | 0.03862 | 0.00 | 8.9 | 15.9 |
| near_fib_21 | VOLATILITY | 0.00000 | 0.00205 | 0.00 | 0.0 | 12.7 |
| rsi_14_lag6 | VOLATILITY | 0.00000 | 0.01624 | 0.00 | 7.2 | 7.8 |
| macd_histogram_lag1 | VOLATILITY | 0.00000 | 0.01673 | 0.00 | 8.5 | 12.1 |
| macd_histogram_lag24 | VOLATILITY | 0.00000 | 0.00409 | 0.00 | 7.1 | 12.5 |
| dow_sin | VOLATILITY | 0.00000 | 0.02478 | 0.00 | 7.9 | 12.5 |
| dow_cos | VOLATILITY | 0.00000 | 0.01477 | 0.00 | 1.8 | 9.1 |
| doy_sin | VOLATILITY | 0.00000 | 0.00878 | 0.00 | 8.4 | 19.2 |
| is_hour_0 | VOLATILITY | 0.00000 | 0.00091 | 0.00 | 0.0 | 6.6 |
| is_hour_4 | VOLATILITY | 0.00000 | 0.00128 | 0.00 | 11.7 | 12.6 |
| day_of_year | VOLATILITY | 0.00000 | 0.00932 | 0.00 | 8.7 | 31.9 |
| day_of_month | VOLATILITY | 0.00000 | 0.00979 | 0.00 | 7.4 | 7.5 |
| month | VOLATILITY | 0.00000 | 0.00509 | 0.00 | 0.0 | 75.1 |
| price_dr | VOLATILITY | 0.00000 | 0.00070 | 0.00 | 0.0 | 25.7 |
| west_planetary_strength | VOLATILITY | 0.00000 | 0.01742 | 0.00 | 7.2 | 12.3 |
| psi | VOLATILITY | 0.00000 | 0.01735 | 0.00 | 12.1 | 10.4 |
| lunar_phase_cos | VOLATILITY | 0.00000 | 0.00002 | 0.00 | 8.2 | 11.3 |
| vedic_nature_encoded | VOLATILITY | 0.00000 | 0.00206 | 0.00 | 11.5 | 54.9 |
| vedic_guna_encoded | VOLATILITY | 0.00000 | 0.00204 | 0.00 | 0.0 | 14.2 |
| bazi_day_stem | VOLATILITY | 0.00000 | 0.00271 | 0.00 | 8.4 | 6.2 |
| arabic_lot_commerce | VOLATILITY | 0.00000 | 0.00246 | 0.00 | 9.7 | 29.2 |
| arabic_lot_increase | VOLATILITY | 0.00000 | 0.02350 | 0.00 | 7.8 | 9.2 |
| arabic_lot_catastrophe | VOLATILITY | 0.00000 | 0.00839 | 0.00 | 9.9 | 10.1 |
| arabic_lot_treachery | VOLATILITY | 0.00000 | 0.01115 | 0.00 | 9.2 | 18.6 |
| tweet_gem_ordinal_mean | VOLATILITY | 0.00000 | 0.00414 | 0.00 | 0.0 | 16.4 |
| tweet_gem_satanic_mean | VOLATILITY | 0.00000 | 0.00600 | 0.00 | 0.0 | 26.3 |
| news_count_4h | VOLATILITY | 0.00000 | 0.00280 | 0.00 | 0.0 | 6.0 |
| news_sentiment_4h | VOLATILITY | 0.00000 | 0.00010 | 0.00 | 0.0 | 10.1 |
| news_gem_ordinal_mean | VOLATILITY | 0.00000 | 0.00321 | 0.00 | 9.2 | 5.8 |
| headline_gem_dr_mode | VOLATILITY | 0.00000 | 0.00575 | 0.00 | 0.0 | 5.7 |
| sport_venue_gem_reduction_mean | VOLATILITY | 0.00000 | 0.00503 | 0.00 | 0.0 | 17.0 |
| onchain_hash_rate | VOLATILITY | 0.00000 | 0.05437 | 0.00 | 7.2 | 13.2 |
| onchain_difficulty | VOLATILITY | 0.00000 | 0.04537 | 0.00 | 0.0 | 18.5 |
| onchain_mempool_size | VOLATILITY | 0.00000 | 0.02000 | 0.00 | 14.8 | 13.5 |
| onchain_hash_rate_roc | VOLATILITY | 0.00000 | 0.02041 | 0.00 | 0.0 | 19.7 |
| macro_dxy | VOLATILITY | 0.00000 | 0.03469 | 0.00 | 7.5 | 22.3 |
| macro_dxy_roc5d | VOLATILITY | 0.00000 | 0.00491 | 0.00 | 8.2 | 12.0 |
| macro_dxy_roc20d | VOLATILITY | 0.00000 | 0.00484 | 0.00 | 10.0 | 27.5 |
| macro_gold | VOLATILITY | 0.00000 | 0.03627 | 0.00 | 0.0 | 13.6 |
| macro_gold_roc5d | VOLATILITY | 0.00000 | 0.01488 | 0.00 | 9.4 | 14.7 |
| macro_gold_roc20d | VOLATILITY | 0.00000 | 0.02243 | 0.00 | 9.1 | 11.5 |
| macro_spx | VOLATILITY | 0.00000 | 0.02417 | 0.00 | 6.8 | 27.5 |
| macro_spx_roc5d | VOLATILITY | 0.00000 | 0.01696 | 0.00 | 9.7 | 13.3 |
| macro_spx_roc20d | VOLATILITY | 0.00000 | 0.00276 | 0.00 | 7.6 | 15.7 |
| macro_vix | VOLATILITY | 0.00000 | 0.01866 | 0.00 | 9.2 | 6.3 |
| macro_vix_roc5d | VOLATILITY | 0.00000 | 0.00471 | 0.00 | 7.6 | 21.4 |
| macro_vix_roc20d | VOLATILITY | 0.00000 | 0.00624 | 0.00 | 9.5 | 8.9 |
| macro_us10y | VOLATILITY | 0.00000 | 0.03420 | 0.00 | 11.8 | 13.2 |
| macro_us10y_roc5d | VOLATILITY | 0.00000 | 0.01585 | 0.00 | 7.7 | 10.1 |
| macro_us10y_roc20d | VOLATILITY | 0.00000 | 0.00416 | 0.00 | 8.7 | 8.2 |
| macro_nasdaq_roc5d | VOLATILITY | 0.00000 | 0.01376 | 0.00 | 10.3 | 19.4 |
| macro_nasdaq_roc20d | VOLATILITY | 0.00000 | 0.01276 | 0.00 | 8.7 | 22.8 |
| macro_russell | VOLATILITY | 0.00000 | 0.01376 | 0.00 | 8.8 | 8.6 |
| macro_russell_roc5d | VOLATILITY | 0.00000 | 0.01125 | 0.00 | 7.6 | 10.0 |
| macro_russell_roc20d | VOLATILITY | 0.00000 | 0.01142 | 0.00 | 7.2 | 14.0 |
| macro_oil_roc5d | VOLATILITY | 0.00000 | 0.00209 | 0.00 | 9.8 | 30.9 |
| macro_oil_roc20d | VOLATILITY | 0.00000 | 0.00885 | 0.00 | 10.8 | 7.5 |
| macro_silver | VOLATILITY | 0.00000 | 0.04056 | 0.00 | 8.2 | 7.0 |
| macro_silver_roc5d | VOLATILITY | 0.00000 | 0.00878 | 0.00 | 6.5 | 6.6 |
| macro_silver_roc20d | VOLATILITY | 0.00000 | 0.01036 | 0.00 | 7.5 | 11.2 |
| macro_mstr_roc5d | VOLATILITY | 0.00000 | 0.01740 | 0.00 | 9.8 | 8.5 |
| macro_mstr_roc20d | VOLATILITY | 0.00000 | 0.00731 | 0.00 | 7.7 | 9.9 |
| macro_hyg_roc5d | VOLATILITY | 0.00000 | 0.00947 | 0.00 | 8.8 | 7.9 |
| macro_hyg_roc20d | VOLATILITY | 0.00000 | 0.00657 | 0.00 | 8.3 | 20.1 |
| fear_greed | VOLATILITY | 0.00000 | 0.01727 | 0.00 | 10.9 | 11.0 |
| fg_roc | VOLATILITY | 0.00000 | 0.01291 | 0.00 | 10.5 | 11.0 |
| fear_greed_lag60 | VOLATILITY | 0.00000 | 0.02405 | 0.00 | 7.1 | 9.5 |
| gtrends_interest_high | VOLATILITY | 0.00000 | 0.00879 | 0.00 | 0.0 | 18.9 |
| funding_rate | VOLATILITY | 0.00000 | 0.03695 | 0.00 | 7.7 | 7.2 |
| tweet_astro_lunar_cos | VOLATILITY | 0.00000 | 0.00633 | 0.00 | 9.0 | 16.5 |
| news_astro_moon_phase_day | VOLATILITY | 0.00000 | 0.00722 | 0.00 | 0.0 | 13.6 |
| d_ema50_dist | VOLATILITY | 0.00000 | 0.03247 | 0.00 | 10.9 | 9.1 |
| d_rsi14 | VOLATILITY | 0.00000 | 0.01710 | 0.00 | 13.0 | 8.8 |
| d_bb_pctb | VOLATILITY | 0.00000 | 0.01855 | 0.00 | 11.5 | 18.4 |
| d_volatility | VOLATILITY | 0.00000 | 0.05223 | 0.00 | 7.6 | 32.6 |
| d_atr_pct | VOLATILITY | 0.00000 | 0.07266 | 0.00 | 9.3 | 63.0 |
| d_vol_ratio | VOLATILITY | 0.00000 | 0.03297 | 0.00 | 10.8 | 32.1 |
| w_return | VOLATILITY | 0.00000 | 0.04309 | 0.00 | 11.7 | 10.4 |
| w_volatility | VOLATILITY | 0.00000 | 0.05014 | 0.00 | 9.2 | 10.2 |
| w_atr_pct | VOLATILITY | 0.00000 | 0.04311 | 0.00 | 9.8 | 14.4 |
| ema50_rising | VOLATILITY | 0.00000 | 0.00572 | 0.00 | 0.0 | 6.4 |
| sw_kp_index | VOLATILITY | 0.00000 | 0.00419 | 0.00 | 5.4 | 12.3 |
| sw_sunspot_number | VOLATILITY | 0.00000 | 0.01453 | 0.00 | 9.8 | 21.9 |
| schumann_133d_sin | VOLATILITY | 0.00000 | 0.00832 | 0.00 | 10.3 | 12.2 |
| schumann_133d_cos | VOLATILITY | 0.00000 | 0.02585 | 0.00 | 8.3 | 15.2 |
| schumann_783d_sin | VOLATILITY | 0.00000 | 0.01425 | 0.00 | 9.0 | 9.9 |
| schumann_783d_cos | VOLATILITY | 0.00000 | 0.01490 | 0.00 | 9.4 | 12.4 |
| chakra_heart_161d_sin | VOLATILITY | 0.00000 | 0.01371 | 0.00 | 6.6 | 11.0 |
| mercury_1216d_sin | VOLATILITY | 0.00000 | 0.01574 | 0.00 | 9.5 | 8.6 |
| mercury_1216d_cos | VOLATILITY | 0.00000 | 0.02158 | 0.00 | 7.1 | 13.3 |
| equinox_proximity | VOLATILITY | 0.00000 | 0.00581 | 0.00 | 7.3 | 13.7 |
| esoteric_vol_score | VOLATILITY | 0.00000 | 0.01250 | 0.00 | 9.8 | 19.6 |
| vol_directional_asymmetry | VOLATILITY | 0.00000 | 0.01422 | 0.00 | 7.5 | 6.8 |
| seasonal_vol_direction | VOLATILITY | 0.00000 | 0.01232 | 0.00 | 5.4 | 18.4 |
| fg_x_moon_phase | VOLATILITY | 0.00000 | 0.00353 | 0.00 | 8.1 | 10.0 |
| cross_fg_greed_x_moon | VOLATILITY | 0.00000 | 0.00796 | 0.00 | 0.0 | 9.9 |
| cross_vol_score_x_trend | VOLATILITY | 0.00000 | 0.02559 | 0.00 | 8.4 | 23.1 |
| cross_fg_greed_x_funding | VOLATILITY | 0.00000 | 0.00787 | 0.00 | 0.0 | 16.6 |
| cross_nakshatra_x_moon | VOLATILITY | 0.00000 | 0.00793 | 0.00 | 10.5 | 13.9 |
| bars_since_high_fear | VOLATILITY | 0.00000 | 0.04386 | 0.00 | 8.9 | 18.5 |
| bars_since_high_greed | VOLATILITY | 0.00000 | 0.02816 | 0.00 | 10.4 | 13.7 |
| knn_pattern_std | VOLATILITY | 0.00000 | 0.05856 | 0.00 | 8.8 | 10.6 |

## Recommendations

### Feature Engineering Priorities
1. DIRECTIONAL features are your edge for entry signals -- create interaction features between the top directional features
2. VOLATILITY features should drive position sizing and stop-loss placement, not entry decisions
3. DUAL features are rare and valuable -- they predict both direction and magnitude, prioritize these in model stacking
4. NOISE features (bottom category) are candidates for removal -- dropping them can reduce overfitting
5. Features with high MI_dir but low model gain may have nonlinear relationships worth exploring with deeper trees or interactions
6. Features with high model gain but low MI may be proxies -- investigate what they actually capture
