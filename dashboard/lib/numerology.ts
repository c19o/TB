// Numerology calculation utilities for the Savage22 system

const CAUTION_NUMBERS = [4, 8, 13, 16, 22, 26];
const PUMP_NUMBERS = [3, 7, 9, 11, 33];
const MASTER_NUMBERS = [11, 22, 33];

export function reduceToSingleDigit(n: number): number {
  while (n > 9 && !MASTER_NUMBERS.includes(n)) {
    n = String(n).split('').reduce((sum, d) => sum + parseInt(d), 0);
  }
  return n;
}

export function getDayOfYear(date: Date): number {
  const start = new Date(date.getFullYear(), 0, 0);
  const diff = date.getTime() - start.getTime();
  return Math.floor(diff / (1000 * 60 * 60 * 24));
}

export function getDaysRemaining(date: Date): number {
  const year = date.getFullYear();
  const isLeap = (year % 4 === 0 && year % 100 !== 0) || (year % 400 === 0);
  const totalDays = isLeap ? 366 : 365;
  return totalDays - getDayOfYear(date);
}

export function getDateReduction(date: Date): number {
  const m = date.getMonth() + 1;
  const d = date.getDate();
  const y = date.getFullYear();
  const sum = reduceToSingleDigit(m) + reduceToSingleDigit(d) + reduceToSingleDigit(y);
  return reduceToSingleDigit(sum);
}

export function isCautionNumber(n: number): boolean {
  return CAUTION_NUMBERS.includes(n) || CAUTION_NUMBERS.includes(reduceToSingleDigit(n));
}

export function isPumpNumber(n: number): boolean {
  return PUMP_NUMBERS.includes(n) || PUMP_NUMBERS.includes(reduceToSingleDigit(n));
}

// Approximate moon phase calculation
export function getMoonPhase(date: Date): string {
  const year = date.getFullYear();
  const month = date.getMonth() + 1;
  const day = date.getDate();

  let c = 0, e = 0, jd = 0, b = 0;
  if (month < 3) {
    c = year - 1;
    e = month + 12;
  } else {
    c = year;
    e = month;
  }
  jd = (365.25 * (c + 4716)) + (30.6001 * (e + 1)) + day - 1524.5;
  b = jd - 2451550.1;
  b = b / 29.530588853;
  b = b - Math.floor(b);
  const age = Math.round(b * 29.530588853);

  if (age < 1) return 'New Moon';
  if (age < 7) return 'Waxing Crescent';
  if (age < 8) return 'First Quarter';
  if (age < 14) return 'Waxing Gibbous';
  if (age < 16) return 'Full Moon';
  if (age < 22) return 'Waning Gibbous';
  if (age < 23) return 'Last Quarter';
  if (age < 29) return 'Waning Crescent';
  return 'New Moon';
}

export function getZodiacSign(date: Date): string {
  const month = date.getMonth() + 1;
  const day = date.getDate();
  const signs: [number, number, string][] = [
    [1, 20, 'Capricorn'], [2, 19, 'Aquarius'], [3, 20, 'Pisces'],
    [4, 20, 'Aries'], [5, 21, 'Taurus'], [6, 21, 'Gemini'],
    [7, 22, 'Cancer'], [8, 23, 'Leo'], [9, 23, 'Virgo'],
    [10, 23, 'Libra'], [11, 22, 'Scorpio'], [12, 22, 'Sagittarius'],
  ];
  for (let i = signs.length - 1; i >= 0; i--) {
    if (month === signs[i][0] && day >= signs[i][1]) return signs[i][2];
    if (month > signs[i][0]) return signs[i][2];
  }
  return 'Capricorn';
}

const PLANETS = ['Saturn', 'Jupiter', 'Mars', 'Sun', 'Venus', 'Mercury', 'Moon'];

export function getPlanetaryHour(date: Date): string {
  const dayOfWeek = date.getDay(); // 0=Sun
  const dayPlanetIndex = [3, 6, 4, 5, 2, 1, 0][dayOfWeek]; // Sun, Moon, Mars, Merc, Jup, Ven, Sat
  const hour = date.getHours();
  const planetIndex = (dayPlanetIndex + hour) % 7;
  return PLANETS[planetIndex];
}

export function computeNumerology(date: Date) {
  const dayOfYear = getDayOfYear(date);
  const dateReduction = getDateReduction(date);
  const daysRemaining = getDaysRemaining(date);
  return {
    day_of_year: dayOfYear,
    date_reduction: dateReduction,
    days_remaining: daysRemaining,
    moon_phase: getMoonPhase(date),
    zodiac_sign: getZodiacSign(date),
    planetary_hour: getPlanetaryHour(date),
    is_caution_number: isCautionNumber(dateReduction),
    is_pump_number: isPumpNumber(dateReduction),
    master_number: MASTER_NUMBERS.includes(dateReduction),
  };
}

// Gematria calculations for tweet analysis
export function simpleGematria(text: string): number {
  return text.toUpperCase().split('').reduce((sum, ch) => {
    const code = ch.charCodeAt(0);
    if (code >= 65 && code <= 90) return sum + (code - 64);
    return sum;
  }, 0);
}

export function ordinalGematria(text: string): number {
  return simpleGematria(text);
}

export function reducedGematria(text: string): number {
  return reduceToSingleDigit(simpleGematria(text));
}
