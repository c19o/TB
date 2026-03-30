'use client';
import { API } from '@/lib/api';

import { useEffect, useState, useCallback } from 'react';
import type { Tweet } from '@/types';

const HANDLES = ['elonmusk', 'JoelKatz', 'IOHK_Charles', 'tyler', 'cameron', 'IAmSteveHarvey'];

export default function TweetFeed() {
  const [tweets, setTweets] = useState<Tweet[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterHandle, setFilterHandle] = useState<string | null>(null);

  useEffect(() => {
    const params = new URLSearchParams({ limit: '100' });
    if (filterHandle) params.set('handle', filterHandle);

    fetch(`${API}/api/tweets?${params}`)
      .then(r => r.json())
      .then(data => {
        setTweets(data.tweets || []);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, [filterHandle]);

  const formatDate = (dateStr: string) => {
    try {
      const d = new Date(dateStr);
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) + ' ' +
             d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
    } catch {
      return dateStr;
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header bar */}
      <div className="flex items-center gap-3 px-3 py-1.5 border-b border-panel-border shrink-0">
        <div className="flex items-center gap-2">
          <svg className="w-3.5 h-3.5 text-cyan-500" fill="currentColor" viewBox="0 0 24 24">
            <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
          </svg>
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Decoded Tweet Feed</h3>
        </div>

        {/* Handle filter */}
        <div className="flex items-center gap-1 ml-auto">
          <button
            onClick={() => setFilterHandle(null)}
            className={`px-2 py-0.5 text-[10px] rounded transition-all ${
              !filterHandle
                ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                : 'text-slate-600 hover:text-slate-400 border border-transparent'
            }`}
          >
            ALL
          </button>
          {HANDLES.map(handle => (
            <button
              key={handle}
              onClick={() => setFilterHandle(filterHandle === handle ? null : handle)}
              className={`px-2 py-0.5 text-[10px] rounded transition-all ${
                filterHandle === handle
                  ? 'bg-cyan-500/15 text-cyan-400 border border-cyan-500/30'
                  : 'text-slate-600 hover:text-slate-400 border border-transparent'
              }`}
            >
              @{handle.length > 8 ? handle.slice(0, 8) + '..' : handle}
            </button>
          ))}
        </div>
      </div>

      {/* Tweet list */}
      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div className="flex items-center justify-center h-full">
            <div className="text-xs text-slate-600">Loading tweets...</div>
          </div>
        )}

        {!loading && tweets.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-xs text-slate-600">No tweets found</div>
          </div>
        )}

        <div className="divide-y divide-panel-border">
          {tweets.map(tweet => (
            <div
              key={tweet.tweet_id}
              className="tweet-card flex gap-3 px-3 py-2 border-l-2 border-transparent hover:border-l-2"
              style={{
                borderLeftColor: tweet.misdirection_flag ? '#f59e0b' :
                  tweet.gematria_hits && tweet.gematria_hits.length > 0 ? '#8b5cf6' : 'transparent',
              }}
            >
              {/* Timestamp + handle */}
              <div className="w-32 shrink-0">
                <div className="text-[10px] text-cyan-500 font-medium">@{tweet.user_handle}</div>
                <div className="text-[9px] text-slate-600 num-display">{formatDate(tweet.created_at)}</div>
                <div className="text-[9px] text-slate-700 num-display">DOY: {tweet.day_of_year}</div>
              </div>

              {/* Tweet text */}
              <div className="flex-1 min-w-0">
                <p className="text-[11px] text-slate-400 leading-relaxed line-clamp-2">
                  {tweet.full_text}
                </p>
              </div>

              {/* Gematria hits + flags */}
              <div className="w-36 shrink-0 flex flex-col items-end gap-1">
                {tweet.gematria_hits && tweet.gematria_hits.length > 0 && (
                  <div className="flex flex-wrap gap-0.5 justify-end">
                    {tweet.gematria_hits.map((hit, i) => (
                      <span
                        key={i}
                        className="px-1.5 py-0.5 text-[9px] rounded bg-esoteric/10 text-esoteric border border-esoteric/20 num-display"
                      >
                        {hit}
                      </span>
                    ))}
                  </div>
                )}
                {tweet.misdirection_flag && (
                  <span className="px-1.5 py-0.5 text-[9px] rounded bg-warning/10 border border-warning/20 font-bold misdirection-badge">
                    MISDIRECTION
                  </span>
                )}
                {tweet.date_gematria && (
                  <span className="text-[8px] text-slate-700 num-display">
                    {(() => {
                      try {
                        const dg = JSON.parse(tweet.date_gematria);
                        return `${dg.month_day} R:${dg.date_reduction}`;
                      } catch { return ''; }
                    })()}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
