'use client';

import { useState, useEffect, useCallback } from 'react';
import {
  MousePointer, Minus, MoveHorizontal, MoveVertical,
  Triangle, Square, Circle, Type, ArrowRight,
  Ruler, Pencil, Trash2,
} from 'lucide-react';

export type DrawingTool =
  | 'select'
  | 'trendline'
  | 'hline'
  | 'vline'
  | 'fib'
  | 'rectangle'
  | 'circle'
  | 'text'
  | 'arrow'
  | 'measure'
  | 'freehand'
  | 'delete'
  | null;

// Mapping from our tool names to KLineChart overlay names
export const TOOL_TO_KLINE_OVERLAY: Record<string, string> = {
  trendline: 'segment',
  hline: 'horizontalStraightLine',
  vline: 'verticalStraightLine',
  fib: 'fibonacciLine',
  rectangle: 'rect',
  circle: 'circle',
  arrow: 'arrow',
  text: 'simpleAnnotation',
  freehand: 'segment',
};

interface ToolDef {
  id: DrawingTool;
  label: string;
  icon: React.ReactNode;
  shortcut?: string;
  klineOverlay?: string; // KLineChart overlay name
}

const TOOLS: ToolDef[] = [
  { id: 'select', label: 'Select', icon: <MousePointer size={16} />, shortcut: 'V' },
  { id: 'trendline', label: 'Trend Line', icon: <Minus size={16} />, shortcut: 'T', klineOverlay: 'segment' },
  { id: 'hline', label: 'Horizontal Line', icon: <MoveHorizontal size={16} />, shortcut: 'H', klineOverlay: 'horizontalStraightLine' },
  { id: 'vline', label: 'Vertical Line', icon: <MoveVertical size={16} />, klineOverlay: 'verticalStraightLine' },
  { id: 'fib', label: 'Fib Retracement', icon: <Triangle size={16} />, shortcut: 'F', klineOverlay: 'fibonacciLine' },
  { id: 'rectangle', label: 'Rectangle', icon: <Square size={16} />, shortcut: 'R', klineOverlay: 'rect' },
  { id: 'circle', label: 'Circle', icon: <Circle size={16} />, klineOverlay: 'circle' },
  { id: 'text', label: 'Text', icon: <Type size={16} />, klineOverlay: 'simpleAnnotation' },
  { id: 'arrow', label: 'Arrow', icon: <ArrowRight size={16} />, klineOverlay: 'arrow' },
  { id: 'measure', label: 'Measure (Shift+Click)', icon: <Ruler size={16} />, shortcut: 'M' },
  { id: 'freehand', label: 'Freehand', icon: <Pencil size={16} />, klineOverlay: 'segment' },
  { id: 'delete', label: 'Delete All', icon: <Trash2 size={16} /> },
];

interface DrawingToolbarProps {
  activeTool: DrawingTool;
  onSelectTool: (tool: DrawingTool) => void;
  onDeleteAll: () => void;
}

export default function DrawingToolbar({
  activeTool,
  onSelectTool,
  onDeleteAll,
}: DrawingToolbarProps) {
  const [hoveredTool, setHoveredTool] = useState<string | null>(null);

  // ESC deselects current tool
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      onSelectTool(null);
    }
    // Keyboard shortcuts (only when not in an input)
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
    const upper = e.key.toUpperCase();
    const match = TOOLS.find(t => t.shortcut === upper);
    if (match) {
      if (match.id === 'delete') {
        onDeleteAll();
      } else {
        onSelectTool(match.id);
      }
    }
  }, [onSelectTool, onDeleteAll]);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  return (
    <div
      className="absolute left-2 top-1/2 -translate-y-1/2 z-40 flex flex-col gap-0.5 p-1.5 rounded-xl border"
      style={{
        background: 'rgba(15, 15, 25, 0.92)',
        borderColor: 'rgba(50, 50, 80, 0.35)',
        backdropFilter: 'blur(12px)',
      }}
    >
      {TOOLS.map((tool, idx) => {
        const isActive = activeTool === tool.id;
        const isDelete = tool.id === 'delete';
        const showDivider = idx === 0 || idx === 4 || idx === 8 || idx === 10;

        return (
          <div key={tool.id}>
            {showDivider && idx !== 0 && (
              <div className="w-6 mx-auto my-0.5 border-t" style={{ borderColor: 'rgba(50,50,80,0.3)' }} />
            )}
            <div className="relative">
              <button
                onClick={() => {
                  if (isDelete) {
                    onDeleteAll();
                  } else {
                    onSelectTool(isActive ? null : tool.id);
                  }
                }}
                onMouseEnter={() => setHoveredTool(tool.id)}
                onMouseLeave={() => setHoveredTool(null)}
                className={`flex items-center justify-center w-8 h-8 rounded-lg transition-all ${
                  isActive
                    ? 'bg-blue-500/20 text-blue-400 shadow-[0_0_8px_rgba(59,130,246,0.2)]'
                    : isDelete
                    ? 'text-slate-500 hover:text-red-400 hover:bg-red-500/10'
                    : 'text-slate-500 hover:text-slate-200 hover:bg-white/5'
                }`}
                style={isActive ? { border: '1px solid rgba(59,130,246,0.4)' } : {}}
              >
                {tool.icon}
              </button>

              {/* Tooltip */}
              {hoveredTool === tool.id && (
                <div
                  className="absolute left-full ml-2 top-1/2 -translate-y-1/2 px-2.5 py-1.5 rounded-md whitespace-nowrap pointer-events-none"
                  style={{
                    background: 'rgba(10, 10, 20, 0.95)',
                    border: '1px solid rgba(50, 50, 80, 0.4)',
                    zIndex: 100,
                  }}
                >
                  <span className="text-xs text-slate-200 font-medium">{tool.label}</span>
                  {tool.shortcut && (
                    <span className="ml-2 text-[10px] text-slate-500 font-mono">{tool.shortcut}</span>
                  )}
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
