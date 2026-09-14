/**
 * Shared primitives for every interactive visual on the site.
 *
 * Rules these encode (see src/styles/custom.css for the tokens):
 *  - Controls live in ONE row above the figure.
 *  - A legend is always present for >= 2 series; identity is never color-alone.
 *  - Wide figures scroll inside .viz-figure, never the page.
 *  - Colors come from --viz-1..8 (categorical, fixed order, never cycled),
 *    --viz-seq-* (magnitude), --viz-good/warning/serious/critical (status only).
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';

/* ------------------------------------------------------------------ shell */

export function VizPanel({
  title,
  subtitle,
  controls,
  children,
  legend,
  stats,
  note,
  table,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  controls?: ReactNode;
  children: ReactNode;
  legend?: ReactNode;
  stats?: ReactNode;
  note?: ReactNode;
  table?: ReactNode;
}) {
  return (
    <figure className="viz" style={{ position: 'relative' }}>
      {title ? <p className="viz-title">{title}</p> : null}
      {subtitle ? <p className="viz-sub">{subtitle}</p> : null}
      {controls ? <div className="viz-controls">{controls}</div> : null}
      <div className="viz-figure">{children}</div>
      {legend}
      {stats}
      {note}
      {table ? (
        <details className="viz-data">
          <summary>Show the numbers</summary>
          {table}
        </details>
      ) : null}
    </figure>
  );
}

/* --------------------------------------------------------------- controls */

export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format,
  disabled,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (n: number) => void;
  format?: (n: number) => string;
  disabled?: boolean;
}) {
  const id = useId();
  return (
    <label className="viz-control" htmlFor={id}>
      <span>
        {label} <output htmlFor={id}>{format ? format(value) : value}</output>
      </span>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.currentTarget.value))}
      />
    </label>
  );
}

export function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label?: string;
  value: T;
  options: readonly { value: T; label: string; title?: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="viz-control">
      {label ? <span>{label}</span> : null}
      <div className="viz-seg" role="group" aria-label={label}>
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            title={o.title}
            aria-pressed={o.value === value}
            onClick={() => onChange(o.value)}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function Choice<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: readonly { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  const id = useId();
  return (
    <label className="viz-control" htmlFor={id}>
      <span>{label}</span>
      <select id={id} value={value} onChange={(e) => onChange(e.currentTarget.value as T)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function Check({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (b: boolean) => void;
}) {
  const id = useId();
  return (
    <label className="viz-control" htmlFor={id} style={{ flexDirection: 'row', alignItems: 'center', gap: '.4rem' }}>
      <input id={id} type="checkbox" checked={checked} onChange={(e) => onChange(e.currentTarget.checked)} />
      <span>{label}</span>
    </label>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  primary,
  title,
}: {
  children: ReactNode;
  onClick: () => void;
  disabled?: boolean;
  primary?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      data-variant={primary ? 'primary' : undefined}
    >
      {children}
    </button>
  );
}

/* ----------------------------------------------------------------- legend */

export type LegendItem = { label: string; color: string; shape?: 'square' | 'line' | 'dot' };

export function Legend({ items }: { items: LegendItem[] }) {
  return (
    <div className="viz-legend">
      {items.map((it) => (
        <span className="viz-legend-item" key={it.label}>
          <span
            className="viz-swatch"
            aria-hidden="true"
            style={{
              background: it.shape === 'line' ? 'transparent' : it.color,
              borderTop: it.shape === 'line' ? `2px solid ${it.color}` : undefined,
              height: it.shape === 'line' ? 0 : undefined,
              borderRadius: it.shape === 'dot' ? '50%' : undefined,
            }}
          />
          {it.label}
        </span>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------- stat tiles */

export function Stats({ items }: { items: { label: string; value: ReactNode; hint?: string }[] }) {
  return (
    <div className="viz-stats">
      {items.map((s) => (
        <div className="viz-stat" key={s.label} title={s.hint}>
          <div className="viz-stat-label">{s.label}</div>
          <div className="viz-stat-value">{s.value}</div>
        </div>
      ))}
    </div>
  );
}

export function Note({ children }: { children: ReactNode }) {
  return (
    <p className="viz-note" role="status">
      {children}
    </p>
  );
}

/* ---------------------------------------------------------------- tooltip */

type TipState = { x: number; y: number; content: ReactNode } | null;
const TipCtx = createContext<(t: TipState) => void>(() => {});

/** Wrap a figure to give its children a shared hover tooltip. */
export function TooltipHost({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<TipState>(null);
  const ref = useRef<HTMLDivElement>(null);
  return (
    <TipCtx.Provider value={setTip}>
      <div ref={ref} style={{ position: 'relative' }} onMouseLeave={() => setTip(null)}>
        {children}
        {tip ? (
          <div
            className="viz-tooltip"
            style={{
              left: Math.max(0, tip.x + 12),
              top: Math.max(0, tip.y - 8),
              transform: 'translateY(-100%)',
            }}
          >
            {tip.content}
          </div>
        ) : null}
      </div>
    </TipCtx.Provider>
  );
}

/** Returns handlers to spread onto an SVG mark: {...useTip(<>…</>)} */
export function useTip() {
  const setTip = useContext(TipCtx);
  return useCallback(
    (content: ReactNode) => ({
      onMouseMove: (e: React.MouseEvent) => {
        const host = (e.currentTarget as Element).closest('.viz-figure')?.parentElement;
        const box = (host ?? (e.currentTarget as Element)).getBoundingClientRect();
        setTip({ x: e.clientX - box.left, y: e.clientY - box.top, content });
      },
      onMouseLeave: () => setTip(null),
      tabIndex: 0,
      onFocus: (e: React.FocusEvent) => {
        const r = (e.currentTarget as Element).getBoundingClientRect();
        const host = (e.currentTarget as Element).closest('.viz-figure')?.parentElement;
        const box = (host ?? (e.currentTarget as Element)).getBoundingClientRect();
        setTip({ x: r.left - box.left + r.width / 2, y: r.top - box.top, content });
      },
      onBlur: () => setTip(null),
    }),
    [setTip],
  );
}

/* ------------------------------------------------------------ animation */

/** requestAnimationFrame ticker that respects prefers-reduced-motion. */
export function useTicker(onTick: (dtMs: number) => void, running: boolean) {
  const cb = useRef(onTick);
  cb.current = onTick;
  useEffect(() => {
    if (!running) return;
    const reduced =
      typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
    let raf = 0;
    let last = performance.now();
    const step = (t: number) => {
      const dt = Math.min(t - last, 64);
      last = t;
      cb.current(reduced ? 1000 / 60 : dt);
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [running]);
}

/** Deterministic PRNG — visuals must look the same on every render/SSR pass. */
export function makeRng(seed = 1) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return ((s >>> 0) % 1e6) / 1e6;
  };
}

/* ------------------------------------------------------------- formatting */

export const SERIES = [
  'var(--viz-1)',
  'var(--viz-2)',
  'var(--viz-3)',
  'var(--viz-4)',
  'var(--viz-5)',
  'var(--viz-6)',
  'var(--viz-7)',
  'var(--viz-8)',
] as const;

export function fmtBytes(n: number) {
  const u = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n < 10 && i > 0 ? n.toFixed(1) : Math.round(n)} ${u[i]}`;
}

/** Nanoseconds → the largest unit that keeps it readable. */
export function fmtTime(ns: number) {
  if (ns < 1e3) return `${ns < 10 ? ns.toFixed(1) : Math.round(ns)} ns`;
  if (ns < 1e6) return `${(ns / 1e3).toFixed(ns < 1e4 ? 1 : 0)} µs`;
  if (ns < 1e9) return `${(ns / 1e6).toFixed(ns < 1e7 ? 1 : 0)} ms`;
  return `${(ns / 1e9).toFixed(2)} s`;
}

export function fmtNum(n: number, digits = 0) {
  return n.toLocaleString('en-US', { maximumFractionDigits: digits });
}

export function useSize(initial = 720) {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, e.contentRect.width)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

export const useMemoOnce = useMemo;
