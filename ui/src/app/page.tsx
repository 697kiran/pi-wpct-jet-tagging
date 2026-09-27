"use client";

import { useState, useCallback } from "react";
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell
} from "recharts";

/* ─────────────────────────────────────────────
   PI-WPCT Dashboard — Editorial Design System
   ───────────────────────────────────────────── */

interface ScatterPoint {
  x: number;
  y: number;
  z: number;
  type: string;
  fill: string;
}

const CLASS_COLORS: Record<string, string> = {
  "b-jet": "#0007cb",
  charm: "#888888",
  light: "#d3cec6",
};

// Custom tooltip for scatter chart
const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-ink text-pure-white px-4 py-3 rounded-[4px] text-[13px] font-saans leading-relaxed">
      <div className="font-medium mb-1">{d.type}</div>
      <div className="text-pure-white/70">b-score: {d.x.toFixed(3)}</div>
      <div className="text-pure-white/70">light-score: {d.y.toFixed(3)}</div>
    </div>
  );
};

export default function Dashboard() {
  const [workingPoint, setWorkingPoint] = useState(0.75);
  const [loading, setLoading] = useState(false);
  const [hasRun, setHasRun] = useState(false);
  const [activeTab, setActiveTab] = useState<"scatter" | "metrics">("scatter");

  const [metrics, setMetrics] = useState({
    lambdaLight: "—",
    lambdaCharm: "—",
    mistagLight: "—",
    mistagCharm: "—",
    purity: "—",
    bEff: "—",
  });

  const [scatterData, setScatterData] = useState<ScatterPoint[]>([]);

  const runSimulation = useCallback(async (wp: number) => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/api/simulate-inference", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_wp: wp }),
      });
      const data = await res.json();

      if (data.status === "success") {
        setMetrics({
          lambdaLight: data.metrics.dynamic_lambda_light.toFixed(4),
          lambdaCharm: data.metrics.dynamic_lambda_charm.toFixed(4),
          mistagLight: data.metrics.mistag_rate_light.toFixed(4),
          mistagCharm: data.metrics.mistag_rate_charm.toFixed(4),
          purity: (data.purity_proxy * 100).toFixed(1) + "%",
          bEff: (data.b_efficiency_achieved * 100).toFixed(1) + "%",
        });

        setScatterData(
          data.scatter_points.map((p: any) => ({
            x: p.score_b,
            y: p.score_light,
            z: p.score_charm,
            type: p.true_class === 2 ? "b-jet" : p.true_class === 1 ? "charm" : "light",
            fill: p.true_class === 2 ? "#0007cb" : p.true_class === 1 ? "#888888" : "#d3cec6",
          }))
        );
        setHasRun(true);
      }
    } catch {
      // Backend not running — silently handle
    } finally {
      setLoading(false);
    }
  }, []);

  const handleSliderMouseUp = () => {
    if (hasRun) runSimulation(workingPoint);
  };

  return (
    <div className="min-h-screen flex flex-col selection:bg-electric-violet/20">

      {/* ═══════════ TOP NAV ═══════════ */}
      <nav className="h-16 px-8 lg:px-16 flex items-center justify-between border-b border-hairline bg-pure-white/80 backdrop-blur-sm sticky top-0 z-50 animate-slide-down">
        <div className="flex items-center gap-10">
          {/* Logo */}
          <a href="#" className="flex items-center gap-2.5 group">
            <div className="w-[18px] h-[18px] rounded-[4px] bg-electric-violet group-hover:scale-110 transition-transform" />
            <span className="font-saans text-[15px] font-medium tracking-[-0.02em] text-ink">
              PI-WPCT
            </span>
          </a>

          {/* Nav links */}
          <div className="hidden md:flex items-center gap-8">
            {["Platform", "Documentation", "Research"].map((item) => (
              <button
                key={item}
                className="font-saans text-[14px] text-iron hover:text-ink transition-colors"
              >
                {item}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => runSimulation(workingPoint)}
            disabled={loading}
            className="font-saans text-[14px] bg-ink text-pure-white px-5 py-2.5 rounded-[4px] hover:bg-carbon/85 transition-all active:scale-[0.98] disabled:opacity-50 flex items-center gap-2"
          >
            {loading && (
              <svg className="animate-spin w-3.5 h-3.5" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" opacity="0.25" />
                <path d="M12 2a10 10 0 019.95 9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
              </svg>
            )}
            Run simulation
          </button>
        </div>
      </nav>

      {/* ═══════════ HERO ═══════════ */}
      <section className="relative overflow-hidden">
        {/* Background image */}
        <div className="absolute inset-0 z-0">
          <img
            src="https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=1920&q=80"
            alt=""
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-b from-canvas-cream/95 via-canvas-cream/80 to-canvas-cream" />
        </div>

        <div className="relative z-10 max-w-[1120px] mx-auto px-8 lg:px-16 pt-24 pb-28">
          {/* Badge */}
          <div className="animate-fade-in-up inline-flex items-center gap-2 mb-8 px-3 py-1.5 rounded-[4px] border border-hairline bg-pure-white/60 backdrop-blur-sm">
            <span className="font-saansmono text-[11px] tracking-[1.2px] uppercase text-electric-violet font-medium">
              V2.0
            </span>
            <span className="w-px h-3 bg-hairline" />
            <span className="font-saans text-[13px] text-iron">
              Conformal prediction sets now available
            </span>
          </div>

          {/* Headline */}
          <h1 className="animate-fade-in-up delay-100 font-saans text-[52px] lg:text-[72px] xl:text-[80px] leading-[0.92] tracking-[-2.4px] font-light text-ink max-w-[720px] mb-8">
            Principled risk
            <br />
            allocation for
            <br />
            <span className="text-electric-violet">jet tagging.</span>
          </h1>

          {/* Sub-copy */}
          <p className="animate-fade-in-up delay-200 font-serrif text-[17px] leading-[1.55] text-iron max-w-[440px] mb-10">
            A non-differentiable conformal pipeline for high-energy physics
            at the LHC. Fine-tune working points, compute dynamic risk weights,
            and validate purity bounds — all in real time.
          </p>

          {/* CTA row */}
          <div className="animate-fade-in-up delay-300 flex items-center gap-3">
            <button
              onClick={() => runSimulation(workingPoint)}
              disabled={loading}
              className="bg-ink text-pure-white font-saans text-[14px] px-6 py-3 rounded-[4px] hover:bg-carbon/85 transition-all active:scale-[0.98] flex items-center gap-2"
            >
              {loading ? (
                <svg className="animate-spin w-4 h-4" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" opacity="0.25" />
                  <path d="M12 2a10 10 0 019.95 9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                </svg>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
              )}
              Start inference
            </button>
            <button className="font-saans text-[14px] text-ink px-6 py-3 rounded-[4px] border border-hairline hover:bg-pure-white/60 transition-all">
              View documentation →
            </button>
          </div>
        </div>
      </section>

      {/* ═══════════ MAIN CONTENT ═══════════ */}
      <main className="max-w-[1120px] mx-auto px-8 lg:px-16 w-full pb-32">

        {/* ─── Control Surface ─── */}
        <section className="py-20 animate-fade-in delay-300">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-1.5 h-1.5 rounded-full bg-electric-violet" />
            <span className="font-saansmono text-[11px] tracking-[1.2px] uppercase text-graphite font-medium">
              Configuration
            </span>
          </div>
          <h2 className="font-saans text-[32px] lg:text-[40px] font-light tracking-[-1.2px] text-ink mb-12">
            Working point & risk parameters
          </h2>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left: Slider Panel */}
            <div className="lg:col-span-5 bg-pure-white border border-hairline rounded-[4px] p-8">
              <div className="font-saansmono text-[11px] tracking-[1.2px] uppercase text-graphite mb-6">
                Target Working Point
              </div>

              <div className="flex items-baseline justify-between mb-4">
                <span className="font-saans text-[56px] font-light tracking-[-1.62px] text-ink leading-none">
                  {workingPoint.toFixed(2)}
                </span>
                <span className="font-saansmono text-[12px] text-smoke">$w$</span>
              </div>

              <input
                type="range"
                min="0.50"
                max="0.99"
                step="0.01"
                value={workingPoint}
                onChange={(e) => setWorkingPoint(parseFloat(e.target.value))}
                onMouseUp={handleSliderMouseUp}
                onTouchEnd={handleSliderMouseUp}
                className="w-full mb-3"
              />
              <div className="flex justify-between font-saansmono text-[11px] text-smoke">
                <span>0.50</span>
                <span>0.75</span>
                <span>0.99</span>
              </div>

              <div className="mt-8 pt-6 border-t border-hairline space-y-4">
                <div className="flex justify-between items-center">
                  <span className="font-saans text-[14px] text-iron">Expected Purity</span>
                  <span className={`font-saansmono text-[14px] text-ink transition-opacity ${loading ? "opacity-30" : ""}`}>
                    {metrics.purity}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="font-saans text-[14px] text-iron">b-Efficiency</span>
                  <span className={`font-saansmono text-[14px] text-ink transition-opacity ${loading ? "opacity-30" : ""}`}>
                    {metrics.bEff}
                  </span>
                </div>
              </div>
            </div>

            {/* Right: Metric Cards */}
            <div className="lg:col-span-7 grid grid-cols-2 gap-6">
              {[
                { label: "λ_LIGHT", value: metrics.lambdaLight, sub: "Dynamic risk weight for light-jet misclassification" },
                { label: "λ_CHARM", value: metrics.lambdaCharm, sub: "Dynamic risk weight for charm-jet misclassification" },
                { label: "MISTAG LIGHT", value: metrics.mistagLight, sub: "Estimated light-jet mistag rate at current WP" },
                { label: "MISTAG CHARM", value: metrics.mistagCharm, sub: "Estimated charm-jet mistag rate at current WP" },
              ].map((card) => (
                <div
                  key={card.label}
                  className="bg-pure-white border border-hairline rounded-[4px] p-6 flex flex-col justify-between min-h-[160px] group hover:border-ink/20 transition-colors"
                >
                  <div>
                    <div className="font-saansmono text-[11px] tracking-[1.2px] uppercase text-graphite mb-1">
                      {card.label}
                    </div>
                    <div className="font-saans text-[13px] text-smoke leading-snug">
                      {card.sub}
                    </div>
                  </div>
                  <div className={`font-saans text-[36px] font-light tracking-[-0.64px] text-ink transition-opacity ${loading ? "opacity-30 animate-pulse-glow" : ""}`}>
                    {card.value}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ─── Visualization ─── */}
        <section className="py-20 border-t border-hairline">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-1.5 h-1.5 rounded-full bg-electric-violet" />
            <span className="font-saansmono text-[11px] tracking-[1.2px] uppercase text-graphite font-medium">
              Telemetry
            </span>
          </div>

          <div className="flex flex-col lg:flex-row lg:items-end justify-between mb-8">
            <h2 className="font-saans text-[32px] lg:text-[40px] font-light tracking-[-1.2px] text-ink">
              Inference results
            </h2>

            {/* Tab switcher */}
            <div className="flex items-center gap-1 mt-4 lg:mt-0 bg-linen rounded-[4px] p-1">
              {(["scatter", "metrics"] as const).map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`font-saansmono text-[12px] tracking-[0.8px] uppercase px-4 py-2 rounded-[4px] transition-all ${
                    activeTab === tab
                      ? "bg-pure-white text-ink border border-hairline"
                      : "text-smoke hover:text-iron"
                  }`}
                >
                  {tab === "scatter" ? "Scatter Plot" : "Risk Table"}
                </button>
              ))}
            </div>
          </div>

          {/* Chart / Table area */}
          <div className="bg-pure-white border border-hairline rounded-[4px] overflow-hidden relative">
            {loading && hasRun && (
              <div className="absolute inset-0 bg-pure-white/70 z-20 flex items-center justify-center backdrop-blur-[2px]">
                <svg className="animate-spin w-6 h-6 text-iron" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" opacity="0.25" />
                  <path d="M12 2a10 10 0 019.95 9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                </svg>
              </div>
            )}

            {!hasRun ? (
              /* Empty state */
              <div className="h-[420px] flex flex-col items-center justify-center p-8">
                <div className="w-12 h-12 rounded-[4px] bg-linen flex items-center justify-center mb-4">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#888888" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12,6 12,12 16,14" />
                  </svg>
                </div>
                <span className="font-saansmono text-[12px] tracking-[1.2px] uppercase text-graphite mb-1">
                  Awaiting sequence
                </span>
                <span className="font-saans text-[14px] text-smoke text-center max-w-sm">
                  Click &ldquo;Run simulation&rdquo; to generate real-time scatter topology and conformal boundaries.
                </span>
              </div>
            ) : activeTab === "scatter" ? (
              /* Scatter Plot */
              <div className="h-[420px] p-6 pl-2">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 24, bottom: 24, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1eee9" vertical={false} />
                    <XAxis
                      type="number"
                      dataKey="x"
                      name="b-jet Score"
                      domain={[0, 1]}
                      tick={{ fill: "#a0a0a0", fontSize: 11, fontFamily: "var(--font-jetbrains-mono, monospace)" }}
                      axisLine={{ stroke: "#dedbd6" }}
                      tickLine={false}
                      label={{ value: "b-jet score", position: "insideBottom", offset: -16, style: { fill: "#888888", fontSize: 12, fontFamily: "var(--font-jetbrains-mono, monospace)" } }}
                    />
                    <YAxis
                      type="number"
                      dataKey="y"
                      name="light Score"
                      domain={[0, 1]}
                      tick={{ fill: "#a0a0a0", fontSize: 11, fontFamily: "var(--font-jetbrains-mono, monospace)" }}
                      axisLine={{ stroke: "#dedbd6" }}
                      tickLine={false}
                      label={{ value: "light score", angle: -90, position: "insideLeft", offset: 10, style: { fill: "#888888", fontSize: 12, fontFamily: "var(--font-jetbrains-mono, monospace)" } }}
                    />
                    <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: "3 3", stroke: "#dedbd6" }} />
                    <ReferenceLine x={workingPoint} stroke="#0007cb" strokeDasharray="6 3" strokeWidth={1.5} />
                    <Scatter name="Jets" data={scatterData}>
                      {scatterData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} opacity={0.7} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            ) : (
              /* Metrics Table */
              <div className="p-6">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-hairline">
                      <th className="text-left font-saansmono text-[11px] tracking-[1.2px] uppercase text-graphite pb-3">Metric</th>
                      <th className="text-right font-saansmono text-[11px] tracking-[1.2px] uppercase text-graphite pb-3">Value</th>
                    </tr>
                  </thead>
                  <tbody className="font-saans text-[15px]">
                    {[
                      ["Dynamic λ (light)", metrics.lambdaLight],
                      ["Dynamic λ (charm)", metrics.lambdaCharm],
                      ["Mistag rate (light)", metrics.mistagLight],
                      ["Mistag rate (charm)", metrics.mistagCharm],
                      ["Expected purity", metrics.purity],
                      ["b-efficiency", metrics.bEff],
                      ["Working point", workingPoint.toFixed(2)],
                    ].map(([label, value]) => (
                      <tr key={label} className="border-b border-hairline/50 last:border-b-0">
                        <td className="py-4 text-iron">{label}</td>
                        <td className="py-4 text-right font-saansmono text-ink">{value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Legend */}
          {hasRun && activeTab === "scatter" && (
            <div className="flex items-center gap-6 mt-5">
              {Object.entries(CLASS_COLORS).map(([label, color]) => (
                <span key={label} className="flex items-center gap-2 font-saansmono text-[11px] tracking-[1.2px] uppercase text-iron">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
                  {label}
                </span>
              ))}
              <span className="flex items-center gap-2 font-saansmono text-[11px] tracking-[1.2px] uppercase text-electric-violet ml-auto">
                <span className="w-4 h-px bg-electric-violet" style={{ borderTop: "1.5px dashed #0007cb" }} />
                Working point threshold
              </span>
            </div>
          )}
        </section>

        {/* ─── About Section ─── */}
        <section className="py-20 border-t border-hairline">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-16">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <div className="w-1.5 h-1.5 rounded-full bg-electric-violet" />
                <span className="font-saansmono text-[11px] tracking-[1.2px] uppercase text-graphite font-medium">
                  Methodology
                </span>
              </div>
              <h2 className="font-saans text-[32px] lg:text-[40px] font-light tracking-[-1.2px] text-ink mb-6">
                How PI-WPCT works
              </h2>
              <p className="font-serrif text-[16px] leading-[1.55] text-iron mb-6">
                PI-WPCT introduces a principled, non-differentiable risk allocation
                framework that dynamically computes class-specific loss weights
                using conformal prediction theory. Unlike fixed-weight approaches,
                our method adapts to the observed mistag landscape at each working point.
              </p>
              <p className="font-serrif text-[16px] leading-[1.55] text-iron">
                The pipeline enforces strict data leakage prevention through
                a canonical 4-way split (train / calibration / validation / test)
                and variance-stabilized weight estimation — ensuring that all
                reported metrics are statistically honest.
              </p>
            </div>

            <div className="space-y-4">
              {[
                { step: "01", title: "Pilot calibration", desc: "Run forward pass on calibration split to estimate class-conditional mistag rates." },
                { step: "02", title: "Variance stabilization", desc: "Apply Wilson interval or bootstrap to stabilize noisy small-sample estimates." },
                { step: "03", title: "Risk allocation", desc: "Compute dynamic λ weights inversely proportional to stabilized mistag rates." },
                { step: "04", title: "Conformal validation", desc: "Validate coverage guarantees and purity bounds on held-out test split." },
              ].map((item) => (
                <div key={item.step} className="flex gap-5 p-5 rounded-[4px] border border-hairline bg-pure-white hover:border-ink/20 transition-colors group">
                  <span className="font-saansmono text-[12px] tracking-[1.2px] text-electric-violet font-medium shrink-0 mt-0.5">
                    {item.step}
                  </span>
                  <div>
                    <div className="font-saans text-[15px] font-medium text-ink mb-1">{item.title}</div>
                    <div className="font-saans text-[14px] text-iron leading-relaxed">{item.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

      </main>

      {/* ═══════════ FOOTER ═══════════ */}
      <footer className="border-t border-hairline py-8 px-8 lg:px-16">
        <div className="max-w-[1120px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-[4px] bg-electric-violet" />
            <span className="font-saans text-[13px] text-smoke">
              PI-WPCT Jet Tagging
            </span>
          </div>
          <div className="font-saansmono text-[11px] tracking-[1.2px] uppercase text-smoke">
            Built for HEP researchers at CERN
          </div>
        </div>
      </footer>

    </div>
  );
}
