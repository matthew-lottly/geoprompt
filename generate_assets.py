"""Generate polished modern SVG assets for geoprompt documentation."""
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(exist_ok=True)


def portfolio_scorecard():
    """Modern horizontal bar scorecard with gradient fills and clear labels."""
    return '''\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 420" width="720" height="420"
     font-family="'Segoe UI', system-ui, -apple-system, sans-serif">
  <defs>
    <linearGradient id="bar1" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#3b82f6"/><stop offset="100%" stop-color="#2563eb"/>
    </linearGradient>
    <linearGradient id="bar2" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#22c55e"/><stop offset="100%" stop-color="#16a34a"/>
    </linearGradient>
    <linearGradient id="bar3" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#f59e0b"/><stop offset="100%" stop-color="#d97706"/>
    </linearGradient>
    <linearGradient id="bar4" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#a78bfa"/><stop offset="100%" stop-color="#7c3aed"/>
    </linearGradient>
    <filter id="shadow" x="-2%" y="-2%" width="104%" height="108%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.10"/>
    </filter>
  </defs>

  <!-- Card background -->
  <rect width="720" height="420" rx="14" fill="#f8fafc"/>
  <rect x="1" y="1" width="718" height="418" rx="14" fill="none" stroke="#e2e8f0" stroke-width="1"/>

  <!-- Title block -->
  <text x="40" y="44" font-size="20" font-weight="700" fill="#0f172a">Portfolio Scorecard</text>
  <text x="40" y="66" font-size="12" fill="#64748b">Composite scores across four key infrastructure categories</text>

  <!-- Axis area: labels at left, bars to the right -->
  <!-- Grid lines (25, 50, 75, 100 marks) mapped to x 200..680 => 200+120n -->
  <g stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4 4">
    <line x1="200" y1="88" x2="200" y2="365"/>
    <line x1="320" y1="88" x2="320" y2="365"/>
    <line x1="440" y1="88" x2="440" y2="365"/>
    <line x1="560" y1="88" x2="560" y2="365"/>
    <line x1="680" y1="88" x2="680" y2="365"/>
  </g>
  <g font-size="10" fill="#94a3b8" text-anchor="middle">
    <text x="200" y="385">0</text>
    <text x="320" y="385">25</text>
    <text x="440" y="385">50</text>
    <text x="560" y="385">75</text>
    <text x="680" y="385">100</text>
  </g>

  <!-- Row 1: Water Distribution  score=80 => width=80/100*480=384 -->
  <text x="190" y="126" font-size="13" fill="#334155" font-weight="500" text-anchor="end">Water Distribution</text>
  <rect x="200" y="108" width="384" height="32" rx="6" fill="url(#bar1)" filter="url(#shadow)"/>
  <text x="592" y="130" font-size="12" font-weight="700" fill="#1e40af">80</text>

  <!-- Row 2: Storm Drainage  score=95 => width=456 -->
  <text x="190" y="196" font-size="13" fill="#334155" font-weight="500" text-anchor="end">Storm Drainage</text>
  <rect x="200" y="178" width="456" height="32" rx="6" fill="url(#bar2)" filter="url(#shadow)"/>
  <text x="664" y="200" font-size="12" font-weight="700" fill="#15803d">95</text>

  <!-- Row 3: Electric Feeders  score=60 => width=288 -->
  <text x="190" y="266" font-size="13" fill="#334155" font-weight="500" text-anchor="end">Electric Feeders</text>
  <rect x="200" y="248" width="288" height="32" rx="6" fill="url(#bar3)" filter="url(#shadow)"/>
  <text x="496" y="270" font-size="12" font-weight="700" fill="#b45309">60</text>

  <!-- Row 4: Telecom Trunk  score=42 => width=201.6 -->
  <text x="190" y="336" font-size="13" fill="#334155" font-weight="500" text-anchor="end">Telecom Trunk</text>
  <rect x="200" y="318" width="202" height="32" rx="6" fill="url(#bar4)" filter="url(#shadow)"/>
  <text x="410" y="340" font-size="12" font-weight="700" fill="#6d28d9">42</text>

  <!-- Target reference line  target=76 => x=200+76/100*480=564.8 -->
  <line x1="565" y1="92" x2="565" y2="362" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="6 3"/>
  <rect x="539" y="92" width="52" height="18" rx="4" fill="#fef2f2"/>
  <text x="565" y="104" font-size="10" fill="#dc2626" text-anchor="middle" font-weight="600">Target 76</text>

  <!-- Legend -->
  <text x="40" y="408" font-size="10" fill="#94a3b8">Higher is better  ·  Scores normalised 0–100</text>
</svg>'''


def before_after_scenario():
    """Side-by-side map-style boxes showing meaningful before/after differences."""
    return '''\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 440" width="820" height="440"
     font-family="'Segoe UI', system-ui, -apple-system, sans-serif">
  <defs>
    <filter id="sh" x="-2%" y="-2%" width="104%" height="108%">
      <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.08"/>
    </filter>
    <linearGradient id="deficitGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#fca5a5"/><stop offset="100%" stop-color="#fecaca"/>
    </linearGradient>
    <linearGradient id="surplusGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#86efac"/><stop offset="100%" stop-color="#bbf7d0"/>
    </linearGradient>
  </defs>

  <!-- Card -->
  <rect width="820" height="440" rx="14" fill="#f8fafc"/>
  <rect x="1" y="1" width="818" height="438" rx="14" fill="none" stroke="#e2e8f0"/>

  <!-- Title -->
  <text x="40" y="40" font-size="20" font-weight="700" fill="#0f172a">Before / After Scenario</text>
  <text x="40" y="62" font-size="12" fill="#64748b">Water-pressure study — proposed main replacement on Elm Street</text>

  <!-- ===== BEFORE panel (left) ===== -->
  <g transform="translate(30,82)">
    <rect width="370" height="310" rx="10" fill="#ffffff" stroke="#cbd5e1" filter="url(#sh)"/>
    <rect width="370" height="34" rx="10" fill="#f1f5f9"/>
    <rect y="17" width="370" height="17" fill="#f1f5f9"/>
    <text x="185" y="24" font-size="13" font-weight="600" fill="#475569" text-anchor="middle">BEFORE — Baseline</text>

    <!-- Map background -->
    <rect x="16" y="46" width="338" height="210" rx="6" fill="#f0f9ff"/>

    <!-- Streets (simplified) -->
    <g stroke="#94a3b8" stroke-width="2" fill="none">
      <line x1="40" y1="100" x2="330" y2="100"/>
      <line x1="40" y1="160" x2="330" y2="160"/>
      <line x1="100" y1="60" x2="100" y2="240"/>
      <line x1="250" y1="60" x2="250" y2="240"/>
    </g>
    <!-- Street labels -->
    <text x="175" y="93" font-size="9" fill="#64748b" text-anchor="middle">Main St</text>
    <text x="175" y="153" font-size="9" fill="#64748b" text-anchor="middle">Elm St</text>

    <!-- Pressure nodes — mostly red/yellow (deficit) -->
    <circle cx="100" cy="100" r="14" fill="#fca5a5" stroke="#dc2626" stroke-width="2"/>
    <text x="100" y="104" font-size="9" font-weight="700" fill="#991b1b" text-anchor="middle">32</text>

    <circle cx="250" cy="100" r="14" fill="#fde68a" stroke="#d97706" stroke-width="2"/>
    <text x="250" y="104" font-size="9" font-weight="700" fill="#92400e" text-anchor="middle">45</text>

    <circle cx="100" cy="160" r="14" fill="#fca5a5" stroke="#dc2626" stroke-width="2"/>
    <text x="100" y="164" font-size="9" font-weight="700" fill="#991b1b" text-anchor="middle">28</text>

    <circle cx="250" cy="160" r="14" fill="#fca5a5" stroke="#dc2626" stroke-width="2"/>
    <text x="250" y="164" font-size="9" font-weight="700" fill="#991b1b" text-anchor="middle">35</text>

    <!-- Pipe highlight (old, thin) -->
    <line x1="100" y1="160" x2="250" y2="160" stroke="#dc2626" stroke-width="4" opacity="0.4"/>

    <!-- Legend labels -->
    <text x="56" y="76" font-size="8" fill="#64748b">Node A</text>
    <text x="206" y="76" font-size="8" fill="#64748b">Node B</text>
    <text x="56" y="186" font-size="8" fill="#64748b">Node C</text>
    <text x="206" y="186" font-size="8" fill="#64748b">Node D</text>

    <!-- Metrics footer -->
    <g transform="translate(16,268)">
      <rect width="338" height="34" rx="5" fill="#fef2f2"/>
      <circle cx="14" cy="17" r="4" fill="#dc2626"/>
      <text x="24" y="21" font-size="11" fill="#991b1b" font-weight="500">Avg pressure: 35 psi  ·  3 of 4 nodes below target</text>
    </g>
  </g>

  <!-- ===== AFTER panel (right) ===== -->
  <g transform="translate(420,82)">
    <rect width="370" height="310" rx="10" fill="#ffffff" stroke="#cbd5e1" filter="url(#sh)"/>
    <rect width="370" height="34" rx="10" fill="#f0fdf4"/>
    <rect y="17" width="370" height="17" fill="#f0fdf4"/>
    <text x="185" y="24" font-size="13" font-weight="600" fill="#166534" text-anchor="middle">AFTER — Proposed</text>

    <!-- Map background -->
    <rect x="16" y="46" width="338" height="210" rx="6" fill="#f0fdf9"/>

    <!-- Streets -->
    <g stroke="#94a3b8" stroke-width="2" fill="none">
      <line x1="40" y1="100" x2="330" y2="100"/>
      <line x1="40" y1="160" x2="330" y2="160"/>
      <line x1="100" y1="60" x2="100" y2="240"/>
      <line x1="250" y1="60" x2="250" y2="240"/>
    </g>
    <text x="175" y="93" font-size="9" fill="#64748b" text-anchor="middle">Main St</text>
    <text x="175" y="153" font-size="9" fill="#64748b" text-anchor="middle">Elm St</text>

    <!-- Pressure nodes — all green/yellow after improvement -->
    <circle cx="100" cy="100" r="14" fill="#86efac" stroke="#16a34a" stroke-width="2"/>
    <text x="100" y="104" font-size="9" font-weight="700" fill="#166534" text-anchor="middle">58</text>

    <circle cx="250" cy="100" r="14" fill="#86efac" stroke="#16a34a" stroke-width="2"/>
    <text x="250" y="104" font-size="9" font-weight="700" fill="#166534" text-anchor="middle">62</text>

    <circle cx="100" cy="160" r="14" fill="#86efac" stroke="#16a34a" stroke-width="2"/>
    <text x="100" y="164" font-size="9" font-weight="700" fill="#166534" text-anchor="middle">52</text>

    <circle cx="250" cy="160" r="14" fill="#fde68a" stroke="#d97706" stroke-width="2"/>
    <text x="250" y="164" font-size="9" font-weight="700" fill="#92400e" text-anchor="middle">48</text>

    <!-- New pipe highlight (thick green) -->
    <line x1="100" y1="160" x2="250" y2="160" stroke="#16a34a" stroke-width="5" opacity="0.5"/>
    <text x="175" y="202" font-size="9" fill="#166534" font-weight="600" text-anchor="middle">New 12" main</text>

    <!-- Improvement arrows -->
    <g font-size="9" fill="#16a34a" font-weight="700">
      <text x="126" y="100" text-anchor="start">+26 &#x25B2;</text>
      <text x="276" y="100" text-anchor="start">+17 &#x25B2;</text>
      <text x="126" y="160" text-anchor="start">+24 &#x25B2;</text>
      <text x="276" y="160" text-anchor="start">+13 &#x25B2;</text>
    </g>

    <!-- Node labels -->
    <text x="56" y="76" font-size="8" fill="#64748b">Node A</text>
    <text x="206" y="76" font-size="8" fill="#64748b">Node B</text>
    <text x="56" y="186" font-size="8" fill="#64748b">Node C</text>
    <text x="206" y="186" font-size="8" fill="#64748b">Node D</text>

    <!-- Metrics footer -->
    <g transform="translate(16,268)">
      <rect width="338" height="34" rx="5" fill="#f0fdf4"/>
      <circle cx="14" cy="17" r="4" fill="#16a34a"/>
      <text x="24" y="21" font-size="11" fill="#166534" font-weight="500">Avg pressure: 55 psi  ·  0 of 4 nodes below target</text>
    </g>
  </g>

  <!-- Delta summary bar -->
  <g transform="translate(30,402)">
    <rect width="760" height="28" rx="6" fill="#eff6ff"/>
    <text x="380" y="19" font-size="12" fill="#1e40af" text-anchor="middle" font-weight="600">
      &#x2705;  Average pressure +20 psi  ·  Deficit nodes reduced 3 &#x2192; 0  ·  Fire-flow compliance 100%
    </text>
  </g>
</svg>'''


def restoration_storyboard():
    """Modern flowchart showing a 4-stage network restoration."""
    return '''\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 840 340" width="840" height="340"
     font-family="'Segoe UI', system-ui, -apple-system, sans-serif">
  <defs>
    <filter id="card" x="-2%" y="-4%" width="104%" height="112%">
      <feDropShadow dx="0" dy="2" stdDeviation="4" flood-opacity="0.10"/>
    </filter>
    <linearGradient id="prog" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#dc2626"/><stop offset="33%" stop-color="#f59e0b"/>
      <stop offset="66%" stop-color="#22c55e"/><stop offset="100%" stop-color="#16a34a"/>
    </linearGradient>
    <marker id="arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6" fill="#94a3b8"/>
    </marker>
  </defs>

  <!-- Background -->
  <rect width="840" height="340" rx="14" fill="#f8fafc"/>
  <rect x="1" y="1" width="838" height="338" rx="14" fill="none" stroke="#e2e8f0"/>

  <!-- Title -->
  <text x="40" y="38" font-size="20" font-weight="700" fill="#0f172a">Restoration Storyboard</text>
  <text x="40" y="58" font-size="12" fill="#64748b">Four-stage outage recovery sequence for feeder F-12</text>

  <!-- Progress bar -->
  <rect x="40" y="72" width="760" height="6" rx="3" fill="#e2e8f0"/>
  <rect x="40" y="72" width="760" height="6" rx="3" fill="url(#prog)"/>

  <!-- Connector arrows -->
  <g>
    <line x1="218" y1="182" x2="238" y2="182" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>
    <line x1="418" y1="182" x2="438" y2="182" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>
    <line x1="618" y1="182" x2="638" y2="182" stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>
  </g>

  <!-- Stage 1: Detect -->
  <g transform="translate(30,100)">
    <rect width="186" height="164" rx="10" fill="#ffffff" stroke="#fca5a5" stroke-width="2" filter="url(#card)"/>
    <rect width="186" height="36" rx="10" fill="#fef2f2"/>
    <rect y="18" width="186" height="18" fill="#fef2f2"/>
    <!-- Icon circle -->
    <circle cx="24" cy="18" r="12" fill="#dc2626"/>
    <text x="24" y="22" font-size="11" fill="#fff" text-anchor="middle" font-weight="700">!</text>
    <text x="44" y="23" font-size="13" font-weight="600" fill="#991b1b">Detect Fault</text>
    <!-- Body -->
    <text x="16" y="58" font-size="11" fill="#475569">SCADA alarm triggers</text>
    <text x="16" y="76" font-size="11" fill="#475569">isolation of faulted</text>
    <text x="16" y="94" font-size="11" fill="#475569">segment on edge e1.</text>
    <!-- Metric chip -->
    <rect x="12" y="116" width="80" height="22" rx="4" fill="#fef2f2" stroke="#fca5a5"/>
    <text x="52" y="131" font-size="10" fill="#dc2626" text-anchor="middle" font-weight="600">t = 0 min</text>
    <rect x="98" y="116" width="76" height="22" rx="4" fill="#fef2f2" stroke="#fca5a5"/>
    <text x="136" y="131" font-size="10" fill="#dc2626" text-anchor="middle" font-weight="600">0% served</text>
  </g>

  <!-- Stage 2: Isolate -->
  <g transform="translate(230,100)">
    <rect width="186" height="164" rx="10" fill="#ffffff" stroke="#fde68a" stroke-width="2" filter="url(#card)"/>
    <rect width="186" height="36" rx="10" fill="#fefce8"/>
    <rect y="18" width="186" height="18" fill="#fefce8"/>
    <circle cx="24" cy="18" r="12" fill="#d97706"/>
    <text x="24" y="23" font-size="14" fill="#fff" text-anchor="middle" font-weight="700">&#x26A0;</text>
    <text x="44" y="23" font-size="13" font-weight="600" fill="#92400e">Isolate Section</text>
    <text x="16" y="58" font-size="11" fill="#475569">Open sectionaliser S3.</text>
    <text x="16" y="76" font-size="11" fill="#475569">Re-energise healthy</text>
    <text x="16" y="94" font-size="11" fill="#475569">upstream segments.</text>
    <rect x="12" y="116" width="80" height="22" rx="4" fill="#fefce8" stroke="#fde68a"/>
    <text x="52" y="131" font-size="10" fill="#d97706" text-anchor="middle" font-weight="600">t = 12 min</text>
    <rect x="98" y="116" width="76" height="22" rx="4" fill="#fefce8" stroke="#fde68a"/>
    <text x="136" y="131" font-size="10" fill="#d97706" text-anchor="middle" font-weight="600">40% served</text>
  </g>

  <!-- Stage 3: Repair -->
  <g transform="translate(430,100)">
    <rect width="186" height="164" rx="10" fill="#ffffff" stroke="#86efac" stroke-width="2" filter="url(#card)"/>
    <rect width="186" height="36" rx="10" fill="#f0fdf4"/>
    <rect y="18" width="186" height="18" fill="#f0fdf4"/>
    <circle cx="24" cy="18" r="12" fill="#16a34a"/>
    <text x="24" y="22" font-size="13" fill="#fff" text-anchor="middle" font-weight="700">&#x2699;</text>
    <text x="44" y="23" font-size="13" font-weight="600" fill="#166534">Repair Edge</text>
    <text x="16" y="58" font-size="11" fill="#475569">Crew dispatched to e1.</text>
    <text x="16" y="76" font-size="11" fill="#475569">Splice replaced, joint</text>
    <text x="16" y="94" font-size="11" fill="#475569">tested and energised.</text>
    <rect x="12" y="116" width="80" height="22" rx="4" fill="#f0fdf4" stroke="#86efac"/>
    <text x="52" y="131" font-size="10" fill="#16a34a" text-anchor="middle" font-weight="600">t = 90 min</text>
    <rect x="98" y="116" width="76" height="22" rx="4" fill="#f0fdf4" stroke="#86efac"/>
    <text x="136" y="131" font-size="10" fill="#16a34a" text-anchor="middle" font-weight="600">85% served</text>
  </g>

  <!-- Stage 4: Verify -->
  <g transform="translate(630,100)">
    <rect width="186" height="164" rx="10" fill="#ffffff" stroke="#86efac" stroke-width="2" filter="url(#card)"/>
    <rect width="186" height="36" rx="10" fill="#f0fdf4"/>
    <rect y="18" width="186" height="18" fill="#f0fdf4"/>
    <circle cx="24" cy="18" r="12" fill="#166534"/>
    <text x="24" y="23" font-size="13" fill="#fff" text-anchor="middle" font-weight="700">&#x2713;</text>
    <text x="44" y="23" font-size="13" font-weight="600" fill="#166534">Verify &amp; Close</text>
    <text x="16" y="58" font-size="11" fill="#475569">Full load restored.</text>
    <text x="16" y="76" font-size="11" fill="#475569">Voltage within limits,</text>
    <text x="16" y="94" font-size="11" fill="#475569">incident report filed.</text>
    <rect x="12" y="116" width="80" height="22" rx="4" fill="#f0fdf4" stroke="#86efac"/>
    <text x="52" y="131" font-size="10" fill="#166534" text-anchor="middle" font-weight="600">t = 105 min</text>
    <rect x="98" y="116" width="76" height="22" rx="4" fill="#f0fdf4" stroke="#86efac"/>
    <text x="136" y="131" font-size="10" fill="#166534" text-anchor="middle" font-weight="600">100% served</text>
  </g>

  <!-- Bottom timeline bar -->
  <g transform="translate(40, 290)">
    <line x1="0" y1="10" x2="760" y2="10" stroke="#cbd5e1" stroke-width="2"/>
    <!-- Time dots -->
    <circle cx="0" cy="10" r="5" fill="#dc2626"/>
    <text x="0" y="32" font-size="10" fill="#64748b" text-anchor="middle">0 min</text>
    <circle cx="253" cy="10" r="5" fill="#d97706"/>
    <text x="253" y="32" font-size="10" fill="#64748b" text-anchor="middle">12 min</text>
    <circle cx="506" cy="10" r="5" fill="#16a34a"/>
    <text x="506" y="32" font-size="10" fill="#64748b" text-anchor="middle">90 min</text>
    <circle cx="760" cy="10" r="5" fill="#166534"/>
    <text x="760" y="32" font-size="10" fill="#64748b" text-anchor="middle">105 min</text>
  </g>
</svg>'''


if __name__ == "__main__":
    (ASSETS / "portfolio-scorecard-example.svg").write_text(
        portfolio_scorecard(), encoding="utf-8"
    )
    print("Wrote portfolio-scorecard-example.svg")

    (ASSETS / "before-after-scenario-example.svg").write_text(
        before_after_scenario(), encoding="utf-8"
    )
    print("Wrote before-after-scenario-example.svg")

    (ASSETS / "restoration-storyboard-example.svg").write_text(
        restoration_storyboard(), encoding="utf-8"
    )
    print("Wrote restoration-storyboard-example.svg")

    print("Done – all 3 SVGs regenerated.")
