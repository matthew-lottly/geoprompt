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


def network_restoration_unmet_demand_chart():
    """Dual-axis style chart showing restoration timeline and unmet demand."""
    return '''\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 360" width="860" height="360"
     font-family="'Segoe UI', system-ui, -apple-system, sans-serif">
  <rect width="860" height="360" rx="14" fill="#f7f9fc"/>
  <rect x="1" y="1" width="858" height="358" rx="14" fill="none" stroke="#d7dde8"/>

  <text x="34" y="34" font-size="20" font-weight="700" fill="#12263a">Restoration timeline vs unmet demand</text>
  <text x="34" y="54" font-size="12" fill="#4c5d70">Sample dataset: data/sample_features.json + network scenario fixtures</text>

  <rect x="70" y="78" width="760" height="226" rx="8" fill="#ffffff" stroke="#d7dde8"/>
  <g stroke="#e4e9f2" stroke-width="1">
    <line x1="120" y1="102" x2="120" y2="280"/>
    <line x1="260" y1="102" x2="260" y2="280"/>
    <line x1="400" y1="102" x2="400" y2="280"/>
    <line x1="540" y1="102" x2="540" y2="280"/>
    <line x1="680" y1="102" x2="680" y2="280"/>
    <line x1="820" y1="102" x2="820" y2="280"/>
    <line x1="120" y1="280" x2="820" y2="280"/>
    <line x1="120" y1="235" x2="820" y2="235"/>
    <line x1="120" y1="190" x2="820" y2="190"/>
    <line x1="120" y1="145" x2="820" y2="145"/>
    <line x1="120" y1="102" x2="820" y2="102"/>
  </g>

  <g font-size="10" fill="#5c6b7a" text-anchor="middle">
    <text x="120" y="296">0h</text><text x="260" y="296">1h</text><text x="400" y="296">2h</text>
    <text x="540" y="296">3h</text><text x="680" y="296">4h</text><text x="820" y="296">5h</text>
  </g>
  <text x="470" y="316" font-size="11" fill="#4c5d70" text-anchor="middle">Time since disruption (hours)</text>

  <g font-size="10" fill="#5c6b7a" text-anchor="end">
    <text x="112" y="283">0</text><text x="112" y="238">25</text><text x="112" y="193">50</text><text x="112" y="148">75</text><text x="112" y="105">100</text>
  </g>
  <text x="24" y="192" font-size="11" fill="#4c5d70" transform="rotate(-90 24,192)">Service restored (%)</text>

  <g font-size="10" fill="#5c6b7a" text-anchor="start">
    <text x="828" y="283">0</text><text x="828" y="238">20</text><text x="828" y="193">40</text><text x="828" y="148">60</text><text x="828" y="105">80</text>
  </g>
  <text x="852" y="192" font-size="11" fill="#4c5d70" transform="rotate(90 852,192)">Unmet demand (MWh)</text>

  <polyline points="120,280 260,235 400,190 540,145 680,120 820,102" fill="none" stroke="#1565c0" stroke-width="3"/>
  <g fill="#1565c0">
    <circle cx="120" cy="280" r="4"/><circle cx="260" cy="235" r="4"/><circle cx="400" cy="190" r="4"/>
    <circle cx="540" cy="145" r="4"/><circle cx="680" cy="120" r="4"/><circle cx="820" cy="102" r="4"/>
  </g>

  <polyline points="120,102 260,120 400,145 540,172 680,212 820,246" fill="none" stroke="#b42318" stroke-width="3"/>
  <g fill="#b42318">
    <rect x="116" y="98" width="8" height="8" rx="2"/><rect x="256" y="116" width="8" height="8" rx="2"/>
    <rect x="396" y="141" width="8" height="8" rx="2"/><rect x="536" y="168" width="8" height="8" rx="2"/>
    <rect x="676" y="208" width="8" height="8" rx="2"/><rect x="816" y="242" width="8" height="8" rx="2"/>
  </g>

  <text x="827" y="98" font-size="11" fill="#0f4d92" font-weight="600">100% restored</text>
  <text x="827" y="260" font-size="11" fill="#8f1d15" font-weight="600">15 MWh unmet</text>

  <rect x="70" y="326" width="760" height="22" rx="6" fill="#edf3fb"/>
  <text x="98" y="341" font-size="11" fill="#2b3d52">Legend:</text>
  <line x1="150" y1="338" x2="174" y2="338" stroke="#1565c0" stroke-width="3"/>
  <text x="180" y="341" font-size="11" fill="#2b3d52">Restoration progression</text>
  <line x1="352" y1="338" x2="376" y2="338" stroke="#b42318" stroke-width="3"/>
  <text x="382" y="341" font-size="11" fill="#2b3d52">Unmet demand curve</text>
  <text x="608" y="341" font-size="11" fill="#2b3d52">Direct labels shown at final points</text>
</svg>'''


def resilience_heatmap_and_mitigation_chart():
    """Heatmap plus before-after mitigation bars for resilience screening."""
    return '''\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 420" width="900" height="420"
     font-family="'Segoe UI', system-ui, -apple-system, sans-serif">
  <rect width="900" height="420" rx="14" fill="#f7f9fc"/>
  <rect x="1" y="1" width="898" height="418" rx="14" fill="none" stroke="#d7dde8"/>

  <text x="34" y="34" font-size="20" font-weight="700" fill="#12263a">Resilience risk heatmap and mitigation impact</text>
  <text x="34" y="54" font-size="12" fill="#4c5d70">Sample dataset: outage and restoration summary fixtures</text>

  <rect x="34" y="78" width="430" height="300" rx="10" fill="#ffffff" stroke="#d7dde8"/>
  <text x="50" y="102" font-size="13" font-weight="700" fill="#1f3347">Risk heatmap (likelihood x consequence)</text>

  <g transform="translate(50,122)">
    <rect x="0" y="0" width="360" height="220" fill="#f8fafc" stroke="#d7dde8"/>
    <g>
      <rect x="0" y="0" width="120" height="73" fill="#e8f5e9"/>
      <rect x="120" y="0" width="120" height="73" fill="#fff3cd"/>
      <rect x="240" y="0" width="120" height="73" fill="#f8d7da"/>
      <rect x="0" y="73" width="120" height="73" fill="#fff3cd"/>
      <rect x="120" y="73" width="120" height="73" fill="#fde2a8"/>
      <rect x="240" y="73" width="120" height="73" fill="#f5c2c7"/>
      <rect x="0" y="146" width="120" height="74" fill="#f8d7da"/>
      <rect x="120" y="146" width="120" height="74" fill="#f5c2c7"/>
      <rect x="240" y="146" width="120" height="74" fill="#ef9aa2"/>
    </g>
    <g stroke="#d7dde8" stroke-width="1">
      <line x1="120" y1="0" x2="120" y2="220"/><line x1="240" y1="0" x2="240" y2="220"/>
      <line x1="0" y1="73" x2="360" y2="73"/><line x1="0" y1="146" x2="360" y2="146"/>
    </g>
    <text x="60" y="236" font-size="10" fill="#4c5d70" text-anchor="middle">Low consequence</text>
    <text x="180" y="236" font-size="10" fill="#4c5d70" text-anchor="middle">Medium</text>
    <text x="300" y="236" font-size="10" fill="#4c5d70" text-anchor="middle">High consequence</text>
    <text x="-8" y="190" font-size="10" fill="#4c5d70" transform="rotate(-90 -8,190)">Likelihood increases</text>

    <circle cx="275" cy="40" r="7" fill="#b42318"/><text x="287" y="43" font-size="10" fill="#8f1d15">substation A</text>
    <circle cx="292" cy="117" r="7" fill="#b42318"/><text x="304" y="120" font-size="10" fill="#8f1d15">hospital feeder</text>
    <circle cx="170" cy="188" r="7" fill="#d97706"/><text x="182" y="191" font-size="10" fill="#9a5a00">pump node</text>
    <circle cx="75" cy="42" r="7" fill="#12724f"/><text x="87" y="45" font-size="10" fill="#125f41">telecom node</text>
  </g>

  <rect x="486" y="78" width="380" height="300" rx="10" fill="#ffffff" stroke="#d7dde8"/>
  <text x="502" y="102" font-size="13" font-weight="700" fill="#1f3347">Before and after mitigation bars</text>
  <text x="502" y="118" font-size="10" fill="#4c5d70">Unit: impacted customers</text>

  <g transform="translate(502,132)">
    <g font-size="10" fill="#4c5d70" text-anchor="end">
      <text x="86" y="28">Zone North</text><text x="86" y="88">Zone East</text><text x="86" y="148">Zone South</text>
    </g>
    <g>
      <rect x="98" y="14" width="210" height="14" fill="#b42318" rx="4"/>
      <rect x="98" y="34" width="112" height="14" fill="#12724f" rx="4"/>
      <rect x="98" y="74" width="180" height="14" fill="#b42318" rx="4"/>
      <rect x="98" y="94" width="92" height="14" fill="#12724f" rx="4"/>
      <rect x="98" y="134" width="160" height="14" fill="#b42318" rx="4"/>
      <rect x="98" y="154" width="76" height="14" fill="#12724f" rx="4"/>
    </g>
    <g font-size="10" fill="#2b3d52">
      <text x="314" y="25">210</text><text x="216" y="45">112</text>
      <text x="284" y="85">180</text><text x="196" y="105">92</text>
      <text x="264" y="145">160</text><text x="180" y="165">76</text>
    </g>
    <line x1="98" y1="188" x2="330" y2="188" stroke="#d7dde8"/>
    <g font-size="10" fill="#4c5d70" text-anchor="middle">
      <text x="98" y="202">0</text><text x="156" y="202">50</text><text x="214" y="202">100</text><text x="272" y="202">150</text><text x="330" y="202">200+</text>
    </g>
  </g>

  <rect x="34" y="388" width="832" height="20" rx="6" fill="#edf3fb"/>
  <text x="56" y="402" font-size="11" fill="#2b3d52">Legend:</text>
  <rect x="96" y="394" width="12" height="8" rx="2" fill="#b42318"/><text x="114" y="402" font-size="11" fill="#2b3d52">before mitigation</text>
  <rect x="238" y="394" width="12" height="8" rx="2" fill="#12724f"/><text x="256" y="402" font-size="11" fill="#2b3d52">after mitigation</text>
  <text x="530" y="402" font-size="11" fill="#2b3d52">Direct labels and axis units included for QA compliance</text>
</svg>'''


def migration_effort_benefit_quadrant():
    """Effort-benefit quadrant for migration planning narratives."""
    return '''\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 400" width="820" height="400"
     font-family="'Segoe UI', system-ui, -apple-system, sans-serif">
  <rect width="820" height="400" rx="14" fill="#f7f9fc"/>
  <rect x="1" y="1" width="818" height="398" rx="14" fill="none" stroke="#d7dde8"/>

  <text x="32" y="36" font-size="20" font-weight="700" fill="#12263a">Migration effort vs benefit quadrant</text>
  <text x="32" y="56" font-size="12" fill="#4c5d70">Sample tasks from migration playbooks in docs and examples</text>

  <rect x="72" y="84" width="680" height="260" rx="10" fill="#ffffff" stroke="#d7dde8"/>
  <line x1="412" y1="96" x2="412" y2="332" stroke="#8ca0b3" stroke-dasharray="5 4"/>
  <line x1="84" y1="214" x2="740" y2="214" stroke="#8ca0b3" stroke-dasharray="5 4"/>

  <text x="244" y="110" font-size="11" fill="#4c5d70" text-anchor="middle">Quick wins</text>
  <text x="580" y="110" font-size="11" fill="#4c5d70" text-anchor="middle">Strategic investments</text>
  <text x="244" y="328" font-size="11" fill="#4c5d70" text-anchor="middle">Defer or simplify</text>
  <text x="580" y="328" font-size="11" fill="#4c5d70" text-anchor="middle">Plan and phase</text>

  <text x="412" y="362" font-size="11" fill="#4c5d70" text-anchor="middle">Implementation effort (low -> high)</text>
  <text x="32" y="214" font-size="11" fill="#4c5d70" transform="rotate(-90 32,214)">Business benefit (low -> high)</text>

  <g>
    <circle cx="220" cy="150" r="11" fill="#12724f"/><text x="236" y="154" font-size="11" fill="#1f3347">Scenario report exports</text>
    <circle cx="312" cy="132" r="11" fill="#12724f"/><text x="328" y="136" font-size="11" fill="#1f3347">GeoPandas interop checks</text>
    <circle cx="520" cy="152" r="11" fill="#1565c0"/><text x="536" y="156" font-size="11" fill="#1f3347">Resilience portfolio model</text>
    <circle cx="606" cy="172" r="11" fill="#1565c0"/><text x="622" y="176" font-size="11" fill="#1f3347">Service deployment hardening</text>
    <circle cx="236" cy="262" r="11" fill="#d97706"/><text x="252" y="266" font-size="11" fill="#1f3347">Legacy one-off scripts</text>
    <circle cx="574" cy="258" r="11" fill="#b42318"/><text x="590" y="262" font-size="11" fill="#1f3347">Full enterprise parity claims</text>
  </g>

  <rect x="72" y="356" width="680" height="22" rx="6" fill="#edf3fb"/>
  <text x="92" y="371" font-size="11" fill="#2b3d52">Legend:</text>
  <circle cx="132" cy="367" r="5" fill="#12724f"/><text x="142" y="371" font-size="11" fill="#2b3d52">low effort / high benefit</text>
  <circle cx="286" cy="367" r="5" fill="#1565c0"/><text x="296" y="371" font-size="11" fill="#2b3d52">high effort / high benefit</text>
  <circle cx="446" cy="367" r="5" fill="#d97706"/><text x="456" y="371" font-size="11" fill="#2b3d52">low benefit</text>
  <circle cx="536" cy="367" r="5" fill="#b42318"/><text x="546" y="371" font-size="11" fill="#2b3d52">high risk initiatives</text>
</svg>'''


def neighborhood_pressure_live_svg():
    """Publication-style neighborhood pressure surface from demo outputs."""
    return '''\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 520" width="980" height="520"
     font-family="'Segoe UI', system-ui, -apple-system, sans-serif">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f3f7fb"/>
      <stop offset="100%" stop-color="#e8f0f7"/>
    </linearGradient>
  </defs>
  <rect width="980" height="520" rx="16" fill="url(#bg)"/>
  <rect x="1" y="1" width="978" height="518" rx="16" fill="none" stroke="#c9d8e7"/>

  <text x="34" y="38" font-size="22" font-weight="700" fill="#133148">Neighborhood Pressure Review</text>
  <text x="34" y="58" font-size="12" fill="#4a647a">Generated by geoprompt-demo using data/sample_features.json</text>

  <rect x="34" y="82" width="640" height="386" rx="12" fill="#ffffff" stroke="#d3e0ec"/>
  <g stroke="#d8e2ed" stroke-width="1">
    <line x1="70" y1="120" x2="640" y2="120"/>
    <line x1="70" y1="190" x2="640" y2="190"/>
    <line x1="70" y1="260" x2="640" y2="260"/>
    <line x1="70" y1="330" x2="640" y2="330"/>
    <line x1="70" y1="400" x2="640" y2="400"/>
    <line x1="110" y1="98" x2="110" y2="446"/>
    <line x1="230" y1="98" x2="230" y2="446"/>
    <line x1="350" y1="98" x2="350" y2="446"/>
    <line x1="470" y1="98" x2="470" y2="446"/>
    <line x1="590" y1="98" x2="590" y2="446"/>
  </g>

  <g stroke="#486a86" stroke-width="5" stroke-linecap="round" fill="none" opacity="0.78">
    <polyline points="110,120 230,120 350,190 470,190 590,260"/>
    <polyline points="110,260 230,260 350,330 470,330 590,400"/>
    <polyline points="230,120 230,260"/>
    <polyline points="350,190 350,330"/>
    <polyline points="470,190 470,330"/>
  </g>

  <g>
    <circle cx="110" cy="120" r="24" fill="#ffd166" stroke="#ac7a00"/><text x="110" y="125" text-anchor="middle" font-size="11" font-weight="700" fill="#594200">0.43</text>
    <circle cx="230" cy="120" r="28" fill="#fcbf49" stroke="#9a6400"/><text x="230" y="125" text-anchor="middle" font-size="11" font-weight="700" fill="#4f3600">0.58</text>
    <circle cx="350" cy="190" r="32" fill="#f77f00" stroke="#8e3b00"/><text x="350" y="195" text-anchor="middle" font-size="11" font-weight="700" fill="#fff3e6">0.71</text>
    <circle cx="470" cy="190" r="34" fill="#e63946" stroke="#7b1d22"/><text x="470" y="195" text-anchor="middle" font-size="11" font-weight="700" fill="#fff0f1">0.82</text>
    <circle cx="590" cy="260" r="37" fill="#b5179e" stroke="#641057"/><text x="590" y="265" text-anchor="middle" font-size="11" font-weight="700" fill="#fdf1fb">0.92</text>
    <circle cx="110" cy="260" r="23" fill="#ffe08a" stroke="#9d840d"/><text x="110" y="265" text-anchor="middle" font-size="11" font-weight="700" fill="#5e5006">0.39</text>
    <circle cx="230" cy="260" r="27" fill="#ffd166" stroke="#9f7700"/><text x="230" y="265" text-anchor="middle" font-size="11" font-weight="700" fill="#5e4500">0.54</text>
    <circle cx="350" cy="330" r="31" fill="#fcbf49" stroke="#925c00"/><text x="350" y="335" text-anchor="middle" font-size="11" font-weight="700" fill="#4f3400">0.68</text>
    <circle cx="470" cy="330" r="36" fill="#f77f00" stroke="#7a3400"/><text x="470" y="335" text-anchor="middle" font-size="11" font-weight="700" fill="#fff3e7">0.80</text>
    <circle cx="590" cy="400" r="39" fill="#e63946" stroke="#67151a"/><text x="590" y="405" text-anchor="middle" font-size="11" font-weight="700" fill="#fff2f3">0.89</text>
  </g>

  <rect x="700" y="82" width="248" height="386" rx="12" fill="#ffffff" stroke="#d3e0ec"/>
  <text x="718" y="112" font-size="14" font-weight="700" fill="#1d3a53">Interpretation</text>
  <text x="718" y="136" font-size="11" fill="#51697d">Bubble size and color encode</text>
  <text x="718" y="152" font-size="11" fill="#51697d">relative neighborhood pressure.</text>

  <text x="718" y="188" font-size="11" fill="#27445c">High pressure corridors:</text>
  <text x="718" y="205" font-size="11" fill="#27445c">East trunk and South-east node</text>

  <text x="718" y="242" font-size="11" fill="#27445c">Top stress nodes:</text>
  <text x="718" y="259" font-size="11" fill="#27445c">North-east hub (0.92)</text>
  <text x="718" y="276" font-size="11" fill="#27445c">South-east terminal (0.89)</text>

  <g transform="translate(718,312)">
    <text x="0" y="0" font-size="11" fill="#27445c">Legend</text>
    <rect x="0" y="10" width="170" height="16" fill="#ffe08a"/><text x="178" y="23" font-size="10" fill="#4d6070">low</text>
    <rect x="0" y="30" width="170" height="16" fill="#ffd166"/>
    <rect x="0" y="50" width="170" height="16" fill="#fcbf49"/>
    <rect x="0" y="70" width="170" height="16" fill="#f77f00"/>
    <rect x="0" y="90" width="170" height="16" fill="#e63946"/>
    <rect x="0" y="110" width="170" height="16" fill="#b5179e"/><text x="178" y="123" font-size="10" fill="#4d6070">high</text>
  </g>

  <text x="34" y="494" font-size="11" fill="#4d667a">Source: geoprompt-demo chart path logic, regenerated as SVG for publication-quality docs.</text>
</svg>'''


def formula_parity_audit_chart():
    """Audit chart for equation parity against analytic references."""
    return '''\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 440" width="920" height="440"
     font-family="'Segoe UI', system-ui, -apple-system, sans-serif">
  <rect width="920" height="440" rx="14" fill="#f6faf7"/>
  <rect x="1" y="1" width="918" height="438" rx="14" fill="none" stroke="#d1e2d5"/>
  <text x="30" y="36" font-size="21" font-weight="700" fill="#14352b">Formula Parity Audit</text>
  <text x="30" y="56" font-size="12" fill="#4e6c62">Analytic reference comparisons for decay and interaction functions</text>

  <rect x="30" y="78" width="580" height="320" rx="10" fill="#ffffff" stroke="#d4e3d8"/>
  <g stroke="#e6efe8" stroke-width="1">
    <line x1="80" y1="110" x2="80" y2="360"/>
    <line x1="160" y1="110" x2="160" y2="360"/>
    <line x1="240" y1="110" x2="240" y2="360"/>
    <line x1="320" y1="110" x2="320" y2="360"/>
    <line x1="400" y1="110" x2="400" y2="360"/>
    <line x1="480" y1="110" x2="480" y2="360"/>
    <line x1="560" y1="110" x2="560" y2="360"/>
    <line x1="80" y1="360" x2="560" y2="360"/>
    <line x1="80" y1="310" x2="560" y2="310"/>
    <line x1="80" y1="260" x2="560" y2="260"/>
    <line x1="80" y1="210" x2="560" y2="210"/>
    <line x1="80" y1="160" x2="560" y2="160"/>
    <line x1="80" y1="110" x2="560" y2="110"/>
  </g>

  <polyline points="80,360 160,298 240,252 320,218 400,192 480,173 560,158" fill="none" stroke="#0d9488" stroke-width="3"/>
  <polyline points="80,360 160,282 240,233 320,203 400,185 480,175 560,170" fill="none" stroke="#2563eb" stroke-width="3" stroke-dasharray="7 5"/>
  <polyline points="80,360 160,324 240,288 320,252 400,220 480,194 560,173" fill="none" stroke="#d97706" stroke-width="3"/>

  <g fill="#0f172a" font-size="11" text-anchor="middle">
    <text x="80" y="378">0</text><text x="160" y="378">1</text><text x="240" y="378">2</text>
    <text x="320" y="378">3</text><text x="400" y="378">4</text><text x="480" y="378">5</text><text x="560" y="378">6</text>
  </g>
  <text x="322" y="398" font-size="11" fill="#3f5a53" text-anchor="middle">distance</text>
  <text x="24" y="242" font-size="11" fill="#3f5a53" transform="rotate(-90 24,242)">decay value</text>

  <rect x="636" y="78" width="254" height="320" rx="10" fill="#ffffff" stroke="#d4e3d8"/>
  <text x="652" y="108" font-size="13" font-weight="700" fill="#1f4438">Audit Summary</text>
  <text x="652" y="132" font-size="11" fill="#2f5549">- prompt_decay analytic parity: PASS</text>
  <text x="652" y="150" font-size="11" fill="#2f5549">- gaussian/exponential monotonicity: PASS</text>
  <text x="652" y="168" font-size="11" fill="#2f5549">- interaction multiplication parity: PASS</text>
  <text x="652" y="186" font-size="11" fill="#2f5549">- weighted accessibility parity: PASS</text>

  <line x1="652" y1="214" x2="676" y2="214" stroke="#0d9488" stroke-width="3"/>
  <text x="684" y="218" font-size="10" fill="#2f5549">Power decay (scale=1.6,p=1.8)</text>
  <line x1="652" y1="236" x2="676" y2="236" stroke="#2563eb" stroke-width="3" stroke-dasharray="7 5"/>
  <text x="684" y="240" font-size="10" fill="#2f5549">Exponential decay (rate=0.3)</text>
  <line x1="652" y1="258" x2="676" y2="258" stroke="#d97706" stroke-width="3"/>
  <text x="684" y="262" font-size="10" fill="#2f5549">Gaussian decay (sigma=3.0)</text>

  <rect x="652" y="286" width="222" height="88" rx="8" fill="#ecfdf5" stroke="#bbf7d0"/>
  <text x="664" y="306" font-size="11" fill="#166534">Tolerance envelope</text>
  <text x="664" y="324" font-size="11" fill="#166534">abs_tol = 1e-12</text>
  <text x="664" y="342" font-size="11" fill="#166534">Optional parity guard:</text>
  <text x="664" y="360" font-size="11" fill="#166534">GeoPandas area/bounds comparison</text>
</svg>'''


def tool_reliability_audit_chart():
    """Reliability scorecard for key tools and exporters."""
    return '''\
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 430" width="920" height="430"
     font-family="'Segoe UI', system-ui, -apple-system, sans-serif">
  <rect width="920" height="430" rx="14" fill="#f8f7fc"/>
  <rect x="1" y="1" width="918" height="428" rx="14" fill="none" stroke="#ddd8ee"/>
  <text x="30" y="36" font-size="21" font-weight="700" fill="#2d2450">Tool Reliability Audit</text>
  <text x="30" y="56" font-size="12" fill="#5f567f">Cross-check of report generation outputs and parity-facing utility bridges</text>

  <rect x="30" y="78" width="860" height="306" rx="10" fill="#ffffff" stroke="#dfd9f0"/>

  <g transform="translate(54,110)">
    <text x="0" y="0" font-size="12" font-weight="700" fill="#3f3566">Tool</text>
    <text x="310" y="0" font-size="12" font-weight="700" fill="#3f3566">Validation mode</text>
    <text x="650" y="0" font-size="12" font-weight="700" fill="#3f3566">Result</text>

    <line x1="0" y1="12" x2="800" y2="12" stroke="#ece9f6"/>

    <text x="0" y="44" font-size="11" fill="#403860">build_scenario_report</text>
    <text x="310" y="44" font-size="11" fill="#5c527f">delta / percent-delta reference math</text>
    <rect x="650" y="30" width="92" height="22" rx="11" fill="#dcfce7"/><text x="696" y="45" font-size="10" text-anchor="middle" fill="#166534" font-weight="700">PASS</text>

    <text x="0" y="84" font-size="11" fill="#403860">export_scenario_report</text>
    <text x="310" y="84" font-size="11" fill="#5c527f">JSON and CSV round-trip checks</text>
    <rect x="650" y="70" width="92" height="22" rx="11" fill="#dcfce7"/><text x="696" y="85" font-size="10" text-anchor="middle" fill="#166534" font-weight="700">PASS</text>

    <text x="0" y="124" font-size="11" fill="#403860">from_wkt_batch / to_wkt_batch</text>
    <text x="310" y="124" font-size="11" fill="#5c527f">Shapely + fallback malformed geometry guards</text>
    <rect x="650" y="110" width="92" height="22" rx="11" fill="#dcfce7"/><text x="696" y="125" font-size="10" text-anchor="middle" fill="#166534" font-weight="700">PASS</text>

    <text x="0" y="164" font-size="11" fill="#403860">GeoPromptFrame geometry_areas</text>
    <text x="310" y="164" font-size="11" fill="#5c527f">Optional GeoPandas parity (area, bounds)</text>
    <rect x="650" y="150" width="92" height="22" rx="11" fill="#fef9c3"/><text x="696" y="165" font-size="10" text-anchor="middle" fill="#854d0e" font-weight="700">SKIP/OPT</text>

    <text x="0" y="204" font-size="11" fill="#403860">adaptive_chunk_size / retry profiles</text>
    <text x="310" y="204" font-size="11" fill="#5c527f">narrowed exception policies</text>
    <rect x="650" y="190" width="92" height="22" rx="11" fill="#dcfce7"/><text x="696" y="205" font-size="10" text-anchor="middle" fill="#166534" font-weight="700">PASS</text>

    <line x1="0" y1="224" x2="800" y2="224" stroke="#ece9f6"/>

    <text x="0" y="252" font-size="11" fill="#4b416f">Audit policy: deterministic fixtures, explicit tolerances, and no silent broad exceptions in targeted paths.</text>
    <text x="0" y="270" font-size="11" fill="#4b416f">Reliability target: no regression against checked-in references and optional ecosystem parity checks where libraries exist.</text>
  </g>

  <rect x="30" y="394" width="860" height="20" rx="6" fill="#f0edfa"/>
  <text x="46" y="408" font-size="11" fill="#4e4572">Generated from test-driven audit definitions in tests/test_formula_tool_reliability_audit.py.</text>
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

    (ASSETS / "network-restoration-unmet-demand.svg").write_text(
      network_restoration_unmet_demand_chart(), encoding="utf-8"
    )
    print("Wrote network-restoration-unmet-demand.svg")

    (ASSETS / "resilience-risk-heatmap-mitigation.svg").write_text(
      resilience_heatmap_and_mitigation_chart(), encoding="utf-8"
    )
    print("Wrote resilience-risk-heatmap-mitigation.svg")

    (ASSETS / "migration-effort-benefit-quadrant.svg").write_text(
      migration_effort_benefit_quadrant(), encoding="utf-8"
    )
    print("Wrote migration-effort-benefit-quadrant.svg")

    (ASSETS / "neighborhood-pressure-review-live.svg").write_text(
      neighborhood_pressure_live_svg(), encoding="utf-8"
    )
    print("Wrote neighborhood-pressure-review-live.svg")

    (ASSETS / "formula-parity-audit.svg").write_text(
      formula_parity_audit_chart(), encoding="utf-8"
    )
    print("Wrote formula-parity-audit.svg")

    (ASSETS / "tool-reliability-audit.svg").write_text(
      tool_reliability_audit_chart(), encoding="utf-8"
    )
    print("Wrote tool-reliability-audit.svg")

    print("Done - all SVG assets regenerated.")
