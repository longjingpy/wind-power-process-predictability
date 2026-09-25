"""Build an editable SVG graphical abstract from the verified process story."""
from pathlib import Path

ROOT = Path("/mnt/d/projects/WindPowerForcast")
OUT = ROOT / "outputs/next_paper/manuscript_figures"


def svg() -> str:
    return r'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">
<defs>
  <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="#263238"/></marker>
  <filter id="shadow" x="-10%" y="-10%" width="120%" height="120%"><feDropShadow dx="0" dy="4" stdDeviation="5" flood-color="#263238" flood-opacity="0.14"/></filter>
</defs>
<rect width="1600" height="900" fill="#ffffff"/>
<text x="800" y="68" text-anchor="middle" font-family="Arial, sans-serif" font-weight="700" font-size="34" fill="#17202a">Conditional predictability of wind-power event structures</text>
<text x="800" y="105" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" fill="#52606d">Lead time × information role × process target</text>

<!-- panel backgrounds -->
<rect x="70" y="150" width="430" height="570" rx="18" fill="#f4f8fb" stroke="#90a4ae" stroke-width="2" filter="url(#shadow)"/>
<rect x="585" y="150" width="430" height="570" rx="18" fill="#f8fafb" stroke="#90a4ae" stroke-width="2" filter="url(#shadow)"/>
<rect x="1100" y="150" width="430" height="570" rx="18" fill="#fbf8fc" stroke="#90a4ae" stroke-width="2" filter="url(#shadow)"/>

<!-- panel 1 -->
<text x="285" y="195" text-anchor="middle" font-family="Arial" font-weight="700" font-size="24" fill="#17202a">1  Shared process target</text>
<text x="285" y="228" text-anchor="middle" font-family="Arial" font-size="17" fill="#52606d">17 native 15-min nodes over four hours</text>
<rect x="105" y="270" width="360" height="210" rx="14" fill="#ffffff" stroke="#607d8b" stroke-width="2"/>
<line x1="130" y1="445" x2="440" y2="445" stroke="#b0bec5" stroke-width="2"/>
<line x1="130" y1="300" x2="130" y2="445" stroke="#b0bec5" stroke-width="2"/>
<path d="M132 390 C165 372, 182 380, 205 333 S252 315, 274 352 S302 428, 329 404 S364 351, 390 370 S416 397, 438 388" fill="none" stroke="#1565c0" stroke-width="5"/>
<circle cx="205" cy="333" r="6" fill="#1565c0"/><circle cx="329" cy="404" r="6" fill="#1565c0"/>
<text x="146" y="292" font-family="Arial" font-size="15" fill="#52606d">power path</text>
<text x="135" y="475" font-family="Arial" font-size="15" fill="#52606d">T</text><text x="421" y="475" font-family="Arial" font-size="15" fill="#52606d">T+4 h</text>
<rect x="115" y="515" width="150" height="55" rx="12" fill="#e3f2fd" stroke="#1565c0" stroke-width="2"/><text x="190" y="548" text-anchor="middle" font-family="Arial" font-size="18" fill="#0d47a1">state risk</text>
<rect x="300" y="515" width="150" height="55" rx="12" fill="#fff3e0" stroke="#ef6c00" stroke-width="2"/><text x="375" y="548" text-anchor="middle" font-family="Arial" font-size="18" fill="#e65100">first passage</text>
<rect x="115" y="590" width="150" height="55" rx="12" fill="#eceff1" stroke="#607d8b" stroke-width="2"/><text x="190" y="623" text-anchor="middle" font-family="Arial" font-size="18" fill="#37474f">range</text>
<rect x="300" y="590" width="150" height="55" rx="12" fill="#eceff1" stroke="#607d8b" stroke-width="2"/><text x="375" y="623" text-anchor="middle" font-family="Arial" font-size="18" fill="#37474f">path order</text>

<!-- connectors -->
<path d="M500 435 H575" fill="none" stroke="#263238" stroke-width="4" marker-end="url(#arrow)"/>
<path d="M1015 435 H1090" fill="none" stroke="#263238" stroke-width="4" marker-end="url(#arrow)"/>

<!-- panel 2 -->
<text x="800" y="195" text-anchor="middle" font-family="Arial" font-weight="700" font-size="24" fill="#17202a">2  Complementary information</text>
<text x="800" y="228" text-anchor="middle" font-family="Arial" font-size="17" fill="#52606d">Matched external three-arm validation</text>
<rect x="620" y="275" width="360" height="125" rx="15" fill="#e3f2fd" stroke="#1565c0" stroke-width="3"/>
<circle cx="662" cy="338" r="25" fill="#1565c0"/><path d="M650 338 q12 -22 24 0 q-12 22 -24 0" fill="none" stroke="#fff" stroke-width="3"/>
<text x="705" y="325" font-family="Arial" font-weight="700" font-size="21" fill="#0d47a1">External weather</text>
<text x="705" y="355" font-family="Arial" font-size="17" fill="#263238">broad process-state information</text>
<rect x="620" y="445" width="360" height="125" rx="15" fill="#fff3e0" stroke="#ef6c00" stroke-width="3"/>
<circle cx="662" cy="508" r="25" fill="#ef6c00"/><path d="M650 508 h24 M662 496 v24" stroke="#fff" stroke-width="3"/>
<text x="705" y="495" font-family="Arial" font-weight="700" font-size="21" fill="#e65100">Power history</text>
<text x="705" y="525" font-family="Arial" font-size="17" fill="#263238">near-term timing information</text>
<text x="800" y="625" text-anchor="middle" font-family="Arial" font-size="17" fill="#52606d">Weather and history are assigned by target.</text>

<!-- panel 3 -->
<text x="1315" y="195" text-anchor="middle" font-family="Arial" font-weight="700" font-size="24" fill="#17202a">3  Target-aligned forecast</text>
<text x="1315" y="228" text-anchor="middle" font-family="Arial" font-size="17" fill="#52606d">State head + timing head</text>
<rect x="1140" y="270" width="350" height="110" rx="15" fill="#e3f2fd" stroke="#1565c0" stroke-width="3"/>
<text x="1315" y="315" text-anchor="middle" font-family="Arial" font-weight="700" font-size="21" fill="#0d47a1">Joint state head</text>
<text x="1315" y="347" text-anchor="middle" font-family="Arial" font-size="17" fill="#263238">weather + power history</text>
<rect x="1140" y="430" width="350" height="110" rx="15" fill="#fff3e0" stroke="#ef6c00" stroke-width="3"/>
<text x="1315" y="475" text-anchor="middle" font-family="Arial" font-weight="700" font-size="21" fill="#e65100">Power timing head</text>
<text x="1315" y="507" text-anchor="middle" font-family="Arial" font-size="17" fill="#263238">power history</text>
<path d="M1085 338 H1128" fill="none" stroke="#1565c0" stroke-width="3" marker-end="url(#arrow)"/>
<path d="M1085 508 H1128" fill="none" stroke="#ef6c00" stroke-width="3" marker-end="url(#arrow)"/>
<rect x="1140" y="590" width="350" height="75" rx="14" fill="#f3e5f5" stroke="#7b1fa2" stroke-width="3"/>
<text x="1315" y="618" text-anchor="middle" font-family="Arial" font-weight="700" font-size="18" fill="#4a148c">Verified gains</text>
<text x="1315" y="646" text-anchor="middle" font-family="Arial" font-size="16" fill="#263238">+9.78% state at 12 h · +8.70/+8.65% timing RPS</text>

<!-- bottom message -->
<rect x="70" y="770" width="1460" height="75" rx="18" fill="#263238"/>
<text x="800" y="816" text-anchor="middle" font-family="Arial" font-weight="700" font-size="22" fill="#ffffff">Forecast information adds value when it is matched to the process target it serves.</text>
</svg>'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "graphical_abstract.svg"
    path.write_text(svg(), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
