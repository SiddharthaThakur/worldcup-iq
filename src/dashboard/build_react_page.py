"""
Build a self-contained React page to view the World Cup predictions.

What this does in simple English:
    Reads the champion+ group-stage predictions and writes a single HTML
    file with the data baked in and React loaded from a CDN. No server,
    no build step — just open dashboard/predictions.html in a browser.

Run: python -m src.dashboard.build_react_page
"""

import json
from pathlib import Path

import pandas as pd

CSV = Path("data/predictions/group_stage_champion_plus.csv")
OUT = Path("dashboard/predictions.html")


def _records() -> list[dict]:
    df = pd.read_csv(CSV)
    recs = []
    for _, r in df.iterrows():
        home, away = str(r["match"]).split(" v ")
        recs.append({
            "date": r["date"], "group": r["group"], "home": home, "away": away,
            "city": r["city"], "pHome": float(r["p_home"]), "pDraw": float(r["p_draw"]),
            "pAway": float(r["p_away"]), "xg": r["xg"], "likely": r["likely"],
            "alt": float(r["alt_adj"]), "conf": float(r["conf"]),
        })
    return recs


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>WorldCupIQ — 2026 Group Stage Predictions</title>
<script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<style>
  :root{--home:#2f9e44;--draw:#868e96;--away:#e8590c;--bg:#0d1117;--card:#161b22;--ink:#e6edf3;--mut:#8b949e}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  .wrap{max-width:900px;margin:0 auto;padding:24px}
  h1{font-size:22px;margin:0 0 4px} .sub{color:var(--mut);font-size:13px;margin-bottom:18px}
  .tabs{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:18px}
  .tab{padding:6px 12px;border-radius:8px;background:var(--card);color:var(--mut);
    cursor:pointer;border:1px solid #30363d;font-size:13px;font-weight:600}
  .tab.on{background:#1f6feb;color:#fff;border-color:#1f6feb}
  .match{background:var(--card);border:1px solid #30363d;border-radius:10px;padding:12px 14px;margin-bottom:10px}
  .top{display:flex;justify-content:space-between;align-items:center;font-size:13px;color:var(--mut);margin-bottom:8px}
  .teams{display:flex;justify-content:space-between;align-items:center;font-weight:700;font-size:16px;margin-bottom:8px}
  .teams .vs{color:var(--mut);font-weight:500;font-size:12px}
  .bar{display:flex;height:26px;border-radius:6px;overflow:hidden;font-size:11px;font-weight:700;color:#fff}
  .bar>div{display:flex;align-items:center;justify-content:center;min-width:24px}
  .meta{display:flex;gap:14px;margin-top:8px;font-size:12px;color:var(--mut);flex-wrap:wrap}
  .flag{background:#3d2c0a;color:#f0b429;padding:1px 7px;border-radius:5px;font-size:11px;font-weight:600}
  .lowconf{background:#3a1d1d;color:#ff8787;padding:1px 7px;border-radius:5px;font-size:11px;font-weight:600}
  .legend{display:flex;gap:16px;font-size:12px;color:var(--mut);margin-bottom:14px}
  .dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:middle}
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">
const DATA = __DATA__;
const GROUPS = [...new Set(DATA.map(d=>d.group))].sort();
const pct = x => Math.round(x*100);

function Match({m}){
  return (
    <div className="match">
      <div className="top"><span>{m.date} · Group {m.group}</span><span>{m.city}</span></div>
      <div className="teams"><span>{m.home}</span><span className="vs">vs</span><span>{m.away}</span></div>
      <div className="bar">
        <div style={{width:pct(m.pHome)+'%',background:'var(--home)'}}>{pct(m.pHome)}%</div>
        <div style={{width:pct(m.pDraw)+'%',background:'var(--draw)'}}>{pct(m.pDraw)}%</div>
        <div style={{width:pct(m.pAway)+'%',background:'var(--away)'}}>{pct(m.pAway)}%</div>
      </div>
      <div className="meta">
        <span>Most likely: <b>{m.likely}</b></span>
        <span>xG: {m.xg}</span>
        {m.alt!==0 && <span className="flag">altitude {m.alt} Elo</span>}
        {m.conf<0.5 && <span className="lowconf">low player-data · leans Elo</span>}
      </div>
    </div>
  );
}

function App(){
  const [g,setG] = React.useState('All');
  const shown = g==='All' ? DATA : DATA.filter(d=>d.group===g);
  return (
    <div className="wrap">
      <h1>WorldCupIQ — 2026 Group Stage</h1>
      <div className="sub">Champion+ model: Elo ensemble + player composition + altitude & rest/travel. {DATA.length} matches.</div>
      <div className="legend">
        <span><span className="dot" style={{background:'var(--home)'}}></span>Home win</span>
        <span><span className="dot" style={{background:'var(--draw)'}}></span>Draw</span>
        <span><span className="dot" style={{background:'var(--away)'}}></span>Away win</span>
      </div>
      <div className="tabs">
        {['All',...GROUPS].map(x=>(
          <div key={x} className={'tab'+(x===g?' on':'')} onClick={()=>setG(x)}>{x==='All'?'All':'Group '+x}</div>
        ))}
      </div>
      {shown.map((m,i)=><Match key={i} m={m}/>)}
    </div>
  );
}
ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
</script>
</body>
</html>
"""


def main() -> Path:
    html = HTML.replace("__DATA__", json.dumps(_records()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    return OUT


if __name__ == "__main__":
    p = main()
    print(f"Wrote {p}  ({p.stat().st_size // 1024} KB)")
    print(f"Open it: file://{p.resolve()}")
