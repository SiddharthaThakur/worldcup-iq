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
CHAMP_CSV = Path("data/predictions/champion_plus_probabilities.csv")
HISTORY = Path("data/predictions/champion_history.csv")
SCORECARD = Path("data/predictions/scorecard.json")
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


def _contenders() -> list[dict]:
    """Top-12 title chances. Uses the daily history (with day-over-day
    movement) if it exists, else the static pre-tournament file."""
    if HISTORY.exists():
        h = pd.read_csv(HISTORY)
        dates = sorted(h["date"].unique())
        today = h[h["date"] == dates[-1]].set_index("team")["p_champion"]
        prev = (h[h["date"] == dates[-2]].set_index("team")["p_champion"]
                if len(dates) >= 2 else None)
        top = today.sort_values(ascending=False).head(12)
        out = []
        for team, champ in top.items():
            mv = (float(champ - prev[team]) if prev is not None and team in prev
                  else None)
            out.append({"team": team, "champ": float(champ), "move": mv})
        return out
    if not CHAMP_CSV.exists():
        return []
    df = pd.read_csv(CHAMP_CSV).head(12)
    return [{"team": r["team"], "champ": float(r["p_champion"]), "move": None}
            for _, r in df.iterrows()]


def _scorecard() -> dict:
    if SCORECARD.exists():
        return json.loads(SCORECARD.read_text())
    return {}


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
  .intro{background:var(--card);border:1px solid #30363d;border-radius:10px;padding:14px 16px;margin-bottom:18px;font-size:13px;line-height:1.55;color:#c9d1d9}
  .intro b{color:var(--ink)}
  .sec{font-size:15px;font-weight:700;margin:24px 0 12px}
  .cont{display:flex;align-items:center;gap:10px;padding:7px 0;border-bottom:1px solid #21262d}
  .cont .nm{width:54px;font-weight:700}
  .cont .track{flex:1;background:#0b0f14;border:1px solid #30363d;border-radius:5px;height:20px;overflow:hidden}
  .cont .fill{display:block;height:100%;background:linear-gradient(90deg,#2f81f7,#56a3ff);border-radius:4px;min-width:3px}
  .cont .v{width:48px;text-align:right;font-variant-numeric:tabular-nums;font-weight:700}
  .cont .mv{width:54px;text-align:right;font-size:12px;font-variant-numeric:tabular-nums}
  .up{color:#3fb950} .down{color:#f85149} .flat{color:var(--mut)}
  .axis{display:flex;justify-content:space-between;font-size:10px;color:var(--mut);margin:2px 0 8px;padding:0 150px 0 64px}
  .score{display:flex;gap:14px;flex-wrap:wrap;margin:6px 0 4px}
  .scard{background:var(--card);border:1px solid #30363d;border-radius:10px;padding:12px 16px;min-width:150px}
  .scard .mn{font-size:12px;color:var(--mut)} .scard .bv{font-size:24px;font-weight:800;margin:2px 0}
  .scard .ng{font-size:11px;color:var(--mut)} .scard.best{border-color:#2f81f7}
  .foot{color:var(--mut);font-size:11px;margin-top:28px;line-height:1.6;border-top:1px solid #21262d;padding-top:14px}
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">
const DATA = __DATA__;
const CONTENDERS = __CONTENDERS__;
const SCORECARD = __SCORECARD__;
const GROUPS = [...new Set(DATA.map(d=>d.group))].sort();
const pct = x => Math.round(x*100);
const pct1 = x => (x*100).toFixed(1);

function Move({m}){
  if(m===null||m===undefined) return <span className="mv flat">·</span>;
  const p=(Math.abs(m)*100).toFixed(1);
  if(m>0.0005) return <span className="mv up">▲ {p}</span>;
  if(m<-0.0005) return <span className="mv down">▼ {p}</span>;
  return <span className="mv flat">–</span>;
}

function Scorecard(){
  const sc = SCORECARD;
  if(!sc.models || !sc.n_completed) return null;
  const names = {champion_plus:'Our model (champion+)', baseline_elo:'Simple Elo baseline', market:'Bookmakers'};
  const entries = Object.entries(sc.models).filter(([k,v])=>v.brier!=null);
  if(!entries.length) return null;
  const best = Math.min(...entries.map(([k,v])=>v.brier));
  return (
    <div>
      <div className="sec">📊 How accurate is it so far? <span style={{color:'var(--mut)',fontWeight:400,fontSize:13}}>({sc.n_completed} games played)</span></div>
      <div style={{color:'var(--mut)',fontSize:12,marginBottom:10}}>Brier score — average error of the probabilities. Lower is better (0 = perfect, 0.67 = a coin-flip). The honest test: does our model beat the simple baseline, and can either match the bookmakers?</div>
      <div className="score">
        {entries.map(([k,v])=>(
          <div className={'scard'+(v.brier===best?' best':'')} key={k}>
            <div className="mn">{names[k]||k}</div>
            <div className="bv">{v.brier.toFixed(3)}</div>
            <div className="ng">{v.n} games{v.brier===best?' · leading':''}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Contenders(){
  if(!CONTENDERS.length) return null;
  // Axis runs 0 -> SCALE so the leader's bar is substantial; bar LENGTH is
  // proportional to the real probability (honest), the % is the exact value.
  const SCALE = 0.15;
  return (
    <div>
      <div className="sec">🏆 Chance of winning the World Cup <span style={{color:'var(--mut)',fontWeight:400,fontSize:13}}>(from 20,000 simulated tournaments)</span></div>
      <div style={{color:'var(--mut)',fontSize:12,marginBottom:10}}>No team is a strong favourite — even the top side wins only about 1 in 8 times. The trophy is wide open. Longer bar = more likely.</div>
      <div className="axis"><span>0%</span><span>5%</span><span>10%</span><span>15%</span></div>
      {CONTENDERS.map(c=>(
        <div className="cont" key={c.team}>
          <span className="nm">{c.team}</span>
          <span className="track"><span className="fill" style={{width:Math.min(100,c.champ/SCALE*100)+'%'}}></span></span>
          <span className="v">{pct1(c.champ)}%</span>
          <Move m={c.move}/>
        </div>
      ))}
      {CONTENDERS.some(c=>c.move!=null) &&
        <div style={{color:'var(--mut)',fontSize:11,marginTop:6}}>▲▼ = change since yesterday's update.</div>}
    </div>
  );
}

function Match({m}){
  // Round each side's expected goals to a whole number (2.5 -> 3)
  const xgRounded = m.xg.split('-').map(s=>Math.round(parseFloat(s))).join('-');
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
        <span>Predicted score: <b>{xgRounded}</b></span>
        <span>Expected goals: <b>{m.xg}</b></span>
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
      <h1>WorldCupIQ — 2026 Predictions</h1>
      <div className="sub">An honest, player-level prediction model for the 2026 World Cup.</div>
      <div className="intro">
        Most World Cup models just use team win-rates. This one predicts from the <b>actual 26-man squads</b> —
        a team of stars in top leagues is rated above a team with great past results but a thin roster.
        It blends two independent strength ratings, adjusts for <b>altitude</b> (Mexico City sits at 2,240m) and
        <b> rest &amp; travel</b> across the continent, then simulates the whole tournament 20,000 times.
        Every prediction is locked in public before kickoff, and the model <b>scores itself</b> against the
        bookmakers afterwards — it doesn't pretend to beat the market, it measures honestly whether it does.
      </div>
      <Contenders/>
      <Scorecard/>
      <div className="sec">Group-stage match predictions</div>
      <div className="legend">
        <span><span className="dot" style={{background:'var(--home)'}}></span>Home / first team</span>
        <span><span className="dot" style={{background:'var(--draw)'}}></span>Draw</span>
        <span><span className="dot" style={{background:'var(--away)'}}></span>Away / second team</span>
      </div>
      <div className="tabs">
        {['All',...GROUPS].map(x=>(
          <div key={x} className={'tab'+(x===g?' on':'')} onClick={()=>setG(x)}>{x==='All'?'All':'Group '+x}</div>
        ))}
      </div>
      {shown.map((m,i)=><Match key={i} m={m}/>)}
      <div className="foot">
        Model: champion+ (ensemble Elo + player-composition blend + altitude &amp; rest/travel). Probabilities are
        90-minute outcomes. "altitude" flags games where thin air penalises non-adapted teams; "leans Elo"
        flags teams whose players we have little data on, where the model falls back to results-based ratings.
        Built with free public data. Not betting advice.
      </div>
    </div>
  );
}
ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
</script>
</body>
</html>
"""


def main() -> Path:
    html = (HTML
            .replace("__DATA__", json.dumps(_records()))
            .replace("__CONTENDERS__", json.dumps(_contenders()))
            .replace("__SCORECARD__", json.dumps(_scorecard())))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    # Also write index.html at the repo root so GitHub Pages serves it at
    # the site root (clean shareable URL).
    Path("index.html").write_text(html)
    return OUT


if __name__ == "__main__":
    p = main()
    print(f"Wrote {p}  ({p.stat().st_size // 1024} KB)")
    print(f"Open it: file://{p.resolve()}")
