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
LOCKED = Path("data/predictions/locked_predictions.csv")
OUT = Path("dashboard/predictions.html")


def _actual_results() -> dict:
    """match_id -> (home_goals, away_goals, result) for played 2026 games."""
    try:
        from src.data.results_loader import load_processed_results
        res = load_processed_results()
    except Exception:
        return {}
    played = res[(res["tournament"] == "FIFA World Cup")
                 & (res["date"].dt.year == 2026) & res["home_score"].notna()]
    out = {}
    for _, r in played.iterrows():
        hg, ag = int(r["home_score"]), int(r["away_score"])
        out[r["match_id"]] = (hg, ag, "H" if hg > ag else "D" if hg == ag else "A")
    return out


def _records() -> list[dict]:
    """Per-match cards. Upcoming games show the LIVE (daily-updating)
    prediction; played games show the actual result plus our frozen,
    pre-match LOCKED call and whether the favourite was right."""
    df = pd.read_csv(CSV)
    actual = _actual_results()
    locked = (pd.read_csv(LOCKED).set_index("match_id").to_dict("index")
              if LOCKED.exists() else {})
    recs = []
    for _, r in df.iterrows():
        home, away = str(r["match"]).split(" v ")
        mid = f"{r['date']}_{home}_{away}"
        rec = {
            "date": r["date"], "group": r["group"], "home": home, "away": away,
            "city": r["city"], "pHome": float(r["p_home"]), "pDraw": float(r["p_draw"]),
            "pAway": float(r["p_away"]), "xg": r["xg"], "likely": r["likely"],
            "alt": float(r["alt_adj"]), "conf": float(r["conf"]),
            "status": "upcoming",
        }
        if mid in actual:
            hg, ag, res = actual[mid]
            rec["status"] = "played"
            rec["score"] = f"{hg}-{ag}"
            rec["result"] = res
            lk = locked.get(mid)
            if lk:
                lp = {"H": lk["champion_plus_H"], "D": lk["champion_plus_D"],
                      "A": lk["champion_plus_A"]}
                fav = max(lp, key=lp.get)
                rec["lockHome"], rec["lockDraw"], rec["lockAway"] = lp["H"], lp["D"], lp["A"]
                rec["lockProb"] = lp[res]          # prob we gave the actual outcome
                rec["correct"] = (fav == res)      # did our favourite win?
        recs.append(rec)
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


ADVANCEMENT = Path("data/predictions/advancement.csv")
_ROUNDS = [("Round of 32", "p_r32"), ("Round of 16", "p_r16"),
           ("Quarter-final", "p_qf"), ("Semi-final", "p_sf"),
           ("Final", "p_final"), ("Champion", "p_champion")]


GOALS_TABLE = Path("data/predictions/goals_table.csv")


def _goals() -> list[dict]:
    """Per-team xG table: group totals (+ actual once played) + knockout rate."""
    if not GOALS_TABLE.exists():
        return []
    df = pd.read_csv(GOALS_TABLE)
    return [{"team": r["team"], "group": r["group"],
             "gxf": float(r["group_xgf"]), "gxa": float(r["group_xga"]),
             "gxd": float(r["group_xgd"]), "af": int(r["group_actual_f"]),
             "aa": int(r["group_actual_a"]), "played": int(r["group_played"]),
             "koF": float(r["ko_xgf"]), "koA": float(r["ko_xga"])}
            for _, r in df.iterrows()]


R32_JSON = Path("data/predictions/bracket.json")


def _r32() -> list[dict]:
    if R32_JSON.exists():
        return json.loads(R32_JSON.read_text())
    return []


def _bracket() -> list[dict]:
    """Per-round 'funnel': top teams most likely to reach each round.

    Honest by construction — these are probabilities of REACHING a round,
    not invented matchups (we don't know the bracket pairings yet)."""
    src = ADVANCEMENT if ADVANCEMENT.exists() else CHAMP_CSV
    if not src.exists():
        return []
    df = pd.read_csv(src)
    cols = {c for c in df.columns}
    out = []
    for label, col in _ROUNDS:
        if col not in cols:
            continue
        top = df.sort_values(col, ascending=False).head(8)
        out.append({"label": label,
                    "teams": [{"team": r["team"], "p": float(r[col])}
                              for _, r in top.iterrows()]})
    return out


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
  .teams .ftscore{font-size:18px;font-weight:800;color:#fff;background:#1f6feb;padding:1px 12px;border-radius:6px}
  .ok{color:#3fb950;font-weight:600} .miss{color:#f0883e;font-weight:600}
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
  .sec.toggle{cursor:pointer;user-select:none;background:var(--card);border:1px solid #30363d;border-radius:10px;padding:12px 14px}
  .sec.toggle:hover{border-color:#1f6feb}
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
  .funnel{display:flex;gap:10px;overflow-x:auto;padding-bottom:8px}
  .col{flex:0 0 140px;background:var(--card);border:1px solid #30363d;border-radius:10px;padding:10px}
  .col h4{margin:0 0 8px;font-size:12px;color:#c9d1d9;text-align:center;text-transform:uppercase;letter-spacing:.3px}
  .col.champ h4{color:#f0b429}
  .brow{display:flex;justify-content:space-between;align-items:center;font-size:13px;padding:4px 2px;border-bottom:1px solid #21262d}
  .brow:last-child{border-bottom:none}
  .brow .bt{font-weight:700} .brow .bp{color:var(--mut);font-variant-numeric:tabular-nums}
  .brow .bp.lead{color:#f0b429}
  table.gt{width:100%;border-collapse:collapse;font-size:13px;margin-bottom:6px}
  table.gt th{text-align:right;color:var(--mut);font-weight:600;padding:6px 8px;border-bottom:1px solid #30363d;font-size:11px;text-transform:uppercase}
  table.gt th:first-child,table.gt td:first-child{text-align:left}
  table.gt td{padding:6px 8px;border-bottom:1px solid #21262d;text-align:right;font-variant-numeric:tabular-nums}
  table.gt td.tm{font-weight:700} table.gt .pos{color:#3fb950} table.gt .neg{color:#f0883e}
  table.gt .act{color:var(--mut);font-size:12px}
  .gtwrap{overflow-x:auto}
  .r32{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
  @media(max-width:640px){.r32{grid-template-columns:1fr}}
  .tie{background:var(--card);border:1px solid #30363d;border-radius:10px;padding:10px 12px}
  .tie .mno{font-size:10px;color:var(--mut);text-transform:uppercase;letter-spacing:.4px;margin-bottom:6px}
  .slot{margin:4px 0}
  .slot .sl{font-size:11px;color:#8b949e;margin-bottom:2px}
  .cand{display:flex;justify-content:space-between;font-size:13px;padding:1px 0}
  .cand .ct{font-weight:600} .cand:first-of-type .ct{color:#fff}
  .cand .cp{color:var(--mut);font-variant-numeric:tabular-nums}
  .tievs{text-align:center;color:var(--mut);font-size:11px;margin:4px 0}
  .tie .locked .ct{color:#3fb950}
</style>
</head>
<body>
<div id="root"></div>
<script type="text/babel">
const DATA = __DATA__;
const CONTENDERS = __CONTENDERS__;
const SCORECARD = __SCORECARD__;
const BRACKET = __BRACKET__;
const GOALS = __GOALS__;
const R32 = __R32__;

function Slot({label, cands}){
  const locked = cands.length===1;
  return (
    <div className="slot">
      <div className="sl">{label}</div>
      {cands.length===0 ? <div className="cand"><span className="ct" style={{color:'var(--mut)'}}>—</span></div> :
        cands.map((c,i)=>(
          <div className={'cand'+(locked?' locked':'')} key={c.team}>
            <span className="ct">{c.team}</span>
            <span className="cp">{locked?'qualified':(c.p*100).toFixed(0)+'%'}</span>
          </div>
        ))}
    </div>
  );
}

function R32Bracket(){
  if(!R32.length) return null;
  return (
    <div>
      <div className="sec">🧩 Round of 32 — who's likely to play whom <span style={{color:'var(--mut)',fontWeight:400,fontSize:13}}>(top 3 per slot)</span></div>
      <div style={{color:'var(--mut)',fontSize:12,marginBottom:10}}>The 16 first-knockout matchups are fixed by group position; the teams aren't. Each slot shows its most likely occupants — this narrows to 2, then 1, as the groups finish. Third-place slots are approximate (a team can appear in more than one).</div>
      <div className="r32">
        {R32.map(t=>(
          <div className="tie" key={t.match}>
            <div className="mno">Match {t.match}</div>
            <Slot label={t.a_label} cands={t.a}/>
            <div className="tievs">vs</div>
            <Slot label={t.b_label} cands={t.b}/>
          </div>
        ))}
      </div>
    </div>
  );
}

function GoalsTable(){
  if(!GOALS.length) return null;
  const grp = [...GOALS].sort((a,b)=>b.gxd-a.gxd);
  const ko = [...GOALS].sort((a,b)=>(b.koF-b.koA)-(a.koF-a.koA)).slice(0,12);
  const anyPlayed = GOALS.some(g=>g.played>0);
  return (
    <div>
      <div className="sec">⚽ Goals outlook <span style={{color:'var(--mut)',fontWeight:400,fontSize:13}}>(expected goals — xG)</span></div>
      <div style={{color:'var(--mut)',fontSize:12,marginBottom:10}}>Group stage: total xG a team is expected to score and concede across its 3 games{anyPlayed?' (actual goals shown once played)':''}.</div>
      <div className="gtwrap">
        <table className="gt">
          <thead><tr><th>Team</th><th>Grp</th><th>xG for</th><th>xG against</th><th>xG diff</th>{anyPlayed&&<th>Actual</th>}</tr></thead>
          <tbody>
            {grp.map(g=>(
              <tr key={g.team}>
                <td className="tm">{g.team}</td><td className="act">{g.group}</td>
                <td>{g.gxf.toFixed(1)}</td><td>{g.gxa.toFixed(1)}</td>
                <td className={g.gxd>=0?'pos':'neg'}>{g.gxd>=0?'+':''}{g.gxd.toFixed(1)}</td>
                {anyPlayed&&<td className="act">{g.played?`${g.af}–${g.aa}`:'·'}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{color:'var(--mut)',fontSize:12,margin:'14px 0 8px'}}><b style={{color:'#c9d1d9'}}>Knockouts</b> — expected goals per match vs a typical qualifier (matchups not known yet; fills in with real games as the bracket forms). Top 12.</div>
      <div className="gtwrap">
        <table className="gt">
          <thead><tr><th>Team</th><th>xG for / match</th><th>xG against / match</th></tr></thead>
          <tbody>
            {ko.map(g=>(
              <tr key={g.team}><td className="tm">{g.team}</td><td>{g.koF.toFixed(2)}</td><td>{g.koA.toFixed(2)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Funnel(){
  if(!BRACKET.length) return null;
  return (
    <div>
      <div className="sec">🗺️ Road to the Final <span style={{color:'var(--mut)',fontWeight:400,fontSize:13}}>(chance of reaching each round)</span></div>
      <div style={{color:'var(--mut)',fontSize:12,marginBottom:10}}>Not a fixed bracket — the matchups aren't decided until the groups finish. Each column shows the teams most likely to REACH that round. Scroll sideways on mobile →</div>
      <div className="funnel">
        {BRACKET.map((col,ci)=>(
          <div className={'col'+(ci===BRACKET.length-1?' champ':'')} key={col.label}>
            <h4>{col.label}</h4>
            {col.teams.map((t,i)=>(
              <div className="brow" key={t.team}>
                <span className="bt">{t.team}</span>
                <span className={'bp'+(i===0?' lead':'')}>{(t.p*100).toFixed(t.p<0.1?1:0)}%</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
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
  const xgRounded = m.xg.split('-').map(s=>Math.round(parseFloat(s))).join('-');
  const played = m.status === 'played';
  return (
    <div className="match">
      <div className="top">
        <span>{m.date} · Group {m.group}</span>
        <span>{played ? 'FULL-TIME' : m.city}</span>
      </div>
      <div className="teams">
        <span>{m.home}</span>
        {played ? <span className="ftscore">{m.score}</span> : <span className="vs">vs</span>}
        <span>{m.away}</span>
      </div>
      {played ? (
        <div className="meta">
          {m.lockProb!==undefined && <span>Our pre-match call gave this result <b>{pct(m.lockProb)}%</b></span>}
          {m.correct!==undefined && (m.correct
            ? <span className="ok">✓ favourite won</span>
            : <span className="miss">✗ upset</span>)}
        </div>
      ) : (
        <>
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
        </>
      )}
    </div>
  );
}

function App(){
  const [g,setG] = React.useState('All');
  const [openMatches,setOpenMatches] = React.useState(false);
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
      <Funnel/>
      <R32Bracket/>
      <GoalsTable/>
      <div className="sec toggle" onClick={()=>setOpenMatches(o=>!o)}>
        {openMatches?'▾':'▸'} All {DATA.length} match predictions
        <span style={{color:'var(--mut)',fontWeight:400,fontSize:13}}> (tap to {openMatches?'hide':'show'})</span>
      </div>
      {openMatches && <>
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
      </>}
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
            .replace("__SCORECARD__", json.dumps(_scorecard()))
            .replace("__BRACKET__", json.dumps(_bracket()))
            .replace("__GOALS__", json.dumps(_goals()))
            .replace("__R32__", json.dumps(_r32())))
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
