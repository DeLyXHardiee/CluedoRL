"""Replay viewer: turn a Game log into one self-contained HTML page.

    python -m cluedo.replay game_log.json replay.html

Per-Seat views are rebuilt by replaying the log through the core `Game` and one
`Deduction` engine per Seat (no rules are re-implemented here).

Each step of `build_replay(log)["steps"]` has an `annotations` dict, rendered as rows by the page:
`flags` (bluff/anchor/probe on Suggestions), `info_gain_bits`, `candidates_left` (Accusations) and,
when the log has `learner` and per-action `probs`, the Learner's top action probabilities.
"""
from __future__ import annotations

import argparse
import json
import sys

from .cli import describe, fmt_cards
from .core import Game
from .core.cards import CARD_NAMES
from .core.deduction import Deduction
from .core.metrics import analyse

OMNISCIENT = "omniscient"


def _describe_action(seat, action):
    kind = action[0]
    if kind == "suggest":
        return f"Seat {seat} suggests {fmt_cards(action[1:])}"
    if kind == "accuse":
        return f"Seat {seat} accuses {fmt_cards(action[1:])}"
    if kind == "pass":
        return f"Seat {seat} does not accuse"
    return f"Seat {seat} reveals {CARD_NAMES[action[1]]}"


def _omniscient_grid(game):
    grid = [[0] * len(CARD_NAMES) for _ in range(game.n + 1)]
    for seat, hand in enumerate(game.hands):
        for c in hand:
            grid[seat][c] = 1
    for c in game.envelope:
        grid[game.n][c] = 1
    return grid


def _views(game, engines):
    views = {}
    for seat, ded in enumerate(engines):
        ded.sync(game.events_for(seat))
        views[str(seat)] = {
            "grid": [[-1 if ded.status(o, c) is None else ded.status(o, c)
                      for c in range(len(CARD_NAMES))] for o in range(game.n + 1)],
            "candidate_count": ded.candidate_count(),
            "open_clauses": [[s, sorted(cs)] for s, cs in ded.open_clauses()],
        }
    views[OMNISCIENT] = {"grid": _omniscient_grid(game), "candidate_count": None,
                         "open_clauses": []}
    return views


def _annotate(log, steps):
    """Behaviour flags (bluff/anchor/probe, see cluedo.core.metrics) on each Suggestion, for the
    log's Learner (`log["learner"]`) or else every Seat; plus the Learner's action probabilities
    (`actions[i]["probs"]` = [[action, p], ...], recorded by cluedo.rl.evaluation.logged_game)."""
    learner = log.get("learner")
    for seat in ([learner] if learner is not None else range(len(log["hands"]))):
        for rec in analyse(log, seat):
            ann = steps[rec["index"] + 1]["annotations"]
            if rec["flags"]:
                ann["flags"] = rec["flags"]
            if "info_gain" in rec:
                ann["info_gain_bits"] = round(rec["info_gain"], 3)
            if "candidates" in rec:
                ann["candidates_left"] = rec["candidates"]
    if learner is not None:
        for i, a in enumerate(log["actions"]):
            if a.get("probs"):
                steps[i + 1]["annotations"]["probs"] = [
                    [_describe_action(a["seat"], tuple(act)), round(p, 4)] for act, p in a["probs"]]


def build_replay(log: dict) -> dict:
    """Step data for a Game log. Step 0 is the deal; step k is the state after action k."""
    g = Game([frozenset(h) for h in log["hands"]], log["envelope"], log["start_seat"],
             seed=log.get("seed"), max_turns=log.get("max_turns", 400))
    engines = [Deduction.for_seat(g, s) for s in range(g.n)]

    def step(index, seat, action, phase):
        return {"index": index, "seat": seat, "phase": phase,
                "action": action,
                "action_text": None if action is None else _describe_action(seat, action),
                "turn_seat": g.turn_seat, "eliminated": list(g.eliminated),
                "events": [describe(e) for e in g.events[seen[0]:]],
                "views": _views(g, engines), "annotations": {}}

    seen = [0]
    steps = [step(0, None, None, None)]
    for i, a in enumerate(log["actions"], start=1):
        seen[0] = len(g.events)
        d = g.decision
        action = tuple(a["action"])
        g.apply(action)
        steps.append(step(i, a["seat"], list(action), d.phase))
    _annotate(log, steps)
    return {"version": 1, "n_players": g.n, "seed": log.get("seed"),
            "card_names": list(CARD_NAMES), "hands": log["hands"],
            "envelope": log["envelope"], "outcome": g.outcome, "steps": steps}


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cluedo replay</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#777;--line:#ccc;--has:#2e7d32;--not:#c62828;--hl:#fff3c4}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#e6e6e6;--mut:#999;--line:#444;--has:#66bb6a;--not:#ef5350;--hl:#4a4020}}
body{background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;margin:0;padding:16px;max-width:900px}
button,select{font:inherit;padding:4px 10px}
table{border-collapse:collapse;margin:8px 0}
td,th{border:1px solid var(--line);padding:2px 8px;text-align:center}
td.name{text-align:left}
.has{color:var(--has);font-weight:bold}.not{color:var(--not)}.unk{color:var(--mut)}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:8px 0}
#events div{padding:2px 0}.mut{color:var(--mut)}
.cur{background:var(--hl)}
</style></head><body>
<h1>Cluedo replay</h1>
<div id="meta" class="mut"></div>
<div class="bar">
<button id="first">&#171;</button><button id="prev">&#8249; Back</button>
<button id="next">Step &#8250;</button><button id="last">&#187;</button>
<span id="pos"></span>
<label>View <select id="view"></select></label>
</div>
<div id="what"></div><div id="events"></div><div id="ann"></div>
<div id="cand"></div><div id="grid"></div>
<script>
const DATA = __DATA__;
const S = DATA.steps; let i = 0, view = "omniscient";
const $ = id => document.getElementById(id);
const esc = t => String(t).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const sel = $("view");
for (const v of ["omniscient", ...Array.from({length: DATA.n_players}, (_, k) => String(k))]) {
  const o = document.createElement("option"); o.value = v;
  o.textContent = v === "omniscient" ? "Omniscient" : "Seat " + v; sel.appendChild(o);
}
sel.onchange = () => { view = sel.value; draw(); };
$("first").onclick = () => { i = 0; draw(); };
$("last").onclick = () => { i = S.length - 1; draw(); };
$("prev").onclick = () => { i = Math.max(0, i - 1); draw(); };
$("next").onclick = () => { i = Math.min(S.length - 1, i + 1); draw(); };
document.onkeydown = e => { if (e.key === "ArrowRight") $("next").click(); if (e.key === "ArrowLeft") $("prev").click(); };
function draw() {
  const s = S[i], v = s.views[view], n = DATA.n_players, o = DATA.outcome;
  $("meta").textContent = n + " players, seed " + DATA.seed + "; outcome: " + o.reason +
    (o.winner === null ? "" : ", winner Seat " + o.winner) + (o.truncated ? " (truncated)" : "") +
    ", " + o.turns + " turns";
  $("pos").textContent = "Step " + i + " / " + (S.length - 1) +
    (s.phase ? " (Turn of Seat " + s.turn_seat + ", " + s.phase + ")" : " (deal)");
  $("what").innerHTML = "<b>" + esc(s.action_text || "Initial deal") + "</b>" +
    (s.eliminated.some(x => x) ? ' <span class="mut">eliminated: ' +
      s.eliminated.map((x, k) => x ? "Seat " + k : null).filter(x => x).join(", ") + "</span>" : "");
  $("events").innerHTML = s.events.map(e => "<div>" + esc(e) + "</div>").join("");
  const ann = Object.entries(s.annotations);
  const badge = f => "<b style='border:1px solid var(--line);border-radius:4px;padding:0 6px;margin-right:4px'>" + esc(f.toUpperCase()) + "</b>";
  const cell = (k, x) => k === "flags" ? x.map(badge).join("") : k === "probs" ?
    x.map(([t, p]) => esc(t) + " <span class='mut'>" + (100 * p).toFixed(1) + "%</span>").join("<br>") : esc(JSON.stringify(x));
  $("ann").innerHTML = ann.length ? "<table>" + ann.map(([k, x]) =>
    "<tr><th>" + esc(k) + "</th><td style='text-align:left'>" + cell(k, x) + "</td></tr>").join("") + "</table>" : "";
  const counts = Array.from({length: n}, (_, k) => "Seat " + k + ": " + s.views[k].candidate_count);
  $("cand").innerHTML = view === "omniscient" ? "Candidates per Seat - " + counts.join(", ")
    : "Candidates: <b>" + v.candidate_count + "</b> &middot; open clauses: " + v.open_clauses.length +
      (v.open_clauses.length ? " (" + v.open_clauses.map(c => "Seat " + c[0] + " has one of " +
        c[1].map(x => DATA.card_names[x]).join("/")).join("; ") + ")" : "");
  let h = "<table><tr><th></th>";
  for (let p = 0; p < n; p++) h += "<th>Seat " + p + "</th>";
  h += "<th>Envelope</th></tr>";
  DATA.card_names.forEach((name, c) => {
    h += "<tr><td class='name'>" + esc(name) + "</td>";
    for (let p = 0; p <= n; p++) {
      const x = v.grid[p][c];
      h += x === 1 ? "<td class='has'>X</td>" : x === 0 ? "<td class='not'>-</td>" : "<td class='unk'>?</td>";
    }
    h += "</tr>";
  });
  $("grid").innerHTML = h + "</table>";
}
draw();
</script></body></html>
"""


def render_html(data: dict) -> str:
    payload = json.dumps(data).replace("</", "<\\/")
    return _PAGE.replace("__DATA__", payload)


def write_replay(log: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(build_replay(log)))


def main(argv=None):
    p = argparse.ArgumentParser(description="Turn a Game log (JSON) into a self-contained HTML replay")
    p.add_argument("log")
    p.add_argument("out")
    args = p.parse_args(argv)
    with open(args.log) as f:
        write_replay(json.load(f), args.out)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
