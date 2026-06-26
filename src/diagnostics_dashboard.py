"""Task 3 — single-file HTML diagnostics dashboard.

``build_dashboard_html(summary)`` turns a ``benchmark_summary.json`` (either the
pipeline's, which carries ``form_diagnostics_summary``, or the Phase-0 eval
harness's, which carries ``aggregate``/``per_form``) into a self-contained
``dashboard.html`` — embedded JSON + vanilla JS + inline CSS, **no external
dependencies, no network**. Read-only; produces a file, changes nothing.
"""
from __future__ import annotations

import json
from typing import Any

_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Form Diagnostics Dashboard</title>
<style>
 body{font:14px/1.4 system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#0f1115;color:#e6e6e6}
 header{padding:16px 22px;background:#161a22;border-bottom:1px solid #2a2f3a}
 h1{font-size:18px;margin:0} h2{font-size:14px;color:#9aa4b2;margin:22px 0 8px;text-transform:uppercase;letter-spacing:.05em}
 main{padding:18px 22px;max-width:1100px}
 .cards{display:flex;flex-wrap:wrap;gap:12px}
 .card{background:#161a22;border:1px solid #2a2f3a;border-radius:8px;padding:12px 14px;min-width:150px}
 .card .v{font-size:24px;font-weight:600} .card .l{color:#9aa4b2;font-size:12px}
 .bad{color:#ff6b6b} .warn{color:#ffd166} .ok{color:#6bcB77}
 .bar{height:18px;border-radius:4px;background:#3b82f6;display:inline-block;vertical-align:middle}
 .row{display:flex;align-items:center;gap:8px;margin:3px 0}
 .row .k{width:140px;color:#9aa4b2} .row .n{width:60px;text-align:right}
 table{border-collapse:collapse;width:100%;margin-top:6px} td,th{border:1px solid #2a2f3a;padding:5px 8px;text-align:left}
 th{color:#9aa4b2;font-weight:500} .pill{padding:2px 8px;border-radius:10px;background:#22303f;font-size:12px}
</style></head><body>
<header><h1>Form Diagnostics Dashboard</h1><div id="sub" class="l"></div></header>
<main id="app"></main>
<script>const DATA = __DATA__;
function el(t,c,h){var e=document.createElement(t);if(c)e.className=c;if(h!=null)e.innerHTML=h;return e;}
function card(label,value,cls){var c=el('div','card');c.appendChild(el('div','v '+(cls||''),value));c.appendChild(el('div','l',label));return c;}
function bars(obj,max,colorFn){var w=el('div');var mx=max||Math.max(1,...Object.values(obj).map(Number));
 for(var k in obj){var v=Number(obj[k]);var r=el('div','row');r.appendChild(el('div','k',k));
  var b=el('span','bar');b.style.width=Math.round(220*v/mx)+'px';if(colorFn)b.style.background=colorFn(k,v);r.appendChild(b);
  r.appendChild(el('div','n',obj[k]));w.appendChild(r);}return w;}
function render(){
 var app=document.getElementById('app');
 var fds=DATA.form_diagnostics_summary||null;
 var agg=DATA.aggregate||null; // eval-harness shape
 document.getElementById('sub').textContent=(DATA.corpus_dir||DATA.pipeline_mode||'benchmark report');
 // Accuracy (only if eval-harness ground-truth metrics present)
 if(agg&&agg.micro){var h=el('div');h.appendChild(el('h2',null,'Accuracy (vs ground truth)'));
  var cs=el('div','cards');cs.appendChild(card('micro F1',agg.micro.f1));cs.appendChild(card('precision',agg.micro.precision));
  cs.appendChild(card('recall',agg.micro.recall));cs.appendChild(card('mean IoU',agg.micro.mean_iou));
  cs.appendChild(card('duplicate rate',agg.duplicate_rate));cs.appendChild(card('fragmentation',agg.fragmentation_rate,'warn'));
  h.appendChild(cs);app.appendChild(h);}
 if(!fds){if(!agg)app.appendChild(el('p',null,'No form_diagnostics_summary in this report.'));return;}
 // KPI cards
 var k=el('div');k.appendChild(el('h2',null,'Form health'));var cs=el('div','cards');
 cs.appendChild(card('total widgets',fds.total_widgets));
 cs.appendChild(card('checkbox share',(fds.checkbox_share*100).toFixed(0)+'%',fds.checkbox_share>0.6?'bad':''));
 cs.appendChild(card('suspicious checkboxes',fds.suspicious_checkboxes,fds.suspicious_checkboxes>0?'warn':''));
 cs.appendChild(card('duplicate rate',(fds.duplicate_rate*100).toFixed(1)+'%',fds.duplicate_rate>0.05?'warn':''));
 cs.appendChild(card('ambiguity count',fds.ambiguity_count,fds.ambiguity_count>0?'warn':''));
 cs.appendChild(card('comb groups',fds.comb_group_count+' ('+fds.comb_cell_count+' cells)'));
 var ps=fds.penalty_saturation||{};
 cs.appendChild(card('penalized',(ps.percent_penalized!=null?ps.percent_penalized:'-')+'%',ps.saturated?'bad':''));
 k.appendChild(cs);app.appendChild(k);
 // Confidence distribution
 if(fds.confidence_distribution){var c=el('div');c.appendChild(el('h2',null,'Confidence distribution'));
  c.appendChild(bars(fds.confidence_distribution,null,function(key){return key=='HIGH'?'#6bcB77':key=='MEDIUM'?'#ffd166':'#ff6b6b';}));app.appendChild(c);}
 // Checkbox explosion per page
 if(fds.page_checkbox_counts){var c=el('div');c.appendChild(el('h2',null,'Checkboxes per page (explosion)'));
  c.appendChild(bars(fds.page_checkbox_counts,null,function(_k,v){return v>40?'#ff6b6b':v>20?'#ffd166':'#3b82f6';}));app.appendChild(c);}
 // Table types
 if(fds.table_types&&Object.keys(fds.table_types).length){var c=el('div');c.appendChild(el('h2',null,'Table classification'));
  c.appendChild(bars(fds.table_types));app.appendChild(c);}
 // Penalties by page
 var pbp=ps.penalties_by_page;
 if(pbp){var c=el('div');c.appendChild(el('h2',null,'Penalties by page'));
  var t=el('table');t.appendChild(el('tr',null,'<th>page</th><th>widgets</th><th>penalized</th><th>mean penalty</th>'));
  Object.keys(pbp).forEach(function(p){var r=pbp[p];t.appendChild(el('tr',null,'<td>'+p+'</td><td>'+r.count+'</td><td>'+r.penalized+'</td><td>'+r.mean_penalty+'</td>'));});
  c.appendChild(t);app.appendChild(c);}
 // Dotted leader recall
 var dl=fds.dotted_leader_recall;
 if(dl&&dl.leaders_detected!=null){var c=el('div');c.appendChild(el('h2',null,'Dotted-leader recall'));
  c.appendChild(bars(dl.filter_reason_counts||{}));app.appendChild(c);}
}
render();
</script></body></html>"""


def _slim(summary: dict[str, Any]) -> dict[str, Any]:
    """Embed only what the dashboard JS reads, so the HTML stays small even when
    the source benchmark_summary carries large nested diagnostic arrays."""
    keep = ("form_diagnostics_summary", "aggregate", "per_form", "corpus_dir", "pipeline_mode")
    return {k: summary[k] for k in keep if k in summary}


def build_dashboard_html(summary: dict[str, Any]) -> str:
    """Return a self-contained dashboard HTML string for a benchmark summary."""
    data = json.dumps(_slim(summary), ensure_ascii=False).replace("</", "<\\/")
    return _TEMPLATE.replace("__DATA__", data)
