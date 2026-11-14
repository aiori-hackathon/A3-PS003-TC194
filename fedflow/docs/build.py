"""Build a notes PDF from a markdown file.
usage: python3 build.py notes/aids2.md out.pdf
Front matter (first lines until a blank line, 'key: value'): title, subtitle, code, exam, color
Mermaid fences ```mermaid ... ``` are rendered to SVG with mmdc and inlined.
"""
import sys, re, subprocess, hashlib, os, html, json
import markdown

src, out = sys.argv[1], sys.argv[2]
text = open(src, encoding="utf-8").read()
meta = {}
head, _, body = text.partition("\n\n")
for line in head.splitlines():
    k, _, v = line.partition(":")
    meta[k.strip()] = v.strip()
color = meta.get("color", "#4f46e5")

os.makedirs("svg", exist_ok=True)
cfg = {"theme": "base", "themeVariables": {"primaryColor": "#eef2ff", "primaryBorderColor": color, "primaryTextColor": "#0f172a",
       "lineColor": "#475569", "fontFamily": "Inter, sans-serif", "fontSize": "14px", "secondaryColor": "#f0fdf4", "tertiaryColor": "#fff7ed"},
       "flowchart": {"htmlLabels": False, "curve": "basis"}}
json.dump(cfg, open("svg/cfg.json", "w"))

def mermaid(m):
    code = m.group(1)
    h = hashlib.md5((code + color).encode()).hexdigest()[:12]
    f = f"svg/{h}.svg"
    if not os.path.exists(f):
        open(f"svg/{h}.mmd", "w").write(code)
        r = subprocess.run(["mmdc", "-i", f"svg/{h}.mmd", "-o", f, "-c", "svg/cfg.json", "-b", "transparent", "-q"], capture_output=True, text=True)
        if r.returncode:
            sys.stderr.write(f"MERMAID FAIL {h}\n{code}\n{r.stderr}\n")
            return f'<pre>{html.escape(code)}</pre>'
    svg = open(f).read()
    svg = re.sub(r'^<\?xml[^>]*>', '', svg)
    return f'<div class="dia">{svg}</div>'

# ensure a blank line before lists that directly follow a paragraph line
lines = body.split("\n"); fixed = []; infence = False
li = re.compile(r"^\s*([-*]|\d+\.)\s")
for i, ln in enumerate(lines):
    if ln.startswith("```"): infence = not infence
    if not infence and li.match(ln) and fixed and fixed[-1].strip() and not li.match(fixed[-1]) and not fixed[-1].startswith(("|", "<", "    ", "  ")) :
        fixed.append("")
    fixed.append(ln)
body = "\n".join(fixed)
body = re.sub(r"```mermaid\n(.*?)```", mermaid, body, flags=re.S)
html_body = markdown.markdown(body, extensions=["tables", "md_in_html", "fenced_code", "sane_lists", "attr_list"])

# wrap each h2 section so headings don't orphan
css = open("style.css").read().replace("--brand:#4f46e5", f"--brand:{color}") + meta.get("css", "")
cover = f"""<div class="cover"><div class="tag">{meta.get('tag','PREP')}</div>
<h1>{meta.get('title','')}</h1><div class="sub">{meta.get('subtitle','')}</div>
<div class="meta">{meta.get('exam','')}</div><div class="for">{meta.get('for','')}</div></div>"""
doc = f"<!doctype html><html><head><meta charset='utf-8'><title>{meta.get('title','')}</title><style>{css}</style></head><body>{cover}<main>{html_body}</main></body></html>"
open(out.replace(".pdf", ".html"), "w").write(doc)

js = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://{os.path.abspath(out.replace(".pdf", ".html"))}'); await p.waitForTimeout(300);
  await p.pdf({{ path: '{os.path.abspath(out)}', preferCSSPageSize: true, printBackground: true, displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: '<div style="width:100%;font-family:Inter;font-size:7.5px;color:#94a3b8;padding:0 14mm;display:flex;justify-content:space-between"><span>{meta.get("short", meta.get("title",""))} · Regionals prep</span><span><span class=pageNumber></span> / <span class=totalPages></span></span></div>' }});
  await b.close();
}})();"""
open("svg/_mk.cjs", "w").write(js)
r = subprocess.run(["node", "svg/_mk.cjs"], capture_output=True, text=True)
if r.returncode:
    sys.stderr.write(r.stderr); sys.exit(1)
print("built", out)
