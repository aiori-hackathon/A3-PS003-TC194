"""Render REPORT.md to docs/Evaluation_Report.pdf (A4) with embedded figures.

    pip install markdown playwright && playwright install chromium
    python scripts/report_pdf.py
"""
import base64
import os
import re

import markdown

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: 'Liberation Serif', 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.42; color: #111; }
h1 { font-size: 17pt; margin: 0 0 4px; color: #0b6b46; }
h2 { break-after: avoid; page-break-after: avoid; font-size: 13pt; margin: 16px 0 6px; color: #0b6b46; border-bottom: 1.5px solid #0b6b46; padding-bottom: 2px; }
p, li { text-align: justify; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 6px 0 10px; page-break-inside: avoid; }
th, td { border: 1px solid #888; padding: 3px 6px; text-align: left; }
th { background: #e8efe9; }
img { max-width: 92%; display: block; margin: 8px auto; page-break-inside: avoid; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.8pt; background: #f2f2f2; padding: 0 3px; }
em { color: #444; }
"""


def inline_images(html: str) -> str:
    def rep(m):
        path = m.group(1)
        with open(path, "rb") as f:
            return f'src="data:image/png;base64,{base64.b64encode(f.read()).decode()}"'
    return re.sub(r'src="(results/figures/[^"]+\.png)"', rep, html)


if __name__ == "__main__":
    md = open("REPORT.md").read()
    body = markdown.markdown(md, extensions=["tables"])
    # keep each heading with the block that follows it
    import re as _re
    body = _re.sub(r"(<h2>.*?</h2>\s*<(p|ul|ol|table)>.*?</\2>)", r'<div style="break-inside:avoid">\1</div>', body, flags=_re.S)
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{inline_images(body)}</body></html>"
    os.makedirs("docs", exist_ok=True)
    with open("docs/_report.html", "w") as f:
        f.write(html)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto("file://" + os.path.abspath("docs/_report.html"))
        pg.pdf(path="docs/Evaluation_Report.pdf", format="A4", print_background=True, prefer_css_page_size=True,
               display_header_footer=True, header_template="<span></span>",
               footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#666'>NeuroForge · PS-003 Evaluation Report · <span class='pageNumber'></span>/<span class='totalPages'></span></div>")
        b.close()
    os.remove("docs/_report.html")
    print("docs/Evaluation_Report.pdf")
