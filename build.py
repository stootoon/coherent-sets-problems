#!/usr/bin/env python3
"""Assemble the static site: wraps each fragment in src/ with the shared template.
Run `python3 build.py` after editing anything in src/, then commit the generated .html files."""
import pathlib, re

ROOT = pathlib.Path(__file__).parent
ACTIVE = ' class="active"'
PAGES = [  # (fragment, output, nav label)
    ("index.md.html",  "index.html",  "Overview"),
    ("blockI.html",    "blockI.html", "I · Why coherent sets"),
    ("block0.html",    "block0.html", "0 · Transfer operators"),
    ("blockA.html",    "blockA.html", "A · Finite chains"),
    ("blockB.html",    "blockB.html", "B · Generators"),
    ("blockC.html",    "blockC.html", "C · Ornstein–Uhlenbeck"),
    ("blockD.html",    "blockD.html", "D · Data-driven"),
]

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="style.css">
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['$','$'], ['\\\\(','\\\\)']], displayMath: [['$$','$$'], ['\\\\[','\\\\]']],
         macros: {{ E: "\\\\mathbb{{E}}", R: "\\\\mathbb{{R}}", tr: "\\\\operatorname{{tr}}", diag: "\\\\operatorname{{diag}}",
                   ip: ["\\\\langle #1 \\\\rangle", 1], K: "\\\\mathcal{{K}}", P: "\\\\mathcal{{P}}", X: "\\\\mathcal{{X}}" }} }},
  options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }}
}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
<nav class="top">
  <a class="brand" href="index.html">Coherent sets · problem set</a>
  <div class="links">{nav}</div>
  <button id="toggle-all" type="button">Show all solutions</button>
</nav>
<main>
{body}
</main>
<footer>Problem set accompanying Pughe-Sanford, Ding, Moore, Sengupta, Epstein, Greengard &amp; Chklovskii,
<em>Neurons as Detectors of Coherent Sets in Sensory Dynamics</em>, NeurIPS 2025. Equation numbers refer to that paper.</footer>
<script>
(function () {{
  const btn = document.getElementById('toggle-all');
  btn.addEventListener('click', () => {{
    const all = Array.from(document.querySelectorAll('details.solution'));
    const open = all.some(d => !d.open);
    all.forEach(d => d.open = open);
    btn.textContent = open ? 'Hide all solutions' : 'Show all solutions';
  }});
}})();
</script>
</body>
</html>
"""

def main():
    for frag, out, _ in PAGES:
        body = (ROOT / "src" / frag).read_text(encoding="utf-8")
        m = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else out
        nav = " ".join(
            f'<a href="{o}"{ACTIVE if o == out else ""}>{label}</a>' for _, o, label in PAGES
        )
        (ROOT / out).write_text(TEMPLATE.format(title=title, nav=nav, body=body), encoding="utf-8")
        print("wrote", out)

if __name__ == "__main__":
    main()
