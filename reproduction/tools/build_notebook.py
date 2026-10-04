"""Rebuild the .ipynb from notebook_src/*.py (cells separated by `# %%` markers).
Usage: python build_notebook.py notebook_src ../PAD_Kaggle_Reproduction.ipynb
"""
import sys, glob, re, nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
S = sys.argv[1]; out = sys.argv[2]
cells = []
for f in sorted(glob.glob(S + "/0*.py")):
    src = open(f).read()
    parts = re.split(r"^# %%(.*)$", src, flags=re.M)
    # parts: [pre, marker1, body1, marker2, body2, ...]
    for marker, body in zip(parts[1::2], parts[2::2]):
        body = body.strip("\n")
        if not body.strip():
            continue
        if "[markdown]" in marker:
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))
            cells.append(new_markdown_cell(text))
        else:
            cells.append(new_code_cell(body))
nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}})
nbformat.validate(nb)
nbformat.write(nb, out)
print(f"{len(cells)} cells -> {out}")
