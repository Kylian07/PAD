"""Build LiteLReID_Kaggle.ipynb = the verified PAD reproduction cells (teacher) + the LiteLReID student cells.

PAD cells are taken from ../../reproduction/tools/notebook_src unchanged, except:
  * the PAD title cell is replaced by 00_intro.py;
  * one patch in 08_driver.py saves the teacher's text anchors after each domain (must match exactly once).
Usage (from the repository root):  python lite_lreid/tools/build_notebook.py
"""
import glob
import os.path as osp
import re

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = osp.dirname(osp.abspath(__file__))
PAD_SRC = osp.join(HERE, "..", "..", "reproduction", "tools", "notebook_src")
LITE_SRC = osp.join(HERE, "notebook_src")
OUT = osp.join(HERE, "..", "LiteLReID_Kaggle.ipynb")

PATCHES = {
    "08_driver.py": [(
        # resume only from the matching teacher folder (PAD_outputs vs PAD_outputs_lf<fraction>)
        '        if osp.basename(d) == domain and "pad_outputs" in d.lower() and osp.isfile(q):\n',
        '        if osp.basename(d) == domain and osp.basename(osp.dirname(d)) == osp.basename(OUT_DIR) and osp.isfile(q):\n',
    ), (
        "        model, _ = train_one_domain(domain, prev_ckpt, is_first=(idx == 0))\n",
        "        model, _ = train_one_domain(domain, prev_ckpt, is_first=(idx == 0))\n"
        "        save_text_bank(model, domain)          # [LiteLReID] teacher text anchors for the student\n",
    )],
}

ORDER = [("lite", "00_intro.py"), ("pad", "01_setup.py"), ("lite", "01b_lite_config.py"),
         ("pad", "02_config.py"), ("pad", "03_data_prep.py"), ("pad", "04_datasets.py"),
         ("pad", "05_loader_clip.py"), ("pad", "06_losses_optim.py"), ("pad", "07_engine.py"),
         ("lite", "07b_labels_textbank.py"), ("pad", "08_driver.py"), ("pad", "09_results.py"),
         ("lite", "10_student_model.py"), ("lite", "11_student_engine.py"), ("lite", "12_student_driver.py"),
         ("lite", "13_efficiency.py"), ("lite", "14_summary.py")]


def cells_of(src):
    parts = re.split(r"^# %%(.*)$", src, flags=re.M)
    out = []
    for marker, body in zip(parts[1::2], parts[2::2]):
        body = body.strip("\n")
        if not body.strip():
            continue
        if "[markdown]" in marker:
            out.append(("md", "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.split("\n"))))
        else:
            out.append(("code", body))
    return out


cells = []
for origin, fname in ORDER:
    src = open(osp.join(PAD_SRC if origin == "pad" else LITE_SRC, fname)).read()
    for old, new in PATCHES.get(fname, []) if origin == "pad" else []:
        assert src.count(old) == 1, f"patch target not found exactly once in {fname}"
        src = src.replace(old, new)
    cs = cells_of(src)
    if (origin, fname) == ("pad", "01_setup.py"):
        assert cs[0][0] == "md" and cs[0][1].startswith("# PAD reproduction"), "unexpected PAD title cell"
        cs = [("md", "# Part 1: teacher (the PAD reproduction from `reproduction/`, unchanged)")] + cs[1:]
    for kind, body in cs:
        cells.append(new_markdown_cell(body) if kind == "md" else new_code_cell(body))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}})
nbformat.validate(nb)
nbformat.write(nb, OUT)
print(f"{len(cells)} cells -> {osp.normpath(OUT)}")
