import sys, nbformat, time
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
nb = nbformat.read(sys.argv[1], as_version=4)
c = NotebookClient(nb, timeout=2400, kernel_name="python3")
t = time.time()
try:
    c.execute()
    ok = True
except CellExecutionError as e:
    ok = False; print("CELL ERROR:\n", str(e)[-4000:])
nbformat.write(nb, sys.argv[2])
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        for o in cell.get("outputs", []):
            txt = o.get("text") or "".join(o.get("data", {}).get("text/plain", "")) or ""
            if o.get("output_type") == "error":
                txt = "\n".join(o["traceback"])[-2000:]
            if txt:
                print(f"--- cell {i} ---\n" + txt[-int(sys.argv[3]) if len(sys.argv) > 3 else -1500:])
print("OK" if ok else "FAILED", f"{time.time()-t:.0f}s")
