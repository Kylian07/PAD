# %% [markdown]
# ## Data discovery and preparation
# Kaggle mounts each dataset under `/kaggle/input/<slug>/...` with a layout chosen by the uploader. This cell walks
# `/kaggle/input`, recognises each dataset by its **folder signature**, and builds the exact layout the official
# loaders expect under `DATA_ROOT`, using symlinks so nothing is copied:
#
# ```
# DATA_ROOT/market1501/{bounding_box_train,query,bounding_box_test}
# DATA_ROOT/dukemtmcreid/{...}
# DATA_ROOT/msmt17/{train,test,list_train.txt,list_val.txt,list_query.txt,list_gallery.txt}
# DATA_ROOT/cuhksysu/{...}   <- converted: {pid}_c1_{idx}.jpg (query uses c2, see below)
# DATA_ROOT/cuhk03/{...}     <- converted from cuhk-03.mat with the CUHK03-NP detected split
# DATA_ROOT/{cuhk01,cuhk02,grid,ilids,prid,sensereid,viper}/...   (optional unseen sets)
# ```

# %%
def _list_dirs(root, max_depth=11):   # Kaggle now mounts at /kaggle/input/datasets/<user>/<slug>/...
    """All directories under root (depth-limited, follows Kaggle's read-only mounts)."""
    out = []
    root = osp.abspath(root)
    base_depth = root.rstrip("/").count("/")
    for cur, dirs, files in os.walk(root, followlinks=True):
        depth = cur.rstrip("/").count("/") - base_depth
        if depth >= max_depth:
            dirs[:] = []
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("__MACOSX",)]
        out.append(cur)
    return out


ALL_INPUT_DIRS = _list_dirs(INPUT_ROOT) if osp.isdir(INPUT_ROOT) else []
print(f"scanned {len(ALL_INPUT_DIRS)} directories under {INPUT_ROOT}")


def _has_images(d, exts=(".jpg", ".jpeg", ".png", ".bmp"), min_count=1):
    try:
        n = 0
        for f in os.scandir(d):
            if f.name.lower().endswith(exts):
                n += 1
                if n >= min_count:
                    return True
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        pass
    return False


def _find(pred, keyword=None):
    """Shallowest input dir satisfying pred (and containing keyword in its path, if given)."""
    cands = [d for d in ALL_INPUT_DIRS if (keyword is None or keyword in d.lower()) and pred(d)]
    cands.sort(key=lambda p: (p.count("/"), len(p)))
    return cands[0] if cands else None


def _market_style(d):
    return all(_has_images(osp.join(d, s)) for s in ("bounding_box_train", "query", "bounding_box_test"))


def _symlink(src, dst):
    if osp.lexists(dst):
        if osp.islink(dst):
            os.unlink(dst)
        else:
            return
    os.makedirs(osp.dirname(dst), exist_ok=True)
    os.symlink(src, dst)


def _link_market_style(src, name):
    dst = osp.join(DATA_ROOT, name)
    os.makedirs(dst, exist_ok=True)
    for s in ("bounding_box_train", "query", "bounding_box_test"):
        _symlink(osp.join(src, s), osp.join(dst, s))
    return dst


PREPARED = {}

# ---------- Market-1501 / DukeMTMC-reID: plain symlinks ---------------------------------------------------------------
for name, kw in (("market1501", "market"), ("dukemtmcreid", "duke")):
    src = DATA_OVERRIDES.get(name) or _find(_market_style, kw)
    if src:
        PREPARED[name] = _link_market_style(src, name)
        print(f"[data] {name:13s} <- {src}")

# ---------- MSMT17 (V1: train/test, V2: mask_train_v2/mask_test_v2) ------------------------------------------------------
def _msmt_root(d):
    return osp.isfile(osp.join(d, "list_train.txt")) and (
        osp.isdir(osp.join(d, "train")) or osp.isdir(osp.join(d, "mask_train_v2")))


src = DATA_OVERRIDES.get("msmt17") or _find(_msmt_root, "msmt")
if src:
    dst = osp.join(DATA_ROOT, "msmt17"); os.makedirs(dst, exist_ok=True)
    tr = "train" if osp.isdir(osp.join(src, "train")) else "mask_train_v2"
    te = "test" if osp.isdir(osp.join(src, "test")) else "mask_test_v2"
    _symlink(osp.join(src, tr), osp.join(dst, "train")); _symlink(osp.join(src, te), osp.join(dst, "test"))
    for f in ("list_train.txt", "list_val.txt", "list_query.txt", "list_gallery.txt"):
        _symlink(osp.join(src, f), osp.join(dst, f))
    PREPARED["msmt17"] = dst
    print(f"[data] msmt17        <- {src}")
else:
    src = _find(_market_style, "msmt")   # some uploads ship MSMT17 already in Market format
    if src:
        PREPARED["msmt17"] = _link_market_style(src, "msmt17")
        print(f"[data] msmt17 (market-style) <- {src}")


# ---------- CUHK-SYSU -> cuhksysu4reid -> PAD naming -----------------------------------------------------------------
def _crop_save(img_path, box_xyxy, out_path):
    im = Image.open(img_path).convert("RGB")
    im.crop(tuple(int(round(v)) for v in box_xyxy)).save(out_path)


def _mat_str(x):
    """Unwrap a MATLAB char/cell value loaded by scipy into a Python str."""
    while isinstance(x, np.ndarray):
        if x.size == 0:
            return ""
        x = x.ravel()[0]
    return str(x)


def _mat_box(x):
    """MATLAB [x y w h] -> float array [x1 y1 x2 y2]; empty array if the field is empty."""
    b = np.asarray(x, dtype=object)
    while b.dtype == object and b.size == 1 and isinstance(b.ravel()[0], np.ndarray):
        b = b.ravel()[0]
    b = np.asarray(b, dtype=np.float64).ravel()
    if b.size < 4:
        return np.zeros(0)
    b = b[:4].copy(); b[2:] += b[:2]
    return b


def _find_file(name):
    for d in ALL_INPUT_DIRS:
        if osp.isfile(osp.join(d, name)):
            return osp.join(d, name)
    return None


def convert_cuhksysu_raw(train_mat_path, test_mat_path, img_dir, out_dir):
    """Raw CUHK-SYSU person-search release -> ReID crops with the AKA `cuhksysu4reid` splits:
    train = all boxes of the 5,532 Train.mat identities; query = the 2,900 TestG50 query boxes;
    gallery = the boxes of the query person found in its 50-image gallery (images without the person are skipped).
    Port of LifelongReID/lreid/datasets/cuhksysu.py with defensive unwrapping of MATLAB fields."""
    from scipy.io import loadmat
    tmp = out_dir + ".partial"
    shutil.rmtree(tmp, ignore_errors=True)
    for sub in ("bounding_box_train", "query", "bounding_box_test"):
        os.makedirs(osp.join(tmp, sub), exist_ok=True)
    train_mat = loadmat(train_mat_path)["Train"].squeeze()
    test_mat = loadmat(test_mat_path)["TestG50"].squeeze()
    k = 0
    for item in train_mat:                                            # 5,532 identities
        rec = item[0, 0] if isinstance(item, np.ndarray) and item.dtype.names is None else item
        rec = rec if not isinstance(rec, np.ndarray) else rec.ravel()[0]
        pid = int(_mat_str(rec[0])[1:])                               # 'p01234' -> 1234
        for scene in np.asarray(rec[2]).ravel():
            box = _mat_box(scene[1])
            if box.size:
                _crop_save(osp.join(img_dir, _mat_str(scene[0])), box,
                           osp.join(tmp, "bounding_box_train", f"{pid:05d}_c1_{k:07d}.jpg")); k += 1
    for query, gallery in zip(test_mat["Query"], test_mat["Gallery"]):   # 2,900 queries
        pid = int(_mat_str(query["idname"])[1:])
        _crop_save(osp.join(img_dir, _mat_str(query["imname"])), _mat_box(query["idlocate"]),
                   osp.join(tmp, "query", f"{pid:05d}_c2_{k:07d}.jpg")); k += 1
        for g in np.asarray(gallery).ravel():
            gbox = _mat_box(g["idlocate"])
            if gbox.size == 0:                                        # gallery image without the target person
                continue
            _crop_save(osp.join(img_dir, _mat_str(g["imname"])), gbox,
                       osp.join(tmp, "bounding_box_test", f"{pid:05d}_c1_{k:07d}.jpg")); k += 1
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)


def relink_cuhksysu4reid(src, out_dir):
    """cuhksysu4reid/{train,query,gallery}/{pid}_s{scene}_{idx}_{hard}.jpg -> PAD naming via symlinks.
    NOTE: the README says 'renamed to {pid:04d}_c1_{idx:05d}.jpg'. If query AND gallery are all camera 1, the standard
    evaluator (which drops same-id/same-camera gallery hits) would discard every true match, so the query split is
    given camera 2 here (exactly what the AKA/PTKP loaders do: query cam=1, gallery cam=0)."""
    pat = re.compile(r"([-\d]+)_s([-\d]+)_([-\d]+)")
    for sub, dsub, cam in (("train", "bounding_box_train", 1), ("query", "query", 2), ("gallery", "bounding_box_test", 1)):
        os.makedirs(osp.join(out_dir, dsub), exist_ok=True)
        for i, p in enumerate(sorted(glob.glob(osp.join(src, sub, "*.jpg")))):
            m = pat.search(osp.basename(p))
            if m is None:
                continue
            _symlink(p, osp.join(out_dir, dsub, f"{int(m.group(1)):05d}_c{cam}_{i:07d}.jpg"))


sysu_out = osp.join(DATA_ROOT, "cuhksysu")
if _market_style(sysu_out):
    PREPARED["cuhksysu"] = sysu_out
    print(f"[data] cuhksysu      <- (already prepared) {sysu_out}")
else:
    src4 = DATA_OVERRIDES.get("cuhksysu") or _find(
        lambda d: all(_has_images(osp.join(d, s)) for s in ("train", "query", "gallery")), "sysu")
    src_ms = _find(_market_style, "sysu")
    sysu_train_mat, sysu_test_mat = _find_file("Train.mat"), _find_file("TestG50.mat")
    sysu_img_dir = _find(lambda d: osp.basename(d) == "SSM" and _has_images(d, (".jpg",)))
    if src4:
        relink_cuhksysu4reid(src4, sysu_out); PREPARED["cuhksysu"] = sysu_out
        print(f"[data] cuhksysu      <- cuhksysu4reid {src4}")
    elif src_ms:
        PREPARED["cuhksysu"] = _link_market_style(src_ms, "cuhksysu")
        print(f"[data] cuhksysu      <- market-style {src_ms} (check that query/gallery cameras differ!)")
    elif sysu_train_mat and sysu_test_mat and sysu_img_dir:
        print(f"[data] cuhksysu      converting raw person-search release (images {sysu_img_dir}) ~5-10 min ...")
        try:
            convert_cuhksysu_raw(sysu_train_mat, sysu_test_mat, sysu_img_dir, sysu_out)
            PREPARED["cuhksysu"] = sysu_out
        except Exception as e:  # noqa: BLE001
            print(f"!! cuhksysu conversion failed ({type(e).__name__}: {e}); the domain will be left out.")
    elif _find(lambda d: True, "sysu"):
        print("!! a CUHK-SYSU-like input is attached but its layout was not recognised "
              "(need Image/SSM + annotation/test/train_test/{Train,TestG50}.mat, or cuhksysu4reid). See the tree below.")


# ---------- CUHK03 (cuhk-03.mat -> CUHK03-NP detected) -----------------------------------------------------------------
CUHK03_NP_URL = ("https://raw.githubusercontent.com/zhunzhong07/person-re-ranking/master/evaluation/data/CUHK03/"
                 "cuhk03_new_protocol_config_detected.mat")   # dataset split file (Zhong et al. CVPR'17), not PAD code


def convert_cuhk03(mat_path, split_path, out_dir, mode="detected"):
    """Same output as the official tool/convert_cuhk03.py ({pid:04d}_c{1..10}_{idx:05d}.jpg, camid = 2*pair-1 for
    view A and 2*pair for view B), but reads the v7.3 .mat with h5py and maps images by the protocol's file list."""
    import h5py
    from scipy.io import loadmat
    f = h5py.File(mat_path, "r")
    pairs = [f[ref][:].T for ref in f[mode][0]]           # 5 camera pairs, each (num_ids, 10) object refs
    sp = loadmat(split_path)
    def _s(x):   # unwrap MATLAB cell -> str
        while isinstance(x, np.ndarray):
            x = x.ravel()[0]
        return str(x)
    files = [_s(x) for x in sp["filelist"]]
    labels = sp["labels"].ravel().astype(int)
    for key, sub in (("train_idx", "bounding_box_train"), ("query_idx", "query"), ("gallery_idx", "bounding_box_test")):
        os.makedirs(osp.join(out_dir, sub), exist_ok=True)
        for i, k in enumerate(sp[key].ravel().astype(int) - 1):
            pair, pid_in_pair, view, img_id = map(int, osp.splitext(files[k])[0].split("_"))
            img = f[pairs[pair - 1][pid_in_pair - 1, img_id - 1]][:].T      # (H, W, 3) uint8
            camid = 2 * pair - (1 if view == 1 else 0)
            Image.fromarray(img.astype(np.uint8)).save(
                osp.join(out_dir, sub, f"{labels[k]:04d}_c{camid}_{i + 1:05d}.jpg"), quality=95)
    f.close()


c03_out = osp.join(DATA_ROOT, "cuhk03")
if _market_style(c03_out):
    PREPARED["cuhk03"] = c03_out
    print(f"[data] cuhk03        <- (already prepared) {c03_out}")
elif DATA_OVERRIDES.get("cuhk03"):
    PREPARED["cuhk03"] = _link_market_style(DATA_OVERRIDES["cuhk03"], "cuhk03")
else:
    mats = [osp.join(d, "cuhk-03.mat") for d in ALL_INPUT_DIRS if osp.isfile(osp.join(d, "cuhk-03.mat"))]
    if mats:
        splits = [osp.join(d, "cuhk03_new_protocol_config_detected.mat") for d in ALL_INPUT_DIRS
                  if osp.isfile(osp.join(d, "cuhk03_new_protocol_config_detected.mat"))]
        split_path = splits[0] if splits else osp.join(DATA_ROOT, "cuhk03_new_protocol_config_detected.mat")
        if not splits and not osp.isfile(split_path):
            print("[data] cuhk03: downloading CUHK03-NP split file ...")
            urllib.request.urlretrieve(CUHK03_NP_URL, split_path)
        print(f"[data] cuhk03        converting {mats[0]} (CUHK03-NP detected, ~3 min) ...")
        convert_cuhk03(mats[0], split_path, c03_out); PREPARED["cuhk03"] = c03_out
    else:
        src = _find(_market_style, "cuhk03")
        if src:
            PREPARED["cuhk03"] = _link_market_style(src, "cuhk03")
            print(f"[data] cuhk03        <- market-style {src}")


# %% [markdown]
# ### Public unseen test sets (automatic download)
# None of the paper's 7 unseen datasets are on Kaggle. Four are publicly downloadable without a request form, so this
# cell fetches them directly (Internet must be on). Each one has several sources tried in order: the original
# university link that torchreid/Open-ReID use, then an Internet Archive (Wayback Machine) copy of the same file.
# Archives are integrity-checked (and VIPeR's MD5 is verified) and extracted safely, and any source that fails is
# skipped with a message, so the rest of the notebook always runs.
#
# | Set | Status | Source |
# |---|---|---|
# | VIPeR | auto | users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip (MD5 `1c2d9fc1…`) |
# | GRID | auto | personal.ie.cuhk.edu.hk/~ccloy/files/datasets/underground_reid.zip |
# | i-LIDS (QMUL pedestrian) | auto | eecs.qmul.ac.uk/~jason/data/i-LIDS_Pedestrian.tgz |
# | SenseReID | auto | Google Drive file `0B56OfSrVI8hubVJLTzkwV2VaOWM` (via `gdown`) |
# | PRID2011 | optional | paste the `prid_2011.zip` link from the TU Graz page into `PRID_ZIP_URL`, or attach it |
# | CUHK01 / CUHK02 | attach only | request form at CUHK; upload as a private Kaggle dataset if you have them |

# %%
import tarfile, urllib.parse, zipfile

UNSEEN_DL_ROOT = osp.join(_tmp_base, "pad_unseen_downloads")
_WB = "https://web.archive.org/web/2020id_/"   # Wayback "id_" = the original file bytes, nearest snapshot to 2020
UNSEEN_PUBLIC_SOURCES = {
    "viper": {"urls": ["http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip",
                       _WB + "http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip"],
              "md5": "1c2d9fc1cc800332567a0da25a1ce68c", "file": "VIPeR.v1.0.zip"},
    "grid": {"urls": ["http://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/underground_reid.zip",
                      _WB + "http://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/underground_reid.zip"],
             "file": "underground_reid.zip"},
    "ilids": {"urls": ["http://www.eecs.qmul.ac.uk/~jason/data/i-LIDS_Pedestrian.tgz",
                       _WB + "http://www.eecs.qmul.ac.uk/~jason/data/i-LIDS_Pedestrian.tgz"],
              "file": "i-LIDS_Pedestrian.tgz"},
    "sensereid": {"gdrive": "0B56OfSrVI8hubVJLTzkwV2VaOWM", "file": "SenseReID.zip"},
}
if PRID_ZIP_URL:
    UNSEEN_PUBLIC_SOURCES["prid"] = {"urls": [PRID_ZIP_URL], "file": "prid_2011.zip"}
if os.environ.get("PAD_UNSEEN_SOURCES_JSON"):   # testing hook: override the source list
    UNSEEN_PUBLIC_SOURCES = json.loads(os.environ["PAD_UNSEEN_SOURCES_JSON"])

# Folder signatures used to tell whether a set is already available (attached input or earlier download).
UNSEEN_SIGNATURES = {
    "viper": lambda d: osp.isdir(osp.join(d, "cam_a")) and osp.isdir(osp.join(d, "cam_b")) and _has_images(osp.join(d, "cam_a"), (".bmp",)),
    "grid": lambda d: osp.isdir(osp.join(d, "probe")) and osp.isfile(osp.join(d, "features_and_partitions.mat")),
    "ilids": lambda d: osp.basename(d) == "Persons" and _has_images(d, (".jpg",)),
    "sensereid": lambda d: osp.isdir(osp.join(d, "test_probe")) and osp.isdir(osp.join(d, "test_gallery")),
    "prid": lambda d: osp.basename(d) == "single_shot" and osp.isdir(osp.join(d, "cam_a")),
}


def _safe_extract(archive, dest):
    """Extract only regular files/dirs that stay inside dest (archives from the web are untrusted)."""
    dest_real = osp.realpath(dest)
    os.makedirs(dest, exist_ok=True)

    def _inside(name):
        target = osp.realpath(osp.join(dest, name))
        return target == dest_real or target.startswith(dest_real + os.sep)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                if _inside(m.filename):
                    z.extract(m, dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            members = [m for m in t.getmembers() if (m.isfile() or m.isdir()) and _inside(m.name)]
            t.extractall(dest, members=members)
    else:
        raise RuntimeError("not a zip/tar archive (probably an HTML error page)")


def _download_url(url, out, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (PAD reproduction notebook)"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(out + ".part", "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(out + ".part", out)


def _wayback_captures(url, limit=3):
    """Ask the Internet Archive index for saved copies (HTTP 200) of `url`; newest first."""
    q = ("https://web.archive.org/cdx/search/cdx?output=json&fl=timestamp,original&filter=statuscode:200"
         f"&limit=-{limit}&url=" + urllib.parse.quote(url.split("://", 1)[-1], safe=""))
    try:
        req = urllib.request.Request(q, headers={"User-Agent": "Mozilla/5.0 (PAD reproduction notebook)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            rows = json.loads(r.read().decode() or "[]")
        return [f"https://web.archive.org/web/{ts}id_/{orig}" for ts, orig in rows[1:]][::-1]
    except Exception:  # noqa: BLE001
        return []


def fetch_public_unseen(name, spec):
    dest = osp.join(UNSEEN_DL_ROOT, name)
    done_flag = osp.join(dest, ".extracted")
    if osp.isfile(done_flag):
        return "already downloaded"
    os.makedirs(dest, exist_ok=True)
    archive = osp.join(UNSEEN_DL_ROOT, spec["file"])
    errors = []
    attempts = [("gdrive", spec["gdrive"])] if spec.get("gdrive") else []
    attempts += [("url", u) for u in spec.get("urls", [])]
    looked_up = False
    i = 0
    while i < len(attempts):
        kind, src = attempts[i]; i += 1
        try:
            if kind == "gdrive":
                try:
                    import gdown
                except ImportError:
                    import subprocess
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
                    import gdown
                if gdown.download(id=src, output=archive, quiet=True) is None:
                    raise RuntimeError("gdown returned nothing (quota or permissions)")
            else:
                _download_url(src, archive)
            if spec.get("md5") and hashlib.md5(open(archive, "rb").read()).hexdigest() != spec["md5"]:
                raise RuntimeError("MD5 mismatch")
            _safe_extract(archive, dest)
            open(done_flag, "w").write(src)
            os.remove(archive)
            return f"ok <- {src}"
        except Exception as e:  # noqa: BLE001
            errors.append(f"{src[:70]}...: {type(e).__name__}: {str(e)[:80]}")
            if osp.exists(archive):
                os.remove(archive)
        if i == len(attempts) and not looked_up:          # all fixed sources failed -> search the archive index
            looked_up = True
            for u in spec.get("urls", []):
                if "web.archive.org" not in u:
                    attempts += [("url", c) for c in _wayback_captures(u) if ("url", c) not in attempts]
    return "FAILED\n      " + "\n      ".join(errors)


if DOWNLOAD_PUBLIC_UNSEEN and RUN_UNSEEN_EVAL:
    os.makedirs(UNSEEN_DL_ROOT, exist_ok=True)
    for name, spec in UNSEEN_PUBLIC_SOURCES.items():
        if name in UNSEEN_SIGNATURES and _find(UNSEEN_SIGNATURES[name]):
            print(f"[unseen] {name:10s} already attached as a Kaggle input, not downloading")
            continue
        print(f"[unseen] {name:10s} {fetch_public_unseen(name, spec)}")
    ALL_INPUT_DIRS += _list_dirs(UNSEEN_DL_ROOT)   # make downloads visible to the discovery below

# %%
# ---------- unseen-domain sets (optional) ----------------------------------------------------------------------------
def _link_unseen(name, src, rel):
    _symlink(src, osp.join(DATA_ROOT, name, rel)); PREPARED[name] = osp.join(DATA_ROOT, name)
    print(f"[data] {name:13s} <- {src}")


def _try(name, pred, rel, kw=None):
    src = DATA_OVERRIDES.get(name) or _find(pred, kw)
    if src:
        _link_unseen(name, src, rel)


_try("viper", lambda d: osp.basename(d).lower() == "viper" and osp.isdir(osp.join(d, "cam_a")) and osp.isdir(osp.join(d, "cam_b")), "VIPeR")
if "viper" not in PREPARED:
    _try("viper", lambda d: osp.isdir(osp.join(d, "cam_a")) and osp.isdir(osp.join(d, "cam_b"))
         and _has_images(osp.join(d, "cam_a"), (".bmp",)), "VIPeR")
_try("prid", lambda d: osp.basename(d) == "single_shot" and osp.isdir(osp.join(d, "cam_a")), "single_shot")
_try("ilids", lambda d: osp.basename(d) == "i-LIDS_Pedestrian" and osp.isdir(osp.join(d, "Persons")), "i-LIDS_Pedestrian")
if "ilids" not in PREPARED:
    _try("ilids", lambda d: osp.basename(d) == "Persons" and _has_images(d, (".jpg",)), "i-LIDS_Pedestrian/Persons", "lids")
_try("cuhk01", lambda d: osp.basename(d).lower() == "campus" and _has_images(d, (".png",)), "campus")
_try("cuhk02", lambda d: osp.isdir(osp.join(d, "P5", "cam1")) and osp.isdir(osp.join(d, "P5", "cam2")), "Dataset")
_try("sensereid", lambda d: osp.isdir(osp.join(d, "test_probe")) and osp.isdir(osp.join(d, "test_gallery")), "SenseReID")
src = DATA_OVERRIDES.get("grid") or _find(lambda d: osp.isdir(osp.join(d, "probe")) and osp.isdir(osp.join(d, "gallery"))
                                          and osp.isfile(osp.join(d, "features_and_partitions.mat")))
if src:
    for s in ("probe", "gallery", "features_and_partitions.mat"):
        _symlink(osp.join(src, s), osp.join(DATA_ROOT, "grid", s))
    PREPARED["grid"] = osp.join(DATA_ROOT, "grid"); print(f"[data] grid          <- {src}")

def _dataset_roots():
    """Top folder of every attached Kaggle dataset (/kaggle/input/<slug> or /kaggle/input/datasets/<user>/<slug>)."""
    roots = set()
    for d in ALL_INPUT_DIRS:
        rel = osp.relpath(d, INPUT_ROOT)
        if rel.startswith(".."):
            continue
        parts = rel.split(os.sep)
        if parts[0] == "datasets" and len(parts) >= 3:
            roots.add(osp.join(INPUT_ROOT, *parts[:3]))
        elif parts[0] not in (".", "datasets") and len(parts) >= 1:
            roots.add(osp.join(INPUT_ROOT, parts[0]))
    return sorted(roots)


def _print_tree(root, depth=3, max_items=6, _lvl=0):
    try:
        entries = sorted(os.scandir(root), key=lambda e: (not e.is_dir(), e.name))
    except OSError:
        return
    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if not e.is_dir()]
    pad = "    " * (_lvl + 1)
    for e in dirs[:max_items]:
        print(f"{pad}{e.name}/")
        if _lvl + 1 < depth:
            _print_tree(e.path, depth, max_items, _lvl + 1)
    if len(dirs) > max_items:
        print(f"{pad}... (+{len(dirs) - max_items} more folders)")
    if files:
        shown = ", ".join(f.name for f in files[:3])
        print(f"{pad}[{len(files)} files: {shown}{', ...' if len(files) > 3 else ''}]")


_used = set()
for link_root, _, fnames in os.walk(DATA_ROOT):
    for n in os.listdir(link_root):
        q = osp.join(link_root, n)
        if osp.islink(q):
            _used.update({osp.abspath(os.readlink(q)), osp.realpath(q)})
for extra in [globals().get("sysu_train_mat"), globals().get("sysu_img_dir")] + list(globals().get("mats") or []):
    if extra:
        _used.update({osp.abspath(extra), osp.realpath(extra)})
for r in _dataset_roots():
    keys = {osp.abspath(r), osp.realpath(r)}
    if "pad_outputs" in r.lower() or any(u == k or u.startswith(k + os.sep) for u in _used for k in keys):
        continue
    if any(osp.isfile(osp.join(d, "ViT-B-16.pt")) for d in ALL_INPUT_DIRS if d.startswith(r)):
        continue
    print(f"\n?? attached input NOT used (layout not recognised): {r}")
    _print_tree(r)

missing = [d for d in TRAIN_ORDER if d not in PREPARED]
print("\nprepared:", sorted(PREPARED))
if missing and SKIP_MISSING_DOMAINS:
    TRAIN_ORDER = [d for d in TRAIN_ORDER if d in PREPARED]
    print(f"!! not attached, LEFT OUT of the sequence: {missing}")
elif missing:
    print(f"!! MISSING seen domains: {missing}. Attach them (see table above) or set DATA_OVERRIDES. "
          f"Training will stop when it reaches the first missing domain.")
assert TRAIN_ORDER, "no seen-domain dataset found under INPUT_ROOT -- attach the Kaggle datasets listed above"
if TRAIN_ORDER[0] != "market1501":
    print("!! Market-1501 is not the first domain: the first-domain schedule (no KD, full tuning) is applied to "
          f"'{TRAIN_ORDER[0]}' with its own per-domain settings, so this no longer matches the paper's order.")
print("training order:", " -> ".join(TRAIN_ORDER))
