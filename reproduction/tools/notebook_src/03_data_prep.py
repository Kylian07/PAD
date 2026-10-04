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
def _list_dirs(root, max_depth=7):
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


def convert_cuhksysu_raw(ds_dir, out_dir):
    """Raw CUHK-SYSU (Image/SSM + annotation/) -> train/query/gallery crops.
    Same splits as the AKA benchmark's `cuhksysu4reid` (Train.mat, TestG50.mat queries + their gallery hits)."""
    from scipy.io import loadmat
    img_dir, ann = osp.join(ds_dir, "Image", "SSM"), osp.join(ds_dir, "annotation")
    train_mat = loadmat(osp.join(ann, "test", "train_test", "Train.mat"))["Train"].squeeze()
    test_mat = loadmat(osp.join(ann, "test", "train_test", "TestG50.mat"))["TestG50"].squeeze()
    for s in ("bounding_box_train", "query", "bounding_box_test"):
        os.makedirs(osp.join(out_dir, s), exist_ok=True)
    k = 0
    for item in train_mat:                                   # 5,532 identities
        pid = int(item[0, 0][0][0][1:])
        for im_name, box, _hard in item[0, 0][2].squeeze():
            box = box.squeeze().astype(np.float64); box[2:] += box[:2]
            _crop_save(osp.join(img_dir, str(im_name[0])), box,
                       osp.join(out_dir, "bounding_box_train", f"{pid:05d}_c1_{k:07d}.jpg")); k += 1
    for query, gallery in zip(test_mat["Query"], test_mat["Gallery"]):   # 2,900 queries
        pid = int(query["idname"][0, 0][0][1:])
        roi = query["idlocate"][0, 0][0].astype(np.float64); roi[2:] += roi[:2]
        _crop_save(osp.join(img_dir, str(query["imname"][0, 0][0])), roi,
                   osp.join(out_dir, "query", f"{pid:05d}_c2_{k:07d}.jpg")); k += 1
        for g in gallery.squeeze():
            groi = g["idlocate"][0].astype(np.float64)
            if groi.size == 0:                               # gallery image without the target -> no crop
                continue
            groi[2:] += groi[:2]
            _crop_save(osp.join(img_dir, str(g["imname"][0])), groi,
                       osp.join(out_dir, "bounding_box_test", f"{pid:05d}_c1_{k:07d}.jpg")); k += 1


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
    src_raw = _find(lambda d: osp.isdir(osp.join(d, "Image", "SSM")) and osp.isdir(osp.join(d, "annotation")))
    if src4:
        relink_cuhksysu4reid(src4, sysu_out); PREPARED["cuhksysu"] = sysu_out
        print(f"[data] cuhksysu      <- cuhksysu4reid {src4}")
    elif src_ms:
        PREPARED["cuhksysu"] = _link_market_style(src_ms, "cuhksysu")
        print(f"[data] cuhksysu      <- market-style {src_ms} (check that query/gallery cameras differ!)")
    elif src_raw:
        print(f"[data] cuhksysu      converting raw person-search release {src_raw} (~5 min) ...")
        convert_cuhksysu_raw(src_raw, sysu_out); PREPARED["cuhksysu"] = sysu_out


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

missing = [d for d in TRAIN_ORDER if d not in PREPARED]
print("\nprepared:", sorted(PREPARED))
if missing:
    print(f"!! MISSING seen domains: {missing}. Attach them (see table above) or set DATA_OVERRIDES. "
          f"Training will stop when it reaches the first missing domain.")
