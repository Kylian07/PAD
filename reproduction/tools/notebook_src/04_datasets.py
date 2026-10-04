# %% [markdown]
# ## Dataset classes (verbatim port of `pad/data/bases.py`, `seen.py`, `unseen.py`, `sampler.py`)
# Each loader returns `(img_path, pid, camid, trackid)` tuples; see the official repo for the protocol notes.

# %%
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path: str) -> Image.Image:
    if not osp.exists(img_path):
        raise IOError(f"{img_path} does not exist")
    while True:
        try:
            return Image.open(img_path).convert("RGB")
        except IOError:
            print(f"[read_image] retrying: {img_path}")


class BaseImageDataset:
    """Common statistics helpers for image-based ReID datasets."""

    def get_imagedata_info(self, data):
        pids, cams, tracks = set(), set(), set()
        for _, pid, camid, trackid in data:
            pids.add(pid); cams.add(camid); tracks.add(trackid)
        return len(pids), len(data), len(cams), len(tracks)

    def print_dataset_statistics(self, train, query, gallery):
        n_tp, n_ti, n_tc, _ = self.get_imagedata_info(train)
        n_qp, n_qi, n_qc, _ = self.get_imagedata_info(query)
        n_gp, n_gi, n_gc, _ = self.get_imagedata_info(gallery)
        print("  ------------------------------------------")
        print("  subset  | # ids | # images | # cameras")
        print("  ------------------------------------------")
        print(f"  train   | {n_tp:5d} | {n_ti:8d} | {n_tc:9d}")
        print(f"  query   | {n_qp:5d} | {n_qi:8d} | {n_qc:9d}")
        print(f"  gallery | {n_gp:5d} | {n_gi:8d} | {n_gc:9d}")
        print("  ------------------------------------------")

    def _finalize(self, train, query, gallery, tag: str, verbose: bool = True):
        self.train = train
        self.query = query
        self.gallery = gallery
        if verbose:
            print(f"=> {tag} loaded")
            self.print_dataset_statistics(train, query, gallery)
        (self.num_train_pids, self.num_train_imgs,
         self.num_train_cams, self.num_train_vids) = self.get_imagedata_info(train)
        self.num_query_pids, _, _, _ = self.get_imagedata_info(query)
        self.num_gallery_pids, _, _, _ = self.get_imagedata_info(gallery)
        # Unseen loaders have no training split; clamp so PADModel can still
        # be constructed safely at inference time.
        if self.num_train_pids <= 0:
            self.num_train_pids = 1


class ImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, pid, camid, trackid = self.data[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, pid, camid, trackid, img_path.split("/")[-1]


class Market1501(BaseImageDataset):
    """Standard Market-1501 (bounding_box_train / query / bounding_box_test)."""

    dataset_dir = "market1501"

    def __init__(self, root: str = "", verbose: bool = True, pid_begin: int = 0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")
        self.pid_begin = pid_begin
        for d in (self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        self._finalize(train, query, gallery, "Market-1501", verbose)

    def _process_dir(self, dir_path: str, relabel: bool = False):
        img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))
        pattern = re.compile(r"([-\d]+)_c(\d)")
        pids = {int(pattern.search(p).group(1)) for p in img_paths
                if int(pattern.search(p).group(1)) != -1}
        pid2label = {pid: lab for lab, pid in enumerate(sorted(pids))}
        data = []
        for p in img_paths:
            pid, camid = map(int, pattern.search(p).groups())
            if pid == -1:
                continue
            assert 0 <= pid <= 1501 and 1 <= camid <= 6
            camid -= 1
            if relabel:
                pid = pid2label[pid]
            data.append((p, self.pid_begin + pid, camid, 0))
        return data


class DukeMTMCreID(BaseImageDataset):
    """Standard Duke layout (bounding_box_train / query / bounding_box_test)."""

    dataset_dir = "dukemtmcreid"

    def __init__(self, root: str = "", verbose: bool = True, pid_begin: int = 0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")
        self.pid_begin = pid_begin
        for d in (self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        self._finalize(train, query, gallery, "DukeMTMC-reID", verbose)

    def _process_dir(self, dir_path: str, relabel: bool = False):
        img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))
        pattern = re.compile(r"([-\d]+)_c(\d)")
        pids = {int(pattern.search(p).group(1)) for p in img_paths}
        pid2label = {pid: lab for lab, pid in enumerate(sorted(pids))}
        data = []
        for p in img_paths:
            pid, camid = map(int, pattern.search(p).groups())
            assert 1 <= camid <= 8
            camid -= 1
            if relabel:
                pid = pid2label[pid]
            data.append((p, self.pid_begin + pid, camid, 0))
        return data


class MSMT17(BaseImageDataset):
    """MSMT17 (train / test folders + list_*.txt annotation files)."""

    dataset_dir = "msmt17"

    def __init__(self, root: str = "", verbose: bool = True, pid_begin: int = 0, **kwargs):
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "train")
        self.test_dir = osp.join(self.dataset_dir, "test")
        lt = osp.join(self.dataset_dir, "list_train.txt")
        lv = osp.join(self.dataset_dir, "list_val.txt")
        lq = osp.join(self.dataset_dir, "list_query.txt")
        lg = osp.join(self.dataset_dir, "list_gallery.txt")
        for d in (self.dataset_dir, self.train_dir, self.test_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        train = self._process_dir(self.train_dir, lt) + self._process_dir(self.train_dir, lv)
        query = self._process_dir(self.test_dir, lq)
        gallery = self._process_dir(self.test_dir, lg)
        self._finalize(train, query, gallery, "MSMT17", verbose)

    def _process_dir(self, dir_path: str, list_path: str):
        data = []
        with open(list_path, "r") as f:
            for line in f:
                img_name, pid = line.strip().split(" ")
                pid = int(pid)
                camid = int(img_name.split("_")[2]) - 1
                data.append((osp.join(dir_path, img_name), self.pid_begin + pid, camid, 0))
        return data


class CUHKSYSU(BaseImageDataset):
    """Filenames follow ``{pid:04d}_c{camid}_{idx:05d}.jpg`` (ReID-style conversion)."""

    dataset_dir = "cuhksysu"

    def __init__(self, root: str = "", verbose: bool = True, pid_begin: int = 0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")
        self.pid_begin = pid_begin
        for d in (self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        self._finalize(train, query, gallery, "CUHK-SYSU", verbose)

    def _process_dir(self, dir_path: str, relabel: bool = False):
        img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))
        pattern = re.compile(r"([-\d]+)_c(\d+)")
        pids = set()
        for p in img_paths:
            m = pattern.search(osp.basename(p))
            if m is not None and int(m.group(1)) != -1:
                pids.add(int(m.group(1)))
        pid2label = {pid: lab for lab, pid in enumerate(sorted(pids))}
        data = []
        for p in img_paths:
            m = pattern.search(osp.basename(p))
            if m is None:
                continue
            pid, camid = map(int, m.groups())
            if pid == -1:
                continue
            assert camid >= 1
            camid -= 1
            if relabel:
                pid = pid2label[pid]
            data.append((p, self.pid_begin + pid, camid, 0))
        return data


class CUHK03(BaseImageDataset):
    """CUHK03 new-protocol (detected) layout; camid in [1, 10]."""

    dataset_dir = "cuhk03"

    def __init__(self, root: str = "", verbose: bool = True, pid_begin: int = 0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")
        self.pid_begin = pid_begin
        for d in (self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        self._finalize(train, query, gallery, "CUHK03", verbose)

    def _process_dir(self, dir_path: str, relabel: bool = False):
        img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))
        pattern = re.compile(r"([-\d]+)_c(\d+)_")
        pids = set()
        for p in img_paths:
            m = pattern.search(osp.basename(p))
            if m is not None:
                pids.add(int(m.group(1)))
        pid2label = {pid: lab for lab, pid in enumerate(sorted(pids))}
        data = []
        for p in img_paths:
            m = pattern.search(osp.basename(p))
            if m is None:
                continue
            pid, camid = map(int, m.groups())
            assert 1 <= camid <= 10, f"unexpected camid {camid} in {p}"
            camid -= 1
            if relabel:
                pid = pid2label[pid]
            data.append((p, self.pid_begin + pid, camid, 0))
        return data


class LPW_S2(BaseImageDataset):
    """LPW-s2 (pre-converted to Market-style ``<pid>_c<camid>s<sid>_*.jpg``)."""

    dataset_dir = "LPW_s2"

    def __init__(self, root: str = "", verbose: bool = True, pid_begin: int = 0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")
        self.pid_begin = pid_begin
        for d in (self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)
        self._finalize(train, query, gallery, "LPW-s2", verbose)

    def _process_dir(self, dir_path: str, relabel: bool = False):
        img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")))
        pattern = re.compile(r"([-\d]+)_c(\d+)s(\d+)")
        pids = set()
        for p in img_paths:
            m = pattern.search(osp.basename(p))
            if m is not None and int(m.group(1)) != -1:
                pids.add(int(m.group(1)))
        pid2label = {pid: lab for lab, pid in enumerate(sorted(pids))}
        data = []
        for p in img_paths:
            m = pattern.search(osp.basename(p))
            if m is None:
                continue
            pid, camid, _ = map(int, m.groups())
            if pid == -1:
                continue
            camid -= 1
            if relabel:
                pid = pid2label[pid]
            data.append((p, self.pid_begin + pid, camid, 0))
        return data


class CUHK01(BaseImageDataset):
    """``<root>/cuhk01/campus/*.png``, filenames ``PPPPCCCC.png``.

    Cams 1-2 form view A, cams 3-4 form view B. Following the standard
    protocol (see e.g. torchreid's ``CUHK01`` loader), identities are
    randomly split in half; only the held-out test half (~486 of 971)
    is evaluated, in both directions (A2B/B2A), averaged over 10 splits
    (20 runs total, matching VIPeR's convention).
    """

    dataset_dir = "cuhk01"

    def __init__(self, root: str = "", verbose: bool = True,
                 split_id: int = 0, seed: int = 123, direction: str = "auto", **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.campus_dir = osp.join(self.dataset_dir, "campus")
        for d in (self.dataset_dir, self.campus_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        view_a, view_b = defaultdict(list), defaultdict(list)
        for p in sorted(glob.glob(osp.join(self.campus_dir, "*.png"))):
            fname = osp.basename(p)
            try:
                pid = int(fname[:4])
                cam = int(fname[4:7]) if fname[4:7].isdigit() else int(fname[4:8])
            except Exception:
                continue
            if cam in (1, 2):
                view_a[pid].append((p, 0, 0))
            elif cam in (3, 4):
                view_b[pid].append((p, 1, 0))

        all_pids = sorted(set(view_a) & set(view_b))
        if not all_pids:
            raise RuntimeError("CUHK01: no identity overlap between view A and view B")

        if direction in ("A2B", "B2A"):
            eval_direction, base_split = direction, split_id
        else:
            eval_direction = "A2B" if split_id % 2 == 0 else "B2A"
            base_split = split_id // 2

        rng = random.Random(seed + base_split)
        shuffled = all_pids[:]
        rng.shuffle(shuffled)
        num_train = int(len(shuffled) * 0.5)
        test_pids = shuffled[num_train:]

        view_first, view_second = (view_a, view_b) if eval_direction == "A2B" else (view_b, view_a)
        query, gallery = [], []
        for pid in test_pids:
            for p, c, t in view_first[pid]:
                query.append((p, pid, c, t))
            for p, c, t in view_second[pid]:
                gallery.append((p, pid, c, t))
        self._finalize([], query, gallery,
                       f"CUHK01 (unseen, {eval_direction}, split {base_split})", verbose)


class CUHK02(BaseImageDataset):
    """``<root>/cuhk02/Dataset/P5/{cam1,cam2}/*.png`` (P5 is the test split)."""

    dataset_dir = "cuhk02"

    def __init__(self, root: str = "", verbose: bool = True,
                 split_id: int = 0, seed: int = 123, direction: str = "A2B", **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.p5_cam1 = osp.join(self.dataset_dir, "Dataset", "P5", "cam1")
        self.p5_cam2 = osp.join(self.dataset_dir, "Dataset", "P5", "cam2")
        for d in (self.dataset_dir, self.p5_cam1, self.p5_cam2):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        cam1 = self._group(self.p5_cam1, camid=0)
        cam2 = self._group(self.p5_cam2, camid=1)
        overlap = sorted(set(cam1) & set(cam2))
        if not overlap:
            raise RuntimeError("CUHK02: no identity overlap between cam1 and cam2 on P5")
        direction = direction if direction in ("A2B", "B2A") else "A2B"
        rng = random.Random(seed + split_id)

        query, gallery = [], []
        for pid in overlap:
            c1_item = rng.choice(cam1[pid])
            c2_item = rng.choice(cam2[pid])
            first, second = (c1_item, c2_item) if direction == "A2B" else (c2_item, c1_item)
            query.append((first[0], pid, first[1], first[2]))
            gallery.append((second[0], pid, second[1], second[2]))
        self._finalize([], query, gallery, f"CUHK02 (unseen, {direction}, split {split_id})", verbose)

    @staticmethod
    def _parse_pid(fname: str) -> int:
        stem = osp.splitext(fname)[0]
        head = stem.split("_")[0]
        if head.isdigit():
            return int(head)
        digits = ""
        for ch in stem:
            if ch.isdigit():
                digits += ch
            else:
                break
        return int(digits) if digits else -1

    def _group(self, dirpath: str, camid: int):
        by_pid = defaultdict(list)
        for p in sorted(glob.glob(osp.join(dirpath, "*.png"))):
            pid = self._parse_pid(osp.basename(p))
            if pid >= 0:
                by_pid[pid].append((p, camid, 0))
        return by_pid


class GRID(BaseImageDataset):
    """``<root>/grid/{probe, gallery}/*.jpeg`` with ``0000_*`` distractors.

    Uses the dataset's own ``features_and_partitions.mat`` (10 official
    folds, 125 test identities each) rather than a hand-picked split.
    """

    dataset_dir = "grid"

    def __init__(self, root: str = "", verbose: bool = True, split_id: int = 0, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.probe_dir = osp.join(self.dataset_dir, "probe")
        self.gallery_dir = osp.join(self.dataset_dir, "gallery")
        for d in (self.dataset_dir, self.probe_dir, self.gallery_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        try:
            from scipy.io import loadmat
        except ImportError as e:
            raise RuntimeError("GRID requires scipy to read the official .mat partitions (pip install scipy)") from e

        part_path = self._locate_partition_mat()
        if part_path is None:
            raise RuntimeError(
                f"'features_and_partitions.mat' not found under {self.dataset_dir} "
                "(expected at the root or under 'underground_reid/')"
            )
        train_idx_all = loadmat(part_path)["trainIdxAll"][0]
        if not (0 <= split_id < len(train_idx_all)):
            raise ValueError(f"split_id={split_id} out of range [0, {len(train_idx_all)-1}]")
        train_ids = set(int(x) for x in train_idx_all[split_id][0][0][2][0].tolist())

        probe_all = self._collect(self.probe_dir, camid=0, allow_distractor=False)
        gallery_all = self._collect(self.gallery_dir, camid=1, allow_distractor=True)

        query = [(p, pid, c, t) for p, pid, c, t, idx, is_dist in probe_all
                 if idx is not None and idx not in train_ids]
        gallery = [(p, pid, c, t) for p, pid, c, t, idx, is_dist in gallery_all
                   if is_dist or (idx is not None and idx not in train_ids)]
        self._finalize([], query, gallery, f"GRID (unseen, split {split_id})", verbose)

    def _locate_partition_mat(self):
        for candidate in (
            osp.join(self.dataset_dir, "features_and_partitions.mat"),
            osp.join(self.dataset_dir, "underground_reid", "features_and_partitions.mat"),
        ):
            if osp.exists(candidate):
                return candidate
        return None

    @staticmethod
    def _parse_pid(fname: str, allow_distractor: bool) -> int:
        if len(fname) < 4 or not fname[:4].isdigit():
            return -1
        pid = int(fname[:4])
        if pid == 0:
            return 0 if allow_distractor else -1
        return pid

    @staticmethod
    def _parse_idx(fname: str):
        base = osp.splitext(fname)[0]
        if len(base) < 4 or not base[:4].isdigit():
            return None
        idx = int(base[:4])
        return idx if idx != 0 else None

    def _collect(self, dirpath: str, camid: int, allow_distractor: bool):
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(glob.glob(osp.join(dirpath, ext)))
        paths = sorted(paths)
        if not paths:
            raise RuntimeError(f"no images under {dirpath}")

        items = []
        for p in paths:
            fname = osp.basename(p)
            pid = self._parse_pid(fname, allow_distractor)
            if pid < 0:
                continue
            items.append((p, pid, camid, 0, self._parse_idx(fname), pid == 0))
        return items


class ILIDS(BaseImageDataset):
    """``<root>/ilids/i-LIDS_Pedestrian/Persons/*.jpg``, filenames ``PPPPSSS.jpg``.

    Following the standard protocol (see e.g. torchreid's ``iLIDS``
    loader), identities are randomly split in half; only the held-out
    test half (~60 of 119) is evaluated, sampling one query and one
    gallery image per identity (single-shot).
    """

    dataset_dir = "ilids"

    def __init__(self, root: str = "", verbose: bool = True,
                 split_id: int = 0, seed: int = 123, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.persons_dir = osp.join(self.dataset_dir, "i-LIDS_Pedestrian", "Persons")
        for d in (self.dataset_dir, self.persons_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(glob.glob(osp.join(self.persons_dir, ext)))
        paths = sorted(paths)
        if not paths:
            raise RuntimeError(f"no images under {self.persons_dir}")

        by_pid = defaultdict(list)
        for p in paths:
            fname = osp.basename(p)
            if len(fname) < 7:
                continue
            try:
                pid = int(fname[:4])
            except Exception:
                continue
            by_pid[pid].append(p)

        all_pids = sorted(by_pid)
        rng = random.Random(seed + split_id)
        shuffled = all_pids[:]
        rng.shuffle(shuffled)
        num_train = int(len(shuffled) * 0.5)
        test_pids = shuffled[num_train:]

        query, gallery = [], []
        for pid in test_pids:
            imgs = by_pid[pid]
            if len(imgs) < 2:
                continue
            q_path, g_path = rng.sample(imgs, 2)
            query.append((q_path, pid, 0, 0))
            gallery.append((g_path, pid, 1, 0))
        self._finalize([], query, gallery, f"iLIDS (unseen, split {split_id})", verbose)


class PRID(BaseImageDataset):
    """``<root>/prid/single_shot/{cam_a, cam_b}/person_XXXX.png``.

    Standard PRID2011 single-shot protocol: 200 identities appear in
    both cameras; each split randomly holds out 100 as test (the other
    100 are unused). Distractors from the non-overlapping identities
    are kept in the gallery.
    """

    dataset_dir = "prid"

    def __init__(self, root: str = "", verbose: bool = True,
                 split_id: int = 0, seed: int = 123, direction: str = "A2B", **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.cam_a_dir = osp.join(self.dataset_dir, "single_shot", "cam_a")
        self.cam_b_dir = osp.join(self.dataset_dir, "single_shot", "cam_b")
        for d in (self.dataset_dir, self.cam_a_dir, self.cam_b_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        a_map = self._build_map(self.cam_a_dir)
        b_map = self._build_map(self.cam_b_dir)
        overlap = sorted(set(a_map) & set(b_map))
        if len(overlap) < 2:
            raise RuntimeError("PRID: not enough overlapping identities between cam_a and cam_b")
        direction = direction if direction in ("A2B", "B2A") else "A2B"

        pool = overlap[:200] if len(overlap) >= 200 else overlap
        rng = random.Random(seed + split_id)
        shuffled = pool[:]
        rng.shuffle(shuffled)
        half = 100 if len(shuffled) >= 200 else max(1, len(shuffled) // 2)
        test_ids = shuffled[half:half * 2] or shuffled[half:half + 1]

        query, gallery = [], []
        if direction == "A2B":
            for pid in sorted(test_ids):
                query.append((a_map[pid], pid, 0, 0))
                gallery.append((b_map[pid], pid, 1, 0))
            for pid in sorted(set(b_map) - set(overlap)):
                gallery.append((b_map[pid], pid, 1, 0))
        else:
            for pid in sorted(test_ids):
                query.append((b_map[pid], pid, 1, 0))
                gallery.append((a_map[pid], pid, 0, 0))
            for pid in sorted(set(a_map) - set(overlap)):
                gallery.append((a_map[pid], pid, 0, 0))
        self._finalize([], query, gallery, f"PRID (unseen, {direction}, split {split_id})", verbose)

    @staticmethod
    def _parse_pid(fname: str):
        if "person_" not in fname:
            return None
        stem = fname.split("person_", 1)[1]
        digits = ""
        for ch in stem:
            if ch.isdigit():
                digits += ch
            else:
                break
        return int(digits) if digits else None

    def _build_map(self, dirpath: str):
        paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            paths.extend(glob.glob(osp.join(dirpath, ext)))
        m = {}
        for p in sorted(paths):
            pid = self._parse_pid(osp.basename(p))
            if pid is not None:
                m.setdefault(pid, p)
        if not m:
            raise RuntimeError(f"no valid images in {dirpath}")
        return m


class SenseReID(BaseImageDataset):
    """``<root>/sensereid/SenseReID/{test_probe, test_gallery}/<pid>_<camid>.jpg``.

    Query/gallery are fixed by the dataset's own release.
    """

    dataset_dir = "sensereid"

    def __init__(self, root: str = "", verbose: bool = True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, "SenseReID", "test_probe")
        self.gallery_dir = osp.join(self.dataset_dir, "SenseReID", "test_gallery")
        for d in (self.dataset_dir, self.query_dir, self.gallery_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        query = self._collect(self.query_dir)
        gallery = self._collect(self.gallery_dir)
        self._finalize([], query, gallery, "SenseReID (unseen)", verbose)

    @staticmethod
    def _parse(fname: str):
        stem = osp.splitext(fname)[0]
        if "_" not in stem:
            return None, None
        a, b = stem.split("_", 1)
        if not a.isdigit() or not b.isdigit():
            return None, None
        return int(a), int(b)

    def _collect(self, dirpath: str):
        paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            paths.extend(glob.glob(osp.join(dirpath, ext)))
        paths = sorted(paths)
        if not paths:
            raise RuntimeError(f"no images under {dirpath}")
        items = []
        for p in paths:
            pid, camid = self._parse(osp.basename(p))
            if pid is not None:
                items.append((p, pid, camid, 0))
        return items


class VIPeR(BaseImageDataset):
    """``<root>/viper/VIPeR/{cam_a, cam_b}/*.bmp``; i-th images share the same identity.

    Standard VIPeR generalisation protocol: a random 50/50 identity
    split (test half only), evaluated in both directions and averaged
    over 10 random splits (20 runs total).
    """

    dataset_dir = "viper"

    def __init__(self, root: str = "", verbose: bool = True,
                 split_id: int = 0, seed: int = 123, direction: str = "auto", **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.cam_a_dir = self._resolve(osp.join(self.dataset_dir, "VIPeR", "cam_a"))
        self.cam_b_dir = self._resolve(osp.join(self.dataset_dir, "VIPeR", "cam_b"))
        for d in (self.dataset_dir, self.cam_a_dir, self.cam_b_dir):
            if not osp.exists(d):
                raise RuntimeError(f"'{d}' is not available")

        a_paths = self._glob(self.cam_a_dir)
        b_paths = self._glob(self.cam_b_dir)
        if not a_paths or not b_paths or len(a_paths) != len(b_paths):
            raise RuntimeError(f"VIPeR: cam_a={len(a_paths)} cam_b={len(b_paths)} (must be equal and nonempty)")
        N = len(a_paths)

        if direction in ("A2B", "B2A"):
            eval_direction, base_split = direction, split_id
        else:
            eval_direction = "A2B" if split_id % 2 == 0 else "B2A"
            base_split = split_id // 2

        rng = random.Random(seed + base_split)
        shuffled = list(range(1, N + 1))
        rng.shuffle(shuffled)
        test_ids = sorted(shuffled[len(shuffled) // 2:])

        def a_of(pid):
            return a_paths[pid - 1]

        def b_of(pid):
            return b_paths[pid - 1]

        if eval_direction == "A2B":
            query = [(a_of(pid), pid, 0, 0) for pid in test_ids]
            gallery = [(b_of(pid), pid, 1, 0) for pid in test_ids]
        else:
            query = [(b_of(pid), pid, 1, 0) for pid in test_ids]
            gallery = [(a_of(pid), pid, 0, 0) for pid in test_ids]
        self._finalize([], query, gallery,
                       f"VIPeR (unseen, {eval_direction}, split {base_split})", verbose)

    @staticmethod
    def _resolve(base: str) -> str:
        maybe = osp.join(base, "images")
        return maybe if osp.isdir(maybe) else base

    @staticmethod
    def _glob(dirpath: str):
        paths = []
        for ext in ("*.bmp", "*.BMP", "*.jpg", "*.jpeg", "*.png"):
            paths.extend(glob.glob(osp.join(dirpath, ext)))
        return sorted(paths)


class RandomIdentitySampler(Sampler):
    """Sample ``num_instances`` images per identity to form each mini-batch."""

    def __init__(self, data_source, batch_size: int, num_instances: int):
        if batch_size < num_instances:
            raise ValueError("batch_size must be >= num_instances")
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        self.length = 0
        for pid in self.pids:
            n = max(len(self.index_dic[pid]), self.num_instances)
            self.length += n - n % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected:
                final_idxs.extend(batch_idxs_dict[pid].pop(0))
                if not batch_idxs_dict[pid]:
                    avai_pids.remove(pid)
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


class GenericMarketStyle(BaseImageDataset):
    """Fallback (not in the official repo) for uploads that ship MSMT17/CUHK03 already in Market format,
    e.g. `0001_c2_0001.png`. Same parsing rules as the official Market/CUHK03 loaders, accepts jpg+png."""

    def __init__(self, root: str = "", name: str = "", verbose: bool = True, **kwargs):
        d = osp.join(root, name)
        self.pid_begin = 0
        train = self._process_dir(osp.join(d, "bounding_box_train"), relabel=True)
        query = self._process_dir(osp.join(d, "query"), relabel=False)
        gallery = self._process_dir(osp.join(d, "bounding_box_test"), relabel=False)
        self._finalize(train, query, gallery, f"{name} (market-style)", verbose)

    def _process_dir(self, dir_path, relabel=False):
        img_paths = sorted(glob.glob(osp.join(dir_path, "*.jpg")) + glob.glob(osp.join(dir_path, "*.png")))
        pattern = re.compile(r"([-\d]+)_c(\d+)")
        parsed = [(p, pattern.search(osp.basename(p))) for p in img_paths]
        parsed = [(p, int(m.group(1)), int(m.group(2))) for p, m in parsed if m is not None and int(m.group(1)) != -1]
        pid2label = {pid: lab for lab, pid in enumerate(sorted({pid for _, pid, _ in parsed}))}
        return [(p, pid2label[pid] if relabel else pid, cam - 1, 0) for p, pid, cam in parsed]


_FACTORY = {
    "market1501": Market1501, "dukemtmcreid": DukeMTMCreID, "msmt17": MSMT17, "cuhksysu": CUHKSYSU,
    "cuhk03": CUHK03, "lpw_s2": LPW_S2,
    "cuhk01": CUHK01, "cuhk02": CUHK02, "grid": GRID, "ilids": ILIDS, "prid": PRID, "sensereid": SenseReID,
    "viper": VIPeR,
}


def build_dataset(name, cfg, verbose=True):
    root = cfg.DATASETS.ROOT_DIR
    kw = dict(split_id=int(cfg.EVAL.SPLIT_ID), seed=int(cfg.EVAL.SEED), direction=str(cfg.EVAL.DIRECTION))
    d = osp.join(root, name)
    if name == "msmt17" and not osp.exists(osp.join(d, "list_train.txt")):
        return GenericMarketStyle(root=root, name=name, verbose=verbose)
    if name == "cuhk03" and not glob.glob(osp.join(d, "bounding_box_train", "*.jpg")):
        return GenericMarketStyle(root=root, name=name, verbose=verbose)
    return _FACTORY[name](root=root, verbose=verbose, **kw)
