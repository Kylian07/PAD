import os, os.path as osp, random, numpy as np, h5py, torch
from PIL import Image
from scipy.io import savemat
R = os.environ["FAKE_ROOT"]; rng = np.random.RandomState(0)
def img(p, ext=None, size=(32, 64)):
    os.makedirs(osp.dirname(p), exist_ok=True)
    Image.fromarray(rng.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(p)
def market(root, ncam=6, ntr=24, nte=12):
    for sub, pids in (("bounding_box_train", range(1, ntr + 1)), ("bounding_box_test", range(100, 100 + nte)), ("query", range(100, 100 + nte))):
        for pid in pids:
            for k in range(4 if sub != "query" else 1):
                cam = (k % ncam) + 1 if sub != "query" else ncam
                img(osp.join(root, sub, f"{pid:04d}_c{cam}s1_{k:06d}_00.jpg"))
market(f"{R}/market-1501/Market-1501-v15.09.15")
market(f"{R}/dukemtmcreid/DukeMTMC-reID", ncam=8)
# MSMT17 V1
m = f"{R}/msmt17/MSMT17_V1"
for lst, sub, pids, n in (("list_train.txt", "train", range(0, 20), 3), ("list_val.txt", "train", range(0, 20), 1),
                          ("list_query.txt", "test", range(0, 10), 1), ("list_gallery.txt", "test", range(0, 10), 3)):
    with open(osp.join(m, lst) if osp.isdir(m) else (os.makedirs(m) or osp.join(m, lst)), "a") as f:
        for pid in pids:
            for k in range(n):
                cam = 1 if lst == "list_query.txt" else 2 + k
                name = f"{pid:04d}/{pid:04d}_{k:03d}_{cam:02d}_0303morning_{lst[5]}{k:04d}_0.jpg"
                img(osp.join(m, sub, name)); f.write(f"{name} {pid}\n")
# cuhksysu4reid
c = f"{R}/cuhk-sysu/cuhksysu4reid"; i = 0
for sub, pids, n in (("train", range(1, 25), 3), ("query", range(500, 510), 1), ("gallery", range(500, 510), 2)):
    for pid in pids:
        for k in range(n):
            img(osp.join(c, sub, f"{pid:05d}_s{1000 + i}_{i:07d}_0.jpg")); i += 1
# CUHK03 raw v7.3-like .mat + new-protocol split
os.makedirs(f"{R}/cuhk03/cuhk03_release", exist_ok=True)
f = h5py.File(f"{R}/cuhk03/cuhk03_release/cuhk-03.mat", "w")
npid = 6; camp_refs = []; filelist = []; labels = []; camid = []; gl = 0
for pair in range(1, 6):
    refs = np.empty((10, npid), dtype=h5py.ref_dtype)
    for pid in range(1, npid + 1):
        gl += 1
        for im in range(1, 11):
            if im in (5, 10):   # empty cells like the real file
                d = f.create_dataset(f"e/{pair}_{pid}_{im}", data=np.zeros(2, dtype=np.uint64))
            else:
                d = f.create_dataset(f"i/{pair}_{pid}_{im}", data=rng.randint(0, 255, (3, 32, 64), dtype=np.uint8))
                filelist.append(f"{pair}_{pid:03d}_{1 if im <= 5 else 2}_{im:02d}.png"); labels.append(gl); camid.append(1 if im <= 5 else 2)
            refs[im - 1, pid - 1] = d.ref
    camp_refs.append(f.create_dataset(f"camp{pair}", data=refs).ref)
top = np.empty((5, 1), dtype=h5py.ref_dtype); top[:, 0] = camp_refs
f.create_dataset("detected", data=top.T)   # stored (1,5)
f.close()
N = len(filelist); labs = np.array(labels)
train_ids = set(range(1, 19)); idx = np.arange(1, N + 1)
tr = idx[[l in train_ids for l in labs]]
te = [k for k in idx if labs[k - 1] not in train_ids]
q = [k for k in te if filelist[k - 1].endswith(("_02.png", "_07.png"))]; g = [k for k in te if k not in q]
savemat(f"{R}/cuhk03/cuhk03_release/cuhk03_new_protocol_config_detected.mat",
        {"filelist": np.array([[np.array([s])] for s in filelist], dtype=object), "labels": labs[:, None].astype(np.uint16),
         "camId": np.array(camid)[:, None].astype(np.uint8), "train_idx": tr[:, None].astype(np.uint16),
         "query_idx": np.array(q)[:, None].astype(np.uint16), "gallery_idx": np.array(g)[:, None].astype(np.uint16)})
# unseen
u = f"{R}/unseen-sets"
for k in range(20):
    img(f"{u}/viper/VIPeR/cam_a/{k:03d}_0.bmp"); img(f"{u}/viper/VIPeR/cam_b/{k:03d}_90.bmp")
for k in range(1, 30):
    img(f"{u}/prid/prid_2011/single_shot/cam_a/person_{k:04d}.png")
for k in range(1, 40):
    img(f"{u}/prid/prid_2011/single_shot/cam_b/person_{k:04d}.png")
for pid in range(1, 21):
    for s in range(3):
        img(f"{u}/ilids/i-LIDS_Pedestrian/Persons/{pid:04d}{s:03d}.jpg")
    for cam in range(1, 5):
        img(f"{u}/cuhk01/campus/{pid:04d}{cam:03d}.png")
    img(f"{u}/cuhk02/Dataset/P5/cam1/{pid:03d}_1.png"); img(f"{u}/cuhk02/Dataset/P5/cam2/{pid:03d}_2.png")
    img(f"{u}/sensereid/SenseReID/test_probe/{pid:04d}_0.jpg"); img(f"{u}/sensereid/SenseReID/test_gallery/{pid:04d}_1.jpg")
# tiny CLIP-shaped random weights (2 visual layers, 1 text layer; real widths 768/512)
sd = {}
W, TW = 768, 512
sd["visual.conv1.weight"] = torch.randn(W, 3, 16, 16) * 0.02
sd["visual.class_embedding"] = torch.randn(W) * 0.02
sd["visual.positional_embedding"] = torch.randn(197, W) * 0.02
for n in ("ln_pre", "ln_post"):
    sd[f"visual.{n}.weight"] = torch.ones(W); sd[f"visual.{n}.bias"] = torch.zeros(W)
sd["visual.proj"] = torch.randn(W, 512) * 0.02
def block(prefix, w):
    sd[f"{prefix}.attn.in_proj_weight"] = torch.randn(3 * w, w) * 0.02; sd[f"{prefix}.attn.in_proj_bias"] = torch.zeros(3 * w)
    sd[f"{prefix}.attn.out_proj.weight"] = torch.randn(w, w) * 0.02; sd[f"{prefix}.attn.out_proj.bias"] = torch.zeros(w)
    for n in ("ln_1", "ln_2"):
        sd[f"{prefix}.{n}.weight"] = torch.ones(w); sd[f"{prefix}.{n}.bias"] = torch.zeros(w)
    sd[f"{prefix}.mlp.c_fc.weight"] = torch.randn(4 * w, w) * 0.02; sd[f"{prefix}.mlp.c_fc.bias"] = torch.zeros(4 * w)
    sd[f"{prefix}.mlp.c_proj.weight"] = torch.randn(w, 4 * w) * 0.02; sd[f"{prefix}.mlp.c_proj.bias"] = torch.zeros(w)
for l in range(2):
    block(f"visual.transformer.resblocks.{l}", W)
block("transformer.resblocks.0", TW)
sd["token_embedding.weight"] = torch.randn(49408, TW) * 0.02
sd["positional_embedding"] = torch.randn(77, TW) * 0.01
sd["ln_final.weight"] = torch.ones(TW); sd["ln_final.bias"] = torch.zeros(TW)
sd["text_projection"] = torch.randn(TW, 512) * 0.02
sd["logit_scale"] = torch.tensor(4.6)
sd = {k: v.half() if v.is_floating_point() else v for k, v in sd.items()}   # mimic fp16 OpenAI checkpoint
os.makedirs(f"{R}/clip-weights", exist_ok=True); torch.save(sd, f"{R}/clip-weights/ViT-B-16.pt")
print("fake input ready at", R)
