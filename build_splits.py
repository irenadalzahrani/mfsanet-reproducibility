import csv, glob, hashlib, json, os, urllib.request
from config import PATHS, SPLITS

def stable_bucket(vid, n=1000):
    return int(hashlib.sha256(vid.encode()).hexdigest(), 16) % n

def write_manifest(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["video_id","split","label","abs_path","source"]); w.writerows(rows)
    counts = {}
    for r in rows: counts[r[1]] = counts.get(r[1], 0) + 1
    print(f"[wrote] {out_path} ({len(rows)} videos) counts={counts}")

FFPP_URLS = {
    "train": "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/train.json",
    "val":   "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/val.json",
    "test":  "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/test.json",
}
FFPP_METHODS = ["Deepfakes","Face2Face","FaceSwap","NeuralTextures","FaceShifter"]

def fetch_ffpp_official(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    splits, ok = {}, True
    for name, url in FFPP_URLS.items():
        dst = os.path.join(cache_dir, f"{name}.json")
        try:
            if not os.path.exists(dst): urllib.request.urlretrieve(url, dst)
            splits[name] = json.load(open(dst))
        except Exception as e:
            print(f"[WARN] fetch {name} failed: {e}"); ok = False
    print("[ffpp]", "using OFFICIAL splits." if ok else "FALLBACK to hash split (disclose in paper!).")
    return (splits, True) if ok else (None, False)

def build_ffpp():
    root = PATHS.ffpp_root
    official, used = fetch_ffpp_official(os.path.join(PATHS.splits_root, "_ffpp_json"))
    def vid_of(p): return os.path.splitext(os.path.basename(p))[0]
    def fallback(v):
        b = stable_bucket(v)
        return "train" if b < 720 else ("val" if b < 860 else "test")
    rows = []
    orig = sorted(glob.glob(os.path.join(root, "original", "*.mp4")))
    print(f"[ffpp] found {len(orig)} real videos under original/")
    for v in orig:
        vid = vid_of(v); sp = None
        if used:
            for name, pairs in official.items():
                if vid in {p[0] for p in pairs} | {p[1] for p in pairs}: sp = name; break
        else: sp = fallback(vid)
        if sp: rows.append([f"real_{vid}", sp, "real", os.path.abspath(v), "ffpp_orig"])
    for method in FFPP_METHODS:
        method_videos = sorted(glob.glob(os.path.join(root, method, "*.mp4")))
        print(f"[ffpp] found {len(method_videos)} videos under {method}/")
        for v in method_videos:
            vid = vid_of(v); src = vid.split("_")[0]; sp = None
            if used:
                for name, pairs in official.items():
                    if any(p[0]==src or p[1]==src for p in pairs): sp = name; break
            else: sp = fallback(src)
            if sp: rows.append([f"{method}_{vid}", sp, "fake", os.path.abspath(v), f"ffpp_{method}"])
    write_manifest(rows, os.path.join(PATHS.splits_root, "ffpp_splits.csv"))

def build_celebdf():
    root = PATHS.celebdf_root
    cand = [os.path.join(root,"List_of_testing_videos.txt"),
            os.path.join(root,"List_of_testing_videos","List_of_testing_videos.txt")]
    tlist = next((p for p in cand if os.path.exists(p)), None)
    test_set = set()
    if tlist:
        for line in open(tlist):
            line = line.strip()
            if line: test_set.add(line.split()[-1].replace("\\","/"))
        print(f"[celebdf] official test list: {len(test_set)} videos")
    else:
        print("[celebdf][WARN] List_of_testing_videos.txt not found -- heuristic split used.")
    def ident(rel):
        b = os.path.basename(rel)
        return b.split("_")[0] if b.startswith("id") else "yt_"+b
    all_v = []
    for sub, lbl in {"Celeb-real":"real","YouTube-real":"real","Celeb-synthesis":"fake"}.items():
        for v in sorted(glob.glob(os.path.join(root, sub, "*.mp4"))):
            rel = os.path.relpath(v, root).replace("\\","/")
            all_v.append((os.path.abspath(v), rel, lbl))
    trainval = [x for x in all_v if x[1] not in test_set]
    testv = [x for x in all_v if x[1] in test_set]
    ids = sorted({ident(r) for (_,r,_) in trainval})
    val_ids = {i for i in ids if stable_bucket(i) < int(1000*SPLITS.celebdf_val_fraction_of_trainval)}
    rows = []
    for v, rel, lbl in trainval:
        sp = "val" if ident(rel) in val_ids else "train"
        rows.append([f"{lbl}_{os.path.splitext(os.path.basename(rel))[0]}", sp, lbl, v, "celebdf"])
    for v, rel, lbl in testv:
        rows.append([f"{lbl}_{os.path.splitext(os.path.basename(rel))[0]}", "test", lbl, v, "celebdf"])
    write_manifest(rows, os.path.join(PATHS.splits_root, "celebdf_splits.csv"))

def build_dfdc():
    parts = PATHS.dfdc_part_dirs
    if not parts:
        print("[dfdc][WARN] PATHS.dfdc_part_dirs فاضية"); return
    n = len(parts)
    n_train, n_val = (max(1,n-2), 1 if n>=3 else 0)
    n_test = n - n_train - n_val
    assign = (["train"]*n_train + ["val"]*n_val + ["test"]*n_test)[:n]
    print("[dfdc] part->split:", dict(zip([os.path.basename(p) for p in parts], assign)))
    rows = []
    for part_dir, sp in zip(parts, assign):
        pname = os.path.basename(part_dir.rstrip("/"))
        meta = json.load(open(os.path.join(part_dir, "metadata.json")))
        for fname, info in meta.items():
            lbl = "fake" if info.get("label","").upper()=="FAKE" else "real"
            v = os.path.join(part_dir, fname)
            if os.path.exists(v):
                rows.append([f"{pname}_{os.path.splitext(fname)[0]}", sp, lbl, os.path.abspath(v), pname])
    write_manifest(rows, os.path.join(PATHS.splits_root, "dfdc_splits.csv"))

if __name__ == "__main__":
    build_ffpp(); build_celebdf(); build_dfdc()
