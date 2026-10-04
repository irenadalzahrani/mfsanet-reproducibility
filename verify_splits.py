"""Split-integrity checks computed directly from the released manifests (splits/*.csv).
Run from the repository root:  python3 verify_splits.py > split_verification.txt
Also (re)writes da_subset_videos.csv (the domain-adaptation subset, see datasets.MixedDomainDataset)."""
import collections, csv, os, random, re

PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))
FRAC = {"celebdf": 0.30, "dfdc": 0.20}  # config.SPLITS.*_train_frac_for_da


def rows(ds):
    return list(csv.DictReader(open(os.path.join("splits", f"{ds}_splits.csv"), newline="")))


def base(r):
    return os.path.splitext(os.path.basename(r["abs_path"]))[0]


print("=== 1. Videos per split and label; video_id overlap across splits ===")
for ds in ("ffpp", "celebdf", "dfdc"):
    R = rows(ds)
    c = collections.Counter((r["split"], r["label"]) for r in R)
    ids = collections.defaultdict(set)
    for r in R:
        ids[r["split"]].add(r["video_id"])
    ov = sum(len(ids[a] & ids[b]) for a, b in PAIRS)
    tot = collections.Counter(r["split"] for r in R)
    print(f"{ds}: total={len(R)} train/val/test={tot['train']}/{tot['val']}/{tot['test']} "
          f"duplicate video_ids={len(R) - len({r['video_id'] for r in R})} video_id overlap across splits={ov}")
    for sp in ("train", "val", "test"):
        print(f"    {sp}: real={c[(sp, 'real')]} fake={c[(sp, 'fake')]}")

print("\n=== 2. FF++ (official split lists): source-video numbers shared across splits ===")
nums = collections.defaultdict(set)
for r in rows("ffpp"):
    nums[r["split"]].update(re.findall(r"\d{3}", base(r)))
for a, b in PAIRS:
    print(f"{a}/{b}: shared source numbers = {len(nums[a] & nums[b])}")
orig_split = {re.findall(r"\d{3}", base(r))[0]: r["split"] for r in rows("ffpp") if r["label"] == "real"}
bad = tot = 0
for r in rows("ffpp"):
    if r["label"] != "fake":
        continue
    a, b = re.findall(r"\d{3}", base(r))[:2]
    tot += 1
    bad += not (orig_split.get(a) == r["split"] == orig_split.get(b))
print(f"manipulated videos whose two source videos are not both in the video's own split: {bad} of {tot}")

print("\n=== 3. CelebDF-v2: identity tags (idN in file names) shared across splits ===")
man = {r["video_id"]: r for r in rows("celebdf")}


def ids_of(vid):
    return set(re.findall(r"id(\d+)", base(man[vid])))


ident = collections.defaultdict(set)
for v, r in man.items():
    ident[r["split"]].update(ids_of(v))
for a, b in PAIRS:
    print(f"{a}/{b}: shared identities = {len(ident[a] & ident[b])} (distinct identities: {a}={len(ident[a])}, {b}={len(ident[b])})")
for sp in ("val", "test"):
    tagged = [v for v, r in man.items() if r["split"] == sp and ids_of(v)]
    leak = sum(1 for v in tagged if ids_of(v) & ident["train"])
    print(f"{sp}: identity-tagged videos={len(tagged)}; sharing >=1 identity with a train video={leak} ({100 * leak / len(tagged):.1f}%)")
print("videos without an identity tag (YouTube-real):",
      {sp: sum(1 for v, r in man.items() if r["split"] == sp and not ids_of(v)) for sp in ("train", "val", "test")})

print("\n=== 4. DFDC: partition by archive part ===")
c = collections.Counter((r["source"], r["split"], r["label"]) for r in rows("dfdc"))
for k, v in sorted(c.items()):
    print(k, v)
print("(actor/subject disjointness is NOT verified; it requires the original DFDC metadata)")

print("\n=== 5. Domain-adaptation subset (video level, sorted ids, random.Random(1234).shuffle) ===")
out = []
for ds, frac in FRAC.items():
    lab = {r["video_id"]: r["label"] for r in rows(ds) if r["split"] == "train"}
    vids = sorted(lab)
    random.Random(1234).shuffle(vids)
    out += [(ds, v, lab[v]) for v in vids[: int(len(vids) * frac)]]
with open("da_subset_videos.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "video_id", "label"])
    w.writerows(sorted(out))
print(dict(sorted(collections.Counter((d, l) for d, _, l in out).items())))
cel = {v for d, v, _ in out if d == "celebdf"}
da_ids = set().union(*[ids_of(v) for v in cel])
test_v = [v for v, r in man.items() if r["split"] == "test" and ids_of(v)]
test_ids = set().union(*[ids_of(v) for v in test_v])
shared = sum(1 for v in test_v if ids_of(v) & da_ids)
print(f"CelebDF DA subset: {len(cel)} videos, {len(da_ids)} identities; test identities also in DA subset: "
      f"{len(test_ids & da_ids)} of {len(test_ids)}; identity-tagged test videos sharing >=1 identity with a DA video: "
      f"{shared} of {len(test_v)}")
print("DA videos that are also test videos:", len(cel & {v for v, r in man.items() if r['split'] == 'test'}))
