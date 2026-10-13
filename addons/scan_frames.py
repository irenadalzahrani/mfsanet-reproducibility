import glob, sys
from multiprocessing import Pool
from PIL import Image
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/data/frames"
OUT = "/workspace/logs/bad_frames.txt" if ROOT == "/workspace/data/frames" else "/tmp/bad_frames.txt"
def chk(f):
    try:
        with Image.open(f) as im:
            im.load()
            return None if im.size == (288, 288) and im.mode == "RGB" else (f, f"bad size/mode {im.size} {im.mode}")
    except Exception as e:
        return (f, str(e)[:60])
if __name__ == "__main__":
    fs = glob.glob(ROOT + "/*/*/*/*.jpg")
    with Pool(8) as p:
        res = [r for r in p.imap_unordered(chk, fs, chunksize=500) if r]
    print("scanned", len(fs), "| bad:", len(res))
    open(OUT, "w").write("\n".join(f"{f}\t{e}" for f, e in sorted(res)))
    for r in sorted(res)[:10]:
        print(r)
