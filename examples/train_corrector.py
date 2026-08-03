"""Train the gated corrector and report it against the identity baseline.

The identity baseline matters more than usual here: 84% of pairs already agree
at cos>0.9, so a model that does nothing scores well. The only number worth
reading is whether it beats identity ON THE CONTACT WINDOW.
"""
import glob, json, os, sys
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
from xembody.adapt import Corrector, featurise
from xembody.pairs import WV

T, S, TCP, PH, G, SRC = [], [], [], [], [], []
for f in sorted(glob.glob(os.path.join(R, "pairs", "*", "[pr]_*.npz"))):
    d = np.load(f); m = json.load(open(f.replace(".npz", "_meta.json")))
    T.append(d["a_target"]); S.append(d["a_source"]); TCP.append(d["tcp"])
    # Files predating the gripper-conditioned collector have no `geom`; NaN is
    # the honest encoding, and featurise() maps it to the Panda's own geometry
    # -- a real zero difference, which is exactly what those pairs are.
    G.append(d["geom"] if "geom" in d.files
             else np.full((len(d["tcp"]), 2), np.nan, np.float32))
    PH += [x["phase"] for x in m]
    SRC += [os.path.basename(os.path.dirname(f))] * len(d["tcp"])
T = np.concatenate(T); S = np.concatenate(S); TCP = np.concatenate(TCP)
G = np.concatenate(G); PH = np.array(PH); SRC = np.array(SRC)

for g in np.unique(SRC):
    sel = SRC == g
    gm = np.nanmean(G[sel], 0) if not np.all(np.isnan(G[sel])) else np.zeros(2)
    print(f"  {g:<12}{sel.sum():>5} pairs   geom "
          f"[{gm[0]*1000:6.1f}, {gm[1]*1000:6.1f}] mm")

X = featurise(T, TCP, G); Y = S[:, :6].astype(np.float32)
# weight by SOURCE magnitude: a near-zero command has an ill-conditioned
# direction, and fitting it teaches the model noise
W = np.linalg.norm(S[:, WV], axis=1).astype(np.float32)
print(f"{len(T)} pairs, feature dim {X.shape[1]}")

def cos_of(pred):
    a, b = pred[:, WV], S[:, WV]
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-9)

c = Corrector(hidden=64)
val = c.fit(X, Y, W, epochs=600, verbose=True)
pred = c(T, TCP, G)
ci, cp = cos_of(T), cos_of(pred)
contact = (PH >= 0.4) & (PH <= 0.6)
late = PH > 0.8
print(f"\n  {'':22}{'identity':>12}{'corrected':>12}")
rows = [("all", np.ones_like(PH, bool)), ("contact 0.4-0.6", contact),
        ("late 0.8-1.0", late)]
rows += [(f"{g} only", SRC == g) for g in np.unique(SRC)]
for lbl, sel in rows:
    print(f"  {lbl:<22}{np.median(ci[sel]):>12.4f}{np.median(cp[sel]):>12.4f}"
          f"   frac<0.9 {100*(ci[sel]<0.9).mean():5.1f}% -> {100*(cp[sel]<0.9).mean():5.1f}%")
print(f"\n  gripper channel untouched: {bool(np.allclose(pred[:,6], T[:,6]))}")
