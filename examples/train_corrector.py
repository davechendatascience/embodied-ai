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

T, S, TCP, PH = [], [], [], []
for f in sorted(glob.glob(os.path.join(R, "pairs", "p_*.npz"))):
    d = np.load(f); m = json.load(open(f.replace(".npz", "_meta.json")))
    T.append(d["a_target"]); S.append(d["a_source"]); TCP.append(d["tcp"])
    PH += [x["phase"] for x in m]
T = np.concatenate(T); S = np.concatenate(S); TCP = np.concatenate(TCP); PH = np.array(PH)

X = featurise(T, TCP); Y = S[:, :6].astype(np.float32)
# weight by SOURCE magnitude: a near-zero command has an ill-conditioned
# direction, and fitting it teaches the model noise
W = np.linalg.norm(S[:, WV], axis=1).astype(np.float32)
print(f"{len(T)} pairs, feature dim {X.shape[1]}")

def cos_of(pred):
    a, b = pred[:, WV], S[:, WV]
    return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-9)

c = Corrector(hidden=64)
val = c.fit(X, Y, W, epochs=600, verbose=True)
pred = c(T, TCP)
ci, cp = cos_of(T), cos_of(pred)
contact = (PH >= 0.4) & (PH <= 0.6)
late = PH > 0.8
print(f"\n  {'':22}{'identity':>12}{'corrected':>12}")
for lbl, sel in (("all", np.ones_like(PH, bool)), ("contact 0.4-0.6", contact),
                 ("late 0.8-1.0", late)):
    print(f"  {lbl:<22}{np.median(ci[sel]):>12.4f}{np.median(cp[sel]):>12.4f}"
          f"   frac<0.9 {100*(ci[sel]<0.9).mean():5.1f}% -> {100*(cp[sel]<0.9).mean():5.1f}%")
print(f"\n  gripper channel untouched: {bool(np.allclose(pred[:,6], T[:,6]))}")
