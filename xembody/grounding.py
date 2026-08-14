"""Pixel boxes -> the string a VLA is actually prompted with.

Pure python. No numpy, no mujoco, no torch -- this is the half of the grounding
probe that can be unit-tested with nothing installed, and the half that has to be
right or every downstream number is measuring a typo.

WHY THERE ARE SEVERAL FORMATS AND NOT ONE.
A box is not a box to a language model; it is a string, and each backbone was
pretrained to read a different one:

  paligemma   `<loc0231><loc0512><loc0388><loc0640> alphabet soup`
              Four location tokens, 1024 bins, order y_min x_min y_max x_max --
              row before column, which is the reverse of every other convention
              in this file and the single easiest thing to get backwards.
              pi-0.5's backbone is PaliGemma-lineage, so this vocabulary is
              pretrained, not improvised.
  qwen_box    `<|box_start|>(231,512),(388,640)<|box_end|> alphabet soup`
              Qwen2-VL grounding syntax, 0-1000 normalised, x before y.
  qwen_json   `{"bbox_2d": [231, 512, 388, 640], "label": "alphabet soup"}`
              Qwen2.5-VL grounding syntax, ABSOLUTE pixels, x before y.
  plain       `alphabet soup at (0.23, 0.50)-(0.39, 0.64)`
              No special vocabulary at all.

VERIFY THE FORMAT AGAINST THE CHECKPOINT BEFORE TRUSTING A NULL RESULT. Which of
qwen_box / qwen_json a given Qwen-family checkpoint was trained on is a property
of that checkpoint, not of the model family, and feeding the wrong one produces
exactly the same signature as a policy that ignores grounding: no effect. A null
result from an unverified format says nothing about grounding.

`plain` IS THE CONTROL, and it is the reason this module has four formats instead
of one. If `plain` moves the policy as much as `paligemma` does, then whatever is
happening is not the pretrained detection vocabulary being recognised -- it is a
language model reading a sentence. That is a different claim, with different
consequences for anything built on top, and it costs one extra condition to
separate.

BOX CONVENTION IN, EVERYWHERE. `(x0, y0, x1, y1)`, inclusive integer pixels,
origin at TOP-LEFT of the image AS THE POLICY RECEIVES IT. If the host flips or
rotates frames before inference, the boxes must be computed on the flipped image
-- see `examples/probe_grounding.py`, where that rotation is the resident trap.
"""

#: PaliGemma's location vocabulary: `<loc0000>`..`<loc1023>`.
PALIGEMMA_BINS = 1024
#: Qwen2-VL normalises grounding coordinates to this range.
QWEN_SCALE = 1000


#: Trailing name segments MuJoCo/LIBERO append for structure, not identity.
#: Measured on libero_10/0, where every object body is `<name>_<idx>_main`:
#: `alphabet_soup_1_main`, `tomato_sauce_1_main`, `basket_1_main`.
STRUCTURAL_SUFFIXES = {"main", "root", "body", "base"}


def pretty(name):
    """MuJoCo body name -> something a language model would read as a noun.

    `alphabet_soup_1_main` -> `alphabet soup`. Indices and structural suffixes
    are scene-construction artefacts, not part of the referent: a policy asked
    for "alphabet soup 1 main" is being handed three tokens it never saw beside
    that object in training, and the grounding segment then differs from the
    baseline in vocabulary as well as in content -- two changes, one measurement.

    Stripping is repeated from the right, because both appear together. A name
    that is ENTIRELY strippable is returned unchanged, so nothing becomes "".
    """
    parts = name.split("_")
    while len(parts) > 1 and (parts[-1].isdigit()
                              or parts[-1].lower() in STRUCTURAL_SUFFIXES):
        parts = parts[:-1]
    return " ".join(p for p in parts if p) or name


def _quantise(v, lo, hi, bins):
    """Continuous position -> bin index, clamped into [0, bins - 1]."""
    if hi <= lo:
        return 0
    f = (v - lo) / (hi - lo)
    return max(0, min(bins - 1, int(round(f * (bins - 1)))))


def fmt_paligemma(boxes, size, bins=PALIGEMMA_BINS):
    """`<locYMIN><locXMIN><locYMAX><locXMAX> label`, joined by ` ; `.

    ROW BEFORE COLUMN. PaliGemma emits y first. Swapping them yields a box that
    is a valid transpose of the real one -- plausible, on an object, and wrong
    on any scene that is not square-symmetric.
    """
    w, h = size
    out = []
    for name, (x0, y0, x1, y1) in _ordered(boxes):
        t = (_quantise(y0, 0, h - 1, bins), _quantise(x0, 0, w - 1, bins),
             _quantise(y1, 0, h - 1, bins), _quantise(x1, 0, w - 1, bins))
        out.append("".join(f"<loc{v:04d}>" for v in t) + f" {pretty(name)}")
    return " ; ".join(out)


def fmt_qwen_box(boxes, size, scale=QWEN_SCALE):
    """`<|box_start|>(x0,y0),(x1,y1)<|box_end|> label`, 0-1000, x before y."""
    w, h = size
    out = []
    for name, (x0, y0, x1, y1) in _ordered(boxes):
        a = (_quantise(x0, 0, w - 1, scale), _quantise(y0, 0, h - 1, scale))
        b = (_quantise(x1, 0, w - 1, scale), _quantise(y1, 0, h - 1, scale))
        out.append(f"<|box_start|>({a[0]},{a[1]}),({b[0]},{b[1]})"
                   f"<|box_end|> {pretty(name)}")
    return " ".join(out)


def fmt_qwen_json(boxes, size):
    """Qwen2.5-VL style JSON list, ABSOLUTE pixels. `size` is unused and kept in
    the signature so every formatter is interchangeable at the call site."""
    del size
    items = [f'{{"bbox_2d": [{x0}, {y0}, {x1}, {y1}], "label": "{pretty(n)}"}}'
             for n, (x0, y0, x1, y1) in _ordered(boxes)]
    return "[" + ", ".join(items) + "]"


def fmt_plain(boxes, size, places=2):
    """`label at (x0, y0)-(x1, y1)` with fractions of the image. The control."""
    w, h = size
    out = []
    for name, (x0, y0, x1, y1) in _ordered(boxes):
        f = [round(x0 / max(1, w - 1), places), round(y0 / max(1, h - 1), places),
             round(x1 / max(1, w - 1), places), round(y1 / max(1, h - 1), places)]
        out.append(f"{pretty(name)} at ({f[0]}, {f[1]})-({f[2]}, {f[3]})")
    return "; ".join(out)


FORMATS = {
    "paligemma": fmt_paligemma,
    "qwen_box": fmt_qwen_box,
    "qwen_json": fmt_qwen_json,
    "plain": fmt_plain,
}


def _ordered(boxes):
    """Deterministic serialisation order.

    Sorted by name, NOT by detector confidence or dict insertion. Two conditions
    that differ only in the order objects were listed are not a controlled
    comparison, and dict order here descends from `np.unique` over a
    segmentation image, which changes when an object moves.
    """
    return sorted(boxes.items())


def serialise(boxes, size, style):
    """{name: (x0, y0, x1, y1)}, (w, h), style -> str. Empty boxes -> ""."""
    if not boxes:
        return ""
    if style not in FORMATS:
        raise KeyError(f"unknown style {style!r}; have {sorted(FORMATS)}")
    return FORMATS[style](boxes, size)


def build_prompt(instruction, boxes, size, style, template=None):
    """The full text handed to the policy for one condition.

    The grounding segment goes AFTER the instruction. Before it, it displaces
    the task text from the position it occupied in every training example, and
    the resulting change in behaviour is a position effect wearing a grounding
    effect's clothes.

    `style=None` or empty `boxes` returns the instruction untouched -- byte
    identical to the ungrounded baseline, which is what makes the baseline a
    baseline rather than a fifth condition.
    """
    if style is None or not boxes:
        return instruction
    seg = serialise(boxes, size, style)
    if not seg:
        return instruction
    tpl = template or "{instruction} ; objects: {boxes}"
    return tpl.format(instruction=instruction, boxes=seg)


def decode_paligemma_items(text, size, bins=PALIGEMMA_BINS):
    """`<loc...>`-tagged string -> [((x0, y0, x1, y1), label), ...].

    Labels matter as much as coordinates: a grounding segment claims *this box
    is that object*, and a consumer that reads only the boxes cannot tell a
    correct annotation from one whose label points somewhere else. Anything that
    checks whether a box was FOLLOWED must read both halves.

    Segments are split on the ` ; ` that `fmt_paligemma` joins with. A trailing
    run of non-`<loc>` text after the four tokens is the label; empty if absent.
    """
    import re

    w, h = size
    out = []
    for seg in text.split(" ; "):
        vals = re.findall(r"<loc(\d{4})>", seg)
        if len(vals) < 4:
            continue
        y0, x0, y1, x1 = (int(v) for v in vals[:4])
        label = re.sub(r"<loc\d{4}>", "", seg).strip()
        # `build_prompt`'s template joins on the same ` ; ` this splits on, so
        # the first grounded segment arrives carrying the template's own lead-in
        # ("objects: ..."). Drop a short leading run ending in a colon; the
        # length bound keeps it from eating a label that merely contains one.
        label = re.sub(r"^[^:]{0,32}:\s*", "", label).strip()
        out.append(((round(x0 / (bins - 1) * (w - 1)),
                     round(y0 / (bins - 1) * (h - 1)),
                     round(x1 / (bins - 1) * (w - 1)),
                     round(y1 / (bins - 1) * (h - 1))), label))
    return out


def decode_paligemma(text, size, bins=PALIGEMMA_BINS):
    """`<loc...>`-tagged string -> [(x0, y0, x1, y1), ...] in pixels.

    Exists to CHECK the encoder, and later to read boxes back out of a policy
    that emits them. A formatter that cannot survive its own round trip is not
    worth running an experiment through.
    """
    import re

    w, h = size
    vals = [int(m) for m in re.findall(r"<loc(\d{4})>", text)]
    out = []
    for i in range(0, len(vals) - 3, 4):
        y0, x0, y1, x1 = vals[i:i + 4]
        out.append((round(x0 / (bins - 1) * (w - 1)),
                    round(y0 / (bins - 1) * (h - 1)),
                    round(x1 / (bins - 1) * (w - 1)),
                    round(y1 / (bins - 1) * (h - 1))))
    return out
