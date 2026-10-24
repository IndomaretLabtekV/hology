"""Mammo-FM helpers: encoder loader + preprocessing variants at the Mammo-FM input size (1520x912, HxW).
Protocol read from github.com/batmanlab/Mammo-FM (preprocess_image_to_png_kaggle.py + imagetext.py):
  trim 5px -> min-max -> ExtractBreast bbox crop -> stretch resize 912x1520 -> RGB -> min-max -> (x-0.3089)/0.2505."""
import io
from pathlib import Path
import numpy as np, cv2, torch
from PIL import Image
from efficientnet_pytorch import EfficientNet
from prep import ROOT, trim_border, breast_mask

FM_H, FM_W = 1520, 912
FM_MEAN, FM_STD = 0.3089279, 0.25053555408335154
CKPT = ROOT / "ext/weights/mammo_fm/Mammo-FM_BatmanlabTrained_CLIP.tar"
ENC = ROOT / "ext/weights/mammo_fm/image_encoder.pt"


def load_fm_encoder():
    if not ENC.exists():  # extract image-encoder weights once (the .tar also holds the text encoder + optimizer)
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        torch.save({k[len("image_encoder."):]: v for k, v in ck["model"].items() if k.startswith("image_encoder.")}, ENC)
    sd = torch.load(ENC, map_location="cpu")
    net = EfficientNet.from_name("efficientnet-b5")
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert all(k.startswith("_fc") for k in missing) and not unexpected, (missing, unexpected)
    return net.eval()


@torch.no_grad()
def fm_features(net, imgs, bs=4):
    """imgs: list of uint8 HxW already at FM size. Returns GAP 2048-d features."""
    out = []
    for k in range(0, len(imgs), bs):
        x = np.stack(imgs[k:k + bs]).astype(np.float32)
        x -= x.min(axis=(1, 2), keepdims=True); x /= np.maximum(x.max(axis=(1, 2), keepdims=True), 1e-6)
        x = torch.from_numpy((x - FM_MEAN) / FM_STD)[:, None].repeat(1, 3, 1, 1)
        out.append(net._avg_pooling(net.extract_features(x)).flatten(1).numpy())
    return np.concatenate(out)


def load_gray_scale(p, scale, harmonize=False):
    im = Image.open(p)
    if harmonize:
        # EDA 15/12: Normal files are JPEG q95, Benign/Malignant q75 (one identical table). Re-encode EVERY image as
        # grayscale q75 so all classes carry the same compression artefacts (idempotent-ish for q75 files).
        buf = io.BytesIO(); im.convert("L").save(buf, "JPEG", quality=75); buf.seek(0); im = Image.open(buf)
    im.draft("L", (im.width // scale, im.height // scale))
    return np.asarray(im.convert("L"))


def extract_breast_bbox(img):
    """Verbatim logic of Mammo-FM np_ExtractBreast (longest run of non-constant cols/rows after zeroing <=40)."""
    def runs(b):
        left = np.arange(len(b)); left[b > 0] = 0; left = np.maximum.accumulate(left)
        rev = b[::-1]; right = np.arange(len(rev)); right[rev > 0] = 0; right = np.maximum.accumulate(right)
        right = len(rev) - 1 - right[::-1]
        return right - left - 1
    orig = img; img = np.where(img <= 40, 0, img); h, _ = img.shape
    c = runs(img[h // 2 - int(h * .4):h // 2 + int(h * .4)].std(axis=0) != 0); ci = np.where(c == c.max())[0]
    img = img[:, ci]; _, w = img.shape
    r = runs(img[:, w // 2 - int(w * .4):w // 2 + int(w * .4)].std(axis=1) != 0); ri = np.where(r == r.max())[0]
    return orig[ri][:, ci]


def remove_text(a, m, k):
    """Bright blobs far from the breast = burned-in text -> inpaint from surrounding background (keeps bg texture/level)."""
    far = ~cv2.dilate(m.astype(np.uint8), np.ones((31 * k, 31 * k), np.uint8)).astype(bool)
    bg = np.median(a[far]) if far.any() else 0
    txt = (far & (np.abs(a.astype(np.int16) - int(bg)) > 25)).astype(np.uint8)  # text AND its label box (darker or brighter than bg)
    txt = cv2.dilate(txt, np.ones((3 * k + 1, 3 * k + 1), np.uint8))
    return cv2.inpaint(np.ascontiguousarray(a), txt, 3, cv2.INPAINT_TELEA) if txt.any() else a


def fm_variant(p, variant, scale=2, dilate=31, harmonize=False):
    """All variants end at 1520x912 uint8. scale = JPEG draft factor (2 -> 2370x1770, 4 -> 1185x885)."""
    raw = load_gray_scale(p, scale, harmonize)
    if variant == "raw_stretch":
        return cv2.resize(raw, (FM_W, FM_H), interpolation=cv2.INTER_AREA)
    a = raw[5:-5, 5:-5].astype(np.float32); a = ((a - a.min()) / max(a.max() - a.min(), 1) * 255).astype(np.uint8)
    if variant == "fm_protocol":
        return cv2.resize(extract_breast_bbox(a), (FM_W, FM_H), interpolation=cv2.INTER_AREA)
    if variant in ("fm_notext", "fm_flip", "fm_flatbg"):  # FM protocol, isolating one change each
        a = trim_border(raw); k = max(1, 4 // scale)
        sm = cv2.resize(a, (a.shape[1] // k, a.shape[0] // k), interpolation=cv2.INTER_AREA) if k > 1 else a
        m = cv2.resize(breast_mask(sm).astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        if variant == "fm_notext":  # inpaint burned-in text with surrounding background (zeroing left black boxes on gray bg)
            a = remove_text(a, m, k)
        elif variant == "fm_flatbg":  # text AND background texture gone: outside the dilated breast -> constant bg level
            far = ~cv2.dilate(m.astype(np.uint8), np.ones((31 * k, 31 * k), np.uint8)).astype(bool)
            near = ~far & ~m
            a = np.where(far, np.uint8(np.median(a[far]) if far.any() else 0), a)
            a = np.where(near, cv2.GaussianBlur(a, (0, 0), 3 * k), a).astype(np.uint8)  # soften the skin-to-bg transition
        elif m[:, : m.shape[1] // 2].sum() < m[:, m.shape[1] // 2:].sum():
            a = a[:, ::-1]
        a = a.astype(np.float32); a = ((a - a.min()) / max(a.max() - a.min(), 1) * 255).astype(np.uint8)
        return cv2.resize(extract_breast_bbox(np.ascontiguousarray(a)), (FM_W, FM_H), interpolation=cv2.INTER_AREA)
    a = trim_border(raw)
    k = max(1, 4 // scale)  # mask at the same physical scale as before (mask built on 1/4 res)
    small = cv2.resize(a, (a.shape[1] // k, a.shape[0] // k), interpolation=cv2.INTER_AREA) if k > 1 else a
    m = cv2.resize(breast_mask(small).astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    if m[:, : m.shape[1] // 2].sum() < m[:, m.shape[1] // 2:].sum():
        a, m = a[:, ::-1], m[:, ::-1]
    mk = cv2.dilate(m.astype(np.uint8), np.ones((dilate * k, dilate * k), np.uint8)).astype(bool)
    v = np.where(mk, a, 0).astype(np.uint8)
    if variant == "bg_only":
        return cv2.resize(np.ascontiguousarray(np.where(mk, 0, a).astype(np.uint8)), (FM_W, FM_H), interpolation=cv2.INTER_AREA)
    if variant == "bg_notext":  # background texture only: breast zeroed AND text inpainted -> isolates JPEG/noise signature
        return cv2.resize(np.ascontiguousarray(np.where(mk, 0, remove_text(a, m, k)).astype(np.uint8)), (FM_W, FM_H), interpolation=cv2.INTER_AREA)
    if variant == "v2_stretch":  # our v2 (full canvas) stretched to FM aspect
        return cv2.resize(np.ascontiguousarray(v), (FM_W, FM_H), interpolation=cv2.INTER_AREA)
    ys, xs = np.where(mk)
    if variant == "hp_tissue":  # tissue high-pass only (noise texture, no anatomy): leak test INSIDE the breast
        f = v.astype(np.float32); hp = f - cv2.GaussianBlur(f, (0, 0), 2 * (2 // min(scale, 2)) or 1)
        hp = np.where(m, np.clip(hp * 8 + 128, 1, 255), 0).astype(np.uint8)[ys.min():ys.max() + 1, :xs.max() + 1]
        return cv2.resize(np.ascontiguousarray(hp), (FM_W, FM_H), interpolation=cv2.INTER_AREA)
    if variant == "silhouette":  # shape only, zero texture: tests whether bg_only skill is breast shape, not JPEG noise
        sil = (mk[ys.min():ys.max() + 1, :xs.max() + 1] * 200).astype(np.uint8)
        return cv2.resize(np.ascontiguousarray(sil), (FM_W, FM_H), interpolation=cv2.INTER_NEAREST)
    crop = np.ascontiguousarray(v[ys.min():ys.max() + 1, :xs.max() + 1])
    if variant == "v2_bbox_pct":  # + robust windowing: tissue p1..p99 -> 0..255 (EDA 11: dark images fail)
        t = crop[crop > 0]; lo, hi = np.percentile(t, [1, 99]) if t.size else (0, 255)
        crop = np.where(crop > 0, np.clip((crop.astype(np.float32) - lo) / max(hi - lo, 1) * 254 + 1, 1, 255), 0).astype(np.uint8)
    return cv2.resize(crop, (FM_W, FM_H), interpolation=cv2.INTER_AREA)  # "v2_bbox": flip + masked + bbox + stretch
