# Tách một ảnh phẳng (poster, banner) thành từng element PNG trong suốt.
#
# Bước 1  propose : SAM tìm mọi mảnh -> gộp sơ bộ thành nhóm -> sinh review.html (bảng duyệt)
# Bước 2  (người) : mở review.html, gộp/tách/đặt tên, bấm "Lưu file nhóm" -> chép groups.json vào thư mục ra
# Bước 3  export  : cắt từng element theo nhóm, lấp nền bằng LaMa, ghi manifest + bảng xem + vùng cắt cho Copilot
#
#   .venv/Scripts/python split.py propose <ảnh vào> <thư mục ra> [--pps 32] [--force]
#   .venv/Scripts/python split.py export  <thư mục ra> [--groups groups.json] [--no-bg]
import argparse
import base64
import io
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).parent
LAMA = Path.home() / ".cache/torch/hub/checkpoints/big-lama.pt"


# ---------------------------------------------------------------- bước 1
def sam_masks(img, pps):
    from transformers import pipeline

    gen = pipeline("mask-generation", model="facebook/sam-vit-base", device=-1)
    res = gen(img, points_per_side=pps, points_per_batch=64, pred_iou_thresh=0.86, stability_score_thresh=0.92)
    return np.array([np.asarray(m, dtype=bool) for m in res["masks"]])


def build_parts(masks, rgb):
    """Bản đồ mảnh: mỗi điểm ảnh thuộc mask NHỎ NHẤT chứa nó; vùng không mask nào phủ cũng thành mảnh."""
    H, W = rgb.shape[:2]
    area = masks.reshape(len(masks), -1).sum(1)
    order = np.argsort(-area)
    masks, area = masks[order], area[order]
    label = np.zeros((H, W), np.int32)
    for i, m in enumerate(masks):
        label[m] = i + 1
    # Vùng SAM bỏ sót thường là các mảng nền lớn dính liền nhau (giấy trắng + nền xanh):
    # chia theo màu trước rồi mới tách vùng liền khối, để mỗi mảng giấy thành một mảnh bấm được.
    from scipy.cluster.vq import kmeans2

    free = label == 0
    nxt = len(masks) + 1
    if free.sum() > 200:
        px = ndimage.median_filter(rgb, size=(5, 5, 1))[free].astype(np.float32)
        rng = np.random.default_rng(0)
        cent, _ = kmeans2(px[rng.choice(len(px), min(len(px), 20000), replace=False)], 6, minit="++", seed=0)
        near = np.argmin(((px[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        cl = np.full((H, W), -1, np.int32)
        cl[free] = near
        for c in range(len(cent)):
            rest, n = ndimage.label(ndimage.binary_opening(cl == c, iterations=1))
            sizes = ndimage.sum_labels(np.ones_like(rest), rest, np.arange(1, n + 1))
            for k, s in enumerate(sizes, start=1):
                if s >= H * W * 0.0005:
                    label[rest == k] = nxt
                    nxt += 1
    # bỏ mảnh vụn, đánh số lại liền nhau
    ids, counts = np.unique(label, return_counts=True)
    remap = np.zeros(label.max() + 1, np.int32)
    src_of = {}  # id mảnh mới -> chỉ số mask gốc (hoặc None nếu là vùng thừa)
    new = 0
    for i, c in zip(ids, counts):
        if i == 0 or c < 40:
            continue
        new += 1
        remap[i] = new
        src_of[new] = int(i) - 1 if i <= len(masks) else None
    return remap[label], src_of, masks, area


def auto_groups(label, src_of, masks, area):
    """Gộp sơ bộ: (1) mảnh nằm trong một mask cha nhỏ (<3% ảnh) theo cha; (2) các nhóm nhỏ nằm sát nhau (chữ, logo)."""
    H, W = label.shape
    P = int(label.max())
    uf = list(range(P + 1))

    def find(a):
        while uf[a] != a:
            uf[a] = uf[uf[a]]
            a = uf[a]
        return a

    part_of_mask = {m: p for p, m in src_of.items() if m is not None}
    small_parent = H * W * 0.03
    for p, mi in src_of.items():
        if mi is None:
            continue
        top = None
        for j in range(mi - 1, -1, -1):  # mask lớn hơn, duyệt từ nhỏ tới lớn
            if area[j] < small_parent and (masks[mi] & masks[j]).sum() > 0.85 * area[mi]:
                top = j
        if top is not None and top in part_of_mask:
            uf[find(p)] = find(part_of_mask[top])

    sizes = np.bincount(label.ravel(), minlength=P + 1)
    gsize = np.zeros(P + 1, np.int64)
    for p in range(1, P + 1):
        gsize[find(p)] += sizes[p]
    small = np.array([False] + [gsize[find(p)] < H * W * 0.006 for p in range(1, P + 1)])
    near = ndimage.binary_dilation(small[label], iterations=max(1, int(min(H, W) * 0.006)))
    comp, _ = ndimage.label(near)
    first = {}
    for p in np.flatnonzero(small):
        ys, xs = np.nonzero(label == p)
        c = comp[ys[0], xs[0]]
        if c in first:
            uf[find(p)] = find(first[c])
        else:
            first[c] = p

    groups = {}
    for p in range(1, P + 1):
        groups.setdefault(find(p), []).append(p)
    out = sorted(groups.values(), key=lambda ps: -sum(sizes[q] for q in ps))
    big_bg = H * W * 0.2
    return [
        {"name": f"e{i:02d}", "mode": "bg" if sum(sizes[q] for q in ps) > big_bg else "keep", "parts": [int(q) for q in ps]}
        for i, ps in enumerate(out, start=1)
    ]


def data_url(im, fmt, **kw):
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def propose(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    img = Image.open(a.src).convert("RGB")
    img.save(out / "src.png")
    W, H = img.size
    cache = out / "masks.npz"
    if cache.exists() and not a.force:
        masks = np.load(cache)["masks"]
        print(f"Dùng lại {len(masks)} mask đã có")
    else:
        t = time.time()
        masks = sam_masks(img, a.pps)
        np.savez_compressed(cache, masks=masks)
        print(f"SAM: {len(masks)} mask trong {time.time() - t:.0f}s")

    label, src_of, masks, area = build_parts(masks, np.array(img))
    groups = auto_groups(label, src_of, masks, area)
    lab_rgb = np.dstack([label & 255, label >> 8, np.zeros_like(label)]).astype(np.uint8)
    Image.fromarray(lab_rgb).save(out / "parts.png")
    (out / "groups.auto.json").write_text(json.dumps({"source": Path(a.src).name, "groups": groups}, ensure_ascii=False, indent=1), encoding="utf-8")

    data = {
        "key": f"layer-split:{out.name}:{W}x{H}:{int(label.max())}",
        "w": W,
        "h": H,
        "src": data_url(img, "JPEG", quality=92),
        "labels": data_url(Image.fromarray(lab_rgb), "PNG"),
        "groups": groups,
    }
    html = (HERE / "review.template.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    (out / "review.html").write_text(html, encoding="utf-8")
    # file bat nhỏ trong thư mục ra: bấm đúp là mở menu của TachAnh.bat cho đúng thư mục này (chỉ đúng khi thư mục ra nằm trong ket-qua/)
    if out.resolve().parent == (HERE / "ket-qua").resolve():
        (out / "TachAnh.bat").write_text(
            "@echo off\r\n"
            "rem Mo menu tach anh cho rieng thu muc nay. File chinh nam o thu muc TachAnh (cach day 2 cap).\r\n"
            'for %%I in ("%~dp0.") do set "TEN=%%~nxI"\r\n'
            'call "%~dp0..\\..\\TachAnh.bat" mo "%TEN%"\r\n',
            encoding="utf-8",
            newline="",
        )
    print(f"{int(label.max())} mảnh -> {len(groups)} nhóm sơ bộ. Mở {out / 'review.html'} để duyệt.")


# ---------------------------------------------------------------- bước 3
def slug(s):
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower() or "element"


def lama_fill(rgb, mask):
    import torch

    model = torch.jit.load(str(LAMA), map_location="cpu").eval()
    H, W = mask.shape
    ph, pw = (-H) % 8, (-W) % 8
    im = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="reflect").astype(np.float32) / 255
    mk = np.pad(mask, ((0, ph), (0, pw)), mode="reflect").astype(np.float32)
    with torch.no_grad():
        res = model(torch.from_numpy(im).permute(2, 0, 1)[None], torch.from_numpy(mk)[None, None])
    res = np.clip(res[0].permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)[:H, :W]
    return np.where(mask[..., None], res, rgb)


def checker(w, h, s=12):
    yy, xx = np.mgrid[0:h, 0:w]
    c = np.where(((yy // s) + (xx // s)) % 2 == 0, 205, 245).astype(np.uint8)
    return Image.fromarray(np.dstack([c, c, c]))


def plain_bg(rgba):
    """Màu nền trơn khác xa màu của vật nhất (trắng / xám nhạt / xám vừa) để Copilot nhận ra đâu là vật, đâu là nền."""
    px = rgba[..., :3][rgba[..., 3] > 128].astype(np.float32)
    if len(px) > 20000:
        px = px[np.random.default_rng(0).choice(len(px), 20000, replace=False)]
    best, fewest = (255, 255, 255), 2.0
    for c in ((255, 255, 255), (214, 214, 214), (128, 128, 128)):
        near = float((np.sqrt(((px - c) ** 2).sum(1)) < 60).mean()) if len(px) else 0.0
        if near < fewest:
            best, fewest = c, near
    return best


def copilot_crop(rgb, mask, bbox, label, part_size):
    """Ảnh đính kèm cho Copilot: element + những mảnh khác nằm lọt (>= 80% diện tích) trong bao lồi của nó;
    phần còn lại thay bằng nền trơn; chừa lề, phóng lên ~1024px.
    Mảnh nằm lọt trong thân vật thường là phần của vật bị gộp nhầm sang nhóm khác (đai vàng của cái loa bị gộp vào tiêu đề):
    vẫn đưa cho Copilot thấy để vẽ đủ. Mảng lớn chỉ chạm vào bao lồi (giấy nền, khối nền bên cạnh) bị bỏ.
    Trả (ảnh, danh sách id mảnh mượn thêm)."""
    from scipy.spatial import ConvexHull

    x, y, w, h = bbox
    p = int(max(w, h) * 0.08) + 4
    H, W = mask.shape
    X0, Y0, X1, Y1 = max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)
    m = mask[Y0:Y1, X0:X1]
    edge = m & ~ndimage.binary_erosion(m)
    pts = np.argwhere(edge)[:, ::-1]
    hull = Image.new("L", (X1 - X0, Y1 - Y0), 0)
    if len(pts) >= 3:
        try:
            ImageDraw.Draw(hull).polygon([tuple(map(float, pts[i])) for i in ConvexHull(pts).vertices], fill=255)
        except Exception:  # điểm thẳng hàng
            hull = Image.fromarray((m * 255).astype(np.uint8))
    hull = np.asarray(hull) > 0
    lab = label[Y0:Y1, X0:X1]
    ids, n_in = np.unique(lab[hull & ~m], return_counts=True)
    borrowed = [int(i) for i, k in zip(ids, n_in) if i and k >= 0.8 * part_size[i]]
    core = m | np.isin(lab, borrowed)
    keep = core | (ndimage.binary_dilation(core, iterations=3) & (lab == 0))  # điểm ảnh chưa thuộc mảnh nào: chỉ lấy sát mép (viền khử răng cưa)
    keep = ndimage.binary_closing(keep, iterations=2) & (hull | m)
    soft = ndimage.gaussian_filter(keep.astype(np.float32), 1.2)[..., None]
    region = rgb[Y0:Y1, X0:X1].astype(np.float32)
    bg = np.array(plain_bg(np.dstack([region.astype(np.uint8), (keep * 255).astype(np.uint8)])), np.float32)
    canvas = np.full((Y1 - Y0 + 2 * p, X1 - X0 + 2 * p, 3), bg, np.float32)
    canvas[p:-p, p:-p] = region * soft + bg * (1 - soft)
    canvas = Image.fromarray(canvas.clip(0, 255).astype(np.uint8))
    s = 1024 / max(canvas.size)
    if s > 1:
        canvas = canvas.resize((round(canvas.width * s), round(canvas.height * s)), Image.LANCZOS)
    return canvas, borrowed


def _rs(arr, size):
    """Đổi cỡ một bản đồ số thực 2 chiều (mask, kênh màu) về size=(rộng, cao)."""
    return np.asarray(Image.fromarray(arr.astype(np.float32), "F").resize(size, Image.BILINEAR))


def fit_regen(V, B, rgb, reg):
    """Tìm tỉ lệ + vị trí đặt ảnh vẽ lại lên ảnh gốc sao cho khớp nhất với phần element nhìn thấy được.

    V: mask element (H, W); B: điểm ảnh là nền lộ ra trong ảnh gốc (ảnh vẽ lại tràn lên đây là sai); reg: ảnh vẽ lại RGBA.
    Điểm tại mỗi vị trí = phần V được phủ, trừ độ lệch màu với ảnh gốc, trừ phần tràn ra nền. Mọi số hạng là tương quan
    giữa một bản đồ của ảnh gốc và một bản đồ của ảnh vẽ lại, nên tính cho mọi vị trí cùng lúc bằng FFT.
    Phần element bị vật khác che không bị phạt: ảnh vẽ lại được phép có thêm phần đó.
    Trả {"place": [x, y, w, h] của cả ảnh vẽ lại theo toạ độ ảnh gốc, "fit": điểm (1 = khớp hoàn toàn)} hoặc None."""
    from scipy.signal import fftconvolve

    H, W = V.shape
    ra = np.asarray(reg.convert("RGBA"), np.float32) / 255
    ys, xs = np.nonzero(ra[..., 3] > 0.5)
    vy, vx = np.nonzero(V)
    if not len(ys) or not len(vy):
        return None
    cx0, cy0, cx1, cy1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    crop = ra[cy0:cy1, cx0:cx1]
    ch, cw = crop.shape[:2]
    vx0, vy0, vx1, vy1 = vx.min(), vy.min(), vx.max() + 1, vy.max() + 1
    vw, vh = vx1 - vx0, vy1 - vy0
    # bản vẽ lại có cả phần bị che nên không nhỏ hơn phần nhìn thấy: tỉ lệ thật >= k0
    k0 = max(vw / cw, vh / ch)
    m = int(0.6 * max(vw, vh))
    X0, Y0, X1, Y1 = max(0, vx0 - m), max(0, vy0 - m), min(W, vx1 + m), min(H, vy1 + m)
    f = min(1.0, 220 / max(X1 - X0, Y1 - Y0))
    size = (max(1, round((X1 - X0) * f)), max(1, round((Y1 - Y0) * f)))
    ov = _rs(V[Y0:Y1, X0:X1], size)
    ob = _rs(B[Y0:Y1, X0:X1], size)
    oc = np.dstack([_rs(rgb[Y0:Y1, X0:X1, c] / 255, size) for c in range(3)])
    nv = ov.sum()
    K, lam = 0.25, 1.0  # K: độ lệch màu (bình phương, thang 0-1) coi như mất hết điểm; lam: mức phạt tràn ra nền
    o_maps = [ov, ov * (oc ** 2).sum(2), ob] + [ov * oc[..., c] for c in range(3)]

    best = None
    for k in k0 * np.geomspace(0.85, 1.6, 24):
        kw, kh = max(2, round(cw * k * f)), max(2, round(ch * k * f))
        r = np.asarray(Image.fromarray((crop * 255).astype(np.uint8), "RGBA").resize((kw, kh), Image.LANCZOS), np.float32) / 255
        al, cc = r[..., 3], r[..., :3]
        pad = lambda o: np.pad(o, ((kh, kh), (kw, kw)))
        corr = lambda o, ker: fftconvolve(pad(o), ker[::-1, ::-1], mode="valid")
        a_ov, a_oc2, a_ob = (corr(o, al) for o in o_maps[:3])
        cross = sum(corr(o_maps[3 + c], al * cc[..., c]) for c in range(3))
        a_cc2 = corr(ov, al * (cc ** 2).sum(2))
        S = (a_ov - (a_oc2 - 2 * cross + a_cc2) / K - lam * a_ob) / nv
        i, j = np.unravel_index(np.argmax(S), S.shape)
        if best is None or S[i, j] > best[0]:
            best = (float(S[i, j]), kw, kh, i, j)

    s, kw, kh, i, j = best
    ke = kw / (cw * f)  # số điểm ảnh gốc trên một điểm ảnh vẽ lại
    x = X0 + (j - kw) / f - cx0 * ke
    y = Y0 + (i - kh) / f - cy0 * (kh / (ch * f))
    return {"place": [round(x, 1), round(y, 1), round(reg.width * ke, 1), round(reg.height * kh / (ch * f), 1)], "fit": round(s, 3)}


def place_on(base, reg, place):
    """Dán ảnh vẽ lại (RGBA) lên ảnh nền base (RGBA, toạ độ ảnh gốc) theo place = [x, y, w, h]."""
    x, y, w, h = place
    im = reg.convert("RGBA").resize((max(1, round(w)), max(1, round(h))), Image.LANCZOS)
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.paste(im, (round(x), round(y)))
    return Image.alpha_composite(base, layer)


def data_url_thumb(im, side=360):
    t = im.convert("RGBA")
    t.thumbnail((side, side), Image.LANCZOS)
    bg = checker(t.width, t.height, 10).convert("RGBA")
    return data_url(Image.alpha_composite(bg, t).convert("RGB"), "JPEG", quality=85)


def export(a):
    out = Path(a.out)
    gfile = Path(a.groups) if a.groups else (out / "groups.json" if (out / "groups.json").exists() else out / "groups.auto.json")
    groups = json.loads(gfile.read_text(encoding="utf-8"))["groups"]
    print(f"Nhóm lấy từ {gfile.name}")
    img = Image.open(out / "src.png").convert("RGB")
    rgb = np.array(img)
    W, H = img.size
    lab = np.array(Image.open(out / "parts.png").convert("RGB")).astype(np.int32)
    label = lab[..., 0] | (lab[..., 1] << 8)
    free = label == 0
    part_size = np.bincount(label.ravel())

    for d in ("elements", "copilot-crops", "layers"):
        (out / d).mkdir(exist_ok=True)
        if d == "layers" and a.no_bg:
            continue
        for f in (out / d).glob("*.png"):
            f.unlink()

    els, used, names = [], np.zeros((H, W), bool), set()
    for g in groups:
        if g.get("mode") == "bg":
            continue
        m = np.isin(label, g["parts"])
        if not m.any():
            continue
        # chỉ nở vào điểm ảnh chưa thuộc mảnh nào (khe giữa các mảnh, viền khử răng cưa), không nuốt mảnh của nhóm khác
        m |= ndimage.binary_closing(m, iterations=3) & free
        m |= ndimage.binary_fill_holes(m) & free
        m |= ndimage.binary_dilation(m, iterations=2) & free
        # khe mảnh và lỗ nhỏ nằm lọt trong element (đường nối tóc-mặt, mảnh vụn của nhóm khác) thì lấp luôn
        m = ndimage.binary_closing(np.pad(m, 4), iterations=2)[4:-4, 4:-4] | m
        holes, n = ndimage.label(ndimage.binary_fill_holes(m) & ~m)
        if n:
            hs = ndimage.sum_labels(np.ones_like(holes), holes, np.arange(1, n + 1))
            m |= np.isin(holes, np.flatnonzero(hs < 0.02 * m.sum()) + 1)
        used |= m
        ys, xs = np.nonzero(m)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        alpha = np.clip(ndimage.gaussian_filter(m.astype(np.float32), 0.7) * 255, 0, 255).astype(np.uint8)
        name = base = slug(g["name"])
        k = 2
        while name in names:
            name, k = f"{base}-{k}", k + 1
        names.add(name)
        els.append({"name": g["name"], "file": f"elements/{name}.png", "mode": g.get("mode", "keep"), "desc": (g.get("desc") or "").strip(), "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)], "area": int(m.sum()),
                    "img": Image.fromarray(np.dstack([rgb, alpha])[y0:y1, x0:x1]), "mask": m})

    # thứ tự chồng ước lượng: element lớn nằm dưới
    for z, e in enumerate(sorted(els, key=lambda e: -e["area"]), start=1):
        e["z"] = z
    for e in els:
        e["img"].save(out / e["file"])
        if e["mode"] == "regen":  # ảnh đính kèm khi gen lại: element và các mảnh nằm lọt trong thân nó, nền trơn
            crop, borrowed = copilot_crop(rgb, e["mask"], e["bbox"], label, part_size)
            crop.save(out / "copilot-crops" / Path(e["file"]).name)
            if borrowed:
                owners = sorted({g["name"] for g in groups for b in borrowed if b in g["parts"]} - {e["name"]})
                print(f"{e['name']}: có mảnh nằm lọt trong thân nhưng thuộc nhóm {', '.join(owners) or 'chưa gộp'}; "
                      f"đã đưa vào ảnh gửi Copilot. Nếu đó là một phần của {e['name']}, gộp lại ở bảng duyệt.")

    # các phương án Copilot đã vẽ: regen/<tên>.png và regen/<tên>-var<k>.png; chấm điểm khớp, chọn bản tốt nhất
    # trừ khi người duyệt đã chọn trong chon-ban-ve.html (regen-pick.json)
    pick_file = out / "regen-pick.json"
    picks = json.loads(pick_file.read_text(encoding="utf-8")).get("chon", {}) if pick_file.exists() else {}
    bg_visible = ~used
    for e in els:
        e["regen"], e["regen_place"], e["regen_fit"], e["options"] = None, None, None, []
        if e["mode"] != "regen":
            continue
        stem = Path(e["file"]).stem
        files = sorted(f for f in (out / "regen").glob(f"{stem}*.png") if re.fullmatch(rf"{re.escape(stem)}(-var\d+)?", f.stem)) if (out / "regen").is_dir() else []
        for f in files:
            reg = Image.open(f).convert("RGBA")
            fit = fit_regen(e["mask"], bg_visible, rgb, reg)
            if fit:
                e["options"].append({"file": f.name, **fit})
        e["options"].sort(key=lambda o: -o["fit"])
        if e["options"]:
            chosen = next((o for o in e["options"] if o["file"] == picks.get(stem)), e["options"][0])
            e["regen"], e["regen_place"], e["regen_fit"] = f"regen/{chosen['file']}", chosen["place"], chosen["fit"]
            how = "bản đã chọn" if picks.get(stem) == chosen["file"] else "khớp nhất"
            print(f"{e['name']}: {len(e['options'])} phương án, dùng {chosen['file']} ({how}, khớp {max(0, chosen['fit']):.0%})")

    cover = used.mean()
    bg_file = "layers/background.png" if (out / "layers" / "background.png").exists() else None
    if not a.no_bg and els:
        t = time.time()
        hole = ndimage.binary_dilation(used, iterations=6)
        Image.fromarray(lama_fill(rgb, hole)).save(out / "layers" / "background.png")
        bg_file = "layers/background.png"
        print(f"Lấp nền {time.time() - t:.0f}s (element phủ {cover:.0%} ảnh)")

    manifest = {
        "source": "src.png", "size": [W, H], "background": bg_file, "element_coverage": round(float(cover), 3),
        # regen_place: [x, y, rộng, cao] của cả ảnh vẽ lại khi đặt lên ảnh gốc; regen_fit: điểm khớp với bản gốc (1 = khớp hoàn toàn)
        "elements": [{k: e[k] for k in ("name", "file", "mode", "desc", "regen", "regen_place", "regen_fit", "z", "bbox", "area")} for e in sorted(els, key=lambda e: e["z"])],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")

    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    cell, pad, cols = 260, 28, 5
    tag = {"keep": "giữ ảnh gốc", "regen": "vẽ lại"}
    tiles = [(f"{e['name']} · {tag[e['mode']]} · {e['bbox'][2]}x{e['bbox'][3]}", e["img"]) for e in sorted(els, key=lambda e: -e["area"])]
    for e in els:
        if e["regen"]:
            tiles.append((f"{e['name']} · bản Copilot vẽ lại", Image.open(out / e["regen"]).convert("RGBA")))
    if bg_file:
        tiles.insert(0, ("nền đã lấp", Image.open(out / bg_file).convert("RGBA")))
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * cell, rows * (cell + pad)), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(tiles):
        cx, cy = (i % cols) * cell, (i // cols) * (cell + pad)
        t = im.copy()
        t.thumbnail((cell - 12, cell - 12), Image.LANCZOS)
        bgc = checker(cell - 4, cell - 4)
        bgc.paste(t, ((cell - 4 - t.width) // 2, (cell - 4 - t.height) // 2), t)
        sheet.paste(bgc, (cx + 2, cy + 2))
        d.text((cx + 6, cy + cell + 5), name[:34], fill=(235, 235, 235), font=font)
    sheet.save(out / "sheet.jpg", quality=88)

    # ghép lại cả ảnh từ nền + element (bản vẽ lại đặt theo regen_place) để soi lệch bằng mắt
    comp = Image.open(out / bg_file).convert("RGBA") if bg_file else img.convert("RGBA")
    for e in sorted(els, key=lambda e: e["z"]):
        if e["regen"]:
            comp = place_on(comp, Image.open(out / e["regen"]), e["regen_place"])
        else:
            x, y, _, _ = e["bbox"]
            layer = Image.new("RGBA", comp.size, (0, 0, 0, 0))
            layer.paste(e["img"], (x, y))
            comp = Image.alpha_composite(comp, layer)
    comp.convert("RGB").save(out / "ghep-lai.jpg", quality=90)

    # trang chọn phương án: mỗi element «Vẽ lại» một hàng, mỗi phương án một thẻ (riêng + đặt vào ảnh gốc)
    items = []
    src_rgba = img.convert("RGBA")
    for e in els:
        if not e["options"]:
            continue
        x, y, w, h = e["bbox"]
        p = int(max(w, h) * 0.25)
        box = (max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p))
        opts = []
        for i, o in enumerate(e["options"]):
            reg = Image.open(out / "regen" / o["file"])
            opts.append({"file": o["file"], "fit": o["fit"], "alone": data_url_thumb(reg), "placed": data_url_thumb(place_on(src_rgba, reg, o["place"]).crop(box))})
        items.append({"stem": Path(e["file"]).stem, "name": e["name"], "chosen": Path(e["regen"]).name, "original": data_url_thumb(src_rgba.crop(box)), "options": opts})
    pick_html = out / "chon-ban-ve.html"
    if items:
        data = {"key": f"layer-split-pick:{out.name}", "items": items}
        html = (HERE / "pick.template.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
        pick_html.write_text(html, encoding="utf-8")
    elif pick_html.exists():
        pick_html.unlink()

    n_regen = sum(e["mode"] == "regen" for e in els)
    print(f"{len(els)} element ({n_regen} cần vẽ lại, vùng cắt ở copilot-crops/). Xem {out / 'sheet.jpg'}")


def tasks(a):
    """Sinh file task + cấu hình cho bộ gen Copilot dùng chung, đặt ngay trong thư mục ra (thư mục ra đóng vai 'project')."""
    out = Path(a.out)
    els = [e for e in json.loads((out / "manifest.json").read_text(encoding="utf-8"))["elements"] if e["mode"] == "regen"]
    if not els:
        sys.exit("Không có element nào đánh dấu «Vẽ lại». Đánh dấu trong review.html rồi chạy export trước.")
    tpl = (HERE / "regen-prompt.txt").read_text(encoding="utf-8").strip()
    items = []
    for e in els:
        f = Path(e["file"]).name
        items.append({"file": f, "group": Path(f).stem, "attach": f"copilot-crops/{f}", "prompt": tpl.replace("{ten}", e.get("desc") or e["name"]), "palette": False})
    body = "// Sinh tự động bởi layer-split (split.py tasks). Sửa câu chữ ở regen-prompt.txt của tool hoặc ô «Mô tả» trong review.html.\n"
    body += "export const TASKS = " + json.dumps(items, ensure_ascii=False, indent=2) + ";\nexport const BLEED = {};\n"
    (out / "copilot-tasks.mjs").write_text(body, encoding="utf-8")
    cfg = {"tasksFile": "copilot-tasks.mjs", "outDir": "regen", "logDir": "regen/logs", "stateDir": ".gen-state", "assetsDir": "regen", "reviewDir": "regen"}
    (out / "gen.config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"{len(items)} task trong {out / 'copilot-tasks.mjs'}:")
    for t in items:
        print(f"  {t['file']}  <- {t['attach']}")
    gen = Path(os.environ.get("COPILOT_IMAGEGEN_DIR") or Path.home() / ".gemini/tools/copilot-imagegen")
    print(f'Chạy TỪ thư mục ra:  node "{(gen / "gen-parallel.mjs").as_posix()}" 3 "*" --variants 3   (thêm --dry-run để xem trước)')


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("propose")
p.add_argument("src")
p.add_argument("out")
p.add_argument("--pps", type=int, default=32, help="số điểm dò mỗi cạnh; tăng lên 48 nếu sót element nhỏ")
p.add_argument("--force", action="store_true", help="chạy lại SAM dù đã có masks.npz")
p.set_defaults(fn=propose)
p = sub.add_parser("export")
p.add_argument("out")
p.add_argument("--groups")
p.add_argument("--no-bg", action="store_true", help="bỏ bước lấp nền")
p.set_defaults(fn=export)
p = sub.add_parser("tasks")
p.add_argument("out")
p.set_defaults(fn=tasks)
args = ap.parse_args()
args.fn(args)
