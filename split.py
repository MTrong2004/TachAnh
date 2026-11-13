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
                    "img": Image.fromarray(np.dstack([rgb, alpha])[y0:y1, x0:x1])})

    # thứ tự chồng ước lượng: element lớn nằm dưới
    for z, e in enumerate(sorted(els, key=lambda e: -e["area"]), start=1):
        e["z"] = z
    for e in els:
        e["img"].save(out / e["file"])
        if e["mode"] == "regen":  # vùng cắt chữ nhật của ảnh gốc để đính kèm khi gen lại
            x, y, w, h = e["bbox"]
            p = int(max(w, h) * 0.04)
            crop = img.crop((max(0, x - p), max(0, y - p), min(W, x + w + p), min(H, y + h + p)))
            s = max(1, round(1024 / max(crop.size)))
            crop.resize((crop.width * s, crop.height * s), Image.LANCZOS).save(out / "copilot-crops" / Path(e["file"]).name)

    for e in els:
        r = out / "regen" / Path(e["file"]).name
        e["regen"] = f"regen/{r.name}" if e["mode"] == "regen" and r.exists() else None

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
        "elements": [{k: e[k] for k in ("name", "file", "mode", "desc", "regen", "z", "bbox", "area")} for e in sorted(els, key=lambda e: e["z"])],
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
    print(f'Chạy TỪ thư mục ra:  node "{(gen / "gen-parallel.mjs").as_posix()}" --dry-run')


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
