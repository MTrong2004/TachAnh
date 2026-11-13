# TachAnh — tách ảnh phẳng thành element PNG

Tách poster/banner đã gộp lớp thành từng element PNG trong suốt, có người duyệt ở giữa.
Trạng thái (2026-11-14): bản thử, đã chạy trọn quy trình trên một poster 1080px.

## Cài đặt (Windows)

Cần Python 3.11. Máy không cần GPU, mọi thứ chạy trên CPU.

```bash
git clone https://github.com/MTrong2004/TachAnh.git
cd TachAnh
py -3.11 -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
```

- Model SAM (`facebook/sam-vit-base`) tự tải từ Hugging Face ở lần chạy đầu.
- Model LaMa để lấp nền phải tải tay một lần về `~/.cache/torch/hub/checkpoints/big-lama.pt`:

```bash
curl -L --create-dirs -o ~/.cache/torch/hub/checkpoints/big-lama.pt https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
```

- Bước «Vẽ lại bằng Copilot» cần thêm bộ gen ảnh Copilot dùng chung (Node.js), mặc định ở `~/.gemini/tools/copilot-imagegen`.
  Để chỗ khác thì đặt biến môi trường `COPILOT_IMAGEGEN_DIR`. Không có bộ này thì vẫn tách và xuất element bình thường.
- Thư mục `ket-qua/` không đưa lên git vì chứa ảnh nguồn.

## Cách nhanh nhất: TachAnh.bat

- Bấm đúp `TachAnh.bat` để mở menu, hoặc **kéo thả một ảnh lên file bat** để tách ngay.
- Menu: 1 tách ảnh mới → 2 xuất element → 3 vẽ lại bằng Copilot → 6 chọn bản vẽ lại → 2 xuất lại; 4 mở lại bảng duyệt; 5 mở thư mục kết quả.
- Bước 2 tự hỏi có lấy `groups.json` và `regen-pick.json` vừa tải về trong Downloads không, khỏi chép tay.
- Bước 3 cho Copilot vẽ 3 phương án mỗi element; đổi số bằng biến môi trường `TACHANH_VARIANTS` (1–8).
- Kết quả nằm ở `ket-qua/<tên ảnh>/`. Bat nhớ ảnh đang làm, bấm Enter là dùng lại.
- Chạy thẳng không qua menu: `TachAnh.bat xuat <tên>`, `TachAnh.bat copilot <tên>` (không hỏi lại, gửi Copilot luôn).

## Quy trình (lệnh gốc mà bat gọi)

```bash
cd TachAnh
.venv/Scripts/python split.py propose <ảnh vào> <thư mục ra>     # ~2 phút trên CPU
# mở <thư mục ra>/review.html, gộp mảnh thành element, đặt tên, chọn cách xử lý, bấm "Lưu file nhóm"
# chép groups.json vừa tải về vào <thư mục ra>
.venv/Scripts/python split.py export <thư mục ra>                # ~20 giây
```

1. **propose**: SAM (`facebook/sam-vit-base`) tìm mọi mảnh; vùng SAM bỏ sót được chia theo màu để vẫn bấm được.
   Gộp sơ bộ: mảnh con theo mask cha nhỏ, các mảnh nhỏ nằm sát nhau (chữ, logo) thành một nhóm.
   Kết quả lưu ở `masks.npz` (chạy lại không tốn thời gian SAM, trừ khi thêm `--force`), `parts.png`, `groups.auto.json`, `review.html`.
2. **review.html**: một file tự chứa, mở thẳng bằng trình duyệt. Bấm vào ảnh để chọn nhóm, `G` để gộp, `Esc` bỏ chọn.
   «Chọn từng mảnh» để lấy một mảnh ra khỏi nhóm. Mỗi nhóm chọn một cách xử lý:
   *Giữ ảnh gốc* (cắt đúng pixel), *Vẽ lại* (cắt pixel + xuất vùng cắt để gen lại), *Là nền* (không xuất).
   Việc đang làm dở được trình duyệt nhớ lại.
3. **export**: `elements/<tên>.png`, `layers/background.png` (nền lấp bằng LaMa), `manifest.json` (vị trí, thứ tự chồng ước lượng),
   `sheet.jpg` (bảng xem), `ghep-lai.jpg` (ghép lại cả ảnh từ nền + element, bản vẽ lại đặt đúng chỗ, để soi lệch),
   `copilot-crops/<tên>.png` (ảnh đính kèm khi gen bằng Copilot, xem dưới).
   Không có `groups.json` thì dùng nhóm gợi ý tự động.

## Vẽ lại bằng Copilot

```bash
.venv/Scripts/python split.py tasks <thư mục ra>      # sinh copilot-tasks.mjs + gen.config.json ngay trong thư mục ra
cd <thư mục ra>
node ~/.gemini/tools/copilot-imagegen/gen-parallel.mjs "*" --variants 3 --dry-run
COPILOT_WORKER=auto node ~/.gemini/tools/copilot-imagegen/gen-assets-v2.mjs <tên>.png   # thử 1 ảnh trước
node ~/.gemini/tools/copilot-imagegen/gen-parallel.mjs 3 "*" --variants 3              # cả lô, 3 phương án mỗi element
cd <thư mục TachAnh>
.venv/Scripts/python split.py export <thư mục ra> --no-bg   # chấm điểm, đặt bản vẽ lại vào chỗ, sinh chon-ban-ve.html
# mở <thư mục ra>/chon-ban-ve.html, chọn bản cho từng element, «Lưu lựa chọn», chép regen-pick.json vào <thư mục ra>
.venv/Scripts/python split.py export <thư mục ra>
```

- Mỗi element «Vẽ lại» thành một task: đính kèm `copilot-crops/<tên>.png`, ảnh ra `regen/<tên>-var<k>.png` (và `regen/<tên>.png` nếu gen không kèm `--variants`).
- **Ảnh đính kèm**: element cùng các mảnh khác nằm lọt (≥ 80% diện tích) trong bao lồi của nó; phần còn lại thay bằng nền trơn
  (trắng / xám, chọn màu khác xa màu của vật). Mảnh nằm lọt thường là phần của vật bị gộp nhầm sang nhóm khác: export in tên nhóm đó
  để sửa ở bảng duyệt. Chữ, giấy nền, vật bên cạnh không lọt vào ảnh nên Copilot không vẽ thêm chúng.
- **Đặt vào chỗ**: export tìm tỉ lệ + vị trí để bản vẽ lại khớp nhất với phần element nhìn thấy trong ảnh gốc (phủ đúng chỗ, đúng màu,
  không tràn ra nền; phần bị che không bị trừ điểm). Kết quả ở `manifest.json`: `regen` (file đang dùng), `regen_place`
  (`[x, y, rộng, cao]` của cả ảnh vẽ lại trên ảnh gốc, giữ nguyên độ phân giải cao của bản vẽ), `regen_fit` (điểm khớp, 1 = khớp hoàn toàn).
- **Chọn phương án**: mặc định dùng bản khớp nhất. `chon-ban-ve.html` hiện từng phương án vẽ riêng và đặt vào ảnh gốc cạnh bản gốc;
  lựa chọn lưu ở `regen-pick.json` (`{"chon": {"<tên>": "<tên>-var2.png"}}`). Điểm khớp chỉ để tham khảo: bản đúng của cái loa được ~72%
  vì màu giấy và phần bị che; bản lật ngang chỉ ~0%.
- Câu lệnh gửi Copilot lấy từ `regen-prompt.txt` (chủ dự án sửa ở đó); `{ten}` thay bằng ô «Mô tả» trong bảng duyệt, không có thì dùng tên element.
- Thư mục ra đóng vai "project" của bộ gen, nên phải `cd` vào đó trước khi chạy.
- Không gửi ảnh người thật, logo, chữ sang Copilot: để chế độ «Giữ ảnh gốc».
- Đã thử 2026-11-14: cái loa của poster ra PNG trong suốt thật, đủ cả phần bản cắt pixel bị thiếu (đai vàng, tay cầm bị gộp nhầm vào tiêu đề).
  Kiểm tra đặt vào chỗ: cùng bản vẽ thêm lề và thu nhỏ 70% vẫn đặt về đúng chỗ, lệch dưới 0,3px.

## Giới hạn đã thấy

- SAM cắt vụn (tóc, mặt, tay riêng): bước duyệt là bắt buộc, gợi ý tự động chưa đủ để dùng ngay.
- Element bị vật khác che thì thiếu phần bị che (cắt đúng pixel không bịa thêm) → đánh dấu «Vẽ lại».
- Nền lấp bằng LaMa mờ nhoè khi element phủ quá nửa ảnh (poster thử: 62%) → nền cũng nên gen lại.
- Mép tóc trên nền màu còn dính viền màu nền; mép là mask nhị phân làm mềm 0,7px, chưa phải alpha matting.
- Thứ tự chồng trong manifest chỉ ước lượng theo diện tích (lớn nằm dưới).
- Nền giấy nhàu bị chia thành nhiều mảnh theo sắc độ; đánh dấu «Là nền» hoặc gộp tay.

## Môi trường

- `.venv` riêng, Python 3.11, PyTorch bản CPU; danh sách gói ở `requirements.txt`.
- Giữ `transformers` 4.x (`>=4.44,<4.50`).
- Model: `facebook/sam-vit-base` (Hugging Face), `~/.cache/torch/hub/checkpoints/big-lama.pt`.

## Đã thử và bỏ

- **LayerD** (`cyberagent/layerd-birefnet`): tách tốt thiết kế phẳng nền trơn, nhưng trên poster cắt dán có giấy nhàu và ảnh chụp
  chỉ bóc được vài logo. Script không còn dùng gói `layerd` (trước đây LaMa được tải về nhờ nó; giờ tải tay theo mục Cài đặt).
  Model 844MB ở `~/.cache/huggingface/hub/models--cyberagent--layerd-birefnet` có thể xoá.
- **rembg** `isnet-anime`, `u2netp`: chỉ bắt được người mẫu.
- **Qwen-Image-Layered**: cần VRAM lớn hơn 4GB của máy.
