---
type: lesson
date: 2026-11-14
title: "Ảnh đính kèm Copilot: giữ mảnh nằm lọt trong bao lồi element"
files: ["README.md", "TachAnh.bat", "split.py", "pick.template.html"]
tags: [lesson, memory]
---

# Ảnh đính kèm Copilot: giữ mảnh nằm lọt trong bao lồi element

## Tóm tắt
Loa vẽ lại có đai vàng tưởng là Copilot bịa, thực ra poster gốc có đai; mảnh đai bị gộp nhầm vào nhóm tieu-de nên bản cắt riêng mất nó. Gửi chỉ bản cắt thì Copilot vẽ thiếu; gửi vùng chữ nhật thì lọt vật bên cạnh.

## File liên quan
- `README.md`
- `TachAnh.bat`
- `split.py`
- `pick.template.html`

## Bài học
- Trước khi kết luận Copilot vẽ sai, so với ảnh gốc chứ không so với bản cắt. copilot_crop giữ element + mảnh nhóm khác nằm >=80% trong bao lồi, bỏ phần ngoài; export in tên nhóm bị gộp nhầm.
