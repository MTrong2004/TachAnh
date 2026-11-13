---
type: decision
date: 2026-11-14
title: "File .bat phải giữ CRLF; ket-qua/ không lên git"
files: ["TachAnh.bat", "split.py"]
tags: [decision, memory]
---

# File .bat phải giữ CRLF; ket-qua/ không lên git

## Tóm tắt
TachAnh.bat và file bat nhỏ do split.py sinh ra phải xuống dòng CRLF; .gitattributes ép *.bat eol=crlf. Chuỗi bat trong split.py viết bằng escape \r\n, không xuống dòng thô trong literal. ket-qua/ bị .gitignore vì chứa ảnh nguồn (người thật, logo).

## File liên quan
- `TachAnh.bat`
- `split.py`

## Quyết định
- Bat dùng CRLF qua .gitattributes; đường dẫn bộ gen Copilot lấy từ %USERPROFILE% hoặc biến COPILOT_IMAGEGEN_DIR, không ghi cứng thư mục người dùng.
