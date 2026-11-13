@echo off
rem Tach anh phang thanh tung element. Bam dup de mo menu, hoac keo tha mot anh len file nay de tach ngay.
chcp 65001 >nul
setlocal EnableExtensions
title Tách ảnh thành element
cd /d "%~dp0"
set "ROOT=%~dp0"
set "PY=%ROOT%.venv\Scripts\python.exe"
set "OUTROOT=%ROOT%ket-qua"
rem Bo gen anh Copilot dung chung; dat bien COPILOT_IMAGEGEN_DIR de doi cho khac
set "GEN=%USERPROFILE%\.gemini\tools\copilot-imagegen"
if defined COPILOT_IMAGEGEN_DIR set "GEN=%COPILOT_IMAGEGEN_DIR%"
rem So phuong an Copilot ve cho moi element (1-8); dat bien TACHANH_VARIANTS de doi
set "VARIANTS=3"
if defined TACHANH_VARIANTS set "VARIANTS=%TACHANH_VARIANTS%"
set "PYTHONIOENCODING=utf-8"
set "PYTHONWARNINGS=ignore"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
if not exist "%PY%" echo Không thấy môi trường Python ở "%PY%". & pause & exit /b 1
if not exist "%OUTROOT%" mkdir "%OUTROOT%"
set "LAST="
set "EMPTY=0"
if exist "%OUTROOT%\.last" set /p LAST=<"%OUTROOT%\.last"

rem Chay thang khong qua menu:  TachAnh.bat xuat <ten>  |  TachAnh.bat copilot <ten>  |  TachAnh.bat <anh>
set "BATCH="
rem "mo <ten>": mo menu voi thu muc ket qua do la thu muc dang lam (file bat nho trong tung thu muc ket qua goi kieu nay)
if /i "%~1"=="mo" set "LAST=%~2" & goto menu
if /i "%~1"=="xuat" set "BATCH=1" & set "NAME=%~2" & goto export
if /i "%~1"=="copilot" set "BATCH=1" & set "NAME=%~2" & goto copilot
set "SRC=%~1"
if defined SRC goto propose

:menu
if defined BATCH exit /b 0
echo.
echo ================= TÁCH ẢNH THÀNH ELEMENT =================
echo   1. Tách ảnh mới            (tìm mảnh, mở bảng duyệt)
echo   2. Xuất element            (sau khi đã lưu file nhóm ở bảng duyệt)
echo   3. Vẽ lại bằng Copilot     (các element đánh dấu «Vẽ lại»)
echo   4. Mở lại bảng duyệt
echo   5. Mở thư mục kết quả
echo   6. Chọn bản vẽ lại          (so các phương án Copilot vẽ, lưu lựa chọn rồi chọn 2)
echo   0. Thoát
if defined LAST echo   Đang làm: %LAST%
echo ===========================================================
set "C="
set /p "C=Chọn số: "
if "%C%"=="1" goto ask_src
if "%C%"=="2" goto export
if "%C%"=="3" goto copilot
if "%C%"=="4" goto review
if "%C%"=="5" start "" "%OUTROOT%" & goto menu
if "%C%"=="6" goto pickregen
if "%C%"=="0" exit /b 0
rem Khong co dau vao (cua so bi dong, chay tu script): thoat sau 3 lan trong lien tiep, khong lap vo han
if defined C set "EMPTY=0"
if not defined C set /a EMPTY+=1
if %EMPTY% geq 3 exit /b 0
goto menu

:ask_src
set "SRC="
set /p "SRC=Kéo thả ảnh vào đây rồi Enter: "
if not defined SRC goto menu
set "SRC=%SRC:"=%"

:propose
if not exist "%SRC%" echo Không thấy ảnh "%SRC%". & goto menu
for %%F in ("%SRC%") do set "NAME=%%~nF"
set "NAME=%NAME: =-%"
set "LAST=%NAME%"
>"%OUTROOT%\.last" echo %LAST%
echo.
echo Đang tìm mảnh trong ảnh (khoảng 2 phút)...
"%PY%" "%ROOT%split.py" propose "%SRC%" "%OUTROOT%\%NAME%"
if errorlevel 1 echo Lỗi ở bước tìm mảnh. & pause & goto menu
echo.
echo Bảng duyệt đang mở trên trình duyệt. Gộp mảnh xong bấm «Lưu file nhóm», rồi quay lại đây chọn 2.
if not defined TACHANH_NO_OPEN start "" "%OUTROOT%\%NAME%\review.html"
goto menu

:pick
rem Hoi ten thu muc ket qua; Enter de dung thu muc dang lam
if defined BATCH goto pick_check
echo.
echo Các thư mục kết quả:
dir /b /ad "%OUTROOT%"
set "NAME=%LAST%"
set /p "NAME=Tên thư mục [%LAST%]: "
:pick_check
if not defined NAME exit /b 1
if not exist "%OUTROOT%\%NAME%\parts.png" echo "%NAME%" chưa qua bước 1. & exit /b 1
set "LAST=%NAME%"
>"%OUTROOT%\.last" echo %LAST%
exit /b 0

:export
call :pick
if errorlevel 1 goto menu
set "DL=%USERPROFILE%\Downloads\groups.json"
if defined BATCH goto export_run
if not exist "%DL%" goto export_run
choice /c CK /n /m "Thấy groups.json trong Downloads. Dùng file này cho %NAME%? (C = có, K = không): "
if errorlevel 2 goto export_run
move /y "%DL%" "%OUTROOT%\%NAME%\groups.json" >nul
:export_run
set "DLP=%USERPROFILE%\Downloads\regen-pick.json"
if defined BATCH goto export_go
if not exist "%DLP%" goto export_go
choice /c CK /n /m "Thấy regen-pick.json (bản vẽ lại đã chọn) trong Downloads. Dùng cho %NAME%? (C = có, K = không): "
if errorlevel 2 goto export_go
move /y "%DLP%" "%OUTROOT%\%NAME%\regen-pick.json" >nul
:export_go
if not exist "%OUTROOT%\%NAME%\groups.json" echo Chưa có groups.json, dùng nhóm gợi ý tự động.
"%PY%" "%ROOT%split.py" export "%OUTROOT%\%NAME%"
if errorlevel 1 echo Lỗi ở bước xuất. & pause & goto menu
if not defined TACHANH_NO_OPEN start "" "%OUTROOT%\%NAME%\sheet.jpg"
goto menu

:copilot
call :pick
if errorlevel 1 goto menu
"%PY%" "%ROOT%split.py" tasks "%OUTROOT%\%NAME%"
if errorlevel 1 pause & goto menu
pushd "%OUTROOT%\%NAME%"
echo.
call node "%GEN%\gen-parallel.mjs" "*" --variants %VARIANTS% --dry-run
echo.
if defined BATCH goto copilot_run
choice /c CK /n /m "Gửi các ảnh trên sang Copilot (%VARIANTS% phương án mỗi element)? (C = có, K = không): "
if errorlevel 2 popd & goto menu
:copilot_run
call node "%GEN%\gen-parallel.mjs" 3 "*" --variants %VARIANTS%
popd
"%PY%" "%ROOT%split.py" export "%OUTROOT%\%NAME%" --no-bg
echo Ảnh Copilot vẽ lại nằm ở "%OUTROOT%\%NAME%\regen".
echo Trang so sánh đang mở: chọn bản giống nhất cho từng element, bấm «Lưu lựa chọn», rồi quay lại đây chọn 2.
if not defined TACHANH_NO_OPEN if exist "%OUTROOT%\%NAME%\chon-ban-ve.html" start "" "%OUTROOT%\%NAME%\chon-ban-ve.html"
goto menu

:pickregen
call :pick
if errorlevel 1 goto menu
if not exist "%OUTROOT%\%NAME%\chon-ban-ve.html" echo "%NAME%" chưa có bản vẽ lại nào. Chọn 3 trước. & goto menu
if not defined TACHANH_NO_OPEN start "" "%OUTROOT%\%NAME%\chon-ban-ve.html"
echo Chọn xong bấm «Lưu lựa chọn», rồi chọn 2 để xuất lại với bản đã chọn.
goto menu

:review
call :pick
if errorlevel 1 goto menu
start "" "%OUTROOT%\%NAME%\review.html"
goto menu
