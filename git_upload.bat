@echo off
REM GitHub仓库初始化和上传脚本 (Windows版本)
REM 使用方法: git_upload.bat

echo ==========================================
echo GitHub仓库初始化和上传
echo ==========================================

REM 进入项目目录
cd /d "E:\2026Data\1019\github"

REM 检查是否已经初始化
if exist ".git" (
    echo Git仓库已存在
) else (
    echo 初始化Git仓库...
    git init
    echo √ Git仓库初始化完成
)

REM 添加所有文件
echo.
echo 添加文件到暂存区...
git add .

REM 查看状态
echo.
echo 当前状态:
git status --short

REM 提交
echo.
echo 创建初始提交...
git commit -m "Initial commit: SAR-Flood-Surrogate project" -m "- Add complete project structure" -m "- Add training code and model definitions" -m "- Add evaluation results and supplementary analysis" -m "- Add documentation (README, LICENSE, requirements.txt)" -m "- Add supplementary figures (S1-S4)" -m "- Add result tables and metrics"

echo.
echo √ 提交完成

REM 显示提交信息
echo.
echo 提交详情:
git log --oneline -1

REM 文件统计
echo.
echo 文件统计:
git ls-files | find /c /v ""
echo 个文件已提交

echo.
echo ==========================================
echo 下一步操作:
echo ==========================================
echo.
echo 1. 在GitHub上创建新仓库:
echo    https://github.com/new
echo    仓库名建议: SAR-Flood-Surrogate
echo.
echo 2. 添加远程仓库 (替换YOUR_USERNAME):
echo    git remote add origin https://github.com/YOUR_USERNAME/SAR-Flood-Surrogate.git
echo.
echo 3. 推送到GitHub:
echo    git branch -M main
echo    git push -u origin main
echo.
echo 4. (可选) 如果使用SSH:
echo    git remote set-url origin git@github.com:YOUR_USERNAME/SAR-Flood-Surrogate.git
echo.
echo 5. 输入GitHub用户名和密码 (或使用Personal Access Token)
echo.
echo ==========================================
echo.
pause
