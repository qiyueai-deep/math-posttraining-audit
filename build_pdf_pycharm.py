"""PyCharm 右键入口：把 reports/tech_report.html 打印成 PDF。

优先用本机 Chrome 的无头模式。找不到 Chrome 时给出提示，不静默失败。
HTML 由 src/core/tech_report.py 从 results.jsonl 生成，所以数据变了要先跑
run_pycharm.py（research 阶段），再右键本文件更新 PDF。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
]


def find_browser() -> Path | None:
    for p in CANDIDATES:
        if p and p.is_file():
            return p
    return None


def main() -> int:
    html = ROOT / "reports" / "tech_report.html"
    pdf = ROOT / "reports" / "tech_report.pdf"
    if not html.exists():
        print("缺少 reports/tech_report.html，请先运行 run_pycharm.py（research 阶段）。")
        return 1
    browser = find_browser()
    if browser is None:
        print("未找到 Chrome 或 Edge。可以直接用浏览器打开下面这个文件，")
        print("在打印对话框里选择「另存为 PDF」，纸张 A4、边距默认：")
        print(" ", html)
        return 1
    cmd = [
        str(browser), "--headless=new", "--disable-gpu", "--no-sandbox",
        "--hide-scrollbars", "--no-pdf-header-footer", "--virtual-time-budget=8000",
        f"--user-data-dir={Path(os.environ.get('TEMP', '.')) / 'wb-tech-report'}",
        f"--print-to-pdf={pdf}", html.as_uri(),
    ]
    print("浏览器:", browser)
    print("输出  :", pdf, flush=True)
    rc = subprocess.run(cmd, capture_output=True, text=True).returncode
    if rc != 0 or not pdf.exists():
        print("打印失败，退出码", rc)
        return 1
    blob = pdf.read_bytes()
    pages = len(re.findall(rb"/Type\s*/Page[^s]", blob))
    print(f"完成：{len(blob)} bytes，{pages} 页")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
