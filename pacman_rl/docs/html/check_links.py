"""Kiểm tra liên kết nội bộ trong docs: file đích có tồn tại không, neo (#id) có tồn tại trong trang đích không.

    python check_links.py

Thoát với mã 1 nếu có liên kết gãy, và cũng báo các trang chưa có trong assets/nav.js (hoặc ngược lại).
Chỉ dùng thư viện chuẩn.
"""
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urldefrag

HERE = Path(__file__).resolve().parent


class Collect(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links = set(), []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        for key in ("href", "src"):
            if key in a and a[key]:
                self.links.append((tag, a[key]))


def parse(path):
    p = Collect()
    p.feed(path.read_text(encoding="utf-8"))
    return p


def main():
    pages = {p: parse(p) for p in HERE.rglob("*.html")}
    bad = 0
    for path, info in sorted(pages.items()):
        rel = path.relative_to(HERE).as_posix()
        for tag, link in info.links:
            if re.match(r"^(https?:|mailto:|data:|javascript:)", link):
                continue
            target, frag = urldefrag(link)
            dest = path if not target else (path.parent / unquote(target)).resolve()
            if not dest.exists():
                print(f"{rel}: <{tag}> trỏ tới file không tồn tại: {link}")
                bad += 1
                continue
            if frag and dest.suffix == ".html":
                dest_info = pages.get(dest) or parse(dest)
                if frag not in dest_info.ids:
                    print(f"{rel}: neo '#{frag}' không có trong {dest.relative_to(HERE).as_posix()}")
                    bad += 1

    nav = (HERE / "assets" / "nav.js").read_text(encoding="utf-8")
    in_nav = set(re.findall(r'path:\s*"([^"]+\.html)"', nav))
    on_disk = {p.relative_to(HERE).as_posix() for p in pages}
    for name in sorted(in_nav - on_disk):
        print(f"nav.js khai báo trang không tồn tại: {name}")
        bad += 1
    for name in sorted(on_disk - in_nav):
        print(f"trang chưa có trong nav.js: {name}")
        bad += 1

    print(f"Đã kiểm tra {len(pages)} trang, {bad} vấn đề.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
