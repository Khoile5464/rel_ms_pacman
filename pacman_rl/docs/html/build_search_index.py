"""Sinh hai file dữ liệu cho docs (chỉ dùng thư viện chuẩn):

    python build_search_index.py

- assets/search-index.js: chỉ mục cho ô tìm kiếm, lấy từ các trang .html. Mỗi h2/h3 có id là một kết quả;
  thiếu id thì in cảnh báo.
- assets/source-index.js: bản đồ "tên -> (file, dòng)" quét từ src/pacman_rl/. docs.js dùng nó để biến mọi
  <code> trùng tên hàm / lớp / hằng / trường config / đường dẫn file thành link mở thẳng trong VS Code
  (chỉ khi mở docs bằng file://). Chạy lại sau khi code đổi để số dòng đúng.
"""
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "src" / "pacman_rl"
MAX_TEXT = 600


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.entries = []          # (heading, anchor, [text...])
        self._cur = ["", "", []]   # phần mở đầu trước h2 đầu tiên
        self._in = None            # thẻ tiêu đề đang đọc
        self._id = ""
        self._buf = []
        self._skip = 0
        self._in_article = False
        self.missing_ids = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "style"):
            self._skip += 1
        elif tag == "article":
            self._in_article = True
        elif self._in_article and tag in ("h1", "h2", "h3"):
            self._in, self._buf = tag, []
            self._id = a.get("id", "")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip -= 1
        elif tag == "article":
            self._in_article = False
        elif tag == self._in:
            text = " ".join("".join(self._buf).split())
            if tag == "h1":
                self.title = text
            else:
                self.entries.append(self._cur)
                if not self._id:
                    self.missing_ids.append(text)
                self._cur = [text, self._id, []]
            self._in = None

    def handle_data(self, data):
        if self._skip or not self._in_article:
            return
        if self._in:
            self._buf.append(data)
        else:
            self._cur[2].append(data)

    def close(self):
        super().close()
        self.entries.append(self._cur)


def build_source_index():
    """Tên (hàm, lớp, hằng, Lớp.thành_viên, đường dẫn file) -> {"f": file tương đối so với src/pacman_rl, "l": dòng}."""
    index = {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        index.setdefault(rel, {"f": rel, "l": 1})
        cls = None
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = re.match(r"(class|def)\s+(\w+)", line)
            if m:
                index.setdefault(m.group(2), {"f": rel, "l": n})
                cls = m.group(2) if m.group(1) == "class" else None
                continue
            m = re.match(r"([A-Z][A-Z0-9_]+)\s*=", line)
            if m:
                index.setdefault(m.group(1), {"f": rel, "l": n})
            if line[:1].strip() and not line.startswith(("@", "#", ")", "]", "}")):
                cls = None                      # hết thân lớp
            elif cls:
                m = re.match(r"    (?:def\s+)?(\w+)\s*[:(]", line)
                if m:
                    index.setdefault(f"{cls}.{m.group(1)}", {"f": rel, "l": n})
    target = HERE / "assets" / "source-index.js"
    target.write_text("window.DOCS_SOURCES = " + json.dumps(index, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"Đã ghi {len(index)} mục vào {target.relative_to(HERE)}")


def main():
    build_source_index()
    out = []
    for path in sorted(HERE.rglob("*.html")):
        rel = path.relative_to(HERE).as_posix()
        p = PageParser()
        p.feed(path.read_text(encoding="utf-8"))
        p.close()
        for name in p.missing_ids:
            print(f"cảnh báo: {rel}: tiêu đề '{name}' thiếu id", file=sys.stderr)
        for heading, anchor, text in p.entries:
            body = " ".join(" ".join(text).split())[:MAX_TEXT]
            if not heading and not body:
                continue
            out.append({"p": rel, "t": p.title, "h": heading, "a": anchor, "x": body})
    target = HERE / "assets" / "search-index.js"
    target.write_text("window.DOCS_SEARCH = " + json.dumps(out, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"Đã ghi {len(out)} mục vào {target.relative_to(HERE)}")


if __name__ == "__main__":
    main()
