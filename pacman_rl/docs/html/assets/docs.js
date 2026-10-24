// Giao diện docs: thanh trên, sidebar, mục lục phải, tìm kiếm, tab, nút copy, tô màu code, chuyển sáng/tối.
// Không phụ thuộc thư viện ngoài nên mở thẳng file .html (file://) vẫn chạy.
(function () {
  "use strict";
  var body = document.body;
  var root = body.getAttribute("data-root") || "./";
  var current = body.getAttribute("data-page") || "";
  var NAV = window.DOCS_NAV || [];
  var INDEX = window.DOCS_SEARCH || [];

  function h(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }
  function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } }

  /* ---------- theme ---------- */
  var themeBtn;
  function applyTheme(t) {
    if (t) document.documentElement.setAttribute("data-theme", t);
    var dark = document.documentElement.getAttribute("data-theme") === "dark" ||
      (!document.documentElement.getAttribute("data-theme") && window.matchMedia("(prefers-color-scheme: dark)").matches);
    if (themeBtn) themeBtn.textContent = dark ? "☀" : "☾";
    return dark;
  }

  /* ---------- top bar ---------- */
  var logo = '<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="15" fill="#f2b705"/>' +
    '<path d="M16 16 L31 8 A15 15 0 0 1 31 24 Z" fill="var(--bg)"/><circle cx="14" cy="9" r="2" fill="#1b1f27"/></svg>';
  var header = h("header", "topbar",
    '<button class="menu-btn" type="button" aria-label="Mở menu">☰</button>' +
    '<a class="brand" href="' + root + 'index.html">' + logo + '<span>pacman_rl</span><small>v' + (window.DOCS_VERSION || "") + '</small></a>' +
    '<div class="search"><input type="search" placeholder="Tìm trong tài liệu…  ( / )" aria-label="Tìm kiếm" autocomplete="off"><div class="results" hidden></div></div>' +
    '<button class="theme-btn" type="button" aria-label="Đổi giao diện sáng/tối"></button>');
  body.insertBefore(header, body.firstChild);
  themeBtn = header.querySelector(".theme-btn");
  applyTheme();
  themeBtn.addEventListener("click", function () {
    var t = applyTheme() ? "light" : "dark";
    applyTheme(t); store("pacman-docs-theme", t);
  });
  header.querySelector(".menu-btn").addEventListener("click", function () { body.classList.toggle("nav-open"); });
  document.addEventListener("click", function (e) {
    if (body.classList.contains("nav-open") && !e.target.closest(".sidebar") && !e.target.closest(".menu-btn")) body.classList.remove("nav-open");
  });

  /* ---------- sidebar ---------- */
  var sidebar = document.getElementById("sidebar");
  var flat = [];
  if (sidebar) {
    NAV.forEach(function (sec) {
      var box = h("div", "sec", "<h4>" + esc(sec.section) + "</h4>");
      sec.pages.forEach(function (p) {
        flat.push({ path: p.path, title: p.title });
        var a = h("a", p.path === current ? "active" : "", esc(p.title));
        a.href = root + p.path;
        box.appendChild(a);
      });
      sidebar.appendChild(box);
    });
    var act = sidebar.querySelector("a.active");
    if (act && act.scrollIntoView) { try { act.scrollIntoView({ block: "center" }); window.scrollTo(0, 0); } catch (e) {} }
  }

  /* ---------- tiêu đề: neo, mục lục ---------- */
  var article = document.querySelector("article.doc");
  var heads = article ? [].slice.call(article.querySelectorAll("h2, h3")) : [];
  var toc = document.getElementById("toc");
  if (article && toc && heads.length > 1) {
    toc.appendChild(h("h4", "", "Trên trang này"));
    heads.forEach(function (hd) {
      if (!hd.id) return;
      var code = hd.querySelector("code");
      var label = (hd.tagName === "H3" && code) ? code.textContent : hd.textContent.replace(/#$/, "").trim();
      var a = h("a", hd.tagName === "H3" ? "l3" : "l2", esc(label));
      a.href = "#" + hd.id;
      toc.appendChild(a);
    });
  }
  heads.forEach(function (hd) {
    if (!hd.id) return;
    var a = h("a", "anchor", "#"); a.href = "#" + hd.id; a.setAttribute("aria-label", "Liên kết tới mục này");
    hd.appendChild(a);
  });
  if (toc && "IntersectionObserver" in window) {
    var links = {};
    [].forEach.call(toc.querySelectorAll("a"), function (a) { links[a.getAttribute("href").slice(1)] = a; });
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting && links[en.target.id]) {
          [].forEach.call(toc.querySelectorAll("a.active"), function (x) { x.classList.remove("active"); });
          links[en.target.id].classList.add("active");
        }
      });
    }, { rootMargin: "-70px 0px -70% 0px" });
    heads.forEach(function (hd) { if (hd.id) io.observe(hd); });
  }

  /* ---------- prev / next, footer ---------- */
  if (article) {
    var i = -1;
    flat.forEach(function (p, k) { if (p.path === current) i = k; });
    var pager = h("nav", "pager");
    if (i > 0) pager.appendChild(h("a", "prev", "<small>← Trước</small>" + esc(flat[i - 1].title))).href = root + flat[i - 1].path;
    if (i >= 0 && i < flat.length - 1) pager.appendChild(h("a", "next", "<small>Tiếp →</small>" + esc(flat[i + 1].title))).href = root + flat[i + 1].path;
    if (pager.children.length) article.appendChild(pager);
    article.appendChild(h("div", "foot", "pacman_rl v" + (window.DOCS_VERSION || "") + " · tài liệu viết theo mã nguồn ở <code>src/pacman_rl/</code>. Thấy chỗ nào sai lệch với code thì sửa file HTML tương ứng."));
  }

  /* ---------- tô màu code ---------- */
  var PY_KW = "def class import from as return if elif else for while in not and or is None True False with try except finally raise lambda yield pass break continue assert global nonlocal del".split(" ");
  function tokenize(code, lang) {
    var re;
    if (lang === "python" || lang === "py") {
      re = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|[fFrRbB]{0,2}"(?:\\.|[^"\\\n])*"|[fFrRbB]{0,2}'(?:\\.|[^'\\\n])*')|(@[A-Za-z_][\w.]*)|(\b\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|(\b[A-Za-z_]\w*\b)(?=(\())?/g;
    } else if (lang === "bash" || lang === "sh" || lang === "powershell") {
      re = /(#[^\n]*)|("(?:\\.|[^"\\\n])*"|'[^'\n]*')|(\s--?[A-Za-z][\w-]*)|(\$[A-Za-z_]\w*)/g;
    } else { return esc(code); }
    var out = "", last = 0, m;
    while ((m = re.exec(code)) !== null) {
      out += esc(code.slice(last, m.index)); last = re.lastIndex;
      var t = m[0];
      if (lang === "python" || lang === "py") {
        if (m[1]) out += '<span class="tok-com">' + esc(t) + "</span>";
        else if (m[2]) out += '<span class="tok-str">' + esc(t) + "</span>";
        else if (m[3]) out += '<span class="tok-fn">' + esc(t) + "</span>";
        else if (m[4]) out += '<span class="tok-num">' + esc(t) + "</span>";
        else if (m[5] && PY_KW.indexOf(m[5]) >= 0) out += '<span class="tok-kw">' + esc(t) + "</span>";
        else if (m[5] && m[6]) out += '<span class="tok-fn">' + esc(m[5]) + "</span>";
        else out += esc(t);
      } else {
        if (m[1]) out += '<span class="tok-com">' + esc(t) + "</span>";
        else if (m[2]) out += '<span class="tok-str">' + esc(t) + "</span>";
        else if (m[3]) out += esc(t.charAt(0)) + '<span class="tok-flag">' + esc(t.slice(1)) + "</span>";
        else out += '<span class="tok-num">' + esc(t) + "</span>";
      }
    }
    return out + esc(code.slice(last));
  }
  var LABEL = { python: "Python", py: "Python", bash: "Terminal", sh: "Terminal", powershell: "PowerShell", text: "Kết quả", json: "JSON" };
  [].forEach.call(document.querySelectorAll("pre > code[class^='language-']"), function (code) {
    var lang = code.className.replace("language-", "");
    var raw = code.textContent.replace(/\n$/, "");
    code.innerHTML = tokenize(raw, lang);
    var pre = code.parentNode;
    var box = h("div", "codeblock");
    var bar = h("div", "bar", "<span>" + (LABEL[lang] || lang) + "</span>");
    var btn = h("button", "", "Copy"); btn.type = "button";
    btn.addEventListener("click", function () {
      var done = function () { btn.textContent = "Đã copy"; setTimeout(function () { btn.textContent = "Copy"; }, 1400); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(raw).then(done, function () {});
      else { var ta = document.createElement("textarea"); ta.value = raw; document.body.appendChild(ta); ta.select(); try { document.execCommand("copy"); done(); } catch (e) {} ta.remove(); }
    });
    bar.appendChild(btn);
    pre.parentNode.insertBefore(box, pre);
    box.appendChild(bar); box.appendChild(pre);
  });

  /* ---------- tab ---------- */
  [].forEach.call(document.querySelectorAll(".tabs"), function (tabs) {
    var panes = [].slice.call(tabs.querySelectorAll(":scope > .tab"));
    var list = h("div", "tab-list");
    panes.forEach(function (p, k) {
      var b = h("button", k === 0 ? "on" : "", esc(p.getAttribute("data-title") || "Tab " + (k + 1))); b.type = "button";
      if (k === 0) p.classList.add("on");
      b.addEventListener("click", function () {
        [].forEach.call(list.children, function (x) { x.classList.remove("on"); });
        panes.forEach(function (x) { x.classList.remove("on"); });
        b.classList.add("on"); p.classList.add("on");
      });
      list.appendChild(b);
    });
    tabs.insertBefore(list, tabs.firstChild);
  });

  /* ---------- tìm kiếm ---------- */
  function norm(s) { return s.normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/đ/g, "d").replace(/Đ/g, "D").toLowerCase(); }
  var input = header.querySelector(".search input");
  var box = header.querySelector(".results");
  var sel = -1, shown = [];
  INDEX.forEach(function (e) { e._t = norm(e.t); e._h = norm(e.h || ""); e._x = norm(e.x || ""); });
  function search(q) {
    var terms = norm(q).split(/\s+/).filter(Boolean);
    if (!terms.length) return [];
    var out = [];
    INDEX.forEach(function (e) {
      var score = 0;
      for (var k = 0; k < terms.length; k++) {
        var t = terms[k], s = 0;
        if (e._t.indexOf(t) >= 0) s += 6;
        if (e._h.indexOf(t) >= 0) s += 8;
        if (e._x.indexOf(t) >= 0) s += 1;
        if (!s) return;
        score += s;
      }
      out.push({ e: e, s: score });
    });
    out.sort(function (a, b) { return b.s - a.s; });
    return out.slice(0, 9).map(function (o) { return o.e; });
  }
  function render() {
    var q = input.value.trim();
    if (!q) { box.hidden = true; return; }
    shown = search(q); sel = shown.length ? 0 : -1;
    box.innerHTML = "";
    if (!shown.length) box.appendChild(h("div", "empty", "Không có kết quả cho “" + esc(q) + "”."));
    shown.forEach(function (e, k) {
      var a = h("a", k === 0 ? "sel" : "", '<div class="r-title">' + esc(e.h ? e.h : e.t) + '</div><div class="r-path">' + esc(e.t) + (e.h ? "" : "") + "</div>");
      a.href = root + e.p + (e.a ? "#" + e.a : "");
      box.appendChild(a);
    });
    box.hidden = false;
  }
  function mark() { [].forEach.call(box.querySelectorAll("a"), function (a, k) { a.classList.toggle("sel", k === sel); }); }
  input.addEventListener("input", render);
  input.addEventListener("focus", render);
  input.addEventListener("keydown", function (e) {
    if (e.key === "ArrowDown") { e.preventDefault(); sel = Math.min(shown.length - 1, sel + 1); mark(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); sel = Math.max(0, sel - 1); mark(); }
    else if (e.key === "Enter" && shown[sel]) { window.location.href = root + shown[sel].p + (shown[sel].a ? "#" + shown[sel].a : ""); }
    else if (e.key === "Escape") { box.hidden = true; input.blur(); }
  });
  document.addEventListener("click", function (e) { if (!e.target.closest(".search")) box.hidden = true; });
  document.addEventListener("keydown", function (e) {
    if (e.key === "/" && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) { e.preventDefault(); input.focus(); }
  });

  /* ---------- liên kết tới mã nguồn: <code> trùng tên hàm / lớp / trường config / file -> mở VS Code ---------- */
  // Dữ liệu (tên -> file, dòng) do build_search_index.py sinh ra ở assets/source-index.js.
  // Chỉ hoạt động khi mở docs bằng file:// (docs nằm ở pacman_rl/docs/html/, nên src/pacman_rl/ có vị trí cố định).
  var SRC = window.DOCS_SOURCES || {};
  if (location.protocol === "file:" && Object.keys(SRC).length) {
    var srcUrl = new URL(root + "../../src/pacman_rl/", location.href);
    var srcDir = decodeURIComponent(srcUrl.pathname);
    if (/^\/[A-Za-z]:/.test(srcDir)) srcDir = srcDir.slice(1);          // Windows: "/D:/..." -> "D:/..."
    var cfg = body.getAttribute("data-cfg");                            // trang config: h3 của tham số -> trường trong dataclass
    var link = function (code, key) {
      var s = SRC[key];
      if (!s) return;
      var a = h("a", "src");
      a.href = "vscode://file/" + encodeURI(srcDir + s.f) + ":" + s.l;
      a.title = "Mở trong VS Code: " + s.f + ":" + s.l;
      code.parentNode.insertBefore(a, code);
      a.appendChild(code);
    };
    [].forEach.call(document.querySelectorAll("article.doc code"), function (code) {
      if (code.closest("pre") || code.closest("a")) return;
      var t = code.textContent.trim();
      if (cfg && code.closest(".param > h3")) link(code, cfg + "." + t);
      else link(code, t);
    });
  }
})();
