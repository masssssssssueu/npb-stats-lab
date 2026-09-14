// table.stats の見出しをクリックすると、その列を基準に降順(再クリックで昇順)に並び替える。
(function () {
  function parseCell(td) {
    const text = td.textContent.trim();
    const cleaned = text.replace(/[%,]/g, "");
    if (cleaned !== "" && !isNaN(cleaned) && /^-?[\d.]+$/.test(cleaned)) {
      return parseFloat(cleaned);
    }
    return text;
  }

  document.querySelectorAll("table.stats").forEach(function (table) {
    const thead = table.querySelector("thead");
    const tbody = table.querySelector("tbody");
    if (!thead || !tbody) return;
    const ths = Array.from(thead.querySelectorAll("th"));

    ths.forEach(function (th, idx) {
      th.addEventListener("click", function () {
        const nextDir = th.dataset.sortDir === "desc" ? "asc" : "desc";
        ths.forEach(function (other) {
          other.dataset.sortDir = "";
          other.classList.remove("sorted-asc", "sorted-desc");
        });
        th.dataset.sortDir = nextDir;
        th.classList.add(nextDir === "desc" ? "sorted-desc" : "sorted-asc");

        const rows = Array.from(tbody.querySelectorAll("tr"));
        rows.sort(function (a, b) {
          const av = parseCell(a.children[idx]);
          const bv = parseCell(b.children[idx]);
          let cmp;
          if (typeof av === "number" && typeof bv === "number") {
            cmp = av - bv;
          } else {
            cmp = String(av).localeCompare(String(bv), "ja");
          }
          return nextDir === "desc" ? -cmp : cmp;
        });
        rows.forEach(function (r) { tbody.appendChild(r); });
      });
    });
  });
})();
