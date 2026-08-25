/* Self-Shelf dashboard application.
   Pure presentation: every economic number is fetched from the backend
   engine — nothing is recomputed client-side. */

(function () {
  "use strict";

  const view = document.getElementById("view");
  const lastUpdated = document.getElementById("last-updated");

  /* ---------- formatting (display only) ---------- */

  const fmt = {
    money(v, opts) {
      if (v == null) return "—";
      const abs = Math.abs(v);
      const digits = (opts && opts.digits != null)
        ? opts.digits : (abs >= 1000 ? 0 : 2);
      const s = abs.toLocaleString("en-US", {
        minimumFractionDigits: digits, maximumFractionDigits: digits,
      });
      return (v < 0 ? "−$" : "$") + s;
    },
    signedMoney(v) {
      if (v == null) return "—";
      return (v > 0 ? "+" : "") + fmt.money(v);
    },
    pct(v, digits) {
      if (v == null) return "—";
      return v.toFixed(digits != null ? digits : 1) + "%";
    },
    ratio(v) { return v == null ? "unbounded" : v.toFixed(2) + "×"; },
    num(v, digits) {
      if (v == null) return "—";
      return v.toLocaleString("en-US", {
        minimumFractionDigits: 0,
        maximumFractionDigits: digits != null ? digits : 0,
      });
    },
    units(v) { return v == null ? "—" : fmt.num(v, 0); },
    days(v) { return v == null ? "∞" : fmt.num(v, 1) + "d"; },
  };

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  const STATUS_LABELS = {
    healthy: "Healthy", watch: "Watch",
    at_risk: "At risk", clearance: "Clearance",
  };

  const STATUS_DESCRIPTIONS = {
    healthy: "Less than 5% of stock at risk of expiring unsold.",
    watch: "5–25% of stock at risk at the current price.",
    at_risk: "25–50% of stock expected to expire unsold.",
    clearance: "Over half the stock will expire unsold without action.",
  };

  function statusChip(status) {
    return `<span class="chip ${status}">${STATUS_LABELS[status] || status}</span>`;
  }

  function actionChip(action, pct) {
    if (action === "markdown") {
      return `<span class="chip markdown">Markdown ${fmt.pct(pct)}</span>`;
    }
    return `<span class="chip hold">Hold</span>`;
  }

  /* ---------- API ---------- */

  let cache = {};

  function clearCache() { cache = {}; }

  /* Active data source, refreshed by boot() and after imports/switches. */
  const appMode = { source: "synthetic", synthetic: true, hasCustom: false };

  function describeDetail(detail) {
    if (typeof detail === "string") return detail;
    if (detail && detail.message) {
      const extra = (detail.errors || []).join("; ");
      return detail.message + (extra ? ` — ${extra}` : "");
    }
    return JSON.stringify(detail);
  }

  async function api(path, { fresh } = {}) {
    if (!fresh && cache[path]) return cache[path];
    const res = await fetch(path);
    if (res.status === 503) {
      const err = new Error("computing");
      err.computing = true;
      throw err;
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(
        body.detail ? describeDetail(body.detail)
          : `Request failed (${res.status})`);
    }
    const data = await res.json();
    cache[path] = data;
    return data;
  }

  async function apiPost(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(
        data.detail ? describeDetail(data.detail)
          : `Request failed (${res.status})`);
    }
    return data;
  }

  async function apiUpload(path, file) {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(path, { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(
        data.detail ? describeDetail(data.detail)
          : `Upload failed (${res.status})`);
    }
    return data;
  }

  function updateModeBadge() {
    const badge = document.getElementById("mode-badge");
    const footer = document.getElementById("footer-note");
    if (appMode.source === "custom") {
      badge.textContent = "Custom data";
      badge.classList.add("custom");
      badge.title = "Recommendations are computed from your imported data";
      footer.textContent = "Self-Shelf pricing engine — running on imported "
        + "custom data. Recommendations depend on the quality of the "
        + "imported history.";
    } else {
      badge.textContent = "Simulation mode";
      badge.classList.remove("custom");
      badge.title = "All data comes from the synthetic retail simulator";
      footer.textContent = "Self-Shelf pricing engine — synthetic retail "
        + "simulation. Results are not real-world retail performance.";
    }
  }

  async function refreshMode() {
    const status = await api("/api/status", { fresh: true });
    appMode.source = status.source;
    appMode.synthetic = status.source === "synthetic";
    appMode.hasCustom = !!(status.custom && status.custom.available);
    updateModeBadge();
    return status;
  }

  /* ---------- shared view states ---------- */

  function renderComputing() {
    view.innerHTML = `
      <div class="state-block">
        <div class="spinner" role="status" aria-label="Loading"></div>
        <h2>Running the pricing engine</h2>
        <p>Training the demand model and optimizing prices for the synthetic
        product set. This runs once and takes a few seconds.</p>
      </div>`;
  }

  function renderError(message, retry) {
    view.innerHTML = `
      <div class="state-block">
        <h2>Something went wrong</h2>
        <p>${esc(message)}</p>
        <button class="btn" id="retry-btn">Try again</button>
      </div>`;
    document.getElementById("retry-btn").addEventListener("click", retry);
  }

  function renderSkeleton() {
    view.innerHTML = `
      <div class="kpis">${'<div class="kpi skeleton" style="height:92px"></div>'.repeat(6)}</div>
      <div class="skeleton" style="height:320px"></div>`;
  }

  async function guarded(renderFn) {
    renderSkeleton();
    try {
      await renderFn();
    } catch (err) {
      if (err.computing) {
        renderComputing();
        setTimeout(() => route(), 1500);
      } else {
        renderError(err.message, () => route());
      }
    }
  }

  /* ---------- overview ---------- */

  async function renderOverview() {
    const data = await api("/api/dashboard");
    const k = data.kpis;
    const wasteDelta = (k.expected_waste_current || 0)
      - (k.expected_waste_recommended || 0);

    view.innerHTML = `
      <h1 class="page-title">Overview</h1>
      <p class="page-subtitle">Expiry-aware pricing recommendations for
      ${k.products} products in ${data.meta && data.meta.synthetic === false
        ? "your imported dataset"
        : "the synthetic simulation"}. Accepting all
      recommendations changes the expected economics below.</p>

      <div class="kpis">
        <div class="kpi">
          <div class="kpi-label">Products</div>
          <div class="kpi-value">${fmt.num(k.products)}</div>
          <div class="kpi-sub">${fmt.num(k.products_at_risk)} at risk</div>
        </div>
        <div class="kpi kpi-accent">
          <div class="kpi-label">Markdowns</div>
          <div class="kpi-value">${fmt.num(k.markdown_recommendations)}</div>
          <div class="kpi-sub">avg depth ${fmt.pct(k.avg_markdown_pct)}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Expected waste</div>
          <div class="kpi-value">${fmt.units(k.expected_waste_recommended)} <span class="unit">units</span></div>
          <div class="kpi-sub"><span class="delta-pos">−${fmt.units(wasteDelta)}</span> vs holding (${fmt.units(k.expected_waste_current)})</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Sell-through</div>
          <div class="kpi-value">${fmt.pct(100 * k.sell_through_recommended, 0)}</div>
          <div class="kpi-sub">${fmt.pct(100 * k.sell_through_current, 0)} <span class="arrow">→</span> ${fmt.pct(100 * k.sell_through_recommended, 0)}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Simulated value gain</div>
          <div class="kpi-value delta-pos">${fmt.signedMoney(k.value_improvement)}</div>
          <div class="kpi-sub">vs holding every price</div>
        </div>
      </div>

      <div class="overview-grid">
        <div class="card">
          <div class="section-title" id="featured-title">Why the optimizer moves a price</div>
          <div class="section-note" id="featured-note"></div>
          <div id="featured-chart"></div>
          <div class="chart-legend">
            <span class="key"><span class="swatch" style="background:#7C5CFF"></span>economic value at each candidate price</span>
            <span class="key"><span class="swatch" style="background:#8B8B94"></span>current price</span>
          </div>
        </div>
        <div class="card">
          <div class="section-title">Hold vs recommended</div>
          <div class="section-note">Day-by-day replay against the simulator's
          ground truth — synthetic simulation, identical demand noise for
          both strategies.</div>
          <div id="backtest-box"></div>
        </div>
      </div>

      <div class="card">
        <div class="section-title">Markdown queue</div>
        <div class="section-note">Recommendations ranked by expected economic
        improvement over holding the current price.</div>
        <div id="queue-box"></div>
      </div>

      <div style="margin-top: 24px">
        <div class="section-title" style="margin-bottom: 12px">Inventory risk</div>
        <div class="risk-groups" id="risk-box"></div>
      </div>`;

    renderBacktest(document.getElementById("backtest-box"), data.backtest);
    renderQueue(document.getElementById("queue-box"), data.queue);
    renderRiskGroups(document.getElementById("risk-box"), data);
    await renderFeaturedCurve(data);
  }

  function renderBacktest(box, backtest) {
    if (!backtest) {
      box.innerHTML = `<div class="empty-note">The counterfactual backtest
        needs the synthetic simulator's ground truth, which does not exist
        for imported data — so Self-Shelf does not fabricate one.</div>`;
      return;
    }
    const rows = [
      ["Revenue", "revenue", fmt.money],
      ["Gross profit", "gross_profit", fmt.money],
      ["Units sold", "units_sold", fmt.units],
      ["Waste units", "waste_units", fmt.units],
      ["Sell-through", "sell_through", (v) => fmt.pct(100 * v, 0)],
      ["Economic value", "economic_value", fmt.money],
    ];
    const col = (name, key, cls) => `
      <div class="backtest-col ${cls}">
        <h4>${name}</h4>
        ${rows.map(([label, field, format]) => `
          <div class="bt-row"><span>${label}</span>
          <span class="v">${format(backtest[key][field])}</span></div>
        `).join("")}
      </div>`;
    box.innerHTML = `<div class="backtest-cols">
      ${col("Hold prices", "hold", "")}
      ${col("Self-Shelf", "recommended", "rec")}
    </div>`;
  }

  function renderQueue(box, queue) {
    if (!queue.length) {
      box.innerHTML = `<div class="empty-note">No markdowns recommended —
        every product holds its current price.</div>`;
      return;
    }
    box.innerHTML = `<div class="table-wrap"><table>
      <thead><tr>
        <th>Product</th><th>Status</th>
        <th class="num">Current</th><th class="num">Recommended</th>
        <th class="num">Change</th><th class="num">Waste avoided</th>
        <th class="num">Value gain</th>
      </tr></thead>
      <tbody>
        ${queue.map((p) => `
          <tr class="rowlink" data-id="${esc(p.id)}" tabindex="0"
              role="link" aria-label="${esc(p.name)} details">
            <td class="strong">${esc(p.name)}</td>
            <td>${statusChip(p.status)}</td>
            <td class="num">${fmt.money(p.pricing.current)}</td>
            <td class="num rec-price">${fmt.money(p.pricing.recommended)}</td>
            <td class="num">−${fmt.pct(p.pricing.markdown_pct)}</td>
            <td class="num">${fmt.units(p.economics.waste_avoided_units)}</td>
            <td class="num delta-pos">${fmt.signedMoney(p.economics.improvement)}</td>
          </tr>`).join("")}
      </tbody></table></div>`;
    wireRowLinks(box);
  }

  function renderRiskGroups(box, data) {
    api("/api/products").then((payload) => {
      const groups = ["clearance", "at_risk", "watch", "healthy"];
      box.innerHTML = groups.map((g) => {
        const members = payload.products.filter((p) => p.status === g);
        const shown = members.slice(0, 4);
        return `
          <div class="risk-group">
            <div class="risk-head">
              <span class="risk-name">${statusChip(g)}</span>
              <span class="risk-count">${members.length}</span>
            </div>
            <div class="risk-desc">${STATUS_DESCRIPTIONS[g]}</div>
            <div class="risk-items">
              ${shown.map((p) => `<a href="#/products/${esc(p.id)}">${esc(p.name)}</a>`).join("")}
              ${members.length > shown.length
                ? `<span class="risk-more">+${members.length - shown.length} more</span>`
                : ""}
            </div>
          </div>`;
      }).join("");
    });
  }

  async function renderFeaturedCurve(data) {
    const featured = data.queue[0];
    const box = document.getElementById("featured-chart");
    if (!featured) {
      box.innerHTML = `<div class="empty-note">No markdown recommendations
        to illustrate.</div>`;
      return;
    }
    document.getElementById("featured-title").textContent =
      `Why ${featured.name} moves to ${fmt.money(featured.pricing.recommended)}`;
    document.getElementById("featured-note").textContent =
      "Expected economic value of every allowed price. The optimizer picks "
      + "the peak; holding sits below it.";
    const sweep = await api(`/api/products/${featured.id}/sweep`);
    drawValueCurve(box, sweep, { height: 260 });
  }

  function drawValueCurve(box, sweep, opts) {
    const pts = sweep.points.map((p) => ({
      x: p.price, y: p.economic_score, meta: p,
    }));
    const at = (price) => {
      let best = sweep.points[0];
      for (const p of sweep.points) {
        if (Math.abs(p.price - price) < Math.abs(best.price - price)) best = p;
      }
      return best.economic_score;
    };
    Charts.lineChart(box, {
      height: (opts && opts.height) || 280,
      series: [{ points: pts, color: Charts.COLORS.accent, area: true }],
      markers: [
        {
          x: sweep.current_price, y: at(sweep.current_price),
          label: "current", color: Charts.COLORS.line,
        },
        {
          x: sweep.recommended_price, y: at(sweep.recommended_price),
          label: sweep.recommended_price === sweep.current_price
            ? "" : "recommended",
          color: Charts.COLORS.accent, dash: "none",
        },
      ],
      zeroLine: 0,
      xFormat: (v) => "$" + v.toFixed(2),
      yFormat: (v) => fmt.money(v, { digits: 0 }),
      ariaLabel: "Economic value across candidate prices",
      tooltip: (p) => `
        <div class="tip-title">${fmt.money(p.x, { digits: 2 })}</div>
        Economic value ${fmt.money(p.meta.economic_score)}<br>
        Demand ${fmt.num(p.meta.daily_demand, 1)}/day ·
        Waste ${fmt.units(p.meta.expected_waste_units)}`,
    });
  }

  /* ---------- products ---------- */

  const FILTERS = [
    { key: "all", label: "All", test: () => true },
    { key: "markdown", label: "Markdown", test: (p) => p.action === "markdown" },
    { key: "hold", label: "Hold", test: (p) => p.action === "hold" },
    {
      key: "risk", label: "At risk",
      test: (p) => p.status === "at_risk" || p.status === "clearance",
    },
    {
      key: "expiry", label: "Near expiry",
      test: (p) => p.inventory.days_to_expiry <= 5,
    },
    {
      key: "pressure", label: "High inventory pressure",
      test: (p) => p.inventory.inventory_pressure >= 0.25,
    },
  ];

  const productsState = { filter: "all", query: "", sort: "improvement", dir: -1 };

  const SORTS = {
    name: (p) => p.name.toLowerCase(),
    department: (p) => p.department,
    current: (p) => p.pricing.current,
    recommended: (p) => p.pricing.recommended,
    markdown: (p) => p.pricing.markdown_pct,
    expiry: (p) => p.inventory.days_to_expiry,
    inventory: (p) => p.inventory.units,
    supply: (p) => p.inventory.days_of_supply == null
      ? Infinity : p.inventory.days_of_supply,
    sellthrough: (p) => p.economics.recommended.sell_through,
    waste: (p) => p.economics.current.waste_units,
    improvement: (p) => p.economics.improvement,
  };

  async function renderProducts() {
    const payload = await api("/api/products");
    view.innerHTML = `
      <h1 class="page-title">Products</h1>
      <p class="page-subtitle">Every product in the current run with its
      recommendation. Click a row for the full economic breakdown.</p>
      <div class="toolbar">
        <input class="search" id="product-search" type="search"
          placeholder="Search products or departments"
          value="${esc(productsState.query)}" aria-label="Search products">
        <div class="filter-tabs" id="filter-tabs" role="group" aria-label="Filters">
          ${FILTERS.map((f) => `
            <button data-filter="${f.key}"
              class="${productsState.filter === f.key ? "active" : ""}">
              ${f.label}</button>`).join("")}
        </div>
        <span class="count-note" id="count-note"></span>
        <a class="btn" href="/api/export/recommendations.csv" download
          title="Download the full recommendation set as CSV">Export CSV</a>
      </div>
      <div class="card" style="padding: 6px 10px">
        <div id="products-table"></div>
      </div>`;

    const searchBox = document.getElementById("product-search");
    searchBox.addEventListener("input", () => {
      productsState.query = searchBox.value;
      drawProductsTable(payload.products);
    });
    document.getElementById("filter-tabs").addEventListener("click", (evt) => {
      const btn = evt.target.closest("button[data-filter]");
      if (!btn) return;
      productsState.filter = btn.dataset.filter;
      document.querySelectorAll("#filter-tabs button").forEach((b) =>
        b.classList.toggle("active", b === btn));
      drawProductsTable(payload.products);
    });
    drawProductsTable(payload.products);
  }

  function drawProductsTable(products) {
    const box = document.getElementById("products-table");
    if (!box) return;
    const filter = FILTERS.find((f) => f.key === productsState.filter);
    const q = productsState.query.trim().toLowerCase();
    let rows = products.filter(filter.test);
    if (q) {
      rows = rows.filter((p) =>
        p.name.toLowerCase().includes(q)
        || p.department.toLowerCase().includes(q));
    }
    const sortKey = SORTS[productsState.sort] || SORTS.improvement;
    rows = rows.slice().sort((a, b) => {
      const ka = sortKey(a), kb = sortKey(b);
      if (ka < kb) return -productsState.dir;
      if (ka > kb) return productsState.dir;
      return 0;
    });

    const note = document.getElementById("count-note");
    if (note) {
      note.textContent = `${rows.length} of ${products.length} products`;
    }

    if (!rows.length) {
      box.innerHTML = `<div class="empty-note">No products match this
        filter${q ? ` and search “${esc(q)}”` : ""}.</div>`;
      return;
    }

    const th = (label, key, numeric) => {
      const active = productsState.sort === key;
      const ind = active ? (productsState.dir > 0 ? "▲" : "▼") : "";
      return `<th class="${numeric ? "num" : ""}">
        <button data-sort="${key}" aria-label="Sort by ${label}">
          ${label} <span class="sort-ind">${ind}</span></button></th>`;
    };

    box.innerHTML = `<div class="table-wrap"><table>
      <thead><tr>
        ${th("Product", "name")}
        ${th("Dept", "department")}
        ${th("Current", "current", true)}
        ${th("Recommended", "recommended", true)}
        ${th("Change", "markdown", true)}
        ${th("Expiry", "expiry", true)}
        ${th("Stock", "inventory", true)}
        ${th("Supply", "supply", true)}
        ${th("Sell-thru", "sellthrough", true)}
        ${th("Waste", "waste", true)}
        ${th("Value Δ", "improvement", true)}
        <th>Status</th>
      </tr></thead>
      <tbody>
        ${rows.map((p) => `
          <tr class="rowlink" data-id="${esc(p.id)}" tabindex="0"
              role="link" aria-label="${esc(p.name)} details">
            <td class="strong cell-name" title="${esc(p.name)}">${esc(p.name)}</td>
            <td>${esc(p.department)}</td>
            <td class="num">${fmt.money(p.pricing.current)}</td>
            <td class="num ${p.action === "markdown" ? "rec-price" : ""}">
              ${p.action === "markdown"
                ? fmt.money(p.pricing.recommended) : "hold"}</td>
            <td class="num">${p.action === "markdown"
              ? "−" + fmt.pct(p.pricing.markdown_pct) : "—"}</td>
            <td class="num">${fmt.num(p.inventory.days_to_expiry)}d</td>
            <td class="num">${fmt.units(p.inventory.units)}</td>
            <td class="num">${fmt.days(p.inventory.days_of_supply)}</td>
            <td class="num">${fmt.pct(100 * p.economics.recommended.sell_through, 0)}</td>
            <td class="num">${fmt.units(p.economics.current.waste_units)}</td>
            <td class="num ${p.economics.improvement > 0 ? "delta-pos" : ""}">
              ${p.economics.improvement > 0
                ? fmt.signedMoney(p.economics.improvement) : "—"}</td>
            <td>${statusChip(p.status)}</td>
          </tr>`).join("")}
      </tbody></table></div>`;

    box.querySelectorAll("[data-sort]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const key = btn.dataset.sort;
        if (productsState.sort === key) {
          productsState.dir = -productsState.dir;
        } else {
          productsState.sort = key;
          productsState.dir = key === "name" || key === "department" ? 1 : -1;
        }
        drawProductsTable(products);
      });
    });
    wireRowLinks(box);
  }

  function wireRowLinks(scope) {
    scope.querySelectorAll("tr.rowlink").forEach((row) => {
      const go = () => { location.hash = `#/products/${row.dataset.id}`; };
      row.addEventListener("click", go);
      row.addEventListener("keydown", (evt) => {
        if (evt.key === "Enter" || evt.key === " ") { evt.preventDefault(); go(); }
      });
    });
  }

  /* ---------- product detail ---------- */

  async function renderProductDetail(id) {
    const [detail, sweep] = await Promise.all([
      api(`/api/products/${id}`),
      api(`/api/products/${id}/sweep`),
    ]);
    const isMarkdown = detail.action === "markdown";
    const e = detail.economics;

    view.innerHTML = `
      <a class="crumb" href="#/products">← Products</a>
      <div class="detail-head">
        <div>
          <h1 class="detail-title">${esc(detail.name)}</h1>
          <div class="detail-dept">${esc(detail.department)}
            &nbsp;·&nbsp; ${statusChip(detail.status)}
            &nbsp; ${actionChip(detail.action, detail.pricing.markdown_pct)}</div>
        </div>
        <div class="price-block">
          ${isMarkdown ? `
            <div class="price-line">
              <span class="old">${fmt.money(detail.pricing.current)}</span>
              <span aria-hidden="true" style="color: var(--text-3)">→</span>
              <span class="new">${fmt.money(detail.pricing.recommended)}</span>
            </div>
            <div class="price-caption">Recommended markdown
              −${fmt.pct(detail.pricing.markdown_pct)}</div>`
          : `
            <div class="price-line">
              <span class="hold-price">${fmt.money(detail.pricing.current)}</span>
            </div>
            <div class="price-caption">Hold — no economically justified
              markdown</div>`}
        </div>
      </div>

      <div class="detail-grid">
        <div class="detail-col">
          <div class="card">
            <div class="section-title">Economic value across allowed prices</div>
            <div class="section-note">The engine evaluates every candidate
              price from ${fmt.money(detail.pricing.min_allowed)} to
              ${fmt.money(detail.pricing.max_allowed)} and picks the peak.</div>
            <div id="value-curve"></div>
          </div>
          <div class="card">
            <div class="section-title">Demand response</div>
            <div class="section-note">Predicted daily demand at each price
              (elasticity ${fmt.num(detail.demand.elasticity, 2)}).</div>
            <div id="demand-curve"></div>
          </div>
          <div class="card">
            <div class="section-title">Keep price vs recommended</div>
            <div class="section-note">Expected outcomes over the evaluation
              window, computed by the pricing engine.</div>
            <div id="compare-box"></div>
          </div>
        </div>
        <div class="detail-col">
          <div class="card">
            <div class="section-title">${isMarkdown ? "Why markdown?" : "Why hold?"}</div>
            <div class="section-note"></div>
            <div id="verdict-box"></div>
            <ul class="reason-list">
              ${detail.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}
            </ul>
          </div>
          <div class="card" id="impact-card">
            <div class="section-title">If you accept this recommendation</div>
            <div class="section-note">Change vs holding the current price
              (synthetic simulation).</div>
            <div class="impact-rows" id="impact-box"></div>
          </div>
          <div class="card" id="breakeven-card">
            <div class="section-title">Break-even economics</div>
            <div class="impact-rows" id="breakeven-box"></div>
          </div>
          <div class="card">
            <div class="section-title">Inventory position</div>
            <div class="impact-rows" id="position-box"></div>
          </div>
        </div>
      </div>

      <div class="card" style="margin-top: 24px">
        <div class="section-title">Recommended price path</div>
        <div class="section-note">Multi-period schedule over the remaining
          shelf life — the optimizer weighs marking down now against
          repricing later, so a discount is only scheduled when its timing
          beats both holding and an immediate markdown.</div>
        <div id="path-box"><div class="skeleton" style="height:200px"></div></div>
      </div>

      <div class="card" style="margin-top: 24px">
        <div class="section-title">Scenario: try a different price</div>
        <div class="section-note">Drag to any allowed price. The economics
          are evaluated live by the backend engine — nothing is estimated in
          the browser.</div>
        <div id="scenario-box"></div>
      </div>`;

    drawValueCurve(document.getElementById("value-curve"), sweep);
    drawDemandCurve(document.getElementById("demand-curve"), sweep, detail);
    drawCompareTable(document.getElementById("compare-box"), detail);
    drawVerdict(document.getElementById("verdict-box"), detail);
    drawImpact(document.getElementById("impact-box"), detail);
    drawBreakEven(document.getElementById("breakeven-box"), detail);
    drawPosition(document.getElementById("position-box"), detail);
    setupScenario(document.getElementById("scenario-box"), detail);
    api(`/api/products/${id}/path`).then((path) => {
      drawPathCard(document.getElementById("path-box"), path);
    }).catch((err) => {
      const box = document.getElementById("path-box");
      if (box) {
        box.innerHTML = `<div class="empty-note">Could not load the price
          path: ${esc(err.message)}</div>`;
      }
    });
  }

  /* ---------- multi-period path ---------- */

  function stepPoints(dailyPrices) {
    /* Duplicate boundary x-values so the polyline renders as day steps. */
    const pts = [];
    dailyPrices.forEach((price, day) => {
      pts.push({ x: day, y: price });
      pts.push({ x: day + 1, y: price });
    });
    return pts;
  }

  function drawPathChart(box, path, extraSeries) {
    const strategies = path.strategies;
    const series = [];
    if (strategies.hold) {
      series.push({
        points: stepPoints(strategies.hold.daily_prices),
        color: Charts.COLORS.line, dash: "3 4", width: 1.5,
      });
    }
    if (strategies.immediate) {
      series.push({
        points: stepPoints(strategies.immediate.daily_prices),
        color: Charts.COLORS.warning, dash: "6 5", width: 1.5,
      });
    }
    series.push({
      points: stepPoints(path.daily_prices),
      color: Charts.COLORS.accent, width: 2.5, area: true,
    });
    if (extraSeries) series.push(extraSeries);
    Charts.lineChart(box, {
      height: 230,
      series,
      xFormat: (v) => Number.isInteger(v) ? `Day ${v}` : "",
      yFormat: (v) => "$" + v.toFixed(2),
      ariaLabel: "Recommended price by day",
    });
  }

  function pathStrategyTable(path) {
    const strategies = path.strategies;
    const rows = [
      ["Revenue", "revenue", fmt.money],
      ["Gross profit", "gross_profit", fmt.money],
      ["Expected waste", "waste_units", (v) => fmt.units(v) + " units"],
      ["Holding cost", "holding_cost", fmt.money],
      ["Economic value", "economic_value", fmt.money],
    ];
    const cols = [
      ["hold", "Hold throughout"],
      ...(strategies.immediate
        ? [["immediate",
            `Immediate ${fmt.money(strategies.immediate.price)}`]] : []),
      ["recommended", "Optimized path"],
    ];
    return `<div class="table-wrap"><table class="compare strategy-table">
      <thead><tr><th>Metric</th>
        ${cols.map(([key, label]) => `
          <th class="num ${key === "recommended" ? "col-rec" : ""}">
            ${label}</th>`).join("")}
      </tr></thead>
      <tbody>
        ${rows.map(([label, key, format]) => `
          <tr><td>${label}</td>
            ${cols.map(([ckey]) => `
              <td class="num ${ckey === "recommended" ? "col-rec" : ""}">
                ${format(strategies[ckey].econ[key])}</td>`).join("")}
          </tr>`).join("")}
      </tbody></table></div>`;
  }

  function drawPathCard(box, path) {
    if (!box) return;
    const scheduleText = path.schedule.map((seg, i) => {
      const start = path.schedule.slice(0, i)
        .reduce((acc, s) => acc + s.days, 0);
      return `<span class="chip ${i === path.schedule.length - 1
        && path.schedule.length > 1 ? "markdown" : "hold"}">
        Day ${start}${seg.days > 1 ? `–${start + seg.days - 1}` : ""}:
        ${fmt.money(seg.price)}</span>`;
    }).join(" ");

    box.innerHTML = `
      <div style="margin-bottom: 10px">${scheduleText}</div>
      <div id="path-chart"></div>
      <div class="chart-legend">
        <span class="key"><span class="swatch" style="background:#7C5CFF"></span>optimized path</span>
        <span class="key"><span class="swatch" style="background:#8B8B94"></span>hold</span>
        ${path.strategies.immediate
          ? `<span class="key"><span class="swatch" style="background:#D6A84F"></span>immediate markdown</span>`
          : ""}
      </div>
      <div style="margin-top: 16px">${pathStrategyTable(path)}</div>
      ${path.action !== "Hold" ? `
        <ul class="reason-list" style="margin-top: 14px">
          ${path.reasons.map((r) => `<li>${esc(r)}</li>`).join("")}
        </ul>` : `
        <div class="section-note" style="margin: 14px 0 0">
          ${esc(path.reasons[0] || "No staged markdown beats holding.")}
        </div>`}`;
    drawPathChart(document.getElementById("path-chart"), path);
  }

  function drawDemandCurve(box, sweep, detail) {
    const pts = sweep.points.map((p) => ({
      x: p.price, y: p.daily_demand, meta: p,
    }));
    Charts.lineChart(box, {
      height: 220,
      series: [{ points: pts, color: Charts.COLORS.line }],
      markers: [
        { x: sweep.current_price, label: "current", color: Charts.COLORS.line },
        ...(sweep.recommended_price !== sweep.current_price
          ? [{ x: sweep.recommended_price, label: "recommended",
               color: Charts.COLORS.accent }] : []),
      ],
      xFormat: (v) => "$" + v.toFixed(2),
      yFormat: (v) => fmt.num(v, 0),
      ariaLabel: "Demand at each candidate price",
      tooltip: (p) => `
        <div class="tip-title">${fmt.money(p.x, { digits: 2 })}</div>
        ${fmt.num(p.meta.daily_demand, 1)} units/day ·
        sell-through ${fmt.pct(100 * p.meta.sell_through, 0)}`,
    });
  }

  function drawCompareTable(box, detail) {
    const cur = detail.economics.current;
    const rec = detail.economics.recommended;
    const rows = [
      ["Price", "price", fmt.money],
      ["Demand / day", "daily_demand", (v) => fmt.num(v, 1)],
      ["Units sold", "units_sold", fmt.units],
      ["Sell-through", "sell_through", (v) => fmt.pct(100 * v, 0)],
      ["Revenue", "revenue", fmt.money],
      ["Gross profit", "gross_profit", fmt.money],
      ["Expected waste", "waste_units", (v) => fmt.units(v) + " units"],
      ["Terminal inventory", "terminal_inventory", (v) => fmt.units(v) + " units"],
      ["Holding cost", "holding_cost", fmt.money],
      ["Economic value", "economic_value", fmt.money],
    ];
    const same = detail.action !== "markdown";
    box.innerHTML = `<div class="table-wrap"><table class="compare">
      <thead><tr><th>Metric</th>
        <th class="num">Keep ${fmt.money(cur.price)}</th>
        <th class="num">${same ? "Recommended (same)" : "Recommended " + fmt.money(rec.price)}</th>
        <th class="num">Δ</th></tr></thead>
      <tbody>
        ${rows.map(([label, key, format]) => {
          const a = cur[key], b = rec[key];
          const delta = (a != null && b != null) ? b - a : null;
          const deltaText = key === "price" || same || delta == null
            ? "" : formatDelta(key, delta);
          return `<tr>
            <td>${label}</td>
            <td class="num">${format(a)}</td>
            <td class="num col-rec">${format(b)}</td>
            <td class="num delta">${deltaText}</td>
          </tr>`;
        }).join("")}
      </tbody></table></div>`;
  }

  function formatDelta(key, delta) {
    if (Math.abs(delta) < 1e-9) return "";
    const better = key === "waste_units" || key === "terminal_inventory"
      || key === "holding_cost" ? delta < 0 : delta > 0;
    const cls = better ? "delta-pos" : "delta-warn";
    let text;
    if (key === "sell_through") text = (delta > 0 ? "+" : "") + fmt.pct(100 * delta, 0);
    else if (key === "daily_demand") text = (delta > 0 ? "+" : "") + fmt.num(delta, 1);
    else if (key === "units_sold" || key === "waste_units"
      || key === "terminal_inventory") {
      text = (delta > 0 ? "+" : "−") + fmt.units(Math.abs(delta));
    } else text = fmt.signedMoney(delta);
    return `<span class="${cls}">${text}</span>`;
  }

  function drawVerdict(box, detail) {
    const e = detail.economics;
    if (detail.action === "markdown") {
      const losesGP = e.gross_profit_delta < 0;
      box.innerHTML = `<div class="verdict">
        ${losesGP
          ? `This markdown <strong>gives up ${fmt.money(Math.abs(e.gross_profit_delta))}
             of gross profit</strong> but avoids enough waste and terminal-stock
             loss to come out <strong>${fmt.signedMoney(e.improvement)}</strong> ahead
             overall. Self-Shelf only discounts when the total economics beat
             doing nothing.`
          : `This markdown improves gross profit on its own
             (<strong>${fmt.signedMoney(e.gross_profit_delta)}</strong>) and the
             total expected outcome by
             <strong>${fmt.signedMoney(e.improvement)}</strong>.`}
      </div>`;
    } else {
      box.innerHTML = `<div class="verdict">
        The current price of <strong>${fmt.money(detail.pricing.current)}</strong>
        already gives the best expected economic value across the entire
        allowed range — no markdown in
        ${fmt.money(detail.pricing.min_allowed)}–${fmt.money(detail.pricing.max_allowed)}
        beats holding.</div>`;
    }
  }

  function drawImpact(box, detail) {
    const e = detail.economics;
    if (detail.action !== "markdown") {
      box.innerHTML = `
        <div class="impact-row"><span class="k">Recommended action</span>
          <span class="v">Keep ${fmt.money(detail.pricing.current)}</span></div>
        <div class="impact-row"><span class="k">Expected waste at this price</span>
          <span class="v">${fmt.units(e.current.waste_units)} units</span></div>
        <div class="impact-row"><span class="k">Expected sell-through</span>
          <span class="v">${fmt.pct(100 * e.current.sell_through, 0)}</span></div>`;
      return;
    }
    const rows = [
      ["Gross profit change", fmt.signedMoney(e.gross_profit_delta),
        e.gross_profit_delta >= 0 ? "delta-pos" : "delta-warn"],
      ["Waste avoided", fmt.units(e.waste_avoided_units) + " units",
        e.waste_avoided_units > 0 ? "delta-pos" : ""],
      ["Holding cost saved", fmt.money(e.holding_cost_saved),
        e.holding_cost_saved > 0 ? "delta-pos" : ""],
      ["Economic value change", fmt.signedMoney(e.improvement), "delta-pos"],
    ];
    box.innerHTML = rows.map(([k, v, cls]) => `
      <div class="impact-row"><span class="k">${k}</span>
        <span class="v ${cls}">${v}</span></div>`).join("");
    if (detail.timing && detail.timing.advantage_now > 0.01) {
      box.innerHTML += `
        <div class="impact-row"><span class="k">Acting now vs waiting
          ${fmt.num(detail.timing.wait_days)}d</span>
          <span class="v delta-pos">preserves
          ${fmt.money(detail.timing.advantage_now)}</span></div>`;
    }
  }

  function drawBreakEven(box, detail) {
    const card = document.getElementById("breakeven-card");
    if (detail.action !== "markdown") { card.style.display = "none"; return; }
    const b = detail.break_even;
    box.innerHTML = `
      <div class="impact-row"><span class="k">Volume needed to hold gross profit</span>
        <span class="v">${fmt.ratio(b.required_uplift)}</span></div>
      <div class="impact-row"><span class="k">Predicted volume uplift</span>
        <span class="v">${fmt.ratio(b.predicted_uplift)}</span></div>
      <div class="impact-row"><span class="k">Margin-only hurdle</span>
        <span class="v ${b.meets_margin_hurdle ? "delta-pos" : "delta-warn"}">
          ${b.meets_margin_hurdle ? "met" : "not met"}</span></div>`;
    if (!b.meets_margin_hurdle) {
      box.innerHTML += `<div class="section-note" style="margin: 10px 0 0">
        The extra volume alone does not pay for the discount — the markdown
        is justified by avoided waste and holding costs, which the hurdle
        does not count.</div>`;
    }
  }

  function drawPosition(box, detail) {
    const inv = detail.inventory;
    box.innerHTML = `
      <div class="impact-row"><span class="k">Inventory on hand</span>
        <span class="v">${fmt.units(inv.units)} units</span></div>
      <div class="impact-row"><span class="k">Shelf life remaining</span>
        <span class="v">${fmt.num(inv.days_to_expiry)} days</span></div>
      <div class="impact-row"><span class="k">Days of supply at current price</span>
        <span class="v">${fmt.days(inv.days_of_supply)}</span></div>
      <div class="impact-row"><span class="k">Stock at risk (current price)</span>
        <span class="v">${fmt.pct(100 * inv.inventory_pressure, 0)}</span></div>
      <div class="impact-row"><span class="k">Expiry pressure</span>
        <span class="v">${fmt.num(inv.expiry_pressure, 2)}</span></div>
      <div class="impact-row"><span class="k">Unit cost</span>
        <span class="v">${fmt.money(detail.pricing.unit_cost)}</span></div>`;
    if (detail.provenance) {
      const prov = detail.provenance;
      const elasticity = prov.elasticity_source === "estimated"
        ? `<span class="estimated">estimated</span> from
           ${fmt.num(prov.elasticity_n_obs)} observed transactions`
        : `<span class="fallback">fallback assumption</span> —
           ${esc(prov.elasticity_reason || "not learned from your data")}`;
      const baseline = prov.baseline_source === "history"
        ? `from ${fmt.num(prov.baseline_n_days)} recent day(s) of this
           product's sales history`
        : `<span class="fallback">fallback</span> — no sales history for
           this product, using the ${prov.baseline_source === "category"
             ? "category median" : "dataset median"} rate`;
      box.innerHTML += `
        <div class="prov-note">
          Elasticity ${fmt.num(detail.demand.elasticity, 2)}: ${elasticity}.
          <br>Baseline demand: ${baseline}.
        </div>`;
    }
  }

  /* ---------- scenario ---------- */

  function setupScenario(box, detail) {
    const min = detail.pricing.min_allowed;
    const max = detail.pricing.max_allowed;
    if (max - min < 0.02) {
      box.innerHTML = `<div class="empty-note">The allowed price range for
        this product collapses to its current price — no scenario to
        explore.</div>`;
      return;
    }
    box.innerHTML = `
      <div class="scenario-controls">
        <span class="scenario-price" id="scenario-price"></span>
        <input type="range" id="scenario-slider"
          min="${min.toFixed(2)}" max="${max.toFixed(2)}" step="0.01"
          value="${detail.pricing.recommended.toFixed(2)}"
          aria-label="Scenario price">
        <span class="section-note" style="margin:0">
          allowed ${fmt.money(min)} – ${fmt.money(max)}</span>
      </div>
      <div class="scenario-hint" id="scenario-hint"></div>
      <div id="scenario-result"></div>`;

    const slider = document.getElementById("scenario-slider");
    const priceLabel = document.getElementById("scenario-price");
    const hint = document.getElementById("scenario-hint");
    const result = document.getElementById("scenario-result");
    let timer = null;
    let requestId = 0;

    async function evaluate() {
      const price = parseFloat(slider.value);
      priceLabel.textContent = fmt.money(price);
      const id = ++requestId;
      try {
        const scenario = await api(
          `/api/products/${detail.id}/scenario?price=${price.toFixed(2)}`);
        if (id !== requestId) return;
        drawScenario(result, hint, scenario, detail);
      } catch (err) {
        if (id !== requestId) return;
        result.innerHTML = `<div class="empty-note">Could not evaluate this
          price: ${esc(err.message)}</div>`;
      }
    }

    slider.addEventListener("input", () => {
      priceLabel.textContent = fmt.money(parseFloat(slider.value));
      clearTimeout(timer);
      timer = setTimeout(evaluate, 160);
    });
    evaluate();
  }

  function drawScenario(result, hint, scenario, detail) {
    const s = scenario.breakdown;
    const base = scenario.baseline;
    const delta = s.economic_value - base.economic_value;
    hint.textContent = scenario.clamped
      ? "Requested price was outside the allowed range and was clamped."
      : "";
    const rows = [
      ["Demand / day", "daily_demand", (v) => fmt.num(v, 1)],
      ["Units sold", "units_sold", fmt.units],
      ["Sell-through", "sell_through", (v) => fmt.pct(100 * v, 0)],
      ["Revenue", "revenue", fmt.money],
      ["Gross profit", "gross_profit", fmt.money],
      ["Expected waste", "waste_units", (v) => fmt.units(v) + " units"],
      ["Holding cost", "holding_cost", fmt.money],
      ["Economic value", "economic_value", fmt.money],
    ];
    result.innerHTML = `<div class="table-wrap"><table class="compare">
      <thead><tr><th>Metric</th>
        <th class="num">Hold ${fmt.money(base.price)}</th>
        <th class="num">Scenario ${fmt.money(scenario.price)}</th>
        <th class="num">Δ</th></tr></thead>
      <tbody>
        ${rows.map(([label, key, format]) => `
          <tr><td>${label}</td>
            <td class="num">${format(base[key])}</td>
            <td class="num col-rec">${format(s[key])}</td>
            <td class="num delta">${formatDelta(key, s[key] - base[key])}</td>
          </tr>`).join("")}
      </tbody></table></div>
      <div class="section-note" style="margin-top: 12px">
        ${delta > 0.005
          ? `This price beats holding by <span class="delta-pos">${fmt.signedMoney(delta)}</span>.
             The engine's recommendation is ${fmt.money(detail.pricing.recommended)}.`
          : delta < -0.005
            ? `This price is <span class="delta-warn">${fmt.money(Math.abs(delta))} worse</span>
               than holding — the engine would not accept it.`
            : "This price performs the same as holding the current price."}
      </div>`;
  }

  /* ---------- pricing (scenario lab) ---------- */

  const pricingState = { selected: null, query: "", mode: "single" };

  async function renderPricing() {
    const payload = await api("/api/products");
    const products = payload.products;
    if (!pricingState.selected
        || !products.some((p) => p.id === pricingState.selected)) {
      const firstMarkdown = products.find((p) => p.action === "markdown");
      pricingState.selected = (firstMarkdown || products[0]).id;
    }
    view.innerHTML = `
      <h1 class="page-title">Pricing</h1>
      <p class="page-subtitle">Explore what any allowed price — or an entire
      multi-day price path — would do to a product's economics. Every
      evaluation runs in the backend pricing engine.</p>
      <div class="toolbar" style="margin-bottom: 18px">
        <div class="mode-toggle" role="group" aria-label="Scenario mode">
          <button data-mode="single"
            class="${pricingState.mode === "single" ? "active" : ""}">
            Single price</button>
          <button data-mode="path"
            class="${pricingState.mode === "path" ? "active" : ""}">
            Price path</button>
        </div>
      </div>
      <div class="detail-grid" style="grid-template-columns: minmax(0,4fr) minmax(0,8fr)">
        <div class="card" style="padding: 14px">
          <input class="search" id="picker-search" type="search"
            placeholder="Filter products" style="width: 100%; margin-bottom: 10px"
            value="${esc(pricingState.query)}" aria-label="Filter products">
          <div class="picker-list" id="picker-list" role="listbox"></div>
        </div>
        <div class="detail-col">
          <div class="card">
            <div class="section-title" id="lab-title"></div>
            <div class="section-note" id="lab-note"></div>
            <div id="lab-curve"></div>
          </div>
          <div class="card">
            <div class="section-title" id="lab-scenario-title">Scenario</div>
            <div id="lab-scenario"></div>
          </div>
        </div>
      </div>`;

    const searchBox = document.getElementById("picker-search");
    searchBox.addEventListener("input", () => {
      pricingState.query = searchBox.value;
      drawPicker(products);
    });
    view.querySelectorAll(".mode-toggle button").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (pricingState.mode === btn.dataset.mode) return;
        pricingState.mode = btn.dataset.mode;
        view.querySelectorAll(".mode-toggle button").forEach((b) =>
          b.classList.toggle("active", b === btn));
        await drawLab(products);
      });
    });
    drawPicker(products);
    await drawLab(products);
  }

  function drawPicker(products) {
    const list = document.getElementById("picker-list");
    const q = pricingState.query.trim().toLowerCase();
    const rows = q
      ? products.filter((p) => p.name.toLowerCase().includes(q)
        || p.department.toLowerCase().includes(q))
      : products;
    if (!rows.length) {
      list.innerHTML = `<div class="empty-note">No matches.</div>`;
      return;
    }
    list.innerHTML = rows.map((p) => `
      <button role="option" data-id="${esc(p.id)}"
        aria-selected="${p.id === pricingState.selected}"
        class="${p.id === pricingState.selected ? "active" : ""}">
        <span>${esc(p.name)}</span>
        <span class="p-price">${p.action === "markdown"
          ? `${fmt.money(p.pricing.current)} → ${fmt.money(p.pricing.recommended)}`
          : fmt.money(p.pricing.current)}</span>
      </button>`).join("");
    list.querySelectorAll("button[data-id]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        pricingState.selected = btn.dataset.id;
        drawPicker(products);
        await drawLab(products);
      });
    });
  }

  async function drawLab(products) {
    if (pricingState.mode === "path") {
      await drawPathLab(products);
      return;
    }
    const summary = products.find((p) => p.id === pricingState.selected);
    const [detail, sweep] = await Promise.all([
      api(`/api/products/${pricingState.selected}`),
      api(`/api/products/${pricingState.selected}/sweep`),
    ]);
    document.getElementById("lab-title").textContent = summary.name;
    document.getElementById("lab-scenario-title").textContent = "Scenario";
    document.getElementById("lab-note").textContent =
      detail.action === "markdown"
        ? `Recommended ${fmt.money(detail.pricing.recommended)} `
          + `(−${fmt.pct(detail.pricing.markdown_pct)}) — the peak of the `
          + "economic value curve."
        : `Recommended hold at ${fmt.money(detail.pricing.current)} — no `
          + "allowed price improves on it.";
    drawValueCurve(document.getElementById("lab-curve"), sweep);
    setupScenario(document.getElementById("lab-scenario"), detail);
  }

  async function drawPathLab(products) {
    const summary = products.find((p) => p.id === pricingState.selected);
    const path = await api(`/api/products/${pricingState.selected}/path`);
    document.getElementById("lab-title").textContent =
      `${summary.name} — recommended path`;
    document.getElementById("lab-note").textContent =
      path.action === "Hold"
        ? "The optimizer keeps the current price for the whole horizon — "
          + "no staged markdown beats holding."
        : `${path.action} over ${path.horizon_days} day(s), worth `
          + `${fmt.signedMoney(path.improvement_vs_hold)} vs holding. `
          + "Edit the day prices below to test your own schedule.";
    const curveBox = document.getElementById("lab-curve");
    drawPathChart(curveBox, path);
    curveBox.insertAdjacentHTML("beforeend", pathStrategyTable(path));
    document.getElementById("lab-scenario-title").textContent =
      "Your own price path";
    setupPathScenario(
      document.getElementById("lab-scenario"), summary, path);
  }

  function setupPathScenario(box, summary, path) {
    const floor = path.constraints.floor;
    const ceiling = path.constraints.ceiling;
    box.innerHTML = `
      <div class="section-note" style="margin-bottom: 4px">
        Set a price per day (allowed ${fmt.money(floor)} –
        ${fmt.money(ceiling)}, drops of at most
        ${fmt.pct(path.constraints.max_daily_drop_pct, 0)} per day, no
        increases). The whole path is valued by the backend engine.</div>
      <div class="path-days" id="path-day-inputs">
        ${path.daily_prices.map((price, day) => `
          <div class="path-day">
            <label for="pd-${day}">Day ${day}</label>
            <input id="pd-${day}" type="number" step="0.01"
              min="${floor.toFixed(2)}" max="${ceiling.toFixed(2)}"
              value="${price.toFixed(2)}" data-day="${day}"
              inputmode="decimal">
          </div>`).join("")}
      </div>
      <div class="path-errors" id="path-errors"></div>
      <div id="path-scenario-result"></div>`;

    const inputs = Array.from(
      box.querySelectorAll("#path-day-inputs input"));
    const errorsBox = document.getElementById("path-errors");
    const resultBox = document.getElementById("path-scenario-result");
    let timer = null;
    let requestId = 0;

    async function evaluate() {
      const prices = inputs.map((input) => parseFloat(input.value));
      if (prices.some((p) => !isFinite(p))) {
        errorsBox.innerHTML = "<div>Every day needs a numeric price.</div>";
        return;
      }
      inputs.forEach((input, day) => input.classList.toggle(
        "changed",
        Math.abs(prices[day] - path.daily_prices[day]) > 0.005));
      const id = ++requestId;
      try {
        const result = await apiPost(
          `/api/products/${summary.id}/path/scenario`,
          { daily_prices: prices });
        if (id !== requestId) return;
        if (result.errors && result.errors.length) {
          errorsBox.innerHTML = result.errors.map(
            (e) => `<div>${esc(e)}</div>`).join("");
          resultBox.innerHTML = "";
          return;
        }
        errorsBox.innerHTML = "";
        const yours = result.econ;
        const rows = [
          ["Revenue", "revenue", fmt.money],
          ["Gross profit", "gross_profit", fmt.money],
          ["Expected waste", "waste_units", (v) => fmt.units(v) + " units"],
          ["Economic value", "economic_value", fmt.money],
        ];
        const cols = [
          ["Hold", result.compare.hold],
          ["Your path", yours],
          ["Optimized", result.compare.recommended],
        ];
        const delta = yours.economic_value
          - result.compare.recommended.economic_value;
        resultBox.innerHTML = `<div class="table-wrap">
          <table class="compare strategy-table">
          <thead><tr><th>Metric</th>
            ${cols.map(([label], i) => `
              <th class="num ${i === 1 ? "col-rec" : ""}">${label}</th>`)
              .join("")}
          </tr></thead>
          <tbody>${rows.map(([label, key, format]) => `
            <tr><td>${label}</td>
              ${cols.map(([, econ], i) => `
                <td class="num ${i === 1 ? "col-rec" : ""}">
                  ${format(econ[key])}</td>`).join("")}
            </tr>`).join("")}
          </tbody></table></div>
          <div class="section-note" style="margin-top: 12px">
            ${Math.abs(delta) <= 0.01
              ? "Your path matches the optimizer's expected value."
              : delta > 0
                ? "Your path beats the optimizer's restricted schedule "
                  + `by <span class="delta-pos">${fmt.signedMoney(delta)}</span>.`
                : `Your path gives up <span class="delta-warn">
                   ${fmt.money(Math.abs(delta))}</span> vs the optimized
                   schedule.`}
          </div>`;
      } catch (err) {
        if (id !== requestId) return;
        errorsBox.innerHTML = `<div>${esc(err.message)}</div>`;
      }
    }

    inputs.forEach((input) => input.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(evaluate, 280);
    }));
    evaluate();
  }

  /* ---------- analytics ---------- */

  async function renderAnalytics() {
    const data = await api("/api/analytics");
    const t = data.totals;
    const wasteCut = (t.waste_current || 0) - (t.waste_recommended || 0);

    view.innerHTML = `
      <h1 class="page-title">Analytics</h1>
      <p class="page-subtitle">Aggregate behavior of the recommendation set —
      synthetic simulation across ${data.price_changes.length} products.</p>

      <div class="kpis">
        <div class="kpi kpi-accent">
          <div class="kpi-label">Markdowns</div>
          <div class="kpi-value">${fmt.num(data.markdowns.count)}</div>
          <div class="kpi-sub">avg depth ${fmt.pct(data.markdowns.avg_depth_pct)}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Waste reduction</div>
          <div class="kpi-value delta-pos">−${fmt.units(wasteCut)} <span class="unit">units</span></div>
          <div class="kpi-sub">${fmt.units(t.waste_current)} → ${fmt.units(t.waste_recommended)}</div>
        </div>
        <div class="kpi">
          <div class="kpi-label">Simulated value gain</div>
          <div class="kpi-value delta-pos">${fmt.signedMoney(t.value_improvement)}</div>
          <div class="kpi-sub">vs holding all prices</div>
        </div>
      </div>

      <div class="analytics-grid">
        <div class="card">
          <div class="section-title">Markdown depth</div>
          <div class="section-note">How deep the recommended markdowns go.</div>
          <div id="depth-chart"></div>
        </div>
        <div class="card">
          <div class="section-title">Inventory risk</div>
          <div class="section-note">Products by share of stock expected to
            expire unsold at the current price.</div>
          <div id="risk-chart"></div>
        </div>
        <div class="card">
          <div class="section-title">Days of supply</div>
          <div class="section-note">Stock cover at current demand.</div>
          <div id="supply-chart"></div>
        </div>
        <div class="card">
          <div class="section-title">Expected waste by department</div>
          <div class="section-note">Units expected to expire unsold —
            holding vs recommended prices.</div>
          <div id="waste-chart"></div>
          <div class="chart-legend">
            <span class="key"><span class="swatch" style="background:#8B8B94"></span>hold</span>
            <span class="key"><span class="swatch" style="background:#7C5CFF"></span>recommended</span>
          </div>
        </div>
      </div>

      <div class="card" style="margin-top: 24px" id="path-backtest-card">
        <div class="section-title">Multi-period strategy backtest</div>
        <div class="section-note">Hold vs immediate markdown vs the
          optimized multi-day path, replayed against the simulator's ground
          truth with identical demand noise. Synthetic simulation only —
          not real-world performance.</div>
        <div id="path-backtest-box"></div>
      </div>

      <div class="card" style="margin-top: 24px">
        <div class="section-title">Price changes</div>
        <div class="section-note">Current → recommended for every
          repriced product.</div>
        <div id="dumbbell-chart"></div>
      </div>`;

    drawPathBacktest(
      document.getElementById("path-backtest-box"), data.path_backtest);

    Charts.barChart(document.getElementById("depth-chart"), {
      items: Object.entries(data.markdowns.depth_distribution).map(
        ([label, value]) => ({
          label, value, color: Charts.COLORS.accent,
          title: `${label}: ${value} markdown${value === 1 ? "" : "s"}`,
        })),
      ariaLabel: "Markdown depth distribution",
    });

    const riskColors = {
      healthy: Charts.COLORS.positive, watch: Charts.COLORS.line,
      at_risk: Charts.COLORS.warning, clearance: Charts.COLORS.danger,
    };
    Charts.barChart(document.getElementById("risk-chart"), {
      items: Object.entries(data.risk_counts).map(([key, value]) => ({
        label: STATUS_LABELS[key], value, color: riskColors[key],
        title: `${STATUS_LABELS[key]}: ${value} products`,
      })),
      ariaLabel: "Products by risk band",
    });

    Charts.barChart(document.getElementById("supply-chart"), {
      items: Object.entries(data.days_of_supply_distribution).map(
        ([label, value]) => ({
          label, value, color: Charts.COLORS.line,
          title: `${label}: ${value} products`,
        })),
      ariaLabel: "Days of supply distribution",
    });

    Charts.groupedBars(document.getElementById("waste-chart"), {
      rows: Object.entries(data.waste_by_department).map(([dept, d]) => ({
        label: dept,
        values: [
          { name: "hold", value: d.waste_current, color: Charts.COLORS.line },
          { name: "recommended", value: d.waste_recommended,
            color: Charts.COLORS.accent },
        ],
      })),
      format: (v) => fmt.units(v),
      ariaLabel: "Expected waste by department",
    });

    const repriced = data.price_changes.filter(
      (p) => p.recommended < p.current);
    const dumbbellBox = document.getElementById("dumbbell-chart");
    if (repriced.length) {
      Charts.dumbbells(dumbbellBox, {
        rows: repriced
          .slice()
          .sort((a, b) => b.markdown_pct - a.markdown_pct)
          .map((p) => ({ label: p.name, from: p.current, to: p.recommended })),
        format: (v) => "$" + v.toFixed(2),
        ariaLabel: "Current versus recommended prices",
      });
    } else {
      dumbbellBox.innerHTML = `<div class="empty-note">No repriced products
        in this run.</div>`;
    }
  }

  function drawPathBacktest(box, backtest) {
    if (!backtest) {
      box.innerHTML = `<div class="empty-note">The strategy backtest needs
        the synthetic simulator's ground truth, which does not exist for
        imported data.</div>`;
      return;
    }
    const rows = [
      ["Revenue", "revenue", fmt.money],
      ["Gross profit", "gross_profit", fmt.money],
      ["Units sold", "units_sold", fmt.units],
      ["Waste units", "waste_units", fmt.units],
      ["Sell-through", "sell_through", (v) => fmt.pct(100 * v, 0)],
      ["Economic value", "economic_value", fmt.money],
    ];
    const cols = [
      ["hold", "Hold"], ["immediate", "Immediate markdown"],
      ["path", "Optimized path"],
    ];
    box.innerHTML = `<div class="table-wrap">
      <table class="compare strategy-table">
      <thead><tr><th>Metric</th>
        ${cols.map(([key, label]) => `
          <th class="num ${key === "path" ? "col-rec" : ""}">${label}</th>`)
          .join("")}
      </tr></thead>
      <tbody>${rows.map(([label, key, format]) => `
        <tr><td>${label}</td>
          ${cols.map(([ckey]) => `
            <td class="num ${ckey === "path" ? "col-rec" : ""}">
              ${format(backtest[ckey][key])}</td>`).join("")}
        </tr>`).join("")}
      </tbody></table></div>
      <div class="section-note" style="margin-top: 10px">
        ${fmt.num(backtest.n_products)} products,
        ${fmt.num(backtest.n_staged_paths)} with staged (hold-first)
        schedules. Identical noise for all three strategies.</div>`;
  }

  /* ---------- data page ---------- */

  const dataState = {
    uploads: { products: null, transactions: null },
    mappings: { products: {}, transactions: {} },
    validation: null,
    busy: false,
  };

  function mappingFromSuggestions(payload) {
    const mapping = {};
    for (const [fieldName, s] of Object.entries(
      payload.suggested_mapping || {})) {
      mapping[fieldName] = s.column;
    }
    return mapping;
  }

  async function renderData() {
    const status = await api("/api/data/status", { fresh: true });
    const custom = status.custom;
    view.innerHTML = `
      <h1 class="page-title">Data</h1>
      <p class="page-subtitle">Run Self-Shelf on your own retail data.
      Upload a product snapshot and daily sales history, map the columns,
      review validation, and import. Nothing is used until you import, and
      imported files stay on this machine.</p>

      <div class="card">
        <div class="section-title">Data source</div>
        <div class="section-note">Synthetic simulation and imported data are
          kept fully separate — switch between them at any time.</div>
        <div class="mode-toggle" role="group" aria-label="Data source">
          <button data-source="synthetic"
            class="${status.active_source === "synthetic" ? "active" : ""}">
            Synthetic demo</button>
          <button data-source="custom"
            class="${status.active_source === "custom" ? "active" : ""}"
            ${custom.available ? "" : "disabled"}>
            Custom data${custom.available ? "" : " (none imported)"}</button>
        </div>
        <div id="custom-summary" style="margin-top: 18px"></div>
      </div>

      <div class="card" style="margin-top: 24px">
        <div class="section-title">1 — Upload files</div>
        <div class="section-note">CSV files. Products: one row per product
          (id, name, price, cost, inventory, days to expiry). Transactions:
          one row per product per day (date, product id, price, units sold)
          — required, because demand is estimated from observed sales, and
          days missing from the history count as zero-sale days.</div>
        <div class="upload-grid">
          ${["products", "transactions"].map((kind) => `
            <div class="upload-box">
              <h4>${kind === "products" ? "Products (current shelf)"
                : "Transactions (sales history)"}</h4>
              <div class="hint">${kind === "products"
                ? "product_id, product_name, category, current_price, cost, inventory, days_to_expiry, …"
                : "date, product_id, price, units_sold, …"}</div>
              <input type="file" accept=".csv,text/csv" data-kind="${kind}"
                aria-label="Upload ${kind} CSV">
              <div class="file-ok" id="upload-ok-${kind}"></div>
            </div>`).join("")}
        </div>
      </div>

      <div class="card" style="margin-top: 24px" id="mapping-card">
        <div class="section-title">2 — Map columns</div>
        <div class="section-note">Match your CSV columns to Self-Shelf
          fields. Suggestions are pre-filled from the headers; correct any
          that are wrong.</div>
        <div id="mapping-box"><div class="empty-note">Upload both files to
          map their columns.</div></div>
      </div>

      <div class="card" style="margin-top: 24px" id="validate-card">
        <div class="section-title">3 — Validate &amp; import</div>
        <div class="section-note">Rows that fail validation are rejected
          with a reason and can be downloaded — they are never silently
          dropped.</div>
        <div style="display:flex; gap: 12px; margin-bottom: 8px">
          <button class="btn" id="validate-btn" disabled>Validate</button>
          <button class="btn btn-accent" id="import-btn" disabled>
            Import &amp; optimize</button>
        </div>
        <div id="validation-box"></div>
      </div>

      <div class="card" style="margin-top: 24px">
        <div class="section-title">Export</div>
        <div class="section-note">Download the current recommendation set
          (active source: ${esc(status.active_source)}).</div>
        <div style="display:flex; gap: 12px; flex-wrap: wrap">
          <a class="btn" href="/api/export/recommendations.csv" download>
            Recommendations CSV</a>
          <a class="btn" href="/api/export/paths.csv" download>
            Price paths CSV</a>
        </div>
      </div>`;

    drawCustomSummary(document.getElementById("custom-summary"), custom);

    view.querySelectorAll(".mode-toggle button[data-source]").forEach(
      (btn) => {
        btn.addEventListener("click", async () => {
          if (btn.disabled || dataState.busy) return;
          try {
            dataState.busy = true;
            await apiPost("/api/data/source",
              { source: btn.dataset.source });
            clearCache();
            await refreshMode();
            route();
          } catch (err) {
            alertBox(`Could not switch source: ${err.message}`);
          } finally {
            dataState.busy = false;
          }
        });
      });

    view.querySelectorAll("input[type=file][data-kind]").forEach((input) => {
      input.addEventListener("change", async () => {
        const kind = input.dataset.kind;
        const file = input.files && input.files[0];
        if (!file) return;
        const okBox = document.getElementById(`upload-ok-${kind}`);
        okBox.textContent = "Uploading…";
        try {
          const payload = await apiUpload(
            `/api/data/upload?kind=${kind}`, file);
          dataState.uploads[kind] = payload;
          dataState.mappings[kind] = mappingFromSuggestions(payload);
          dataState.validation = null;
          okBox.textContent = `${payload.filename} — `
            + `${fmt.num(payload.row_count)} rows, `
            + `${payload.columns.length} columns`;
          drawMapping();
          syncDataButtons();
        } catch (err) {
          okBox.textContent = "";
          alertBox(`Upload failed: ${err.message}`);
        }
      });
    });

    document.getElementById("validate-btn").addEventListener(
      "click", () => runValidation(false));
    document.getElementById("import-btn").addEventListener(
      "click", () => runValidation(true));

    drawMapping();
    syncDataButtons();
  }

  function alertBox(message) {
    const box = document.getElementById("validation-box");
    if (box) {
      box.innerHTML = `<div class="path-errors"><div>${esc(message)}</div></div>`
        + box.innerHTML;
    }
  }

  function drawCustomSummary(box, custom) {
    if (!custom.available || !custom.meta) {
      box.innerHTML = `<div class="section-note" style="margin:0">
        No custom dataset imported yet. The dashboard is running on the
        synthetic demo simulation.</div>`;
      if (custom.error) {
        box.innerHTML += `<div class="path-errors">
          <div>${esc(custom.error)}</div></div>`;
      }
      return;
    }
    const meta = custom.meta;
    const q = meta.quality || {};
    const range = q.date_range
      ? `${q.date_range.start} → ${q.date_range.end}` : "—";
    box.innerHTML = `
      <div class="quality-grid">
        <div class="quality-cell"><div class="n">${fmt.num(q.products)}</div>
          <div class="l">products</div></div>
        <div class="quality-cell"><div class="n">${fmt.num(q.transactions)}</div>
          <div class="l">transactions</div></div>
        <div class="quality-cell"><div class="n">${fmt.num(q.products_with_history)}</div>
          <div class="l">with sales history</div></div>
        <div class="quality-cell"><div class="n">${fmt.num(q.products_without_history)}</div>
          <div class="l">using fallback demand</div></div>
        <div class="quality-cell"><div class="n" style="font-size:13px; margin-top:4px">${range}</div>
          <div class="l">history range</div></div>
      </div>
      <div class="section-note" style="margin: 14px 0 0">
        Imported ${meta.imported_at
          ? new Date(meta.imported_at).toLocaleString() : "—"} ·
        elasticities by category:</div>
      <ul class="issue-list">
        ${Object.entries(meta.elasticities || {}).map(([cat, e]) => `
          <li class="${e.source === "estimated" ? "ok" : ""}">
            <strong>${esc(cat)}</strong>: ${fmt.num(e.elasticity, 2)}
            — ${e.source === "estimated"
              ? `estimated from ${fmt.num(e.n_observations)} observations`
              : `fallback (${esc(e.reason || "insufficient data")})`}
          </li>`).join("")}
      </ul>`;
  }

  function drawMapping() {
    const box = document.getElementById("mapping-box");
    if (!box) return;
    const kinds = ["products", "transactions"];
    if (!kinds.every((k) => dataState.uploads[k])) {
      box.innerHTML = `<div class="empty-note">Upload both files to map
        their columns.</div>`;
      return;
    }
    box.innerHTML = kinds.map((kind) => {
      const upload = dataState.uploads[kind];
      const fields = [
        ...upload.required_fields.map((f) => [f, true]),
        ...upload.optional_fields.map((f) => [f, false]),
      ];
      return `
        <div style="margin-top: 8px">
          <h4 style="font-size:13.5px; margin-bottom: 2px">
            ${kind === "products" ? "Products file" : "Transactions file"}
            <span class="map-conf">(${esc(upload.filename)})</span></h4>
          <div class="map-grid">
            ${fields.map(([fieldName, required]) => {
              const chosen = dataState.mappings[kind][fieldName] || "";
              const suggestion = (upload.suggested_mapping || {})[fieldName];
              return `
                <span class="fname">${fieldName}
                  ${required ? '<span class="req">*</span>' : ""}</span>
                <select class="select" data-kind="${kind}"
                  data-field="${fieldName}"
                  aria-label="${kind} column for ${fieldName}">
                  <option value="">— not in file —</option>
                  ${upload.columns.map((c) => `
                    <option value="${esc(c)}"
                      ${c === chosen ? "selected" : ""}>${esc(c)}</option>`)
                    .join("")}
                </select>
                <span class="map-conf ${suggestion
                  && suggestion.column === chosen
                  && suggestion.confidence === "exact" ? "exact" : ""}">
                  ${suggestion && suggestion.column === chosen
                    ? suggestion.confidence + " match" : ""}</span>`;
            }).join("")}
          </div>
        </div>`;
    }).join("");

    box.querySelectorAll("select[data-kind]").forEach((select) => {
      select.addEventListener("change", () => {
        const kind = select.dataset.kind;
        if (select.value) {
          dataState.mappings[kind][select.dataset.field] = select.value;
        } else {
          delete dataState.mappings[kind][select.dataset.field];
        }
        dataState.validation = null;
        syncDataButtons();
      });
    });
  }

  function syncDataButtons() {
    const validateBtn = document.getElementById("validate-btn");
    const importBtn = document.getElementById("import-btn");
    if (!validateBtn) return;
    const uploaded = dataState.uploads.products
      && dataState.uploads.transactions;
    validateBtn.disabled = !uploaded || dataState.busy;
    importBtn.disabled = !uploaded || dataState.busy
      || !(dataState.validation && dataState.validation.can_import);
  }

  async function runValidation(thenImport) {
    const box = document.getElementById("validation-box");
    dataState.busy = true;
    syncDataButtons();
    box.innerHTML = `<div class="section-note">Validating…</div>`;
    try {
      const payload = {
        products_mapping: dataState.mappings.products,
        transactions_mapping: dataState.mappings.transactions,
      };
      const report = await apiPost("/api/data/validate", payload);
      dataState.validation = report;
      drawValidationReport(box, report);
      if (thenImport && report.can_import) {
        box.insertAdjacentHTML("afterbegin",
          `<div class="section-note">Importing and optimizing…</div>`);
        const result = await apiPost("/api/data/import", payload);
        clearCache();
        await refreshMode();
        dataState.validation = null;
        route();
        return;
      }
    } catch (err) {
      box.innerHTML = `<div class="path-errors">
        <div>${esc(err.message)}</div></div>`;
    } finally {
      dataState.busy = false;
      syncDataButtons();
    }
  }

  function drawValidationReport(box, report) {
    const section = (kind, r) => {
      const issues = (r.issues || []).map(
        (i) => `<li>${esc(i.message)}</li>`).join("");
      const errors = (r.errors || []).map(
        (e) => `<li class="err">${esc(e)}</li>`).join("");
      const warnings = (r.warnings || []).map(
        (w) => `<li>${esc(w)}</li>`).join("");
      const preview = (r.preview && r.preview.length)
        ? `<div class="table-wrap" style="margin-top:10px"><table>
            <thead><tr>${Object.keys(r.preview[0]).map(
              (c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
            <tbody>${r.preview.map((row) => `<tr>${Object.values(row).map(
              (v) => `<td>${esc(v == null ? "—" : v)}</td>`).join("")}
              </tr>`).join("")}</tbody>
          </table></div>` : "";
      return `
        <div style="margin-top: 16px">
          <h4 style="font-size: 13.5px">
            ${kind === "products" ? "Products" : "Transactions"} —
            <span class="${r.rows_rejected ? "delta-warn" : "delta-pos"}">
              ${fmt.num(r.rows_valid)} valid</span> /
            ${fmt.num(r.rows_total)} rows
            ${r.rows_rejected
              ? `· <a href="/api/data/rejected?kind=${kind}" download>
                  download ${fmt.num(r.rows_rejected)} rejected</a>` : ""}
          </h4>
          ${errors ? `<ul class="issue-list">${errors}</ul>` : ""}
          ${issues ? `<ul class="issue-list">${issues}</ul>` : ""}
          ${warnings ? `<ul class="issue-list">${warnings}</ul>` : ""}
          ${preview}
        </div>`;
    };
    box.innerHTML = `
      ${report.can_import
        ? `<div class="section-note" style="color: var(--positive)">
            Validation passed — ready to import.</div>`
        : `<div class="path-errors"><div>Fix the highlighted problems
            before importing.</div></div>`}
      ${section("products", report.products)}
      ${section("transactions", report.transactions)}`;
  }

  /* ---------- router / boot ---------- */

  const routes = [
    { pattern: /^#\/products\/(.+)$/, nav: "products",
      render: (m) => renderProductDetail(decodeURIComponent(m[1])) },
    { pattern: /^#\/products$/, nav: "products", render: renderProducts },
    { pattern: /^#\/pricing$/, nav: "pricing", render: renderPricing },
    { pattern: /^#\/analytics$/, nav: "analytics", render: renderAnalytics },
    { pattern: /^#\/data$/, nav: "data", render: renderData },
    { pattern: /^#\/overview$|^$|^#\/?$/, nav: "overview",
      render: renderOverview },
  ];

  function route() {
    const hash = location.hash || "#/overview";
    for (const r of routes) {
      const match = hash.match(r.pattern);
      if (match) {
        document.querySelectorAll(".nav a").forEach((a) =>
          a.classList.toggle("active", a.dataset.nav === r.nav));
        guarded(() => r.render(match));
        view.focus({ preventScroll: true });
        window.scrollTo(0, 0);
        return;
      }
    }
    location.hash = "#/overview";
  }

  async function boot() {
    renderComputing();
    try {
      const status = await refreshMode();
      if (!status.ready) {
        if (status.error) {
          renderError(status.error, boot);
          return;
        }
        setTimeout(boot, 1200);
        return;
      }
      if (status.generated_at) {
        const at = new Date(status.generated_at);
        lastUpdated.textContent = "Updated " + at.toLocaleTimeString([], {
          hour: "2-digit", minute: "2-digit",
        });
      }
      route();
    } catch (err) {
      renderError("The Self-Shelf server is not responding. " + err.message,
        boot);
    }
  }

  window.addEventListener("hashchange", route);
  boot();
})();
