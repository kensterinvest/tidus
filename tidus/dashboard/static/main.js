/* Tidus Dashboard — main.js
 * Fetches /api/v1/dashboard/summary every 30s and updates all panels.
 * No build step required — vanilla ES2020.
 */

const REFRESH_INTERVAL_MS = 30_000;
let costChart = null;
let activeDays = 7;

// ── Safety: HTML escape to prevent XSS ────────────────────────────────────────

function esc(str) {
  const d = document.createElement("div");
  d.textContent = String(str ?? "");
  return d.innerHTML;
}

// ── Fetch ──────────────────────────────────────────────────────────────────────

async function fetchSummary(days) {
  const resp = await fetch(`/api/v1/dashboard/summary?days=${days}`);
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  return resp.json();
}

// ── Days toggle ────────────────────────────────────────────────────────────────

function initDaysToggle() {
  document.getElementById("days-toggle").addEventListener("click", (e) => {
    const btn = e.target.closest(".toggle-btn");
    if (!btn) return;
    activeDays = parseInt(btn.dataset.days, 10);
    document.querySelectorAll(".toggle-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    refresh();
  });
}

// ── Overview KPIs ──────────────────────────────────────────────────────────────

function renderOverview(cost, savings, days) {
  const dayLabel = days === 1 ? "today" : `last ${days} days`;
  document.getElementById("kpi-cost-label").textContent  = `${days}d Cost`;
  document.getElementById("kpi-requests-label").textContent = "Total Requests";
  document.getElementById("kpi-requests-sub").textContent   = dayLabel;

  document.getElementById("kpi-total-cost").textContent =
    `$${cost.total_cost_usd.toFixed(4)}`;
  document.getElementById("kpi-total-requests").textContent =
    cost.total_requests.toLocaleString();
  document.getElementById("kpi-requests-today").textContent =
    cost.requests_today.toLocaleString();
  document.getElementById("kpi-avg-cost").textContent =
    `$${cost.avg_cost_per_request_usd.toFixed(6)}`;
  document.getElementById("kpi-most-used").textContent =
    cost.most_used_model ?? "—";

  // Savings KPI
  if (savings && savings.savings_usd > 0) {
    const savedFmt = savings.savings_usd >= 1
      ? `$${savings.savings_usd.toFixed(2)}`
      : `$${savings.savings_usd.toFixed(4)}`;
    document.getElementById("kpi-savings-usd").textContent =
      `${savedFmt} (${savings.savings_pct.toFixed(1)}%)`;
    document.getElementById("kpi-baseline-model").textContent =
      savings.baseline_model_id;
  } else if (cost.total_requests === 0) {
    document.getElementById("kpi-savings-usd").textContent = "—";
  } else {
    document.getElementById("kpi-savings-usd").textContent = "$0.00 (0%)";
  }
}

// ── Empty state ────────────────────────────────────────────────────────────────

function setEmptyState(isEmpty) {
  document.getElementById("empty-state").hidden  = !isEmpty;
  document.getElementById("chart-section").hidden = isEmpty;
}

// ── Cost-by-Model Chart ────────────────────────────────────────────────────────

function renderCostChart(rows, days) {
  document.getElementById("chart-title").textContent =
    `Cost by Model (${days === 1 ? "today" : days + " days"})`;

  const active = rows
    .filter(r => r.requests > 0)
    .sort((a, b) => b.total_cost_usd - a.total_cost_usd)
    .slice(0, 12);

  const labels = active.map(r => r.model_id);
  const data   = active.map(r => r.total_cost_usd);
  const colors = active.map(r => tierColor(r.tier));

  const ctx = document.getElementById("cost-chart").getContext("2d");

  if (costChart) {
    costChart.data.labels = labels;
    costChart.data.datasets[0].data = data;
    costChart.data.datasets[0].backgroundColor = colors;
    costChart.data.datasets[0].label = `Cost (USD, ${days}d)`;
    costChart.update();
    return;
  }

  costChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: `Cost (USD, ${days}d)`,
        data,
        backgroundColor: colors,
        borderRadius: 4,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `$${ctx.parsed.y.toFixed(6)}`,
          },
        },
      },
      scales: {
        x: {
          ticks: { color: "#94a3b8", maxRotation: 35 },
          grid: { color: "#2a2d3a" },
        },
        y: {
          ticks: {
            color: "#94a3b8",
            callback: v => `$${v.toFixed(4)}`,
          },
          grid: { color: "#2a2d3a" },
        },
      },
    },
  });
}

function tierColor(tier) {
  switch (tier) {
    case 1: return "#f87171";  // premium → red
    case 2: return "#fbbf24";  // mid     → yellow
    case 3: return "#4f8ef7";  // economy → blue
    default: return "#34d399"; // local   → green
  }
}

// ── Budget Bars ────────────────────────────────────────────────────────────────

function renderBudgets(budgets) {
  const el = document.getElementById("budget-list");

  if (!budgets.length) {
    el.textContent = "No budget policies configured";
    el.className = "empty";
    return;
  }

  el.textContent = "";
  el.className = "budget-list";

  budgets.forEach(b => {
    const pct    = b.utilisation_pct ?? 0;
    const warnAt = (b.warn_threshold_pct ?? 0.8) * 100;
    const fillClass = b.is_hard_stopped ? "alert" : pct >= warnAt ? "warn" : "";
    const limitStr  = b.limit_usd != null ? `$${b.limit_usd.toFixed(2)}` : "unlimited";

    const row = document.createElement("div");
    row.className = "budget-row";

    const header = document.createElement("div");
    header.className = "header";

    const teamEl = document.createElement("span");
    teamEl.className = "team";
    teamEl.textContent = b.team_id;

    const right = document.createElement("div");
    right.style.cssText = "display:flex;gap:8px;align-items:center";

    const badge = document.createElement("span");
    badge.className = `badge ${b.is_hard_stopped ? "red" : pct >= warnAt ? "yellow" : "green"}`;
    badge.textContent = b.is_hard_stopped ? "STOPPED" : pct >= warnAt ? "WARN" : "OK";

    const pctEl = document.createElement("span");
    pctEl.className = "pct";
    pctEl.textContent = `$${b.spent_usd.toFixed(4)} / ${limitStr} (${pct.toFixed(1)}%)`;

    right.append(badge, pctEl);
    header.append(teamEl, right);

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = `bar-fill ${fillClass}`;
    fill.style.width = `${Math.min(pct, 100)}%`;

    track.append(fill);
    row.append(header, track);
    el.append(row);
  });
}

// ── Sessions Table ─────────────────────────────────────────────────────────────

function renderSessions(sessions) {
  const tbody = document.getElementById("sessions-tbody");
  tbody.textContent = "";

  if (!sessions.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 6;
    td.className = "empty";
    td.textContent = "No active agent sessions";
    tr.append(td);
    tbody.append(tr);
    return;
  }

  sessions.forEach(s => {
    const depth = s.agent_depth;
    const badgeClass = depth >= 4 ? "red" : depth >= 2 ? "yellow" : "green";
    const started = new Date(s.started_at).toLocaleTimeString();

    const tr = document.createElement("tr");

    function td(text, mono = false) {
      const el = document.createElement("td");
      el.textContent = text;
      if (mono) { el.style.fontFamily = "monospace"; el.style.fontSize = "11px"; }
      return el;
    }

    const depthTd = document.createElement("td");
    const depthBadge = document.createElement("span");
    depthBadge.className = `badge ${badgeClass}`;
    depthBadge.textContent = depth;
    depthTd.append(depthBadge);

    tr.append(
      td(`${s.session_id.slice(0, 12)}…`, true),
      td(s.team_id),
      depthTd,
      td(s.step_count),
      td(s.total_tokens.toLocaleString()),
      td(started),
    );
    tbody.append(tr);
  });
}

// ── Registry Health ─────────────────────────────────────────────────────────────

function renderHealth(healthMap) {
  const entries = Object.entries(healthMap);
  const enabled = entries.filter(([, ok]) => ok).length;
  document.getElementById("health-summary").textContent =
    `${enabled} / ${entries.length} models enabled`;

  const el = document.getElementById("health-list");
  el.textContent = "";

  entries
    .sort(([a], [b]) => a.localeCompare(b))
    .forEach(([model, ok]) => {
      const span = document.createElement("span");
      span.className = `badge ${ok ? "green" : "red"}`;
      span.title = model;
      span.style.cssText = "margin:3px;cursor:default";
      span.textContent = `${ok ? "●" : "✕"} ${model}`;
      el.append(span);
    });
}

// ── Main render ────────────────────────────────────────────────────────────────

async function refresh() {
  try {
    const data = await fetchSummary(activeDays);
    const isEmpty = data.cost.total_requests === 0;

    renderOverview(data.cost, data.savings, activeDays);
    setEmptyState(isEmpty);

    if (!isEmpty) {
      renderCostChart(data.cost_by_model, activeDays);
    }

    renderBudgets(data.budgets);
    renderSessions(data.sessions);
    renderHealth(data.registry_health);

    document.getElementById("last-updated").textContent =
      new Date().toLocaleTimeString();
    document.getElementById("error-banner").hidden = true;
  } catch (err) {
    const banner = document.getElementById("error-banner");
    banner.textContent = `Refresh failed: ${err.message}`;
    banner.hidden = false;
    console.error("Dashboard refresh error:", err);
  }
}

// ── Bootstrap ──────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  initDaysToggle();
  refresh();
  setInterval(refresh, REFRESH_INTERVAL_MS);
});
