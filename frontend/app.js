// ═══════════════════════════════════════════════════════════════
// Stock Data Intelligence Platform — Dashboard Frontend Logic
// Author: Mahim Yadav | JarNox SWE Internship — Part 3
//
// Features:
//   • Closing price chart with gradient fill + SMA overlay
//   • Volume bar chart (synchronized)
//   • AI prediction line (linear regression via backend)
//   • Head-to-head stock comparison with scoring
//   • Market heatmap (all stocks at a glance)
//   • Sector performance bars
//   • Time-range filters (1M, 3M, 6M, 1Y, ALL)
//   • Search, toast notifications, responsive
// ═══════════════════════════════════════════════════════════════

// Detect how the page is being served:
//   - Via FastAPI server → use same origin (e.g. http://127.0.0.1:8000)
//   - Direct file open (file://) → fall back to localhost:8000
const API_BASE = (
    !window.location.origin ||
    window.location.origin === 'null' ||
    window.location.protocol === 'file:' ||
    ((window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost') && window.location.port !== '8000')
) ? 'http://127.0.0.1:8000' : window.location.origin;

// ── State ────────────────────────────────────────────────────
let allCompanies = [];
let activeSymbol = null;
let activeDays = 30;
let mainChartInstance = null;
let volumeChartInstance = null;
let compareChartInstance = null;
let isPredictMode = false;
let cachedStockData = {};  // Lightweight cache

// ── DOM Refs ─────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const elStockList = $('#stock-list');
const elSearch = $('#stock-search');
const elStockCount = $('#stock-count');
const elTopMovers = $('#top-movers');
const elChartLoader = $('#chart-loader');

// ═══════════════════════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════════════════════
async function init() {
    // Chart.js global defaults
    Chart.defaults.color = '#64748b';
    Chart.defaults.font.family = "'Inter', 'Outfit', sans-serif";
    Chart.defaults.font.size = 11;
    Chart.defaults.elements.point.radius = 0;
    Chart.defaults.elements.point.hoverRadius = 5;

    await Promise.all([
        fetchCompanies(),
        fetchTopMovers(),
        fetchSectors(),
    ]);

    bindEvents();
}

function bindEvents() {
    elSearch.addEventListener('input', (e) => filterStocks(e.target.value));

    // Timeframe toggles
    $$('.toggle-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            $$('.toggle-btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            activeDays = parseInt(e.target.dataset.days);
            if (activeSymbol) loadStockData(activeSymbol);
        });
    });

    // Predict
    $('#btn-predict').addEventListener('click', togglePredict);

    // Compare Modal
    $('#btn-compare').addEventListener('click', openCompareModal);
    $('#close-compare').addEventListener('click', () => {
        $('#compare-modal').classList.add('hidden');
    });
    $('#compare-modal').addEventListener('click', (e) => {
        if (e.target === $('#compare-modal')) {
            $('#compare-modal').classList.add('hidden');
        }
    });
    $('#btn-run-compare').addEventListener('click', runCompare);
}

// ═══════════════════════════════════════════════════════════════
// Toast Notifications
// ═══════════════════════════════════════════════════════════════
function showToast(msg, type = 'info') {
    const container = $('#toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const icons = {
        info: 'ph-info',
        success: 'ph-check-circle',
        error: 'ph-warning-circle',
    };

    toast.innerHTML = `<i class="ph ${icons[type] || icons.info}"></i> ${msg}`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(30px)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}

// ═══════════════════════════════════════════════════════════════
// API Fetchers
// ═══════════════════════════════════════════════════════════════
async function fetchCompanies() {
    try {
        const res = await fetch(`${API_BASE}/companies`);
        const data = await res.json();
        allCompanies = data.companies;
        elStockCount.innerText = data.count;
        renderStockList(allCompanies);

        // Auto-select first stock
        if (allCompanies.length > 0) {
            selectStock(allCompanies[0].symbol);

            // Populate compare selects
            const s1 = $('#compare-stock-1');
            const s2 = $('#compare-stock-2');
            s1.innerHTML = '';
            s2.innerHTML = '';
            allCompanies.forEach(c => {
                const opt1 = `<option value="${c.symbol}">${c.symbol} — ${c.name}</option>`;
                s1.innerHTML += opt1;
                s2.innerHTML += opt1;
            });
        }

        // Build heatmap
        buildHeatmap(allCompanies);
    } catch (err) {
        console.error("Failed to load companies", err);
        showToast("Cannot connect to backend API. Start with: python app.py", "error");
    }
}

async function fetchTopMovers() {
    try {
        const res = await fetch(`${API_BASE}/market/movers?n=3`);
        const data = await res.json();

        let html = '';
        if (data.gainers?.length > 0) {
            const g = data.gainers[0];
            html += `<div class="mover-chip gainer">
                <i class="ph ph-trend-up"></i> ${g.symbol} <strong>+${g.return_30d_pct}%</strong>
            </div>`;
        }
        if (data.losers?.length > 0) {
            const l = data.losers[0];
            html += `<div class="mover-chip loser">
                <i class="ph ph-trend-down"></i> ${l.symbol} <strong>${l.return_30d_pct}%</strong>
            </div>`;
        }
        elTopMovers.innerHTML = html;
    } catch (err) {
        elTopMovers.innerHTML = '<span class="text-dim" style="font-size:0.8rem">Offline</span>';
    }
}

async function fetchSectors() {
    try {
        const res = await fetch(`${API_BASE}/sectors`);
        const data = await res.json();
        renderSectorBars(data.sectors || []);
    } catch (err) {
        // Silently fail — sector overview is supplementary
    }
}

// ═══════════════════════════════════════════════════════════════
// Sidebar — Stock List Rendering
// ═══════════════════════════════════════════════════════════════
function renderStockList(companies) {
    elStockList.innerHTML = '';
    companies.forEach((c, idx) => {
        const li = document.createElement('li');
        li.className = `stock-item ${c.symbol === activeSymbol ? 'active' : ''} fade-in`;
        li.style.animationDelay = `${idx * 25}ms`;
        li.onclick = () => selectStock(c.symbol);

        const isNearHigh = c.pct_from_52w_high > -10;
        const isDeepLow = c.pct_from_52w_high < -30;
        const colorClass = isNearHigh ? 'up' : (isDeepLow ? 'down' : 'text-dim');

        li.innerHTML = `
            <div class="stock-meta">
                <strong>${c.symbol}</strong>
                <span>${truncate(c.sector, 18)}</span>
            </div>
            <div class="stock-price">
                <span class="price">₹${formatNumber(c.latest_close)}</span>
                <span class="change ${colorClass}">${c.pct_from_52w_high}% 52WH</span>
            </div>
        `;
        elStockList.appendChild(li);
    });
}

function filterStocks(query) {
    const q = query.toLowerCase().trim();
    const filtered = allCompanies.filter(c =>
        c.symbol.toLowerCase().includes(q) ||
        c.name.toLowerCase().includes(q) ||
        c.sector.toLowerCase().includes(q)
    );
    renderStockList(filtered);
}

// ═══════════════════════════════════════════════════════════════
// Stock Selection & Data Loading
// ═══════════════════════════════════════════════════════════════
function selectStock(symbol) {
    activeSymbol = symbol;
    isPredictMode = false;
    $('#btn-predict').classList.remove('active-predict');

    // Update sidebar active state
    $$('.stock-item').forEach(el => el.classList.remove('active'));
    const items = $$('.stock-item');
    items.forEach(el => {
        const strong = el.querySelector('.stock-meta strong');
        if (strong && strong.textContent === symbol) el.classList.add('active');
    });

    // Load Data  
    loadStockData(symbol);
    loadStockSummary(symbol);

    // Update heatmap active state
    $$('.heat-cell').forEach(el => {
        el.classList.toggle('active', el.dataset.symbol === symbol);
    });
}

async function loadStockData(symbol) {
    elChartLoader.classList.remove('hidden');
    try {
        const cacheKey = `${symbol}_${activeDays}`;
        let data;

        if (cachedStockData[cacheKey]) {
            data = cachedStockData[cacheKey];
        } else {
            const res = await fetch(`${API_BASE}/data/${symbol}?days=${activeDays}`);
            data = await res.json();
            cachedStockData[cacheKey] = data;
        }

        // Update header info
        $('#active-symbol').innerText = data.symbol;
        $('#active-company').innerText = data.company;

        const company = allCompanies.find(c => c.symbol === data.symbol);
        const sectorEl = $('#active-sector');
        if (company) {
            sectorEl.innerText = company.sector;
            sectorEl.classList.remove('hidden');
        }

        const latest = data.data[data.data.length - 1];
        const first = data.data[0];
        $('#active-price').innerText = `₹${formatNumber(latest.close)}`;

        const pctChange = ((latest.close - first.close) / first.close * 100).toFixed(2);
        const elChange = $('#active-change');
        elChange.innerText = `${pctChange > 0 ? '+' : ''}${pctChange}%`;
        elChange.className = `change-badge ${pctChange >= 0 ? 'up' : 'down'}`;

        renderMainChart(data.data, data.symbol);
        renderVolumeChart(data.data);
    } catch (err) {
        showToast(`Failed to load data for ${symbol}`, 'error');
    }
    elChartLoader.classList.add('hidden');
}

async function loadStockSummary(symbol) {
    const panel = $('#insights-panel');
    panel.innerHTML = '<div class="loader-spinner"></div>';

    try {
        const res = await fetch(`${API_BASE}/summary/${symbol}`);
        const data = await res.json();
        renderInsights(data);
    } catch (err) {
        panel.innerHTML = '<div class="empty-state"><i class="ph ph-warning"></i><p>Failed to load analytics.</p></div>';
    }
}

// ═══════════════════════════════════════════════════════════════
// Insights Rendering
// ═══════════════════════════════════════════════════════════════
function renderInsights(data) {
    const panel = $('#insights-panel');
    const { risk, custom_metrics, technicals } = data;

    // Determine RSI color
    const rsi = technicals.rsi || 50;
    const rsiColor = rsi > 70 ? 'var(--down)' : rsi < 30 ? 'var(--up)' : 'var(--accent)';
    const rsiLabel = rsi > 70 ? 'Overbought' : rsi < 30 ? 'Oversold' : 'Neutral';

    // Momentum -> gauge
    const momentum = custom_metrics.momentum_score || 50;
    const momClass = momentum > 65 ? 'bullish' : momentum < 35 ? 'bearish' : '';

    // Sentiment
    const sentLabel = custom_metrics.sentiment_label || 'Neutral';
    const sentIdx = custom_metrics.sentiment_index || 50;
    const sentColor = sentLabel === 'Bullish' ? 'var(--up)' :
                      sentLabel === 'Bearish' ? 'var(--down)' : 'var(--warning)';

    // Sharpe
    const sharpe = risk.sharpe_ratio ?? '—';
    const sharpeColor = sharpe > 0.5 ? 'var(--up)' : sharpe > 0 ? 'var(--warning)' : 'var(--down)';

    // Beta
    const beta = risk.beta != null ? risk.beta.toFixed(2) : '—';

    // Max Drawdown
    const maxDD = risk.max_drawdown_pct != null ? risk.max_drawdown_pct.toFixed(1) : '—';

    // VaR
    const var95 = risk.var_95_daily_pct != null ? risk.var_95_daily_pct.toFixed(2) : '—';

    // 52W info
    const pctFrom52WH = data.pct_from_52w_high != null ? data.pct_from_52w_high : 0;
    const posIn52WRange = data.position_in_52w_range != null ? data.position_in_52w_range : 50;

    panel.innerHTML = `
        <div class="metric-grid-2">
            <!-- Momentum Score with gauge -->
            <div class="metric-card fade-in fade-in-delay-1">
                <div class="metric-label">Momentum Score</div>
                <div class="metric-value" style="color: ${momentum > 65 ? 'var(--up)' : momentum < 35 ? 'var(--down)' : 'var(--accent)'}">${momentum}</div>
                <div class="metric-sub">${momentum > 65 ? '🔥 Strong Bullish' : momentum > 50 ? '📈 Bullish' : momentum > 35 ? '↔ Neutral' : '📉 Bearish'}</div>
                <div class="mini-gauge"><div class="mini-gauge-fill ${momClass}" style="width: ${momentum}%"></div></div>
            </div>

            <!-- Sentiment -->
            <div class="metric-card fade-in fade-in-delay-2">
                <div class="metric-label">AI Sentiment</div>
                <div class="metric-value" style="color: ${sentColor}">${sentLabel}</div>
                <div class="metric-sub">Index: ${sentIdx}</div>
                <div class="mini-gauge"><div class="mini-gauge-fill ${sentLabel === 'Bullish' ? 'bullish' : sentLabel === 'Bearish' ? 'bearish' : ''}" style="width: ${Math.min(sentIdx, 100)}%"></div></div>
            </div>

            <!-- RSI -->
            <div class="metric-card fade-in fade-in-delay-3">
                <div class="metric-label">RSI (${technicals.rsi_signal || 'N/A'})</div>
                <div class="metric-value" style="color: ${rsiColor}">${rsi}</div>
                <div class="metric-sub">${rsiLabel}</div>
                <div class="mini-gauge"><div class="mini-gauge-fill" style="width: ${Math.min(rsi, 100)}%; background: ${rsiColor}"></div></div>
            </div>

            <!-- Sharpe Ratio -->
            <div class="metric-card fade-in fade-in-delay-4">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value" style="color: ${sharpeColor}">${sharpe}</div>
                <div class="metric-sub">Risk/Reward</div>
            </div>

            <!-- Beta -->
            <div class="metric-card fade-in">
                <div class="metric-label">Beta</div>
                <div class="metric-value">${beta}</div>
                <div class="metric-sub">vs NIFTY 50</div>
            </div>

            <!-- Volatility -->
            <div class="metric-card fade-in">
                <div class="metric-label">Annual Vol</div>
                <div class="metric-value" style="color: var(--warning)">${risk.annual_volatility_pct != null ? risk.annual_volatility_pct.toFixed(1) + '%' : '—'}</div>
                <div class="metric-sub">Annualized</div>
            </div>

            <!-- 52-Week Range Position -->
            <div class="metric-card full-width fade-in">
                <div class="metric-label">52-Week Range Position</div>
                <div style="display:flex; justify-content:space-between; align-items:baseline; margin-top:0.3rem;">
                    <span style="font-size:0.8rem; color:var(--text-dim)">₹${formatNumber(data.low_52w)}</span>
                    <span class="metric-value" style="font-size:1.1rem">${posIn52WRange.toFixed(0)}%</span>
                    <span style="font-size:0.8rem; color:var(--text-dim)">₹${formatNumber(data.high_52w)}</span>
                </div>
                <div class="mini-gauge" style="margin-top:0.4rem">
                    <div class="mini-gauge-fill" style="width: ${posIn52WRange}%; background: linear-gradient(90deg, var(--down), var(--warning), var(--up))"></div>
                </div>
                <div class="metric-sub" style="text-align:center; margin-top:0.25rem">${pctFrom52WH}% from 52W High</div>
            </div>

            <!-- Risk Snapshot -->
            <div class="metric-card fade-in">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value" style="color: var(--down)">${maxDD}%</div>
                <div class="metric-sub">Peak-to-Trough</div>
            </div>

            <div class="metric-card fade-in">
                <div class="metric-label">VaR (95%)</div>
                <div class="metric-value">${var95}%</div>
                <div class="metric-sub">Daily worst case</div>
            </div>

            <!-- Returns -->
            <div class="metric-card fade-in">
                <div class="metric-label">30D Return</div>
                <div class="metric-value ${data.return_30d_pct >= 0 ? 'up' : 'down'}">${data.return_30d_pct != null ? (data.return_30d_pct >= 0 ? '+' : '') + data.return_30d_pct + '%' : '—'}</div>
            </div>

            <div class="metric-card fade-in">
                <div class="metric-label">YTD Return</div>
                <div class="metric-value ${data.return_ytd_pct >= 0 ? 'up' : 'down'}">${data.return_ytd_pct != null ? (data.return_ytd_pct >= 0 ? '+' : '') + data.return_ytd_pct + '%' : '—'}</div>
            </div>
        </div>
    `;
}

// ═══════════════════════════════════════════════════════════════
// Chart Rendering — Main Price Chart
// ═══════════════════════════════════════════════════════════════
function renderMainChart(historyData, symbol, predictionData = null) {
    const ctx = $('#mainChart').getContext('2d');

    if (mainChartInstance) mainChartInstance.destroy();

    const labels = historyData.map(d => d.date);
    const closePrices = historyData.map(d => d.close);
    const smaValues = historyData.map(d => d.sma_7);

    // Price gradient fill
    const gradient = ctx.createLinearGradient(0, 0, 0, ctx.canvas.clientHeight || 300);
    gradient.addColorStop(0, 'rgba(59, 130, 246, 0.18)');
    gradient.addColorStop(0.5, 'rgba(59, 130, 246, 0.06)');
    gradient.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

    const datasets = [
        {
            label: `${symbol} Close`,
            data: closePrices,
            borderColor: '#3b82f6',
            backgroundColor: gradient,
            borderWidth: 2.2,
            tension: 0.25,
            fill: true,
            pointHoverBackgroundColor: '#3b82f6',
            pointHoverBorderColor: '#fff',
            pointHoverBorderWidth: 2,
        },
    ];

    // SMA-7 line (subtle)
    if (smaValues.some(v => v !== null && v !== undefined)) {
        datasets.push({
            label: 'SMA-7',
            data: smaValues,
            borderColor: 'rgba(168, 85, 247, 0.5)',
            borderWidth: 1.5,
            borderDash: [4, 4],
            tension: 0.3,
            fill: false,
            pointRadius: 0,
            pointHoverRadius: 0,
        });
    }

    // Prediction line
    if (predictionData) {
        const lastIdx = labels.length - 1;
        const lastVal = closePrices[lastIdx];

        predictionData.predictions.forEach(p => {
            labels.push(p.date);
        });

        const predArray = Array(labels.length).fill(null);
        predArray[lastIdx] = lastVal;

        predictionData.predictions.forEach((p, idx) => {
            predArray[lastIdx + 1 + idx] = p.predicted_close;
        });

        const isUp = predictionData.trend_direction === "Up";
        const predColor = isUp ? '#22c55e' : '#ef4444';

        // Prediction gradient
        const predGradient = ctx.createLinearGradient(0, 0, 0, ctx.canvas.clientHeight || 300);
        predGradient.addColorStop(0, isUp ? 'rgba(34, 197, 94, 0.12)' : 'rgba(239, 68, 68, 0.12)');
        predGradient.addColorStop(1, 'transparent');

        datasets.push({
            label: `Prediction (${predictionData.model_type})`,
            data: predArray,
            borderColor: predColor,
            backgroundColor: predGradient,
            borderWidth: 2,
            borderDash: [6, 4],
            fill: true,
            pointRadius: 3,
            pointBackgroundColor: predColor,
            pointBorderColor: 'rgba(255,255,255,0.3)',
            tension: 0,
        });
    }

    mainChartInstance = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    display: predictionData !== null || smaValues.some(v => v != null),
                    position: 'top',
                    align: 'end',
                    labels: {
                        boxWidth: 12,
                        boxHeight: 2,
                        padding: 15,
                        font: { size: 11, weight: '500' },
                        usePointStyle: false,
                    },
                },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    borderColor: 'rgba(148, 163, 184, 0.15)',
                    borderWidth: 1,
                    padding: 12,
                    titleFont: { size: 12, weight: '600' },
                    bodyFont: { size: 11 },
                    displayColors: true,
                    callbacks: {
                        label: function(ctx) {
                            if (ctx.raw == null) return null;
                            return ` ${ctx.dataset.label}: ₹${formatNumber(ctx.raw)}`;
                        }
                    }
                },
            },
            scales: {
                x: {
                    grid: { color: 'rgba(148, 163, 184, 0.04)', lineWidth: 1 },
                    ticks: {
                        maxTicksLimit: 8,
                        font: { size: 10 },
                    },
                    border: { color: 'rgba(148, 163, 184, 0.08)' },
                },
                y: {
                    grid: { color: 'rgba(148, 163, 184, 0.04)', lineWidth: 1 },
                    ticks: {
                        font: { size: 10 },
                        callback: v => '₹' + formatNumber(v),
                    },
                    border: { display: false },
                    beginAtZero: false,
                },
            },
            animation: {
                duration: 600,
                easing: 'easeOutQuart',
            },
        },
    });
}

// ═══════════════════════════════════════════════════════════════
// Volume Bar Chart
// ═══════════════════════════════════════════════════════════════
function renderVolumeChart(historyData) {
    const ctx = $('#volumeChart').getContext('2d');

    if (volumeChartInstance) volumeChartInstance.destroy();

    const labels = historyData.map(d => d.date);
    const volumes = historyData.map(d => d.volume);
    const colors = historyData.map((d, i) => {
        if (i === 0) return 'rgba(59, 130, 246, 0.3)';
        return d.close >= historyData[i-1].close
            ? 'rgba(34, 197, 94, 0.35)'
            : 'rgba(239, 68, 68, 0.30)';
    });

    volumeChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                data: volumes,
                backgroundColor: colors,
                borderWidth: 0,
                borderRadius: 1,
                barPercentage: 0.85,
                categoryPercentage: 0.9,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    borderColor: 'rgba(148, 163, 184, 0.15)',
                    borderWidth: 1,
                    padding: 10,
                    callbacks: {
                        label: ctx => ` Vol: ${(ctx.raw / 1e6).toFixed(2)}M`,
                    },
                },
            },
            scales: {
                x: {
                    display: false,
                },
                y: {
                    display: false,
                    beginAtZero: true,
                },
            },
            animation: { duration: 400 },
        },
    });
}

// ═══════════════════════════════════════════════════════════════
// AI Prediction Toggle
// ═══════════════════════════════════════════════════════════════
async function togglePredict() {
    if (!activeSymbol) {
        showToast("Select a stock first", "error");
        return;
    }

    const btn = $('#btn-predict');

    if (isPredictMode) {
        isPredictMode = false;
        btn.classList.remove('active-predict');
        loadStockData(activeSymbol);
        return;
    }

    isPredictMode = true;
    btn.classList.add('active-predict');
    elChartLoader.classList.remove('hidden');

    try {
        // First get standard data
        const resData = await fetch(`${API_BASE}/data/${activeSymbol}?days=${activeDays}`);
        const baseData = await resData.json();

        // Then get prediction
        const resPred = await fetch(`${API_BASE}/predict/${activeSymbol}?lookback=${activeDays}&days=7`);
        if (!resPred.ok) throw new Error('Prediction failed');
        const predictData = await resPred.json();

        renderMainChart(baseData.data, activeSymbol, predictData);

        const trendEmoji = predictData.trend_direction === 'Up' ? '📈' : predictData.trend_direction === 'Down' ? '📉' : '↔';
        showToast(`${trendEmoji} ${predictData.model_type} · R² = ${predictData.r_squared} · Trend: ${predictData.trend_direction}`, 'success');
    } catch (err) {
        showToast("Prediction engine unavailable. Ensure scipy is installed.", "error");
        isPredictMode = false;
        btn.classList.remove('active-predict');
    }

    elChartLoader.classList.add('hidden');
}

// ═══════════════════════════════════════════════════════════════
// Stock Comparison Modal
// ═══════════════════════════════════════════════════════════════
function openCompareModal() {
    if (activeSymbol) {
        $('#compare-stock-1').value = activeSymbol;
    }
    $('#compare-results').classList.add('hidden');
    $('#compare-modal').classList.remove('hidden');
}

async function runCompare() {
    const s1 = $('#compare-stock-1').value;
    const s2 = $('#compare-stock-2').value;

    if (s1 === s2) {
        showToast("Pick two different stocks!", "error");
        return;
    }

    const btn = $('#btn-run-compare');
    btn.innerHTML = '<i class="ph ph-spinner ph-spin"></i> Analyzing...';
    btn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/compare?symbol1=${s1}&symbol2=${s2}`);
        const data = await res.json();

        $('#compare-results').classList.remove('hidden');

        // Render score cards
        const { stock1, stock2, verdict, correlation } = data;
        const scoresEl = $('#compare-scores');
        scoresEl.innerHTML = `
            <div class="compare-score-card">
                <div class="label">Sharpe Winner</div>
                <div class="value text-accent">${verdict.sharpe_winner}</div>
            </div>
            <div class="compare-score-card">
                <div class="label">Return Leader</div>
                <div class="value up">${verdict.return_winner}</div>
            </div>
            <div class="compare-score-card">
                <div class="label">Lower Risk</div>
                <div class="value" style="color: var(--cyan)">${verdict.lower_risk}</div>
            </div>
            <div class="compare-score-card">
                <div class="label">Correlation</div>
                <div class="value" style="color: var(--purple)">${correlation.return_correlation != null ? correlation.return_correlation.toFixed(2) : '—'}</div>
            </div>
        `;

        // Verdict
        $('#compare-verdict').innerHTML = `
            <strong style="color:var(--accent)">📊 Algorithmic Verdict</strong><br/>
            <span style="font-size:0.92rem; line-height:1.6; display:block; margin-top:0.4rem;">${verdict.overall_verdict}</span>
            <span style="font-size:0.8rem; color: var(--text-dim); margin-top:0.5rem; display:block;">
                ${verdict.reasoning}
            </span>
        `;

        // Comparison Chart
        renderCompareChart(data.price_history, s1, s2);
        showToast(`Comparison complete: ${s1} vs ${s2}`, 'success');
    } catch (err) {
        showToast("Comparison failed", "error");
    }

    btn.innerHTML = '<i class="ph ph-play-circle"></i> Run Analysis';
    btn.disabled = false;
}

function renderCompareChart(history, symbol1, symbol2) {
    const ctx = $('#compareChart').getContext('2d');

    if (compareChartInstance) compareChartInstance.destroy();

    compareChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: history.dates,
            datasets: [
                {
                    label: symbol1,
                    data: history[symbol1],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.05)',
                    borderWidth: 2,
                    tension: 0.2,
                    fill: true,
                },
                {
                    label: symbol2,
                    data: history[symbol2],
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.05)',
                    borderWidth: 2,
                    tension: 0.2,
                    fill: true,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    align: 'end',
                    labels: {
                        boxWidth: 12, boxHeight: 2, padding: 15,
                        font: { size: 11, weight: '500' },
                    },
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                    borderColor: 'rgba(148, 163, 184, 0.15)',
                    borderWidth: 1,
                },
            },
            scales: {
                x: {
                    grid: { color: 'rgba(148, 163, 184, 0.04)' },
                    ticks: { maxTicksLimit: 6, font: { size: 10 } },
                },
                y: {
                    grid: { color: 'rgba(148, 163, 184, 0.04)' },
                    title: {
                        display: true,
                        text: 'Normalized (Base 100)',
                        font: { size: 10, weight: '500' },
                        color: '#64748b',
                    },
                    ticks: { font: { size: 10 } },
                },
            },
            animation: { duration: 500 },
        },
    });
}

// ═══════════════════════════════════════════════════════════════
// Market Heatmap
// ═══════════════════════════════════════════════════════════════
function buildHeatmap(companies) {
    const grid = $('#heat-grid');
    if (!grid) return;

    grid.innerHTML = '';
    companies.forEach(c => {
        const pct = c.pct_from_52w_high;
        const cell = document.createElement('div');
        cell.className = 'heat-cell';
        cell.dataset.symbol = c.symbol;

        // Color based on distance from 52w high
        let bg, fg;
        if (pct > -5) {
            bg = 'rgba(34, 197, 94, 0.3)'; fg = '#22c55e';
        } else if (pct > -15) {
            bg = 'rgba(34, 197, 94, 0.15)'; fg = '#86efac';
        } else if (pct > -25) {
            bg = 'rgba(245, 158, 11, 0.15)'; fg = '#fbbf24';
        } else if (pct > -40) {
            bg = 'rgba(239, 68, 68, 0.15)'; fg = '#fca5a5';
        } else {
            bg = 'rgba(239, 68, 68, 0.3)'; fg = '#ef4444';
        }

        cell.style.background = bg;
        cell.style.color = fg;
        cell.innerHTML = `<span class="symbol">${c.symbol}</span><span class="pct">${pct}%</span>`;
        cell.onclick = () => selectStock(c.symbol);
        grid.appendChild(cell);
    });
}

// ═══════════════════════════════════════════════════════════════
// Sector Performance Bars (Sidebar)
// ═══════════════════════════════════════════════════════════════
function renderSectorBars(sectors) {
    const container = $('#sector-bars');
    if (!container || sectors.length === 0) return;

    // Sort by return
    sectors.sort((a, b) => (b.avg_annual_return_pct || 0) - (a.avg_annual_return_pct || 0));

    const maxAbs = Math.max(...sectors.map(s => Math.abs(s.avg_annual_return_pct || 0)), 1);

    container.innerHTML = '';
    sectors.forEach(s => {
        const ret = s.avg_annual_return_pct || 0;
        const pct = Math.min(Math.abs(ret) / maxAbs * 100, 100);
        const isUp = ret >= 0;
        const color = isUp ? 'var(--up)' : 'var(--down)';

        // Shorten sector name
        let shortName = s.sector;
        if (shortName.includes('/')) shortName = shortName.split('/')[0].trim();
        if (shortName.length > 12) shortName = shortName.substring(0, 11) + '…';

        container.innerHTML += `
            <div class="sector-bar-row">
                <span class="name">${shortName}</span>
                <span class="bar-track">
                    <span class="bar-fill" style="width: ${pct}%; background: ${color}"></span>
                </span>
                <span class="pct" style="color: ${color}">${isUp ? '+' : ''}${ret.toFixed(1)}%</span>
            </div>
        `;
    });
}

// ═══════════════════════════════════════════════════════════════
// Utility Functions
// ═══════════════════════════════════════════════════════════════
function formatNumber(n) {
    if (n == null) return '—';
    return parseFloat(n).toLocaleString('en-IN', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
    });
}

function truncate(str, len) {
    if (!str) return '';
    return str.length > len ? str.substring(0, len - 1) + '…' : str;
}

// ═══════════════════════════════════════════════════════════════
// Boot
// ═══════════════════════════════════════════════════════════════
init();
