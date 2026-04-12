// Polls the detector /insights endpoint and renders a firefighter-ready
// flight report (summary, victims, risks, timeline).

import { fetchInsights } from './cv_ingest.js';

const POLL_MS = 2500;

const ui = {
  pollInterval: null,
  lastReport:   null,
};

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function renderSummaryText(value) {
  return escapeHtml(value)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');
}

function fmtGps(gps) {
  if (!gps || gps.lat == null) return 'GPS unknown';
  return `${gps.lat.toFixed(5)}, ${gps.lon.toFixed(5)}`;
}

function fmtTs(ms) {
  const d = new Date(ms);
  return d.toTimeString().slice(0, 8);
}

function priorityTag(p) {
  if (p >= 0.8) return { text: 'CRITICAL', color: '#ff3c3c' };
  if (p >= 0.5) return { text: 'HIGH',     color: '#ff8c00' };
  if (p >= 0.3) return { text: 'MEDIUM',   color: '#f5c518' };
  return { text: 'LOW', color: '#5a7080' };
}

function renderSummary(container, s) {
  const v = (x, d = 0) => (x == null ? '—' : (typeof x === 'number' ? x.toFixed(d) : x));
  const tsl = s.time_since_last_victim_s;
  container.innerHTML = `
    <div class="ins-metrics">
      <div><span>${v(s.victims_count)}</span><label>Victims</label></div>
      <div><span>${v(s.critical_victims)}</span><label>Critical</label></div>
      <div><span>${v(s.eeg_corroborated_victims)}</span><label>EEG-verified</label></div>
      <div><span>${v(s.risk_score, 0)}</span><label>Risk /100</label></div>
      <div><span>${v(s.peak_simultaneous_persons)}</span><label>Peak</label></div>
      <div><span>${v(s.person_frames)}</span><label>Person fr.</label></div>
      <div><span>${v(s.fire_events)}</span><label>Fire evt</label></div>
      <div><span>${v(s.smoke_events)}</span><label>Smoke evt</label></div>
      <div><span>${v(s.fire_coverage_pct, 1)}%</span><label>Fire %</label></div>
      <div><span>${v(s.smoke_coverage_pct, 1)}%</span><label>Smoke %</label></div>
      <div><span>${v(s.max_fire_severity != null ? s.max_fire_severity * 100 : null, 1)}%</span><label>Peak fire</label></div>
      <div><span>${v(s.max_smoke_severity != null ? s.max_smoke_severity * 100 : null, 1)}%</span><label>Peak smoke</label></div>
      <div><span>${v(s.avg_confidence != null ? s.avg_confidence * 100 : null, 0)}%</span><label>Avg conf</label></div>
      <div><span>${v(s.detections_per_min, 1)}</span><label>Det/min</label></div>
      <div><span>${v(s.person_detection_rate != null ? s.person_detection_rate * 100 : null, 0)}%</span><label>Person %</label></div>
      <div><span>${v(s.observed_fps, 1)}</span><label>CV fps</label></div>
      <div><span>${v(s.fire_exposure_s, 0)}s</span><label>Fire time</label></div>
      <div><span>${v(s.smoke_exposure_s, 0)}s</span><label>Smoke time</label></div>
      <div><span>${v(s.coverage_diagonal_m, 0)}m</span><label>Coverage</label></div>
      <div><span>${tsl == null ? '—' : tsl.toFixed(0) + 's'}</span><label>Since last</label></div>
      <div><span>${v(s.total_frames)}</span><label>Frames</label></div>
      <div><span>${v(s.duration_s, 0)}s</span><label>Duration</label></div>
      <div><span>${v(s.vehicle_frames)}</span><label>Vehicle fr.</label></div>
    </div>
    <ul class="ins-lines">
      ${(s.lines || []).map(l => `<li>${l}</li>`).join('')}
    </ul>
  `;
}

function renderVictims(container, victims) {
  if (!victims.length) {
    container.innerHTML = '<div class="ins-empty">No victims detected.</div>';
    return;
  }
  container.innerHTML = victims.slice(0, 10).map(v => {
    const tag = priorityTag(v.priority);
    const eeg = v.eeg_corroborations > 0
      ? `<span class="ins-eeg">EEG×${v.eeg_corroborations}</span>` : '';
    return `
      <div class="ins-victim">
        <div class="ins-row">
          <span class="ins-id">${v.victim_id}</span>
          <span class="ins-badge" style="background:${tag.color}">${tag.text}</span>
          ${eeg}
        </div>
        <div class="ins-meta">
          conf ${(v.max_confidence * 100).toFixed(0)}% ·
          ${v.detections} hit${v.detections === 1 ? '' : 's'} ·
          ${fmtGps(v.gps)}
        </div>
      </div>`;
  }).join('');
}

function renderRisks(container, risks) {
  if (!risks.length) {
    container.innerHTML = '<div class="ins-empty">No fire or smoke events.</div>';
    return;
  }
  container.innerHTML = risks.map(r => {
    const kindColor = r.kind === 'fire' ? '#ff3c3c' : '#9aa7b4';
    return `
      <div class="ins-risk">
        <div class="ins-row">
          <span class="ins-badge" style="background:${kindColor}">${r.kind.toUpperCase()}</span>
          <span class="ins-meta">severity ${(r.max_severity * 100).toFixed(1)}% · ${r.frame_count} frames</span>
        </div>
        <div class="ins-meta">${fmtTs(r.first_ts)} → ${fmtTs(r.last_ts)} · ${fmtGps(r.gps)}</div>
      </div>`;
  }).join('');
}

function renderTimeline(container, timeline) {
  const recent = timeline.slice(-20).reverse();
  if (!recent.length) {
    container.innerHTML = '<div class="ins-empty">No events yet.</div>';
    return;
  }
  container.innerHTML = recent.map(e => {
    const tag = priorityTag(e.priority || 0);
    return `
      <div class="ins-event">
        <span class="ins-time">${fmtTs(e.t)}</span>
        <span class="ins-dot" style="background:${tag.color}"></span>
        <span class="ins-label">${e.label}</span>
      </div>`;
  }).join('');
}

function pct(value) {
  return Math.max(0, Math.min(100, Math.round((value || 0) * 100)));
}

function renderReviewVisuals(container, report) {
  if (!container) return;
  const s = report?.summary || {};
  const bars = [
    { label: 'Risk score', value: s.risk_score || 0, max: 100, cls: 'risk' },
    { label: 'Person frames', value: pct(s.person_detection_rate), max: 100, cls: 'person' },
    { label: 'Fire coverage', value: s.fire_coverage_pct || 0, max: 100, cls: 'fire' },
    { label: 'Smoke coverage', value: s.smoke_coverage_pct || 0, max: 100, cls: 'smoke' },
  ];
  container.innerHTML = `
    <div class="visual-grid">
      <div class="visual-stat"><span>${s.total_frames ?? 0}</span><label>Frames</label></div>
      <div class="visual-stat"><span>${s.duration_s ?? 0}s</span><label>Duration</label></div>
      <div class="visual-stat"><span>${s.victims_count ?? 0}</span><label>Victims</label></div>
      <div class="visual-stat"><span>${s.risk_score ?? 0}</span><label>Risk</label></div>
    </div>
    <div class="visual-bars">
      ${bars.map(b => {
        const w = Math.max(0, Math.min(100, (b.value / b.max) * 100));
        return `<div class="visual-bar-row">
          <div class="visual-bar-label"><span>${b.label}</span><b>${Math.round(b.value)}${b.label === 'Risk score' ? '/100' : '%'}</b></div>
          <div class="visual-bar"><i class="${b.cls}" style="width:${w}%"></i></div>
        </div>`;
      }).join('')}
    </div>
  `;
}

function renderImportantFrames(container, frames) {
  if (!container) return;
  if (!frames?.length) {
    container.innerHTML = '<div class="ins-empty">No important frames yet.</div>';
    return;
  }
  container.innerHTML = frames.slice(0, 12).map(f => {
    const reasonText = (f.reasons || []).join(' · ');
    const tag = priorityTag(f.score || 0);
    const frameLabel = `F${String(f.frame_id).padStart(4, '0')}`;
    const media = f.thumbnail
      ? `<img src="${f.thumbnail}" alt="${frameLabel}">`
      : `<span>${frameLabel}</span>`;
    return `
      <div class="important-frame">
        <div class="frame-tile ${f.thumbnail ? 'has-image' : ''}">
          ${media}
          <span class="frame-label">${frameLabel}</span>
          <i style="height:${Math.max(12, Math.round((f.score || 0) * 100))}%"></i>
        </div>
        <div class="frame-detail">
          <div class="ins-row">
            <span class="ins-id">${fmtTs(f.ts)}</span>
            <span class="ins-badge" style="background:${tag.color}">${Math.round((f.score || 0) * 100)}%</span>
          </div>
          <div class="ins-meta">${escapeHtml(reasonText)}</div>
          <div class="ins-meta">${fmtGps(f.gps)}</div>
        </div>
      </div>`;
  }).join('');
}

async function poll() {
  const report = await fetchInsights();
  const badge = document.getElementById('insights-status');
  if (!report) {
    if (badge) {
      badge.textContent = 'detector offline';
      badge.style.color = '#ff3c3c';
    }
    return;
  }
  ui.lastReport = report;
  if (badge) {
    badge.textContent = 'live';
    badge.style.color = '#39e97b';
  }
  renderSummary(document.getElementById('ins-summary'), report.summary);
  renderVictims(document.getElementById('ins-victims'), report.victims);
  renderRisks(document.getElementById('ins-risks'),   report.risks);
  renderTimeline(document.getElementById('ins-timeline'), report.timeline);
}

export async function refreshInsights() {
  await poll();
  return ui.lastReport;
}

export function startInsightsPoll() {
  if (ui.pollInterval) return;
  poll();
  ui.pollInterval = setInterval(poll, POLL_MS);
}

export function stopInsightsPoll() {
  if (ui.pollInterval) clearInterval(ui.pollInterval);
  ui.pollInterval = null;
}

export function downloadReport() {
  if (!ui.lastReport) return;
  const blob = new Blob([JSON.stringify(ui.lastReport, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `flight-report-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  a.click();
  URL.revokeObjectURL(url);
}

window.downloadFlightReport = downloadReport;

export async function generateTripSummary({ outputId = 'llm-summary-output', buttonId = 'btn-llm-summary', report = ui.lastReport } = {}) {
  const output = document.getElementById(outputId);
  const btn = document.getElementById(buttonId);
  if (!output) return;
  const wasDisabled = !!btn?.disabled;

  output.textContent = 'Generating scan summary...';
  output.classList.add('ins-empty');
  if (btn) btn.disabled = true;

  try {
    const res = await fetch('/api/trip-summary', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ report }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    output.classList.remove('ins-empty');
    output.innerHTML = `<div class="llm-provider">Scan summary</div><div class="llm-text">${renderSummaryText(data.summary)}</div>`;
    return data;
  } catch (err) {
    output.classList.add('ins-empty');
    output.textContent = `Could not generate scan summary: ${err.message}`;
    return { ok: false, error: err.message };
  } finally {
    if (btn) btn.disabled = wasDisabled;
  }
}

window.generateTripSummary = generateTripSummary;

export function renderSessionReview(report, llmData = null) {
  if (!report) return;
  const summaryEl = document.getElementById('review-ins-summary');
  const victimsEl = document.getElementById('review-ins-victims');
  const risksEl = document.getElementById('review-ins-risks');
  const timelineEl = document.getElementById('review-ins-timeline');
  const visualsEl = document.getElementById('review-visuals');
  const importantEl = document.getElementById('review-important-frames');
  const llmEl = document.getElementById('review-llm-summary');

  if (summaryEl) renderSummary(summaryEl, report.summary || {});
  if (victimsEl) renderVictims(victimsEl, report.victims || []);
  if (risksEl) renderRisks(risksEl, report.risks || []);
  if (timelineEl) renderTimeline(timelineEl, report.timeline || []);
  renderReviewVisuals(visualsEl, report);
  renderImportantFrames(importantEl, report.important_frames || []);
  if (llmEl && llmData?.summary) {
    llmEl.classList.remove('ins-empty');
    llmEl.innerHTML = `<div class="llm-provider">Scan summary</div><div class="llm-text">${renderSummaryText(llmData.summary)}</div>`;
  }
}

window.renderSessionReview = renderSessionReview;
