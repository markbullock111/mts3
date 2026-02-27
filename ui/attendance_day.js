const $ = (sel) => document.querySelector(sel);

function todayStr() {
  return new Date().toISOString().slice(0, 10);
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function asIsoOrEmpty(value) {
  if (!value) return '';
  try {
    return new Date(value).toISOString();
  } catch {
    return String(value);
  }
}

function normalizeTimeHHMM(value, fallback = '09:00') {
  const raw = String(value ?? '').trim();
  const m = raw.match(/^(\d{1,2}):(\d{2})/);
  if (!m) return fallback;
  const hh = Number(m[1]);
  const mm = Number(m[2]);
  if (!Number.isFinite(hh) || !Number.isFinite(mm) || hh < 0 || hh > 23 || mm < 0 || mm > 59) return fallback;
  return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}`;
}

function targetUtcMs(dateStr, hhmm) {
  const d = String(dateStr || '').trim();
  const t = normalizeTimeHHMM(hhmm, '');
  if (!d || !t) return Number.NaN;
  const ts = Date.parse(`${d}T${t}:00Z`);
  return Number.isFinite(ts) ? ts : Number.NaN;
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      detail = data.detail || JSON.stringify(data);
    } catch {
      // noop
    }
    throw new Error(detail);
  }
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) return res.json();
  return res.text();
}

function renderTableRows(tbody, rowsHtml) {
  tbody.innerHTML = rowsHtml || '<tr><td colspan="99">No rows</td></tr>';
}

function eventImageThumb(r, label = 'snapshot') {
  const url = r.image_url || (r.image_path ? `file:///${String(r.image_path).replaceAll('\\', '/')}` : '');
  if (!url) return '<span class="muted">-</span>';
  return `<a class="image-link image-thumb-link" target="_blank" href="${escapeHtml(url)}"><img class="event-thumb" src="${escapeHtml(url)}" alt="${escapeHtml(label)}" loading="lazy" /></a>`;
}

function getDateFromQuery() {
  const date = new URLSearchParams(window.location.search).get('date');
  if (!date) return todayStr();
  return date;
}

function sortRows(rows) {
  const out = [...rows];
  out.sort((a, b) => {
    const aHas = Boolean(a.entrance_ts);
    const bHas = Boolean(b.entrance_ts);
    if (aHas !== bHas) return aHas ? -1 : 1;
    const aT = aHas ? new Date(a.entrance_ts).getTime() : Number.POSITIVE_INFINITY;
    const bT = bHas ? new Date(b.entrance_ts).getTime() : Number.POSITIVE_INFINITY;
    if (aT !== bT) return aT - bT;
    return String(a.full_name || '').localeCompare(String(b.full_name || ''), undefined, { sensitivity: 'base' });
  });
  return out;
}

function buildSummaryRows(employees, events) {
  const employeeById = new Map((employees || []).map(e => [Number(e.id), e]));
  const earliestByEmployee = new Map();
  const unknownRows = [];

  for (const evt of events || []) {
    if (!evt) continue;
    if (evt.employee_id == null || evt.method === 'unknown') {
      unknownRows.push(evt);
      continue;
    }
    const empId = Number(evt.employee_id);
    if (!Number.isFinite(empId)) {
      unknownRows.push(evt);
      continue;
    }
    const existing = earliestByEmployee.get(empId);
    const evtTimeMs = new Date(evt.ts).getTime();
    if (!existing || evtTimeMs < new Date(existing.ts).getTime()) {
      earliestByEmployee.set(empId, evt);
    }
  }

  const rows = [];
  for (const [empId, evt] of earliestByEmployee.entries()) {
    const emp = employeeById.get(empId);
    rows.push({
      full_name: emp?.full_name || evt.employee_name || `Employee ${empId}`,
      employee_code: emp?.employee_code || evt.employee_code || '',
      status: emp?.status || '',
      entrance_ts: evt.ts || null,
      method: evt.method || '',
      confidence: evt.confidence ?? null,
      image_url: evt.image_url || null,
      image_path: evt.image_path || null,
    });
  }

  unknownRows.sort((a, b) => new Date(a.ts).getTime() - new Date(b.ts).getTime());
  unknownRows.forEach((evt) => {
    rows.push({
      full_name: 'UNKNOWN',
      employee_code: '',
      status: 'unknown',
      entrance_ts: evt.ts || null,
      method: evt.method || 'unknown',
      confidence: evt.confidence ?? null,
      image_url: evt.image_url || null,
      image_path: evt.image_path || null,
    });
  });

  return sortRows(rows);
}

function effectiveTargetTime(policy, dateStr) {
  const standard = normalizeTimeHHMM(policy.standardTime, '09:00');
  if (String(policy.overrideDate || '') === String(dateStr || '') && String(policy.overrideTime || '').trim()) {
    return normalizeTimeHHMM(policy.overrideTime, standard);
  }
  return standard;
}

function targetTimeLabel(policy, dateStr) {
  const standard = normalizeTimeHHMM(policy.standardTime, '09:00');
  const hasOverrideForDate =
    String(policy.overrideDate || '') === String(dateStr || '') &&
    String(policy.overrideTime || '').trim().length > 0;
  if (hasOverrideForDate) {
    const setTime = normalizeTimeHHMM(policy.overrideTime, standard);
    return `Set ${setTime} UTC (Standard ${standard} UTC)`;
  }
  return `Standard ${standard} UTC`;
}

async function loadDay(dateStr) {
  const status = $('#dayViewStatus');
  const hint = $('#dayViewHint');
  const title = $('#dayViewTitle');
  const tbody = $('#dayViewTable tbody');
  status.textContent = 'Loading...';
  try {
    const [settingsResp, employees, events] = await Promise.all([
      api('/settings'),
      api('/employees'),
      api(`/events?date=${encodeURIComponent(dateStr)}`),
    ]);
    const settings = settingsResp?.values || {};
    const policy = {
      standardTime: normalizeTimeHHMM(settings.standard_attendance_time, '09:00'),
      overrideDate: String(settings.attendance_today_override_date || ''),
      overrideTime: normalizeTimeHHMM(settings.attendance_today_override_time, ''),
    };
    const target = effectiveTargetTime(policy, dateStr);
    const targetLabel = targetTimeLabel(policy, dateStr);
    const targetMs = targetUtcMs(dateStr, target);
    const rows = buildSummaryRows(employees || [], events || []);
    renderTableRows(
      tbody,
      rows.map((r) => `
        <tr>
          <td>${escapeHtml(r.full_name)}</td>
          <td>${escapeHtml(r.employee_code)}</td>
          <td>${
            r.entrance_ts
              ? (() => {
                  const enteredMs = new Date(r.entrance_ts).getTime();
                  const cls = Number.isFinite(enteredMs) && Number.isFinite(targetMs) && enteredMs <= targetMs
                    ? 'attendance-time-on'
                    : 'attendance-time-late';
                  return `<span class="${cls}">${escapeHtml(asIsoOrEmpty(r.entrance_ts))}</span>`;
                })()
              : '<span class="muted">Not checked in</span>'
          }</td>
          <td>${escapeHtml(r.method || '')}</td>
          <td>${r.confidence == null ? '' : Number(r.confidence).toFixed(3)}</td>
          <td>${escapeHtml(r.status || '')}</td>
          <td>${eventImageThumb(r, `${r.full_name || 'event'} snapshot`)}</td>
        </tr>
      `).join('')
    );
    title.textContent = `Attendance Summary (${dateStr})`;
    hint.textContent = `Attendance time: ${targetLabel} | Rows: ${rows.length}`;
    status.textContent = 'Loaded';
  } catch (err) {
    renderTableRows(tbody, '');
    hint.textContent = `Load failed: ${err.message}`;
    status.textContent = 'Error';
  }
}

function bind() {
  const dateInput = $('#dayViewDate');
  const loadBtn = $('#dayViewLoadBtn');
  const exportBtn = $('#dayViewExportBtn');
  const initialDate = getDateFromQuery();
  dateInput.value = initialDate;

  loadBtn.addEventListener('click', async () => {
    const date = dateInput.value || todayStr();
    const url = `/ui/attendance_day.html?date=${encodeURIComponent(date)}`;
    window.history.replaceState({}, '', url);
    await loadDay(date);
  });

  exportBtn.addEventListener('click', () => {
    const date = dateInput.value || todayStr();
    window.location.href = `/reports/daily.csv?date=${encodeURIComponent(date)}`;
  });
}

async function init() {
  bind();
  await loadDay($('#dayViewDate').value || todayStr());
}

init().catch((err) => {
  const status = $('#dayViewStatus');
  if (status) status.textContent = `Error: ${err.message}`;
});
