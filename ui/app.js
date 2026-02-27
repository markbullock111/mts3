const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  selectedEmployeeId: null,
  employees: [],
  cameras: [],
  employeesSearchQuery: '',
  employeesSortBy: 'id',
  employeesSortDir: 'asc',
  attendanceSummaryRows: [],
  attendanceSummaryQuery: '',
  attendanceSelectedDate: '',
  attendanceTargetTime: '09:00',
  attendancePolicy: {
    standardTime: '09:00',
    overrideDate: '',
    overrideTime: '',
  },
  attendanceLoading: false,
  attendanceAutoTimer: null,
};

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

function getEffectiveAttendanceTimeForDate(dateStr) {
  const p = state.attendancePolicy || {};
  const standard = normalizeTimeHHMM(p.standardTime, '09:00');
  if (String(p.overrideDate || '') === String(dateStr || '') && String(p.overrideTime || '').trim()) {
    return normalizeTimeHHMM(p.overrideTime, standard);
  }
  return standard;
}

function refreshAttendanceTimeInfo(dateStr) {
  const info = $('#attendanceTimeInfo');
  if (!info) return;
  const standard = normalizeTimeHHMM(state.attendancePolicy?.standardTime, '09:00');
  const target = getEffectiveAttendanceTimeForDate(dateStr);
  const usingOverride = String(state.attendancePolicy?.overrideDate || '') === String(dateStr || '') && !!state.attendancePolicy?.overrideTime;
  info.textContent = `Standard ${standard} UTC | Effective ${target} UTC${usingOverride ? ' (today override)' : ''}`;
}

function normalizeEmployeePayload(form, { includeCode = false } = {}) {
  const fd = new FormData(form);
  const payload = {
    full_name: String(fd.get('full_name') || '').trim(),
    status: String(fd.get('status') || 'active'),
    birth_date: String(fd.get('birth_date') || '').trim() || null,
    job_title: String(fd.get('job_title') || '').trim() || null,
    address: String(fd.get('address') || '').trim() || null,
  };
  if (includeCode) payload.employee_code = String(fd.get('employee_code') || '').trim();
  return payload;
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      detail = data.detail || JSON.stringify(data);
    } catch {}
    throw new Error(detail);
  }
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) return res.json();
  return res.text();
}

function setHealth(text, ok = true) {
  const badge = $('#healthBadge');
  badge.textContent = text;
  badge.style.borderColor = ok ? 'rgba(43,122,75,0.35)' : 'rgba(166,60,36,0.35)';
  badge.style.color = ok ? '#2b7a4b' : '#a63c24';
}

function switchTab(tabName) {
  $$('.tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === tabName));
  $$('.tab').forEach(t => t.classList.toggle('active', t.id === `tab-${tabName}`));
}

function bindTabs() {
  $$('.tabs button').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
}

async function checkHealth() {
  try {
    const h = await api('/health');
    setHealth(`Backend: ${h.status} | DB: ${h.db ? 'ok' : 'fail'}`, h.status === 'ok');
  } catch (e) {
    setHealth(`Backend error: ${e.message}`, false);
  }
}

function renderTableRows(tbody, rowsHtml) {
  tbody.innerHTML = rowsHtml || '<tr><td colspan="99">No rows</td></tr>';
}

function eventImageLink(r) {
  const url = r.image_url || (r.image_path ? `file:///${String(r.image_path).replaceAll('\\', '/')}` : '');
  if (!url) return '';
  return `<a class="image-link" target="_blank" href="${escapeHtml(url)}">open</a>`;
}

function eventImageThumb(r, label = 'snapshot') {
  const url = r.image_url || (r.image_path ? `file:///${String(r.image_path).replaceAll('\\', '/')}` : '');
  if (!url) return '<span class="muted">-</span>';
  return `<a class="image-link image-thumb-link" target="_blank" href="${escapeHtml(url)}"><img class="event-thumb" src="${escapeHtml(url)}" alt="${escapeHtml(label)}" loading="lazy" /></a>`;
}

function employeeMainThumb(r) {
  const url = r.main_photo_url || r.main_photo?.media_url || '';
  if (!url) return '<span class="muted">-</span>';
  const alt = `${r.full_name || 'employee'} main picture`;
  return `<a class="image-link image-thumb-link" target="_blank" href="${escapeHtml(url)}"><img class="event-thumb" src="${escapeHtml(url)}" alt="${escapeHtml(alt)}" loading="lazy" /></a>`;
}

function employeePhotoUrl(r) {
  return r?.main_photo_url || r?.main_photo?.media_url || '';
}

function employeeInitial(r) {
  const name = String(r?.full_name || '').trim();
  if (!name) return '?';
  return name.slice(0, 1).toUpperCase();
}

async function loadEmployees() {
  renderTableRows($('#employeesTable tbody'), '<tr><td colspan="99">Loading employees...</td></tr>');
  const rows = await api('/employees');
  state.employees = rows;
  renderEmployeesTable();
  renderAttendanceEmployeeOptions();
}

function getFilteredEmployees() {
  const q = String(state.employeesSearchQuery || '').trim().toLowerCase();
  const filtered = !q ? [...(state.employees || [])] : (state.employees || []).filter(r => {
    const idStr = String(r.id ?? '').toLowerCase();
    const code = String(r.employee_code ?? '').toLowerCase();
    const name = String(r.full_name ?? '').toLowerCase();
    return idStr.includes(q) || code.includes(q) || name.includes(q);
  });
  const sortBy = String(state.employeesSortBy || 'id');
  const dir = state.employeesSortDir === 'desc' ? -1 : 1;
  filtered.sort((a, b) => {
    if (sortBy === 'name') {
      return String(a.full_name || '').localeCompare(String(b.full_name || ''), undefined, { sensitivity: 'base' }) * dir;
    }
    if (sortBy === 'code') {
      return String(a.employee_code || '').localeCompare(String(b.employee_code || ''), undefined, { sensitivity: 'base' }) * dir;
    }
    if (sortBy === 'status') {
      return String(a.status || '').localeCompare(String(b.status || ''), undefined, { sensitivity: 'base' }) * dir;
    }
    return (Number(a.id || 0) - Number(b.id || 0)) * dir;
  });
  return filtered;
}

function renderEmployeesTable() {
  const rows = getFilteredEmployees();
  const tbody = $('#employeesTable tbody');
  renderTableRows(
    tbody,
    rows.map(r => `
      <tr data-employee-row-id="${r.id}">
        <td>${r.id}</td>
        <td>${escapeHtml(r.employee_code)}</td>
        <td>${escapeHtml(r.full_name)}</td>
        <td>${employeeMainThumb(r)}</td>
        <td>${escapeHtml(r.birth_date || '')}</td>
        <td>${escapeHtml(r.job_title || '')}</td>
        <td>${escapeHtml(r.status)}</td>
        <td>${r.face_embeddings_count ?? 0}</td>
        <td>${r.reid_embeddings_count ?? 0}</td>
        <td>${r.uploaded_images_count ?? 0}</td>
        <td><button data-open-detail-id="${r.id}">View</button></td>
      </tr>`).join('')
  );

  $$('#employeesTable button[data-open-detail-id]').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = Number(btn.dataset.openDetailId);
      if (!id) return;
      window.location.href = `/ui/employee_detail.html?id=${encodeURIComponent(String(id))}`;
    });
  });

  const hint = $('#employeeDetailHint');
  const q = String(state.employeesSearchQuery || '').trim();
  if (hint && q && !rows.length) {
    hint.textContent = `No employees found for search: ${q}`;
  }
  const stats = $('#employeesStats');
  if (stats) {
    const total = Array.isArray(state.employees) ? state.employees.length : 0;
    stats.textContent = q
      ? `Showing ${rows.length} of ${total} employees`
      : `Total employees: ${total}`;
  }
}

function renderAttendanceEmployeeOptions() {
  const dl = $('#attendanceEmployeeOptions');
  if (!dl) return;
  const rows = Array.isArray(state.employees) ? state.employees : [];
  dl.innerHTML = rows.map((e) => {
    const id = Number(e.id);
    const name = String(e.full_name || '').trim();
    const code = String(e.employee_code || '').trim();
    const label = [name, code].filter(Boolean).join(' | ');
    return [
      `<option value="${escapeHtml(String(id))}" label="${escapeHtml(label)}"></option>`,
      `<option value="${escapeHtml(name)}" label="${escapeHtml(`ID ${id}${code ? ` | ${code}` : ''}`)}"></option>`,
      code ? `<option value="${escapeHtml(code)}" label="${escapeHtml(`ID ${id} | ${name}`)}"></option>` : '',
    ].join('');
  }).join('');
}

async function ensureEmployeesLoaded() {
  if (Array.isArray(state.employees) && state.employees.length) return state.employees;
  await loadEmployees();
  return state.employees;
}

function bindEmployeeForm() {
  const openAddBtn = $('#openAddEmployeePageBtn');
  const openEnrollmentBtn = $('#openEnrollmentPageBtn');
  const form = $('#employeeForm');
  const refreshBtn = $('#refreshEmployeesBtn');
  const searchInput = $('#employeesSearchInput');
  const clearBtn = $('#clearEmployeesSearchBtn');
  const sortByInput = $('#employeesSortBy');
  const sortDirInput = $('#employeesSortDir');

  if (openAddBtn) {
    openAddBtn.addEventListener('click', () => {
      window.location.href = '/ui/add_employee.html';
    });
  }
  if (openEnrollmentBtn) {
    openEnrollmentBtn.addEventListener('click', () => {
      window.location.href = '/ui/enrollment.html';
    });
  }

  if (form) {
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const payload = normalizeEmployeePayload(form, { includeCode: true });
      try {
        await api('/employees', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        form.reset();
        await loadEmployees();
      } catch (err) {
        alert(`Add employee failed: ${err.message}`);
      }
    });
  }

  refreshBtn?.addEventListener('click', () => loadEmployees().catch(err => alert(err.message)));
  searchInput?.addEventListener('input', (e) => {
    state.employeesSearchQuery = e.target.value || '';
    renderEmployeesTable();
  });
  clearBtn?.addEventListener('click', () => {
    state.employeesSearchQuery = '';
    if (searchInput) searchInput.value = '';
    renderEmployeesTable();
  });
  sortByInput?.addEventListener('change', (e) => {
    state.employeesSortBy = String(e.target.value || 'id');
    renderEmployeesTable();
  });
  sortDirInput?.addEventListener('change', (e) => {
    state.employeesSortDir = String(e.target.value || 'asc');
    renderEmployeesTable();
  });
}

function bindEnrollmentForm() {
  const formEl = $('#faceEnrollForm');
  if (!formEl) return;
  formEl.addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const fd = new FormData(form);
    const employeeId = fd.get('employee_id');
    const kind = fd.get('kind');
    const files = form.querySelector('input[name="files"]').files;
    const submitBtn = $('#enrollSubmitBtn');
    const statusWrap = $('#enrollUploadStatus');
    const statusText = $('#enrollUploadStatusText');
    const targetInfo = $('#enrollTargetInfo');
    if (!files.length) {
      alert('Select at least one image');
      return;
    }
    const upload = new FormData();
    for (const f of files) upload.append('files', f);
    const employeeIdText = String(employeeId || '').trim();
    const modeText = String(kind || '').toUpperCase();
    if (targetInfo) {
      targetInfo.textContent = `Target Employee ID: ${employeeIdText} | Mode: ${modeText} | Files: ${files.length}`;
    }
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.dataset.originalText = submitBtn.dataset.originalText || submitBtn.textContent || 'Upload Enrollment Images';
      submitBtn.textContent = 'Uploading...';
    }
    if (statusWrap && statusText) {
      statusWrap.classList.remove('hidden');
      statusWrap.classList.add('active');
      statusText.textContent = `Uploading ${files.length} image(s) for Employee ID ${employeeIdText} (${modeText})...`;
    }
    $('#enrollResult').textContent = `Uploading to employee_id=${employeeIdText} (${modeText})...`;
    try {
      const data = await api(`/employees/${employeeId}/enroll/${kind}`, { method: 'POST', body: upload });
      $('#enrollResult').textContent = `Upload complete for employee_id=${employeeIdText} (${modeText})\\n` + JSON.stringify(data, null, 2);
      await loadEmployees();
      if (Number(employeeId) === state.selectedEmployeeId) {
        await loadEmployeeDetail(state.selectedEmployeeId);
      }
    } catch (err) {
      $('#enrollResult').textContent = `ERROR uploading employee_id=${employeeIdText} (${modeText}): ${err.message}`;
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = submitBtn.dataset.originalText || 'Upload Enrollment Images';
      }
      if (statusWrap && statusText) {
        statusText.textContent = `Idle`;
        statusWrap.classList.remove('active');
        // keep visible briefly to show state transition; then hide
        window.setTimeout(() => {
          statusWrap.classList.add('hidden');
        }, 600);
      }
    }
  });
}

async function loadCameras() {
  const rows = await api('/cameras');
  state.cameras = Array.isArray(rows) ? rows : [];
  const tbody = $('#camerasTable tbody');
  renderTableRows(
    tbody,
    (rows || []).map(r => `
      <tr>
        <td>${escapeHtml(r.id)}</td>
        <td>${escapeHtml(r.name || '')}</td>
        <td>${escapeHtml(r.rtsp_url || '')}</td>
        <td>${escapeHtml(r.location || '')}</td>
        <td>${escapeHtml(String(r.enabled))}</td>
        <td>
          <button type="button" data-preview-camera-id="${r.id}" ${r.enabled ? '' : 'disabled'}>View</button>
          <button type="button" class="danger-btn" data-remove-camera-id="${r.id}">Remove</button>
        </td>
      </tr>`).join('')
  );

  $$('#camerasTable button[data-preview-camera-id]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = Number(btn.dataset.previewCameraId);
      if (!id) return;
      const cam = (state.cameras || []).find(c => Number(c.id) === id);
      const source = cam?.rtsp_url || '';
      await startCameraPreview(id, source);
    });
  });

  $$('#camerasTable button[data-remove-camera-id]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = Number(btn.dataset.removeCameraId);
      if (!id) return;
      if (!window.confirm(`Remove camera ${id}?`)) return;
      btn.disabled = true;
      try {
        await api(`/cameras/${id}`, { method: 'DELETE' });
        const img = $('#cameraPreviewImg');
        if (img && Number(img.dataset.activeCameraId || 0) === id) {
          await stopCameraPreview({ releaseBackend: false });
        }
        await loadCameras();
      } catch (err) {
        alert(`Remove camera failed: ${err.message}`);
      } finally {
        btn.disabled = false;
      }
    });
  });
}

function isWebcamSource(source) {
  const s = String(source || '').trim().toLowerCase();
  if (!s) return false;
  if (s === 'webcam' || s === 'camera' || s === 'cam') return true;
  if (/^(webcam|camera|cam):\d+$/.test(s)) return true;
  if (/^\d+$/.test(s)) return true;
  return false;
}

async function requestBrowserCameraPermission(source) {
  if (!isWebcamSource(source)) return { ok: true, skipped: true, message: '' };

  // If browser already reports granted permission, skip active probing to avoid timeout noise.
  if (navigator.permissions && typeof navigator.permissions.query === 'function') {
    try {
      const status = await navigator.permissions.query({ name: 'camera' });
      if (status?.state === 'granted') {
        return { ok: true, skipped: false, message: 'Browser camera permission already granted.' };
      }
      if (status?.state === 'denied') {
        return {
          ok: false,
          skipped: false,
          message: 'Browser camera permission is denied for this site.',
        };
      }
    } catch {
      // Ignore; not all browsers support querying camera permission.
    }
  }

  if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
    return {
      ok: false,
      skipped: false,
      message: 'This browser does not support camera permission requests (getUserMedia unavailable).',
    };
  }
  try {
    const mediaPromise = navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    const timeoutPromise = new Promise((_, reject) =>
      window.setTimeout(() => reject(new Error('Timeout starting video source')), 5000)
    );
    const stream = await Promise.race([mediaPromise, timeoutPromise]);
    stream.getTracks().forEach(t => t.stop());
    return { ok: true, skipped: false, message: '' };
  } catch (err) {
    const msg = String(err?.message || err || '').toLowerCase();
    if (msg.includes('timeout')) {
      return {
        ok: false,
        skipped: false,
        message: 'Browser camera probe timed out.',
      };
    }
    return {
      ok: false,
      skipped: false,
      message: `Camera permission denied/blocked in browser: ${err?.message || err}.`,
    };
  }
}

async function startCameraPreview(cameraId, cameraSource = '') {
  const img = $('#cameraPreviewImg');
  const hint = $('#cameraPreviewHint');
  if (!img || !hint) return;
  const perm = await requestBrowserCameraPermission(cameraSource);
  if (!perm.ok && isWebcamSource(cameraSource)) {
    // Browser permission is requested for UX, but backend stream may still work.
    hint.textContent = `${perm.message} Trying backend preview anyway...`;
  }
  img.src = `/cameras/${cameraId}/preview.mjpeg?t=${Date.now()}`;
  img.dataset.activeCameraId = String(cameraId);
  if (perm.ok || !isWebcamSource(cameraSource)) {
    hint.textContent = `Live view for Camera ID ${cameraId}. Colored rectangles indicate tracked people; labels show recognized names when available.`;
  }
}

async function stopCameraPreview({ releaseBackend = true } = {}) {
  const img = $('#cameraPreviewImg');
  const hint = $('#cameraPreviewHint');
  if (!img || !hint) return;
  const activeId = Number(img.dataset.activeCameraId || 0);
  if (releaseBackend && activeId) {
    try {
      await api(`/cameras/${activeId}/preview/stop`, { method: 'POST' });
    } catch {
      // ignore; still clear frontend preview state
    }
  }
  img.removeAttribute('src');
  img.dataset.activeCameraId = '';
  hint.textContent = 'Click View in the table to show live stream with colored person rectangles and names.';
}

function bindCameras() {
  const form = $('#cameraForm');
  const refreshBtn = $('#refreshCamerasBtn');
  const stopBtn = $('#stopCameraPreviewBtn');
  const previewImg = $('#cameraPreviewImg');
  const previewHint = $('#cameraPreviewHint');
  if (!form || !refreshBtn) return;

  if (previewImg && previewHint) {
    previewImg.addEventListener('error', () => {
      const id = previewImg.dataset.activeCameraId || '?';
      previewHint.textContent = `Preview failed for Camera ID ${id}. Check camera source value, camera enabled flag, and backend logs.`;
    });
  }

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const fd = new FormData(form);
    fd.set('enabled', String(fd.get('enabled') === 'true'));
    try {
      await api('/cameras', {
        method: 'POST',
        body: fd,
      });
      form.reset();
      await loadCameras();
    } catch (err) {
      alert(`Add camera failed: ${err.message}`);
    }
  });

  refreshBtn.addEventListener('click', () => loadCameras().catch(err => alert(`Load cameras failed: ${err.message}`)));
  if (stopBtn) {
    stopBtn.addEventListener('click', () => stopCameraPreview().catch(() => {}));
  }
}

async function loadEvents() {
  if (state.attendanceLoading) return;
  state.attendanceLoading = true;
  const date = $('#eventsDate').value || todayStr();
  renderTableRows($('#attendanceSummaryTable tbody'), '<tr><td colspan="99">Loading attendance...</td></tr>');
  try {
    const rows = await api(`/events?date=${date}`);
    await renderAttendanceSummary(rows, date);
  } finally {
    state.attendanceLoading = false;
  }
}

function openAttendanceDateTab(dateStr) {
  const date = String(dateStr || '').trim() || todayStr();
  const url = `/ui/attendance_day.html?date=${encodeURIComponent(date)}`;
  window.open(url, '_blank', 'noopener,noreferrer');
}

function buildAttendanceSummaryRows(employees, events) {
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
      row_key: `emp-${empId}`,
      employee_id: empId,
      full_name: emp?.full_name || evt.employee_name || `Employee ${empId}`,
      employee_code: emp?.employee_code || evt.employee_code || '',
      status: emp?.status || '',
      entrance_ts: evt.ts || null,
      method: evt.method || '',
      confidence: evt.confidence ?? null,
      image_url: evt.image_url || null,
      image_path: evt.image_path || null,
      is_unknown: false,
    });
  }

  unknownRows.sort((a, b) => new Date(a.ts).getTime() - new Date(b.ts).getTime());
  unknownRows.forEach((evt) => {
    rows.push({
      row_key: `unknown-${evt.id || evt.track_uid || Math.random()}`,
      source_event_id: evt.id || null,
      employee_id: null,
      full_name: 'UNKNOWN',
      employee_code: '',
      status: 'unknown',
      entrance_ts: evt.ts || null,
      method: evt.method || 'unknown',
      confidence: evt.confidence ?? null,
      image_url: evt.image_url || null,
      image_path: evt.image_path || null,
      is_unknown: true,
    });
  });

  return rows;
}

function sortAttendanceSummaryRows(rows, sortBy, sortDir) {
  const dir = sortDir === 'desc' ? -1 : 1;
  const out = [...(rows || [])];
  out.sort((a, b) => {
    if (sortBy === 'time') {
      const aHas = Boolean(a.entrance_ts);
      const bHas = Boolean(b.entrance_ts);
      if (aHas !== bHas) return aHas ? -1 : 1; // rows with times first; absent rows last
      if (!aHas && !bHas) {
        return String(a.full_name || '').localeCompare(String(b.full_name || ''), undefined, { sensitivity: 'base' });
      }
      const aT = new Date(a.entrance_ts).getTime();
      const bT = new Date(b.entrance_ts).getTime();
      if (aT !== bT) return (aT - bT) * dir;
      return String(a.full_name || '').localeCompare(String(b.full_name || ''), undefined, { sensitivity: 'base' });
    }
    const nameCmp = String(a.full_name || '').localeCompare(String(b.full_name || ''), undefined, { sensitivity: 'base' });
    if (nameCmp !== 0) return nameCmp * dir;
    const aT = a.entrance_ts ? new Date(a.entrance_ts).getTime() : Number.POSITIVE_INFINITY;
    const bT = b.entrance_ts ? new Date(b.entrance_ts).getTime() : Number.POSITIVE_INFINITY;
    return aT - bT;
  });
  return out;
}

function filterAttendanceSummaryRows(rows, query) {
  const q = String(query || '').trim().toLowerCase();
  if (!q) return [...(rows || [])];
  return (rows || []).filter((r) => {
    const name = String(r.full_name || '').toLowerCase();
    const code = String(r.employee_code || '').toLowerCase();
    const method = String(r.method || '').toLowerCase();
    const status = String(r.status || '').toLowerCase();
    return name.includes(q) || code.includes(q) || method.includes(q) || status.includes(q);
  });
}

function renderAttendanceMetrics(rows, targetMs) {
  const list = rows || [];
  let recognized = 0;
  let unknown = 0;
  let onTime = 0;
  let late = 0;
  for (const r of list) {
    if (r.is_unknown) {
      unknown += 1;
      continue;
    }
    recognized += 1;
    if (!r.entrance_ts) continue;
    const enteredMs = new Date(r.entrance_ts).getTime();
    if (!Number.isFinite(enteredMs) || !Number.isFinite(targetMs)) continue;
    if (enteredMs <= targetMs) onTime += 1;
    else late += 1;
  }
  const rowsEl = $('#attendanceMetricRows');
  const recognizedEl = $('#attendanceMetricRecognized');
  const unknownEl = $('#attendanceMetricUnknown');
  const onTimeEl = $('#attendanceMetricOnTime');
  const lateEl = $('#attendanceMetricLate');
  if (rowsEl) rowsEl.textContent = String(list.length);
  if (recognizedEl) recognizedEl.textContent = String(recognized);
  if (unknownEl) unknownEl.textContent = String(unknown);
  if (onTimeEl) onTimeEl.textContent = String(onTime);
  if (lateEl) lateEl.textContent = String(late);
}

function findEmployeeRecommendations(query, limit = 6) {
  const q = String(query || '').trim().toLowerCase();
  const rows = Array.isArray(state.employees) ? state.employees : [];
  const scored = [];
  for (const emp of rows) {
    const id = String(emp?.id ?? '');
    const code = String(emp?.employee_code || '').toLowerCase();
    const name = String(emp?.full_name || '').toLowerCase();
    let score = 0;

    if (!q) {
      score = 1;
    } else if (id === q) {
      score = 220;
    } else if (name === q) {
      score = 210;
    } else if (code === q) {
      score = 200;
    } else if (name.startsWith(q)) {
      score = 160;
    } else if (code.startsWith(q)) {
      score = 140;
    } else if (id.startsWith(q)) {
      score = 130;
    } else if (name.includes(q)) {
      score = 110;
    } else if (code.includes(q)) {
      score = 90;
    } else if (id.includes(q)) {
      score = 70;
    }

    if (score > 0) scored.push({ emp, score });
  }

  scored.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return String(a.emp?.full_name || '').localeCompare(String(b.emp?.full_name || ''), undefined, { sensitivity: 'base' });
  });
  return scored.slice(0, Math.max(1, limit)).map((x) => x.emp);
}

function hideUnknownRecommendationPanels(exceptEventId = 0) {
  $$('#attendanceSummaryTable [data-summary-unknown-recommendations]').forEach((panel) => {
    const eventId = Number(panel.dataset.summaryUnknownRecommendations || 0);
    if (exceptEventId && eventId === exceptEventId) return;
    panel.classList.add('hidden');
    panel.innerHTML = '';
  });
}

function recommendationValueLabel(emp) {
  const id = Number(emp?.id || 0);
  const name = String(emp?.full_name || '').trim();
  return `ID ${id} | ${name}`;
}

function renderUnknownRecommendationPanel(eventId, query) {
  const panel = $(`#attendanceSummaryTable [data-summary-unknown-recommendations="${eventId}"]`);
  if (!panel) return;
  const list = findEmployeeRecommendations(query, 6);
  if (!list.length) {
    panel.classList.add('hidden');
    panel.innerHTML = '';
    return;
  }

  panel.innerHTML = list.map((emp) => {
    const photoUrl = employeePhotoUrl(emp);
    const job = String(emp?.job_title || '').trim() || '-';
    const code = String(emp?.employee_code || '').trim();
    return `
      <button type="button" class="unknown-recommendation-item" data-summary-unknown-pick-event="${eventId}" data-summary-unknown-pick-employee="${emp.id}">
        <span class="unknown-rec-photo-wrap">
          ${photoUrl
            ? `<img class="unknown-rec-photo" src="${escapeHtml(photoUrl)}" alt="${escapeHtml(emp?.full_name || 'employee')}" loading="lazy" />`
            : `<span class="unknown-rec-photo-fallback">${escapeHtml(employeeInitial(emp))}</span>`}
        </span>
        <span class="unknown-rec-meta">
          <span class="unknown-rec-name">${escapeHtml(emp?.full_name || '')}</span>
          <span class="unknown-rec-job">${escapeHtml(job)}</span>
          <span class="unknown-rec-extra">ID ${escapeHtml(String(emp?.id ?? ''))}${code ? ` | ${escapeHtml(code)}` : ''}</span>
        </span>
      </button>
    `;
  }).join('');

  panel.classList.remove('hidden');
  panel.querySelectorAll('[data-summary-unknown-pick-employee]').forEach((btn) => {
    btn.addEventListener('mousedown', (ev) => ev.preventDefault());
    btn.addEventListener('click', () => {
      const employeeId = Number(btn.dataset.summaryUnknownPickEmployee || 0);
      if (!employeeId) return;
      const input = $(`#attendanceSummaryTable input[data-summary-unknown-input="${eventId}"]`);
      const emp = (state.employees || []).find((e) => Number(e.id) === employeeId);
      if (input) {
        input.value = recommendationValueLabel(emp || { id: employeeId, full_name: '' });
      }
      hideUnknownRecommendationPanels();
    });
  });
}

function bindUnknownOverrideRecommendationInputs() {
  $$('#attendanceSummaryTable input[data-summary-unknown-input]').forEach((input) => {
    const eventId = Number(input.dataset.summaryUnknownInput || 0);
    if (!eventId) return;

    const refresh = () => {
      hideUnknownRecommendationPanels(eventId);
      renderUnknownRecommendationPanel(eventId, input.value);
    };

    input.addEventListener('focus', refresh);
    input.addEventListener('input', refresh);
    input.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape') {
        hideUnknownRecommendationPanels();
      }
    });
    input.addEventListener('blur', () => {
      window.setTimeout(() => hideUnknownRecommendationPanels(), 120);
    });
  });
}

function renderAttendanceSummaryTable() {
  const sortBy = $('#attendanceSummarySortBy')?.value || 'time';
  const sortDir = $('#attendanceSummarySortDir')?.value || 'asc';
  const filteredRows = filterAttendanceSummaryRows(state.attendanceSummaryRows || [], state.attendanceSummaryQuery);
  const rows = sortAttendanceSummaryRows(filteredRows, sortBy, sortDir);
  const dateStr = String(state.attendanceSelectedDate || todayStr());
  const target = normalizeTimeHHMM(state.attendanceTargetTime, '09:00');
  const targetMs = targetUtcMs(dateStr, target);
  renderAttendanceMetrics(rows, targetMs);
  renderTableRows(
    $('#attendanceSummaryTable tbody'),
    rows.map(r => `
      <tr class="${r.is_unknown ? 'row-unknown' : ''}">
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
        <td>${
          r.is_unknown && r.source_event_id
            ? `<div class="unknown-override-wrap">
                <div class="controls">
                  <input class="small-input" data-summary-unknown-input="${r.source_event_id}" type="text" placeholder="id or name" autocomplete="off" />
                  <button type="button" data-summary-unknown-override="${r.source_event_id}">Override</button>
                </div>
                <div class="unknown-recommendations hidden" data-summary-unknown-recommendations="${r.source_event_id}"></div>
              </div>`
            : '<span class="muted">-</span>'
        }</td>
      </tr>`).join('')
  );

  $$('#attendanceSummaryTable button[data-summary-unknown-override]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const eventId = Number(btn.dataset.summaryUnknownOverride || 0);
      if (!eventId) return;
      const input = $(`#attendanceSummaryTable input[data-summary-unknown-input="${eventId}"]`);
      const resolved = resolveEmployeeInputToId(input?.value || '');
      if (!resolved.ok) {
        alert(resolved.message);
        return;
      }
      const employeeId = resolved.employeeId;
      btn.disabled = true;
      try {
        await api(`/events/${eventId}/override`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ employee_id: employeeId }),
        });
        await loadEvents().catch(() => {});
      } catch (err) {
        alert(`Override failed: ${err.message}`);
      } finally {
        btn.disabled = false;
      }
    });
  });
  bindUnknownOverrideRecommendationInputs();
}

async function renderAttendanceSummary(events, dateStr) {
  const employees = await ensureEmployeesLoaded();
  state.attendanceSelectedDate = String(dateStr || todayStr());
  state.attendanceTargetTime = getEffectiveAttendanceTimeForDate(state.attendanceSelectedDate);
  state.attendanceSummaryRows = buildAttendanceSummaryRows(employees, events);
  const title = $('#attendanceSummaryTitle');
  if (title) {
    const today = todayStr();
    const target = state.attendanceTargetTime;
    title.textContent = dateStr === today
      ? `Today's Check-in List (${dateStr}) | Target ${target} UTC`
      : `Check-in List (${dateStr}) | Target ${target} UTC`;
  }
  refreshAttendanceTimeInfo(state.attendanceSelectedDate);
  renderAttendanceSummaryTable();
}

function bindAttendance() {
  $('#eventsDate').value = todayStr();
  const todayInput = $('#todayAttendanceTimeInput');
  if (todayInput) {
    todayInput.value = normalizeTimeHHMM(state.attendancePolicy?.standardTime, '09:00');
  }
  $('#loadEventsBtn').addEventListener('click', () => {
    const date = $('#eventsDate').value || todayStr();
    openAttendanceDateTab(date);
  });
  $('#attendanceTodayBtn')?.addEventListener('click', async () => {
    $('#eventsDate').value = todayStr();
    await loadEvents().catch(err => alert(err.message));
  });
  $('#refreshTodayBtn').addEventListener('click', async () => {
    $('#eventsDate').value = todayStr();
    await loadEvents().catch(err => alert(err.message));
  });
  $('#exportCsvBtn').addEventListener('click', () => {
    const date = $('#eventsDate').value;
    window.location.href = `/reports/daily.csv?date=${date}`;
  });
  $('#saveTodayAttendanceTimeBtn').addEventListener('click', async () => {
    const today = todayStr();
    const val = normalizeTimeHHMM($('#todayAttendanceTimeInput')?.value, '');
    if (!val) {
      alert('Enter a valid time (HH:MM)');
      return;
    }
    try {
      await api('/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          values: {
            attendance_today_override_date: today,
            attendance_today_override_time: val,
          },
        }),
      });
      state.attendancePolicy.overrideDate = today;
      state.attendancePolicy.overrideTime = val;
      state.attendanceTargetTime = getEffectiveAttendanceTimeForDate(state.attendanceSelectedDate || today);
      refreshAttendanceTimeInfo(state.attendanceSelectedDate || today);
      renderAttendanceSummaryTable();
      if ($('#eventsDate').value === today) {
        await loadEvents().catch(() => {});
      }
    } catch (err) {
      alert(`Save today's attendance time failed: ${err.message}`);
    }
  });
  $('#clearTodayAttendanceTimeBtn').addEventListener('click', async () => {
    try {
      await api('/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          values: {
            attendance_today_override_date: '',
            attendance_today_override_time: '',
          },
        }),
      });
      state.attendancePolicy.overrideDate = '';
      state.attendancePolicy.overrideTime = '';
      if ($('#todayAttendanceTimeInput')) {
        $('#todayAttendanceTimeInput').value = normalizeTimeHHMM(state.attendancePolicy.standardTime, '09:00');
      }
      state.attendanceTargetTime = getEffectiveAttendanceTimeForDate(state.attendanceSelectedDate || todayStr());
      refreshAttendanceTimeInfo(state.attendanceSelectedDate || todayStr());
      renderAttendanceSummaryTable();
    } catch (err) {
      alert(`Reset to standard attendance time failed: ${err.message}`);
    }
  });
  $('#attendanceSummarySortBy').addEventListener('change', () => renderAttendanceSummaryTable());
  $('#attendanceSummarySortDir').addEventListener('change', () => renderAttendanceSummaryTable());
  $('#attendanceSummarySearchInput')?.addEventListener('input', (e) => {
    state.attendanceSummaryQuery = String(e.target.value || '');
    renderAttendanceSummaryTable();
  });
  $('#clearAttendanceSearchBtn')?.addEventListener('click', () => {
    state.attendanceSummaryQuery = '';
    const input = $('#attendanceSummarySearchInput');
    if (input) input.value = '';
    renderAttendanceSummaryTable();
  });

  const autoRefreshCheckbox = $('#attendanceAutoRefresh');
  const startAutoRefresh = () => {
    if (state.attendanceAutoTimer) {
      clearInterval(state.attendanceAutoTimer);
      state.attendanceAutoTimer = null;
    }
    if (!autoRefreshCheckbox?.checked) return;
    state.attendanceAutoTimer = window.setInterval(async () => {
      if (!$('#tab-attendance')?.classList.contains('active')) return;
      if ($('#eventsDate')?.value !== todayStr()) return;
      await loadEvents().catch(() => {});
    }, 5000);
  };
  autoRefreshCheckbox?.addEventListener('change', startAutoRefresh);
  startAutoRefresh();

  document.addEventListener('click', (ev) => {
    const target = ev.target;
    if (!(target instanceof Element)) {
      hideUnknownRecommendationPanels();
      return;
    }
    if (!target.closest('.unknown-override-wrap')) {
      hideUnknownRecommendationPanels();
    }
  });
}

function resolveEmployeeInputToId(raw) {
  const value = String(raw || '').trim();
  if (!value) return { ok: false, message: 'Enter employee ID or name.' };

  const rows = Array.isArray(state.employees) ? state.employees : [];
  if (!rows.length) return { ok: false, message: 'Employee list is not loaded yet.' };

  if (/^\d+$/.test(value)) {
    const id = Number(value);
    const exists = rows.some((e) => Number(e.id) === id);
    return exists ? { ok: true, employeeId: id } : { ok: false, message: `Employee ID ${id} not found.` };
  }

  const idToken = value.match(/\bID\s*[:#-]?\s*(\d+)\b/i);
  if (idToken) {
    const id = Number(idToken[1]);
    const exists = rows.some((e) => Number(e.id) === id);
    return exists ? { ok: true, employeeId: id } : { ok: false, message: `Employee ID ${id} not found.` };
  }

  const norm = value.toLowerCase();
  const byName = rows.filter((e) => String(e.full_name || '').trim().toLowerCase() === norm);
  if (byName.length === 1) return { ok: true, employeeId: Number(byName[0].id) };
  if (byName.length > 1) {
    return { ok: false, message: `Multiple employees found for name "${value}". Enter employee ID.` };
  }

  const byCode = rows.filter((e) => String(e.employee_code || '').trim().toLowerCase() === norm);
  if (byCode.length === 1) return { ok: true, employeeId: Number(byCode[0].id) };
  if (byCode.length > 1) {
    return { ok: false, message: `Multiple employees found for code "${value}". Enter employee ID.` };
  }

  return { ok: false, message: `No employee matched "${value}". Use suggestions or enter employee ID.` };
}

function renderEmployeePhotos(items = []) {
  const grid = $('#employeePhotoGrid');
  const summary = $('#employeePhotosSummary');
  const result = $('#employeePhotoActionResult');
  if (!items.length) {
    summary.textContent = 'No uploaded pictures yet';
    grid.innerHTML = '<div class="empty-box">No uploaded face/ReID pictures for this employee yet.</div>';
    if (result && !result.textContent) result.textContent = '';
    return;
  }
  const faceCount = items.filter(x => x.kind === 'face').length;
  const reidCount = items.filter(x => x.kind === 'reid').length;
  summary.textContent = `Total ${items.length} | Face ${faceCount} | ReID ${reidCount}`;
  grid.innerHTML = items.map(p => {
    const url = p.media_url || '';
    return `
      <div class="photo-card">
        <a href="${escapeHtml(url)}" target="_blank">
          <img loading="lazy" src="${escapeHtml(url)}" alt="${escapeHtml(p.original_filename || 'uploaded photo')}" />
        </a>
        <div class="photo-meta">
          <div><span class="pill">${escapeHtml((p.kind || '').toUpperCase())}</span></div>
          <div title="${escapeHtml(p.original_filename || '')}">${escapeHtml(p.original_filename || '(no name)')}</div>
          <div>${escapeHtml(asIsoOrEmpty(p.created_at))}</div>
          <div><button type="button" class="danger-btn" data-photo-delete-id="${p.id}">Remove</button></div>
        </div>
      </div>`;
  }).join('');

  $$('#employeePhotoGrid button[data-photo-delete-id]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const employeeId = Number(state.selectedEmployeeId);
      const photoId = Number(btn.dataset.photoDeleteId);
      if (!employeeId || !photoId) return;
      if (!window.confirm(`Remove photo ${photoId}? Linked embedding will also be removed if available.`)) return;
      btn.disabled = true;
      try {
        const data = await api(`/employees/${employeeId}/photos/${photoId}`, { method: 'DELETE' });
        if (result) result.textContent = JSON.stringify(data, null, 2);
        await loadEmployees();
        await loadEmployeeDetail(employeeId);
      } catch (err) {
        if (result) result.textContent = `ERROR: ${err.message}`;
      } finally {
        btn.disabled = false;
      }
    });
  });
}

function renderEmployeeHistory(items = []) {
  renderTableRows($('#employeeHistoryTable tbody'), items.map(r => `
    <tr>
      <td>${r.id}</td>
      <td>${escapeHtml(asIsoOrEmpty(r.ts))}</td>
      <td>${escapeHtml(r.method)}</td>
      <td>${Number(r.confidence || 0).toFixed(3)}</td>
      <td>${escapeHtml(r.camera_id || '')}</td>
      <td>${escapeHtml(r.track_uid || '')}</td>
      <td>${eventImageLink(r)}</td>
    </tr>`).join(''));
}

async function loadEmployeeHistory() {
  const employeeId = state.selectedEmployeeId;
  if (!employeeId) {
    renderEmployeeHistory([]);
    return;
  }
  const from = $('#employeeHistoryDateFrom').value;
  const to = $('#employeeHistoryDateTo').value;
  const q = new URLSearchParams();
  if (from) q.set('date_from', from);
  if (to) q.set('date_to', to);
  const data = await api(`/employees/${employeeId}/attendance?${q.toString()}`);
  renderEmployeeHistory(data.items || []);
}

function fillEmployeeDetailForm(data) {
  const form = $('#employeeDetailForm');
  form.id.value = data.id ?? '';
  form.full_name.value = data.full_name ?? '';
  form.employee_code.value = data.employee_code ?? '';
  form.birth_date.value = data.birth_date ?? '';
  form.job_title.value = data.job_title ?? '';
  form.address.value = data.address ?? '';
  form.status.value = data.status ?? 'active';

  $('#employeeDetailTitle').textContent = `Profile: ${data.full_name || ''}`;
  $('#employeeDetailCounts').textContent = [
    `ID ${data.id}`,
    `Face emb ${data.face_embeddings_count ?? 0}`,
    `ReID emb ${data.reid_embeddings_count ?? 0}`,
    `Photos ${data.uploaded_images_count ?? 0}`,
  ].join(' | ');

  $('#employeeDetailHint').textContent = `Employee ${data.employee_code || ''} selected.`;
  $('#employeeDetailResult').textContent = JSON.stringify(data, null, 2);

  if (data.history_default_date_from) $('#employeeHistoryDateFrom').value = data.history_default_date_from;
  if (data.history_default_date_to) $('#employeeHistoryDateTo').value = data.history_default_date_to;
  renderEmployeePhotos(data.uploaded_images || []);
}

async function loadEmployeeDetail(employeeId) {
  const data = await api(`/employees/${employeeId}`);
  fillEmployeeDetailForm(data);
  await loadEmployeeHistory();
  return data;
}

async function setSelectedEmployee(employeeId, { openTab = true } = {}) {
  state.selectedEmployeeId = Number(employeeId) || null;
  if (!state.selectedEmployeeId) return;
  if (openTab) switchTab('employee-details');
  try {
    await loadEmployeeDetail(state.selectedEmployeeId);
  } catch (err) {
    $('#employeeDetailResult').textContent = `ERROR: ${err.message}`;
    throw err;
  }
}

function bindEmployeeDetails() {
  $('#loadEmployeeDetailBtn').addEventListener('click', async () => {
    const id = Number($('#detailEmployeeIdInput').value);
    if (!id) return alert('Enter employee ID');
    try {
      await setSelectedEmployee(id, { openTab: true });
    } catch (err) {
      alert(`Load employee failed: ${err.message}`);
    }
  });

  $('#loadEmployeeHistoryBtn').addEventListener('click', () => {
    loadEmployeeHistory().catch(err => alert(`Load history failed: ${err.message}`));
  });

  $('#employeeDetailForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const id = Number(form.id.value);
    if (!id) return alert('No employee selected');
    const payload = normalizeEmployeePayload(form, { includeCode: false });
    try {
      const data = await api(`/employees/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      $('#employeeDetailResult').textContent = JSON.stringify(data, null, 2);
      await loadEmployees();
      await loadEmployeeDetail(id);
    } catch (err) {
      $('#employeeDetailResult').textContent = `ERROR: ${err.message}`;
    }
  });

  $('#employeePhotoAddForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const employeeId = Number(state.selectedEmployeeId);
    const result = $('#employeePhotoActionResult');
    if (!employeeId) {
      alert('Select an employee first');
      return;
    }
    const form = e.target;
    const fd = new FormData(form);
    const kind = String(fd.get('kind') || 'face').toLowerCase();
    const files = form.querySelector('input[name="files"]')?.files;
    if (!files || !files.length) {
      alert('Select at least one image');
      return;
    }
    const upload = new FormData();
    for (const f of files) upload.append('files', f);
    const btn = $('#employeePhotoAddBtn');
    if (btn) {
      btn.disabled = true;
      btn.dataset.originalText = btn.dataset.originalText || btn.textContent || 'Add Images To Employee';
      btn.textContent = 'Uploading...';
    }
    if (result) result.textContent = `Uploading ${files.length} ${kind} image(s) to employee ${employeeId}...`;
    try {
      const data = await api(`/employees/${employeeId}/enroll/${kind}`, { method: 'POST', body: upload });
      if (result) result.textContent = JSON.stringify(data, null, 2);
      form.reset();
      await loadEmployees();
      await loadEmployeeDetail(employeeId);
    } catch (err) {
      if (result) result.textContent = `ERROR: ${err.message}`;
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = btn.dataset.originalText || 'Add Images To Employee';
      }
    }
  });
}

async function loadSettings() {
  const data = await api('/settings');
  const v = data.values || {};
  state.attendancePolicy.standardTime = normalizeTimeHHMM(v.standard_attendance_time, '09:00');
  state.attendancePolicy.overrideDate = String(v.attendance_today_override_date || '');
  state.attendancePolicy.overrideTime = normalizeTimeHHMM(v.attendance_today_override_time, '');

  const form = $('#settingsForm');
  form.face_threshold.value = v.face_threshold ?? '';
  form.reid_threshold.value = v.reid_threshold ?? '';
  form.standard_attendance_time.value = state.attendancePolicy.standardTime;
  form.morning_window_start.value = v.morning_window_start ?? '';
  form.morning_window_end.value = v.morning_window_end ?? '';
  form.snapshot_retention_days.value = v.snapshot_retention_days ?? '';
  form.save_snapshots_default.value = String(v.save_snapshots_default ?? false);
  form.roi_polygon.value = JSON.stringify(v.roi_polygon ?? [], null, 2);
  form.entry_line.value = JSON.stringify(v.entry_line ?? {}, null, 2);
  const selectedDate = $('#eventsDate')?.value || todayStr();
  const today = todayStr();
  const effectiveToday = getEffectiveAttendanceTimeForDate(today);
  if ($('#todayAttendanceTimeInput')) {
    $('#todayAttendanceTimeInput').value = String(state.attendancePolicy.overrideDate || '') === today && state.attendancePolicy.overrideTime
      ? state.attendancePolicy.overrideTime
      : effectiveToday;
  }
  state.attendanceSelectedDate = selectedDate;
  state.attendanceTargetTime = getEffectiveAttendanceTimeForDate(selectedDate);
  refreshAttendanceTimeInfo(selectedDate);
  renderAttendanceSummaryTable();
  $('#settingsResult').textContent = JSON.stringify(v, null, 2);
}

function bindSettings() {
  $('#reloadSettingsBtn').addEventListener('click', () => loadSettings().catch(err => alert(err.message)));
  $('#settingsForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = e.target;
    const values = {
      face_threshold: Number(f.face_threshold.value),
      reid_threshold: Number(f.reid_threshold.value),
      standard_attendance_time: normalizeTimeHHMM(f.standard_attendance_time.value, '09:00'),
      morning_window_start: f.morning_window_start.value,
      morning_window_end: f.morning_window_end.value,
      snapshot_retention_days: Number(f.snapshot_retention_days.value),
      save_snapshots_default: f.save_snapshots_default.value === 'true',
      roi_polygon: JSON.parse(f.roi_polygon.value),
      entry_line: JSON.parse(f.entry_line.value),
    };
    try {
      const resp = await api('/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ values })
      });
      $('#settingsResult').textContent = JSON.stringify(resp.values, null, 2);
    } catch (err) {
      $('#settingsResult').textContent = `ERROR: ${err.message}`;
    }
  });
}

async function init() {
  bindTabs();
  bindEmployeeForm();
  bindCameras();
  bindAttendance();
  bindSettings();

  $('#eventsDate').value = todayStr();

  await checkHealth();
  await Promise.allSettled([loadEmployees(), loadCameras(), loadEvents(), loadSettings()]);
}

init().catch(err => {
  console.error(err);
  setHealth(`Init error: ${err.message}`, false);
});
