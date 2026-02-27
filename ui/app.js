const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  selectedEmployeeId: null,
  employees: [],
  cameras: [],
  employeesSearchQuery: '',
  attendanceSummaryRows: [],
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

function eventRow(r) {
  const time = asIsoOrEmpty(r.ts);
  return `
    <tr>
      <td>${escapeHtml(r.id)}</td>
      <td>${escapeHtml(time)}</td>
      <td>${escapeHtml(r.employee_name || '')}</td>
      <td>${escapeHtml(r.employee_code || '')}</td>
      <td>${escapeHtml(r.method)}</td>
      <td>${Number(r.confidence || 0).toFixed(3)}</td>
      <td>${escapeHtml(r.camera_id || '')}</td>
      <td>${escapeHtml(r.track_uid || '')}</td>
      <td>${eventImageLink(r)}</td>
    </tr>`;
}

async function loadEmployees() {
  const rows = await api('/employees');
  state.employees = rows;
  renderEmployeesTable();
}

function getFilteredEmployees() {
  const q = String(state.employeesSearchQuery || '').trim().toLowerCase();
  if (!q) return [...(state.employees || [])];
  return (state.employees || []).filter(r => {
    const idStr = String(r.id ?? '').toLowerCase();
    const code = String(r.employee_code ?? '').toLowerCase();
    const name = String(r.full_name ?? '').toLowerCase();
    return idStr.includes(q) || code.includes(q) || name.includes(q);
  });
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
        <td>${escapeHtml(r.birth_date || '')}</td>
        <td>${escapeHtml(r.job_title || '')}</td>
        <td>${escapeHtml(r.status)}</td>
        <td>${r.face_embeddings_count ?? 0}</td>
        <td>${r.reid_embeddings_count ?? 0}</td>
        <td>${r.uploaded_images_count ?? 0}</td>
        <td><button data-open-detail-id="${r.id}">Open</button></td>
      </tr>`).join('')
  );

  $$('#employeesTable button[data-open-detail-id]').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = Number(btn.dataset.openDetailId);
      if (!id) return;
      $('#detailEmployeeIdInput').value = String(id);
      setSelectedEmployee(id, { openTab: true });
    });
  });

  const hint = $('#employeeDetailHint');
  const q = String(state.employeesSearchQuery || '').trim();
  if (hint && q && !rows.length) {
    hint.textContent = `No employees found for search: ${q}`;
  }
}

async function ensureEmployeesLoaded() {
  if (Array.isArray(state.employees) && state.employees.length) return state.employees;
  await loadEmployees();
  return state.employees;
}

function bindEmployeeForm() {
  $('#employeeForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const payload = normalizeEmployeePayload(form, { includeCode: true });
    try {
      const created = await api('/employees', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      form.reset();
      await loadEmployees();
      if (created?.id) {
        $('#detailEmployeeIdInput').value = String(created.id);
        await setSelectedEmployee(Number(created.id), { openTab: false });
      }
    } catch (err) {
      alert(`Add employee failed: ${err.message}`);
    }
  });
  $('#refreshEmployeesBtn').addEventListener('click', () => loadEmployees().catch(err => alert(err.message)));
  $('#employeesSearchInput').addEventListener('input', (e) => {
    state.employeesSearchQuery = e.target.value || '';
    renderEmployeesTable();
  });
  $('#clearEmployeesSearchBtn').addEventListener('click', () => {
    state.employeesSearchQuery = '';
    $('#employeesSearchInput').value = '';
    renderEmployeesTable();
  });
}

function bindEnrollmentForm() {
  $('#faceEnrollForm').addEventListener('submit', async (e) => {
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
  const date = $('#eventsDate').value;
  const rows = await api(`/events?date=${date}`);
  renderTableRows($('#eventsTable tbody'), rows.map(eventRow).join(''));
  await renderAttendanceSummary(rows, date);
}

function buildAttendanceSummaryRows(employees, events) {
  const earliestByEmployee = new Map();
  for (const evt of events || []) {
    if (!evt || evt.employee_id == null) continue;
    const empId = Number(evt.employee_id);
    if (!Number.isFinite(empId)) continue;
    const existing = earliestByEmployee.get(empId);
    const evtTime = new Date(evt.ts);
    if (!existing || evtTime < new Date(existing.ts)) {
      earliestByEmployee.set(empId, evt);
    }
  }

  return (employees || []).map(emp => {
    const evt = earliestByEmployee.get(Number(emp.id));
    return {
      employee_id: emp.id,
      full_name: emp.full_name || '',
      employee_code: emp.employee_code || '',
      status: emp.status || '',
      entrance_ts: evt?.ts || null,
      method: evt?.method || '',
      confidence: evt?.confidence ?? null,
    };
  });
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

function renderAttendanceSummaryTable() {
  const sortBy = $('#attendanceSummarySortBy')?.value || 'name';
  const sortDir = $('#attendanceSummarySortDir')?.value || 'asc';
  const rows = sortAttendanceSummaryRows(state.attendanceSummaryRows || [], sortBy, sortDir);
  renderTableRows(
    $('#attendanceSummaryTable tbody'),
    rows.map(r => `
      <tr>
        <td>${escapeHtml(r.full_name)}</td>
        <td>${escapeHtml(r.employee_code)}</td>
        <td>${r.entrance_ts ? escapeHtml(asIsoOrEmpty(r.entrance_ts)) : '<span class="muted">Not checked in</span>'}</td>
        <td>${escapeHtml(r.method || '')}</td>
        <td>${r.confidence == null ? '' : Number(r.confidence).toFixed(3)}</td>
        <td>${escapeHtml(r.status || '')}</td>
      </tr>`).join('')
  );
}

async function renderAttendanceSummary(events, dateStr) {
  const employees = await ensureEmployeesLoaded();
  state.attendanceSummaryRows = buildAttendanceSummaryRows(employees, events);
  const title = $('#attendanceSummaryTitle');
  if (title) {
    const today = todayStr();
    title.textContent = dateStr === today ? `Today's Entrance Summary (${dateStr})` : `Entrance Summary (${dateStr})`;
  }
  renderAttendanceSummaryTable();
}

function bindAttendance() {
  $('#eventsDate').value = todayStr();
  $('#loadEventsBtn').addEventListener('click', () => loadEvents().catch(err => alert(err.message)));
  $('#exportCsvBtn').addEventListener('click', () => {
    const date = $('#eventsDate').value;
    window.location.href = `/reports/daily.csv?date=${date}`;
  });
  $('#attendanceSummarySortBy').addEventListener('change', () => renderAttendanceSummaryTable());
  $('#attendanceSummarySortDir').addEventListener('change', () => renderAttendanceSummaryTable());
}

async function loadUnknowns() {
  const date = $('#unknownDate').value;
  const rows = await api(`/events?date=${date}`);
  const unknowns = rows.filter(r => r.method === 'unknown' || r.employee_id == null);
  renderTableRows($('#unknownsTable tbody'), unknowns.map(r => `
    <tr>
      <td>${r.id}</td>
      <td>${escapeHtml(asIsoOrEmpty(r.ts))}</td>
      <td>${escapeHtml(r.method)}</td>
      <td>${Number(r.confidence).toFixed(3)}</td>
      <td>${escapeHtml(r.track_uid)}</td>
      <td><input class="small-input" data-event-id="${r.id}" type="number" min="1" placeholder="employee id" /></td>
      <td><button data-override-id="${r.id}">Override</button></td>
    </tr>`).join(''));

  $$('#unknownsTable button[data-override-id]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.dataset.overrideId;
      const input = $(`#unknownsTable input[data-event-id="${id}"]`);
      const employeeId = Number(input.value);
      if (!employeeId) return alert('Enter employee ID');
      try {
        await api(`/events/${id}/override`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ employee_id: employeeId })
        });
        await loadUnknowns();
        await loadEvents().catch(() => {});
      } catch (err) {
        alert(`Override failed: ${err.message}`);
      }
    });
  });
}

function bindUnknowns() {
  $('#unknownDate').value = todayStr();
  $('#loadUnknownsBtn').addEventListener('click', () => loadUnknowns().catch(err => alert(err.message)));
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
  const form = $('#settingsForm');
  form.face_threshold.value = v.face_threshold ?? '';
  form.reid_threshold.value = v.reid_threshold ?? '';
  form.morning_window_start.value = v.morning_window_start ?? '';
  form.morning_window_end.value = v.morning_window_end ?? '';
  form.snapshot_retention_days.value = v.snapshot_retention_days ?? '';
  form.save_snapshots_default.value = String(v.save_snapshots_default ?? false);
  form.roi_polygon.value = JSON.stringify(v.roi_polygon ?? [], null, 2);
  form.entry_line.value = JSON.stringify(v.entry_line ?? {}, null, 2);
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
  bindEmployeeDetails();
  bindCameras();
  bindEnrollmentForm();
  bindAttendance();
  bindUnknowns();
  bindSettings();

  $('#eventsDate').value = todayStr();
  $('#unknownDate').value = todayStr();

  await checkHealth();
  await Promise.allSettled([loadEmployees(), loadCameras(), loadEvents(), loadUnknowns(), loadSettings()]);
}

init().catch(err => {
  console.error(err);
  setHealth(`Init error: ${err.message}`, false);
});
