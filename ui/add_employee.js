const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

let lastCreatedEmployeeId = null;
let currentMainPhotoId = null;
let currentEmployeeMeta = null;

function employeeIdFromQuery() {
  const raw = new URLSearchParams(window.location.search).get('employee_id');
  const id = Number(raw);
  if (!Number.isFinite(id) || id <= 0) return null;
  return id;
}

function setStatus(text, ok = true) {
  const badge = $('#addEmployeeStatus');
  if (!badge) return;
  badge.textContent = text;
  badge.style.borderColor = ok ? 'rgba(43,122,75,0.35)' : 'rgba(166,60,36,0.35)';
  badge.style.color = ok ? '#2b7a4b' : '#a63c24';
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      detail = data.detail || JSON.stringify(data);
    } catch {
      // ignore parse errors
    }
    throw new Error(detail);
  }
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) return res.json();
  return res.text();
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

function normalizeEmployeePayload(form) {
  const fd = new FormData(form);
  return {
    full_name: String(fd.get('full_name') || '').trim(),
    employee_code: String(fd.get('employee_code') || '').trim(),
    status: String(fd.get('status') || 'active'),
    birth_date: String(fd.get('birth_date') || '').trim() || null,
    job_title: String(fd.get('job_title') || '').trim() || null,
    address: String(fd.get('address') || '').trim() || null,
  };
}

function setUploadTargetEmployeeId(id) {
  const n = Number(id || 0);
  const hint = $('#uploadTargetHint');
  const uploadBtn = $('#addEmployeeUploadBtn');
  const clearMainBtn = $('#addEmployeeClearMainPhotoBtn');
  if (hint) {
    hint.textContent = n > 0 ? `Target Employee: ${n}` : 'Target Employee: not selected';
  }
  if (uploadBtn) uploadBtn.disabled = n <= 0;
  if (clearMainBtn) clearMainBtn.disabled = n <= 0;
}

function renderCurrentEmployeeMeta() {
  const meta = $('#addEmployeeCreatedMeta');
  if (!meta) return;
  const id = Number(lastCreatedEmployeeId || 0);
  if (!id || !currentEmployeeMeta) {
    meta.textContent = 'No employee selected yet.';
    return;
  }
  const name = currentEmployeeMeta.full_name || '-';
  const code = currentEmployeeMeta.employee_code || '-';
  meta.textContent = `Current Employee: ID ${id} | ${name} | Code ${code}`;
}

function renderUploadedPhotoGrid(employee) {
  const grid = $('#addEmployeePhotoGrid');
  const summary = $('#addEmployeePhotosSummary');
  if (!grid || !summary) return;

  const photos = Array.isArray(employee?.uploaded_images) ? employee.uploaded_images : [];
  currentMainPhotoId = employee?.main_photo_id ?? null;

  if (!photos.length) {
    summary.textContent = 'No uploaded pictures';
    grid.innerHTML = '<div class="empty-box">No uploaded pictures yet. Upload face/reid images first.</div>';
    return;
  }

  const faceCount = photos.filter((x) => x.kind === 'face').length;
  const reidCount = photos.filter((x) => x.kind === 'reid').length;
  summary.textContent = `Total ${photos.length} | Face ${faceCount} | ReID ${reidCount} | Main ${currentMainPhotoId || '-'}`;

  grid.innerHTML = photos.map((p) => {
    const url = p.media_url || '';
    const isMain = Number(p.id) === Number(currentMainPhotoId || 0);
    return `
      <div class="photo-card">
        <a href="${escapeHtml(url)}" target="_blank">
          <img loading="lazy" src="${escapeHtml(url)}" alt="${escapeHtml(p.original_filename || 'uploaded photo')}" />
        </a>
        <div class="photo-meta">
          <div>
            <span class="pill">${escapeHtml((p.kind || '').toUpperCase())}</span>
            ${isMain ? '<span class="pill">MAIN</span>' : ''}
          </div>
          <div title="${escapeHtml(p.original_filename || '')}">${escapeHtml(p.original_filename || '(no name)')}</div>
          <div>${escapeHtml(asIsoOrEmpty(p.created_at))}</div>
          <div class="controls">
            <button type="button" data-add-set-main-photo-id="${p.id}" ${isMain ? 'disabled' : ''}>${isMain ? 'Main' : 'Set Main'}</button>
          </div>
        </div>
      </div>
    `;
  }).join('');

  $$('#addEmployeePhotoGrid button[data-add-set-main-photo-id]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const employeeId = Number(lastCreatedEmployeeId || 0);
      const photoId = Number(btn.dataset.addSetMainPhotoId || 0);
      const result = $('#addEmployeeUploadResult');
      if (!employeeId || !photoId) return;
      btn.disabled = true;
      try {
        const data = await api(`/employees/${employeeId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ main_photo_id: photoId }),
        });
        if (result) result.textContent = JSON.stringify(data, null, 2);
        await loadCurrentEmployeeDetails();
        setStatus(`Set main picture for employee ${employeeId}`, true);
      } catch (err) {
        if (result) result.textContent = `ERROR: ${err.message}`;
        setStatus('Set main picture failed', false);
      } finally {
        btn.disabled = false;
      }
    });
  });
}

async function loadCurrentEmployeeDetails() {
  const employeeId = Number(lastCreatedEmployeeId || 0);
  if (!employeeId) {
    renderUploadedPhotoGrid({ uploaded_images: [], main_photo_id: null });
    return null;
  }
  const data = await api(`/employees/${employeeId}`);
  currentEmployeeMeta = data;
  renderUploadedPhotoGrid(data);
  renderCurrentEmployeeMeta();
  return data;
}

function bind() {
  const queryId = employeeIdFromQuery();
  if (queryId) {
    lastCreatedEmployeeId = queryId;
    setStatus(`Using employee ${queryId}`, true);
    setUploadTargetEmployeeId(queryId);
    loadCurrentEmployeeDetails().catch(() => {});
  } else {
    setUploadTargetEmployeeId(null);
    renderUploadedPhotoGrid({ uploaded_images: [], main_photo_id: null });
    renderCurrentEmployeeMeta();
  }

  $('#addEmployeeForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const result = $('#addEmployeeResult');
    const payload = normalizeEmployeePayload(form);
    if (!payload.full_name || !payload.employee_code) {
      setStatus('Missing required fields', false);
      if (result) result.textContent = 'Full Name and Employee Code are required.';
      return;
    }
    try {
      const data = await api('/employees', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      lastCreatedEmployeeId = Number(data?.id || 0) || null;
      currentEmployeeMeta = data;
      if (result) {
        result.textContent = `Employee created successfully.\n${JSON.stringify(data, null, 2)}`;
      }
      setUploadTargetEmployeeId(lastCreatedEmployeeId);
      renderCurrentEmployeeMeta();
      await loadCurrentEmployeeDetails();
      setStatus(`Created employee ${lastCreatedEmployeeId || ''}`.trim(), true);
      form.reset();
    } catch (err) {
      if (result) result.textContent = `ERROR: ${err.message}`;
      setStatus('Create failed', false);
    }
  });

  $('#addEmployeeUploadForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const fd = new FormData(form);
    const employeeId = Number(lastCreatedEmployeeId || 0);
    const kind = String(fd.get('kind') || 'face').toLowerCase();
    const files = form.querySelector('input[name="files"]')?.files;
    const result = $('#addEmployeeUploadResult');
    const submitBtn = $('#addEmployeeUploadBtn');
    const statusWrap = $('#addEmployeeUploadStatus');
    const statusText = $('#addEmployeeUploadStatusText');

    if (!employeeId) {
      if (result) result.textContent = 'Add employee first, then upload images.';
      setStatus('Upload failed', false);
      return;
    }
    if (!files || !files.length) {
      if (result) result.textContent = 'Select at least one image.';
      setStatus('Upload failed', false);
      return;
    }

    const upload = new FormData();
    for (const f of files) upload.append('files', f);

    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.dataset.originalText = submitBtn.dataset.originalText || submitBtn.textContent || 'Upload Images';
      submitBtn.textContent = 'Uploading...';
    }
    if (statusWrap && statusText) {
      statusWrap.classList.remove('hidden');
      statusWrap.classList.add('active');
      statusText.textContent = `Uploading ${files.length} image(s) to employee ${employeeId} (${kind.toUpperCase()})...`;
    }
    if (result) result.textContent = 'Uploading...';

    try {
      const data = await api(`/employees/${employeeId}/enroll/${kind}`, {
        method: 'POST',
        body: upload,
      });
      lastCreatedEmployeeId = employeeId;
      setUploadTargetEmployeeId(employeeId);
      if (result) result.textContent = JSON.stringify(data, null, 2);
      await loadCurrentEmployeeDetails();
      setStatus(`Upload complete for employee ${employeeId}`, true);
    } catch (err) {
      if (result) result.textContent = `ERROR: ${err.message}`;
      setStatus('Upload failed', false);
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = submitBtn.dataset.originalText || 'Upload Images';
      }
      if (statusWrap && statusText) {
        statusText.textContent = 'Idle';
        statusWrap.classList.remove('active');
        window.setTimeout(() => statusWrap.classList.add('hidden'), 600);
      }
    }
  });

  $('#addEmployeeClearMainPhotoBtn')?.addEventListener('click', async () => {
    const employeeId = Number(lastCreatedEmployeeId || 0);
    const result = $('#addEmployeeUploadResult');
    if (!employeeId) {
      if (result) result.textContent = 'Add employee first.';
      setStatus('Clear main picture failed', false);
      return;
    }
    try {
      const data = await api(`/employees/${employeeId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ main_photo_id: null }),
      });
      if (result) result.textContent = JSON.stringify(data, null, 2);
      await loadCurrentEmployeeDetails();
      setStatus(`Cleared main picture for employee ${employeeId}`, true);
    } catch (err) {
      if (result) result.textContent = `ERROR: ${err.message}`;
      setStatus('Clear main picture failed', false);
    }
  });
}

bind();
