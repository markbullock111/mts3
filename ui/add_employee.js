const $ = (sel) => document.querySelector(sel);

let lastCreatedEmployeeId = null;

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
  const input = $('#addEmployeeUploadEmployeeId');
  const hint = $('#uploadTargetHint');
  if (input && n > 0) input.value = String(n);
  if (hint) {
    hint.textContent = n > 0 ? `Target Employee: ${n}` : 'Target Employee: not selected';
  }
}

function bind() {
  const queryId = employeeIdFromQuery();
  if (queryId) {
    lastCreatedEmployeeId = queryId;
    setStatus(`Using employee ${queryId}`, true);
    setUploadTargetEmployeeId(queryId);
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
      if (result) {
        result.textContent = `Employee created successfully.\n${JSON.stringify(data, null, 2)}`;
      }
      setUploadTargetEmployeeId(lastCreatedEmployeeId);
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
    const employeeId = Number(fd.get('employee_id') || 0);
    const kind = String(fd.get('kind') || 'face').toLowerCase();
    const files = form.querySelector('input[name="files"]')?.files;
    const result = $('#addEmployeeUploadResult');
    const submitBtn = $('#addEmployeeUploadBtn');
    const statusWrap = $('#addEmployeeUploadStatus');
    const statusText = $('#addEmployeeUploadStatusText');

    if (!employeeId) {
      if (result) result.textContent = 'Employee ID is required.';
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
}

bind();
