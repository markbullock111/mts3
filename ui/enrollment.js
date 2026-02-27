const $ = (sel) => document.querySelector(sel);

function setStatus(text, ok = true) {
  const badge = $('#enrollmentStatusBadge');
  if (!badge) return;
  badge.textContent = text;
  badge.style.borderColor = ok ? 'rgba(43,122,75,0.35)' : 'rgba(166,60,36,0.35)';
  badge.style.color = ok ? '#2b7a4b' : '#a63c24';
}

function employeeIdFromQuery() {
  const raw = new URLSearchParams(window.location.search).get('employee_id');
  const id = Number(raw);
  if (!Number.isFinite(id) || id <= 0) return null;
  return id;
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      detail = data.detail || JSON.stringify(data);
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) return res.json();
  return res.text();
}

function bind() {
  const prefillId = employeeIdFromQuery();
  if (prefillId && $('#enrollEmployeeIdInput')) {
    $('#enrollEmployeeIdInput').value = String(prefillId);
  }

  $('#backToAddEmployeeBtn')?.addEventListener('click', () => {
    const id = Number($('#enrollEmployeeIdInput')?.value || 0);
    const url = id > 0 ? `/ui/add_employee.html?employee_id=${encodeURIComponent(String(id))}` : '/ui/add_employee.html';
    window.location.href = url;
  });

  $('#backToAdminFromEnrollBtn')?.addEventListener('click', () => {
    window.location.href = '/ui/';
  });

  $('#enrollmentForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const fd = new FormData(form);
    const employeeId = String(fd.get('employee_id') || '').trim();
    const kind = String(fd.get('kind') || 'face').trim().toLowerCase();
    const files = form.querySelector('input[name="files"]')?.files;
    const result = $('#enrollResult');
    const submitBtn = $('#enrollSubmitBtn');
    const statusWrap = $('#enrollUploadStatus');
    const statusText = $('#enrollUploadStatusText');
    const targetInfo = $('#enrollTargetInfo');

    if (!employeeId) {
      setStatus('Missing employee ID', false);
      if (result) result.textContent = 'Employee ID is required.';
      return;
    }
    if (!files || !files.length) {
      setStatus('No images selected', false);
      if (result) result.textContent = 'Select at least one image.';
      return;
    }

    const upload = new FormData();
    for (const f of files) upload.append('files', f);

    if (targetInfo) {
      targetInfo.textContent = `Target Employee ID: ${employeeId} | Mode: ${kind.toUpperCase()} | Files: ${files.length}`;
    }
    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.dataset.originalText = submitBtn.dataset.originalText || submitBtn.textContent || 'Upload Enrollment Images';
      submitBtn.textContent = 'Uploading...';
    }
    if (statusWrap && statusText) {
      statusWrap.classList.remove('hidden');
      statusWrap.classList.add('active');
      statusText.textContent = `Uploading ${files.length} image(s) for Employee ID ${employeeId} (${kind.toUpperCase()})...`;
    }
    if (result) result.textContent = 'Uploading...';
    setStatus('Uploading', true);

    try {
      const data = await api(`/employees/${employeeId}/enroll/${kind}`, {
        method: 'POST',
        body: upload,
      });
      if (result) result.textContent = JSON.stringify(data, null, 2);
      setStatus('Upload complete', true);
    } catch (err) {
      if (result) result.textContent = `ERROR: ${err.message}`;
      setStatus('Upload failed', false);
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = submitBtn.dataset.originalText || 'Upload Enrollment Images';
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
