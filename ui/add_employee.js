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

function goEnrollmentPage() {
  const id = Number(lastCreatedEmployeeId || 0);
  const url = id > 0
    ? `/ui/enrollment.html?employee_id=${encodeURIComponent(String(id))}`
    : '/ui/enrollment.html';
  window.location.href = url;
}

function bind() {
  const queryId = employeeIdFromQuery();
  if (queryId) {
    lastCreatedEmployeeId = queryId;
    setStatus(`Using employee ${queryId}`, true);
  }

  $('#backToAdminBtn')?.addEventListener('click', () => {
    window.location.href = '/ui/';
  });

  $('#goEnrollmentBtn')?.addEventListener('click', () => {
    goEnrollmentPage();
  });

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
      setStatus(`Created employee ${lastCreatedEmployeeId || ''}`.trim(), true);
      form.reset();
    } catch (err) {
      if (result) result.textContent = `ERROR: ${err.message}`;
      setStatus('Create failed', false);
    }
  });
}

bind();
