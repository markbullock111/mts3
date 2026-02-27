const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  selectedEmployeeId: null,
};

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

function renderTableRows(tbody, rowsHtml) {
  tbody.innerHTML = rowsHtml || '<tr><td colspan="99">No rows</td></tr>';
}

function eventImageLink(r) {
  const url = r.image_url || (r.image_path ? `file:///${String(r.image_path).replaceAll('\\', '/')}` : '');
  if (!url) return '';
  return `<a class="image-link" target="_blank" href="${escapeHtml(url)}">open</a>`;
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

function normalizeEmployeePayload(form) {
  const fd = new FormData(form);
  return {
    full_name: String(fd.get('full_name') || '').trim(),
    status: String(fd.get('status') || 'active'),
    birth_date: String(fd.get('birth_date') || '').trim() || null,
    job_title: String(fd.get('job_title') || '').trim() || null,
    address: String(fd.get('address') || '').trim() || null,
  };
}

function getEmployeeIdFromQuery() {
  const raw = new URLSearchParams(window.location.search).get('id');
  const id = Number(raw);
  if (!Number.isFinite(id) || id <= 0) return null;
  return id;
}

function setHealth(text, ok = true) {
  const badge = $('#detailHealthBadge');
  if (!badge) return;
  badge.textContent = text;
  badge.style.borderColor = ok ? 'rgba(43,122,75,0.35)' : 'rgba(166,60,36,0.35)';
  badge.style.color = ok ? '#2b7a4b' : '#a63c24';
}

function renderEmployeePhotos(items = []) {
  const grid = $('#employeePhotoGrid');
  const summary = $('#employeePhotosSummary');
  const result = $('#employeePhotoActionResult');
  if (!grid || !summary) return;

  if (!items.length) {
    summary.textContent = 'No uploaded pictures yet';
    grid.innerHTML = '<div class="empty-box">No uploaded face/ReID pictures for this employee yet.</div>';
    if (result && !result.textContent) result.textContent = '';
    return;
  }

  const faceCount = items.filter((x) => x.kind === 'face').length;
  const reidCount = items.filter((x) => x.kind === 'reid').length;
  summary.textContent = `Total ${items.length} | Face ${faceCount} | ReID ${reidCount}`;
  grid.innerHTML = items.map((p) => {
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

  $$('#employeePhotoGrid button[data-photo-delete-id]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const employeeId = Number(state.selectedEmployeeId);
      const photoId = Number(btn.dataset.photoDeleteId);
      if (!employeeId || !photoId) return;
      if (!window.confirm(`Remove photo ${photoId}? Linked embedding will also be removed if available.`)) return;
      btn.disabled = true;
      try {
        const data = await api(`/employees/${employeeId}/photos/${photoId}`, { method: 'DELETE' });
        if (result) result.textContent = JSON.stringify(data, null, 2);
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
  const tbody = $('#employeeHistoryTable tbody');
  if (!tbody) return;
  renderTableRows(tbody, items.map((r) => `
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
  const from = $('#employeeHistoryDateFrom')?.value;
  const to = $('#employeeHistoryDateTo')?.value;
  const q = new URLSearchParams();
  if (from) q.set('date_from', from);
  if (to) q.set('date_to', to);
  const data = await api(`/employees/${employeeId}/attendance?${q.toString()}`);
  renderEmployeeHistory(data.items || []);
}

function fillEmployeeDetailForm(data) {
  const form = $('#employeeDetailForm');
  if (!form) return;
  form.id.value = data.id ?? '';
  form.full_name.value = data.full_name ?? '';
  form.employee_code.value = data.employee_code ?? '';
  form.birth_date.value = data.birth_date ?? '';
  form.job_title.value = data.job_title ?? '';
  form.address.value = data.address ?? '';
  form.status.value = data.status ?? 'active';

  const title = $('#employeeDetailTitle');
  if (title) title.textContent = `Profile: ${data.full_name || ''}`;

  const counts = $('#employeeDetailCounts');
  if (counts) {
    counts.textContent = [
      `ID ${data.id}`,
      `Face emb ${data.face_embeddings_count ?? 0}`,
      `ReID emb ${data.reid_embeddings_count ?? 0}`,
      `Photos ${data.uploaded_images_count ?? 0}`,
    ].join(' | ');
  }

  const hint = $('#employeeDetailHint');
  if (hint) hint.textContent = `Employee ${data.employee_code || ''} selected.`;

  const result = $('#employeeDetailResult');
  if (result) result.textContent = JSON.stringify(data, null, 2);

  if (data.history_default_date_from && $('#employeeHistoryDateFrom')) {
    $('#employeeHistoryDateFrom').value = data.history_default_date_from;
  }
  if (data.history_default_date_to && $('#employeeHistoryDateTo')) {
    $('#employeeHistoryDateTo').value = data.history_default_date_to;
  }
  renderEmployeePhotos(data.uploaded_images || []);
}

async function loadEmployeeDetail(employeeId) {
  const id = Number(employeeId);
  if (!Number.isFinite(id) || id <= 0) throw new Error('Invalid employee ID');
  const data = await api(`/employees/${id}`);
  state.selectedEmployeeId = id;
  if ($('#detailEmployeeIdInput')) $('#detailEmployeeIdInput').value = String(id);
  fillEmployeeDetailForm(data);
  await loadEmployeeHistory();
  const url = `/ui/employee_detail.html?id=${encodeURIComponent(String(id))}`;
  window.history.replaceState({}, '', url);
  setHealth(`Loaded employee ${id}`, true);
  return data;
}

function bind() {
  $('#backToAdminBtn')?.addEventListener('click', () => {
    window.location.href = '/ui/';
  });

  $('#loadEmployeeDetailBtn')?.addEventListener('click', async () => {
    const id = Number($('#detailEmployeeIdInput')?.value);
    if (!id) {
      alert('Enter employee ID');
      return;
    }
    try {
      await loadEmployeeDetail(id);
    } catch (err) {
      setHealth(`Load failed: ${err.message}`, false);
      alert(`Load employee failed: ${err.message}`);
    }
  });

  $('#loadEmployeeHistoryBtn')?.addEventListener('click', () => {
    loadEmployeeHistory().catch((err) => {
      setHealth(`History load failed: ${err.message}`, false);
      alert(`Load history failed: ${err.message}`);
    });
  });

  $('#employeeDetailForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const id = Number(form.id.value);
    if (!id) {
      alert('No employee selected');
      return;
    }
    const payload = normalizeEmployeePayload(form);
    try {
      const data = await api(`/employees/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      $('#employeeDetailResult').textContent = JSON.stringify(data, null, 2);
      await loadEmployeeDetail(id);
      setHealth(`Saved employee ${id}`, true);
    } catch (err) {
      $('#employeeDetailResult').textContent = `ERROR: ${err.message}`;
      setHealth(`Save failed: ${err.message}`, false);
    }
  });

  $('#employeePhotoAddForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const employeeId = Number(state.selectedEmployeeId);
    const result = $('#employeePhotoActionResult');
    if (!employeeId) {
      alert('Load an employee first');
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
      await loadEmployeeDetail(employeeId);
      setHealth(`Uploaded ${kind} images`, true);
    } catch (err) {
      if (result) result.textContent = `ERROR: ${err.message}`;
      setHealth(`Upload failed: ${err.message}`, false);
    } finally {
      if (btn) {
        btn.disabled = false;
        btn.textContent = btn.dataset.originalText || 'Add Images To Employee';
      }
    }
  });
}

async function init() {
  bind();
  const id = getEmployeeIdFromQuery();
  if (!id) {
    setHealth('Ready', true);
    return;
  }
  if ($('#detailEmployeeIdInput')) $('#detailEmployeeIdInput').value = String(id);
  await loadEmployeeDetail(id);
}

init().catch((err) => {
  setHealth(`Init failed: ${err.message}`, false);
});
