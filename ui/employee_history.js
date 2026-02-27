const $ = (sel) => document.querySelector(sel);

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
  return res.json();
}

function renderTableRows(rowsHtml) {
  const tbody = $('#historyTable tbody');
  if (!tbody) return;
  tbody.innerHTML = rowsHtml || '<tr><td colspan="99">No rows</td></tr>';
}

function eventImageLink(r) {
  const url = r.image_url || (r.image_path ? `file:///${String(r.image_path).replaceAll('\\', '/')}` : '');
  if (!url) return '';
  return `<a class="image-link" target="_blank" href="${escapeHtml(url)}">open</a>`;
}

function buildHistoryUrl({ employeeId, dateFrom, dateTo }) {
  const q = new URLSearchParams();
  q.set('employee_id', String(employeeId));
  if (dateFrom) q.set('date_from', String(dateFrom));
  if (dateTo) q.set('date_to', String(dateTo));
  return `/ui/employee_history.html?${q.toString()}`;
}

function getCurrentInputPayload() {
  const employeeId = Number($('#historyEmployeeId')?.value || 0);
  const dateFrom = String($('#historyDateFrom')?.value || '').trim();
  const dateTo = String($('#historyDateTo')?.value || '').trim();
  return { employeeId, dateFrom, dateTo };
}

function updateStatus(text) {
  const el = $('#historyStatus');
  if (el) el.textContent = text;
}

function renderRows(items = []) {
  renderTableRows(
    (items || [])
      .map(
        (r) => `
    <tr>
      <td>${r.id}</td>
      <td>${escapeHtml(asIsoOrEmpty(r.ts))}</td>
      <td>${escapeHtml(r.method)}</td>
      <td>${Number(r.confidence || 0).toFixed(3)}</td>
      <td>${escapeHtml(r.camera_id || '')}</td>
      <td>${escapeHtml(r.track_uid || '')}</td>
      <td>${eventImageLink(r)}</td>
    </tr>`
      )
      .join('')
  );
  const count = $('#historyCount');
  if (count) count.textContent = `${items.length} row(s)`;
}

async function loadHistoryFromQuery() {
  const q = new URLSearchParams(window.location.search);
  const employeeId = Number(q.get('employee_id') || 0);
  const dateFrom = (q.get('date_from') || '').trim();
  const dateTo = (q.get('date_to') || '').trim();

  if ($('#historyEmployeeId')) $('#historyEmployeeId').value = employeeId ? String(employeeId) : '';
  if ($('#historyDateFrom')) $('#historyDateFrom').value = dateFrom;
  if ($('#historyDateTo')) $('#historyDateTo').value = dateTo;

  if (!employeeId) {
    updateStatus('Employee ID is required. Open this page from Employees > History.');
    renderRows([]);
    return;
  }

  updateStatus(`Loading employee ${employeeId}...`);

  const emp = await api(`/employees/${employeeId}`);
  const histQ = new URLSearchParams();
  if (dateFrom) histQ.set('date_from', dateFrom);
  if (dateTo) histQ.set('date_to', dateTo);
  const hist = await api(`/employees/${employeeId}/attendance${histQ.toString() ? `?${histQ.toString()}` : ''}`);

  if (!dateFrom && hist?.date_from && $('#historyDateFrom')) $('#historyDateFrom').value = hist.date_from;
  if (!dateTo && hist?.date_to && $('#historyDateTo')) $('#historyDateTo').value = hist.date_to;

  const title = $('#historyEmployeeTitle');
  if (title) title.textContent = `${emp.full_name || 'Employee'} (ID ${emp.id})`;

  const meta = $('#historyEmployeeMeta');
  if (meta) {
    meta.textContent = `Code ${emp.employee_code || '-'} | Status ${emp.status || '-'} | Face ${emp.face_embeddings_count ?? 0} | ReID ${emp.reid_embeddings_count ?? 0}`;
  }

  const rangeTitle = $('#historyRangeTitle');
  if (rangeTitle) {
    rangeTitle.textContent = `History Result (${hist.date_from || '-'} to ${hist.date_to || '-'})`;
  }

  renderRows(hist.items || []);
  updateStatus(`Loaded ${hist.items?.length || 0} row(s) for employee ${employeeId}.`);
}

function bindActions() {
  const form = $('#historyFilterForm');
  const openNewTabBtn = $('#openNewTabBtn');
  const loadHereBtn = $('#loadHereBtn');

  const openNewTab = () => {
    const { employeeId, dateFrom, dateTo } = getCurrentInputPayload();
    if (!employeeId) {
      alert('Employee ID is required');
      return;
    }
    const url = buildHistoryUrl({ employeeId, dateFrom, dateTo });
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  const loadHere = () => {
    const { employeeId, dateFrom, dateTo } = getCurrentInputPayload();
    if (!employeeId) {
      alert('Employee ID is required');
      return;
    }
    const url = buildHistoryUrl({ employeeId, dateFrom, dateTo });
    window.location.href = url;
  };

  if (form) {
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      openNewTab();
    });
  }
  if (openNewTabBtn) openNewTabBtn.addEventListener('click', openNewTab);
  if (loadHereBtn) loadHereBtn.addEventListener('click', loadHere);
}

async function init() {
  bindActions();
  try {
    await loadHistoryFromQuery();
  } catch (err) {
    updateStatus(`ERROR: ${err.message}`);
    renderRows([]);
  }
}

init();
