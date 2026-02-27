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

function normalizeName(value) {
  return String(value || '').normalize('NFKC').trim().toLowerCase();
}

function inferKindFromFileName(fileName) {
  const lower = String(fileName || '').trim().toLowerCase();
  if (lower.startsWith('face_')) return 'face';
  if (lower.startsWith('re_')) return 'reid';
  return null;
}

function inferFolderNameFromFile(file) {
  const rel = String(file?.webkitRelativePath || '').replaceAll('\\', '/');
  const parts = rel.split('/').filter(Boolean);
  if (parts.length >= 2) return parts[parts.length - 2];
  return '';
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

async function uploadEmployeeImages(employeeId, kind, files) {
  const form = new FormData();
  for (const f of files) form.append('files', f);
  return api(`/employees/${employeeId}/enroll/${kind}`, {
    method: 'POST',
    body: form,
  });
}

function setModeUI(mode) {
  const employeeIdInput = $('#enrollEmployeeIdInput');
  const filesWrap = $('#enrollFilesWrap');
  const filesInput = $('#enrollFilesInput');
  const folderWrap = $('#enrollFolderWrap');
  const folderInput = $('#enrollFolderInput');
  const targetInfo = $('#enrollTargetInfo');

  const folderMode = mode === 'folder';
  if (employeeIdInput) {
    employeeIdInput.disabled = folderMode;
    employeeIdInput.required = !folderMode;
  }
  if (filesWrap) filesWrap.classList.toggle('hidden', folderMode);
  if (filesInput) {
    filesInput.required = !folderMode;
    if (folderMode) filesInput.value = '';
  }
  if (folderWrap) folderWrap.classList.toggle('hidden', !folderMode);
  if (folderInput) {
    folderInput.required = folderMode;
    if (!folderMode) folderInput.value = '';
  }
  if (targetInfo) {
    targetInfo.textContent = folderMode
      ? 'Folder mode: employee ID not required. Folder name must match employee full name.'
      : 'Employee ID and mode will be shown during upload.';
  }
}

async function handleSingleMode({ employeeId, mode, files }) {
  const data = await uploadEmployeeImages(employeeId, mode, files);
  return {
    summary: `Uploaded ${files.length} ${mode.toUpperCase()} image(s) for employee ${employeeId}.`,
    details: data,
  };
}

async function handleFolderMode(files, setProgressText) {
  const grouped = new Map(); // folderName => { face: File[], reid: File[] }
  const skipped = [];

  for (const file of files) {
    const folderName = inferFolderNameFromFile(file);
    const kind = inferKindFromFileName(file.name);
    if (!folderName) {
      skipped.push(`Skipped "${file.name}" (could not infer parent folder).`);
      continue;
    }
    if (!kind) {
      skipped.push(`Skipped "${file.webkitRelativePath || file.name}" (filename must start with face_ or re_).`);
      continue;
    }
    const key = folderName;
    if (!grouped.has(key)) grouped.set(key, { face: [], reid: [] });
    grouped.get(key)[kind].push(file);
  }

  const employees = await api('/employees');
  const nameMap = new Map(); // normalized full_name => employee[]
  for (const emp of employees || []) {
    const key = normalizeName(emp.full_name);
    if (!nameMap.has(key)) nameMap.set(key, []);
    nameMap.get(key).push(emp);
  }

  const lines = [];
  const stats = {
    folders_total: grouped.size,
    employees_matched: 0,
    upload_requests: 0,
    uploaded_files: 0,
    skipped_files: skipped.length,
    failed_groups: 0,
  };
  if (skipped.length) lines.push(...skipped);

  const totalGroups = Array.from(grouped.values()).reduce((acc, g) => acc + (g.face.length ? 1 : 0) + (g.reid.length ? 1 : 0), 0);
  let completedGroups = 0;

  for (const [folderName, pack] of grouped.entries()) {
    const matches = nameMap.get(normalizeName(folderName)) || [];
    if (!matches.length) {
      stats.failed_groups += (pack.face.length ? 1 : 0) + (pack.reid.length ? 1 : 0);
      lines.push(`No employee found for folder "${folderName}".`);
      continue;
    }
    if (matches.length > 1) {
      stats.failed_groups += (pack.face.length ? 1 : 0) + (pack.reid.length ? 1 : 0);
      lines.push(`Ambiguous employee name "${folderName}" (multiple employees have this full name).`);
      continue;
    }
    const emp = matches[0];
    stats.employees_matched += 1;

    for (const kind of ['face', 'reid']) {
      const filesForKind = pack[kind];
      if (!filesForKind.length) continue;
      completedGroups += 1;
      setProgressText(`Uploading ${completedGroups}/${Math.max(totalGroups, 1)}: ${folderName} (${kind.toUpperCase()})...`);
      try {
        const resp = await uploadEmployeeImages(emp.id, kind, filesForKind);
        stats.upload_requests += 1;
        stats.uploaded_files += filesForKind.length;
        lines.push(`OK ${folderName} -> employee_id ${emp.id} (${kind.toUpperCase()} x${filesForKind.length})`);
        if (resp?.failed_files?.length) {
          for (const f of resp.failed_files) {
            lines.push(`  failed file: ${f.file} (${f.reason})`);
          }
        }
      } catch (err) {
        stats.failed_groups += 1;
        lines.push(`FAILED ${folderName} (${kind.toUpperCase()}): ${err.message}`);
      }
    }
  }

  return {
    summary: `Folder batch complete. Uploaded files: ${stats.uploaded_files}, requests: ${stats.upload_requests}, failed groups: ${stats.failed_groups}, skipped files: ${stats.skipped_files}.`,
    details: {
      stats,
      logs: lines,
    },
  };
}

function bind() {
  const prefillId = employeeIdFromQuery();
  if (prefillId && $('#enrollEmployeeIdInput')) {
    $('#enrollEmployeeIdInput').value = String(prefillId);
  }

  const modeSelect = $('#enrollModeSelect');
  setModeUI(modeSelect?.value || 'face');
  modeSelect?.addEventListener('change', () => setModeUI(modeSelect.value || 'face'));

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
    const mode = String(fd.get('mode') || 'face').trim().toLowerCase();
    const employeeId = Number(fd.get('employee_id') || 0);
    const filesInput = $('#enrollFilesInput');
    const folderInput = $('#enrollFolderInput');
    const files = mode === 'folder'
      ? Array.from(folderInput?.files || [])
      : Array.from(filesInput?.files || []);
    const result = $('#enrollResult');
    const submitBtn = $('#enrollSubmitBtn');
    const statusWrap = $('#enrollUploadStatus');
    const statusText = $('#enrollUploadStatusText');
    const targetInfo = $('#enrollTargetInfo');

    const setProgressText = (text) => {
      if (statusWrap && statusText) {
        statusWrap.classList.remove('hidden');
        statusWrap.classList.add('active');
        statusText.textContent = text;
      }
      if (targetInfo) targetInfo.textContent = text;
    };

    if (mode !== 'folder' && !employeeId) {
      setStatus('Missing employee ID', false);
      if (result) result.textContent = 'Employee ID is required for Face/ReID modes.';
      return;
    }
    if (!files.length) {
      setStatus('No files selected', false);
      if (result) result.textContent = mode === 'folder' ? 'Select a folder with image files.' : 'Select at least one image.';
      return;
    }

    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.dataset.originalText = submitBtn.dataset.originalText || submitBtn.textContent || 'Upload Enrollment Images';
      submitBtn.textContent = 'Uploading...';
    }
    setProgressText(mode === 'folder'
      ? `Processing folder batch (${files.length} file(s))...`
      : `Uploading ${files.length} image(s) for Employee ID ${employeeId} (${mode.toUpperCase()})...`);
    if (result) result.textContent = 'Uploading...';
    setStatus('Uploading', true);

    try {
      const outcome = mode === 'folder'
        ? await handleFolderMode(files, setProgressText)
        : await handleSingleMode({ employeeId, mode, files });

      if (result) {
        result.textContent = `${outcome.summary}\n${JSON.stringify(outcome.details, null, 2)}`;
      }
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
