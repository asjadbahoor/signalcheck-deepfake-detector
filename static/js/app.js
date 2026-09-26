(() => {
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const analysis = document.getElementById('analysis');
  const previewImg = document.getElementById('preview-img');
  const scanline = document.getElementById('scanline');
  const report = document.getElementById('report');
  const verdictLabel = document.getElementById('verdict-label');
  const verdictConfidence = document.getElementById('verdict-confidence');
  const reportNote = document.getElementById('report-note');
  const probbars = document.getElementById('probbars');
  const feedbackActions = document.getElementById('feedback-actions');
  const feedbackCorrection = document.getElementById('feedback-correction');
  const feedbackDone = document.getElementById('feedback-done');
  const btnCorrect = document.getElementById('btn-correct');
  const btnIncorrect = document.getElementById('btn-incorrect');
  const btnReset = document.getElementById('btn-reset');
  const stageError = document.getElementById('stage-error');

  const statTotal = document.getElementById('stat-total');
  const statCorrections = document.getElementById('stat-corrections');
  const progressFill = document.getElementById('progress-fill');
  const progressLabel = document.getElementById('progress-label');
  const lastRetrain = document.getElementById('last-retrain');
  const btnRetrain = document.getElementById('btn-retrain');

  const noticeToggle = document.getElementById('notice-toggle');
  const noticeBody = document.getElementById('notice-body');

  let currentToken = null;
  let currentResult = null;

  // -- Helpers ------------------------------------------------------------

  function showError(message) {
    stageError.textContent = message;
    stageError.hidden = false;
  }

  function clearError() {
    stageError.hidden = true;
    stageError.textContent = '';
  }

  function resetStage() {
    analysis.hidden = true;
    dropzone.hidden = false;
    report.hidden = true;
    scanline.hidden = true;
    feedbackDone.hidden = true;
    feedbackCorrection.hidden = true;
    feedbackActions.hidden = false;
    fileInput.value = '';
    currentToken = null;
    currentResult = null;
    clearError();
  }

  function renderResult(result) {
    const { label, confidence, probabilities, low_confidence } = result;

    report.dataset.verdict = label;
    verdictLabel.textContent = label === 'fake' ? 'Likely manipulated' : 'Likely real';
    verdictConfidence.textContent = `${(confidence * 100).toFixed(1)}% confidence`;

    reportNote.hidden = true;
    if (label === 'fake' && confidence >= 0.85) {
      reportNote.dataset.tone = 'warning';
      reportNote.textContent = 'High-confidence detection. If this was shared as genuine, treat it as a strong signal of misinformation or fraud, and verify through an independent source before acting on it.';
      reportNote.hidden = false;
    } else if (low_confidence) {
      reportNote.dataset.tone = 'info';
      reportNote.textContent = 'This prediction has relatively low confidence. Feedback on cases like this is especially valuable for improving the model.';
      reportNote.hidden = false;
    }

    probbars.innerHTML = '';
    Object.entries(probabilities).forEach(([cls, prob]) => {
      const row = document.createElement('div');
      row.className = 'probbar';
      row.dataset.cls = cls;
      row.innerHTML = `
        <div class="probbar__label"><span>${cls}</span><span>${(prob * 100).toFixed(1)}%</span></div>
        <div class="probbar__track"><div class="probbar__fill" style="width:${(prob * 100).toFixed(1)}%"></div></div>
      `;
      probbars.appendChild(row);
    });

    report.hidden = false;
  }

  async function refreshStats() {
    try {
      const res = await fetch('/api/stats');
      if (!res.ok) return;
      const stats = await res.json();
      statTotal.textContent = stats.total_feedback;
      statCorrections.textContent = stats.corrections_received;

      const pct = Math.min(100, (stats.pending_feedback / stats.retrain_threshold) * 100);
      progressFill.style.width = `${pct}%`;
      progressLabel.textContent = `${stats.pending_feedback} / ${stats.retrain_threshold} toward next retrain`;

      if (stats.last_retrain) {
        const [ts, accepted, newAcc] = stats.last_retrain;
        const status = accepted ? 'accepted' : 'rejected (safety check failed)';
        const accStr = newAcc !== null && newAcc !== undefined ? `${(newAcc * 100).toFixed(1)}%` : 'n/a';
        lastRetrain.textContent = `Last retrain: ${ts.slice(0, 19)} UTC — ${status} (val acc: ${accStr})`;
      }
    } catch (e) {
      // Stats are a nice-to-have; fail silently.
    }
  }

  // -- Upload flow ----------------------------------------------------------

  async function analyzeFile(file) {
    clearError();
    if (!file.type.startsWith('image/')) {
      showError('Please choose a JPG or PNG image.');
      return;
    }

    previewImg.src = URL.createObjectURL(file);
    dropzone.hidden = true;
    analysis.hidden = false;
    report.hidden = true;
    scanline.hidden = false;

    const formData = new FormData();
    formData.append('image', file);

    try {
      const res = await fetch('/api/predict', { method: 'POST', body: formData });
      const data = await res.json();
      scanline.hidden = true;

      if (!res.ok) {
        showError(data.error || 'Analysis failed. Please try again.');
        return;
      }

      currentToken = data.token;
      currentResult = data;
      renderResult(data);
    } catch (e) {
      scanline.hidden = true;
      showError('Could not reach the server. Is the Flask app running?');
    }
  }

  dropzone.addEventListener('click', () => fileInput.click());
  dropzone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) analyzeFile(fileInput.files[0]);
  });

  ['dragenter', 'dragover'].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add('is-dragover'); });
  });
  ['dragleave', 'drop'].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove('is-dragover'); });
  });
  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files[0];
    if (file) analyzeFile(file);
  });

  btnReset.addEventListener('click', resetStage);

  // -- Feedback ---------------------------------------------------------

  async function submitFeedback(correctLabel) {
    if (!currentToken || !currentResult) return;
    try {
      const res = await fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token: currentToken,
          predicted_label: currentResult.label,
          confidence: currentResult.confidence,
          correct_label: correctLabel,
        }),
      });
      const data = await res.json();

      feedbackActions.hidden = true;
      feedbackCorrection.hidden = true;
      feedbackDone.hidden = false;

      if (data.retrain && data.retrain.accepted) {
        feedbackDone.textContent = 'Recorded — thank you. Enough feedback came in to update the model, and the update passed its safety check.';
      } else if (data.retrain && data.retrain.accepted === false && !data.retrain.skipped) {
        feedbackDone.textContent = 'Recorded — thank you. An update was attempted but rejected by the accuracy safety check, so the model is unchanged.';
      }

      refreshStats();
    } catch (e) {
      showError('Could not submit feedback. Is the Flask app running?');
    }
  }

  btnCorrect.addEventListener('click', () => submitFeedback(currentResult.label));
  btnIncorrect.addEventListener('click', () => {
    feedbackActions.hidden = true;
    feedbackCorrection.hidden = false;
  });
  feedbackCorrection.querySelectorAll('button[data-label]').forEach((btn) => {
    btn.addEventListener('click', () => submitFeedback(btn.dataset.label));
  });

  // -- Ledger actions -----------------------------------------------------

  btnRetrain.addEventListener('click', async () => {
    btnRetrain.disabled = true;
    btnRetrain.textContent = 'Retraining…';
    try {
      const res = await fetch('/api/retrain', { method: 'POST' });
      const data = await res.json();
      if (data.skipped) {
        lastRetrain.textContent = data.reason || 'Nothing to retrain on yet.';
      } else {
        lastRetrain.textContent = data.accepted
          ? `Model updated just now — ${data.reason}`
          : `Update rejected — ${data.reason}`;
      }
    } catch (e) {
      lastRetrain.textContent = 'Could not reach the server.';
    } finally {
      btnRetrain.disabled = false;
      btnRetrain.textContent = 'Retrain now';
      refreshStats();
    }
  });

  noticeToggle.addEventListener('click', () => {
    const expanded = noticeToggle.getAttribute('aria-expanded') === 'true';
    noticeToggle.setAttribute('aria-expanded', String(!expanded));
    noticeBody.hidden = expanded;
  });

  refreshStats();
})();
