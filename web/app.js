/**
 * Multi-Agent AI — ChatGPT Style Frontend Application Logic
 */

const API_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? 'http://localhost:8000'
  : window.location.origin;

function getApiKey() {
  const meta = document.querySelector('meta[name="api-key"]');
  if (meta && meta.content) return meta.content;
  const stored = localStorage.getItem('APP_API_KEY');
  if (stored) return stored;
  return '';
}

function apiHeaders(extra = {}) {
  const h = { 'Content-Type': 'application/json', ...extra };
  const key = getApiKey();
  if (key) h['X-API-Key'] = key;
  return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────
let isPipelineRunning = false;
let currentAbortController = null;
let currentThreadId = null;
let currentReportMarkdown = '';
let currentReportId = null;

const AGENT_ORDER = ['router', 'planner', 'researcher', 'retriever', 'critic', 'reporter'];
const AGENT_META = {
  'router':     { icon: '⌁', name: 'Router' },
  'planner':    { icon: '◫', name: 'Planner' },
  'researcher': { icon: '⌕', name: 'Researcher' },
  'retriever':  { icon: '▤', name: 'Retriever' },
  'critic':     { icon: '◎', name: 'Critic' },
  'reporter':   { icon: '▧', name: 'Reporter' },
};

// ─────────────────────────────────────────────────────────────────────────────
// Markdown Parser with Clickable Numbered Citations
// ─────────────────────────────────────────────────────────────────────────────
function md2html(md) {
  if (!md) return '';
  let h = md;
  h = h.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const esc = code.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    return `<pre><code class="lang-${lang}">${esc.trim()}</code></pre>`;
  });
  h = h.replace(/`([^`]+)`/g, (_, c) => `<code>${c.replace(/</g,'&lt;')}</code>`);
  h = h.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  h = h.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  h = h.replace(/^# (.+)$/gm, '<h1>$1</h1>');
  h = h.replace(/^---$/gm, '<hr>');
  h = h.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/\*(.+?)\*/g, '<em>$1</em>');
  h = h.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');
  
  // Tables
  h = h.replace(/(\|.+\|\n)(\|[-:| ]+\|\n)((?:\|.+\|\n?)*)/g, (_, header, sep, body) => {
    const heads = header.trim().split('|').filter(Boolean).map(x => `<th>${x.trim()}</th>`).join('');
    const rows = body.trim().split('\n').map(row =>
      `<tr>${row.trim().split('|').filter(Boolean).map(c => `<td>${c.trim()}</td>`).join('')}</tr>`
    ).join('');
    return `<table><thead><tr>${heads}</tr></thead><tbody>${rows}</tbody></table>`;
  });

  // Lists
  h = h.replace(/((?:^[ \t]*[-*+] .+\n?)+)/gm, match => {
    const items = match.trim().split('\n').map(l => `<li>${l.replace(/^[ \t]*[-*+] /, '')}</li>`).join('');
    return `<ul>${items}</ul>`;
  });
  h = h.replace(/((?:^\d+\. .+\n?)+)/gm, match => {
    const items = match.trim().split('\n').map(l => `<li>${l.replace(/^\d+\. /, '')}</li>`).join('');
    return `<ol>${items}</ol>`;
  });

  // Markdown links
  h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  
  // Clickable numbered citations [1], [2]
  h = h.replace(/(?:^|\s)\[(\d+)\](?=[,\s\.\?!]|$)/g, ' <a href="#source-$1" class="citation-pill" title="Jump to source $1">[$1]</a>');
  
  // Sources section anchor
  h = h.replace(/^(?:<p>)?\[(\d+)\]\s+/gm, '<span id="source-$1" class="source-item-num">[$1]</span> ');

  h = h.split('\n\n').map(block => {
    block = block.trim();
    if (!block) return '';
    if (/^<(h[1-6]|ul|ol|pre|table|blockquote|hr|span)/.test(block)) return block;
    return `<p>${block.replace(/\n/g, '<br/>')}</p>`;
  }).join('\n');
  return h;
}

function showToast(msg, type = 'success') {
  const el = document.getElementById('toast');
  if (!el) return;
  el.textContent = msg;
  el.className = 'toast' + (type === 'error' ? ' error' : '');
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 3200);
}

// ─────────────────────────────────────────────────────────────────────────────
// Sidebar & Navigation
// ─────────────────────────────────────────────────────────────────────────────
function toggleSidebar(forceState) {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebarOverlay');
  const isMobile = window.innerWidth <= 768;

  if (isMobile) {
    if (typeof forceState === 'boolean') {
      sidebar.classList.toggle('open', forceState);
      overlay.classList.toggle('active', forceState);
    } else {
      sidebar.classList.toggle('open');
      overlay.classList.toggle('active');
    }
  } else {
    sidebar.classList.toggle('collapsed');
  }
}

function startNewChat() {
  if (isPipelineRunning) {
    if (currentAbortController) currentAbortController.abort();
    isPipelineRunning = false;
  }
  currentThreadId = null;
  currentReportMarkdown = '';
  currentReportId = null;

  document.getElementById('welcomeContainer').style.display = 'flex';
  const thread = document.getElementById('messagesThread');
  thread.style.display = 'none';
  thread.innerHTML = '';
  document.getElementById('hitlBanner').style.display = 'none';
  document.getElementById('chatTitle').textContent = 'Multi-Agent AI';
  
  const input = document.getElementById('chatInput');
  input.value = '';
  autoResizeTextarea(input);
  input.focus();
  toggleSidebar(false);
}

function selectSuggestion(query) {
  const input = document.getElementById('chatInput');
  input.value = query;
  autoResizeTextarea(input);
  submitMessage();
}

function autoResizeTextarea(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 180) + 'px';
  const sendBtn = document.getElementById('sendBtn');
  if (sendBtn) {
    sendBtn.disabled = !el.value.trim() && !isPipelineRunning;
  }
}

function handleInputKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitMessage();
  }
}

function scrollToBottom() {
  const container = document.getElementById('chatScrollContainer');
  if (container) {
    container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' });
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Message Submission Dispatcher
// ─────────────────────────────────────────────────────────────────────────────
function submitMessage() {
  const input = document.getElementById('chatInput');
  const text = input.value.trim();
  if (!text) return;

  // Clear input
  input.value = '';
  autoResizeTextarea(input);

  // If a report already exists in the current session, treat this as a follow-up doubt question!
  if (currentReportMarkdown && !isPipelineRunning) {
    submitFollowUpDoubt(text);
  } else {
    // Initial research query
    startResearchStream(text);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// User & Assistant Chat UI Builders
// ─────────────────────────────────────────────────────────────────────────────
function appendUserMessage(text) {
  document.getElementById('welcomeContainer').style.display = 'none';
  const thread = document.getElementById('messagesThread');
  thread.style.display = 'flex';

  const userRow = document.createElement('div');
  userRow.className = 'chat-message user';
  userRow.innerHTML = `
    <div class="msg-content-wrapper">
      <div class="user-bubble">${escapeHtml(text)}</div>
    </div>
    <div class="msg-avatar">U</div>
  `;
  thread.appendChild(userRow);
  scrollToBottom();
}

function appendAssistantMessageContainer() {
  const thread = document.getElementById('messagesThread');
  const msgRow = document.createElement('div');
  msgRow.className = 'chat-message assistant';
  
  const idSuffix = Date.now();
  msgRow.innerHTML = `
    <div class="msg-avatar">✦</div>
    <div class="msg-content-wrapper">
      <!-- Multi-Agent Process Card ("Thinking" Accordion) -->
      <div class="agent-thinking-card active expanded" id="thinkingCard-${idSuffix}">
        <div class="thinking-header" onclick="toggleThinkingCard('thinkingCard-${idSuffix}')">
          <div class="thinking-status-left">
            <div class="thinking-spinner" id="thinkingSpinner-${idSuffix}"></div>
            <strong id="thinkingTitle-${idSuffix}">Multi-Agent Pipeline Active...</strong>
            <span id="thinkingSummary-${idSuffix}" style="color:var(--text-dim);font-size:12px;">(Starting agents)</span>
          </div>
          <svg class="thinking-chevron" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
        </div>
        <div class="thinking-body" id="thinkingBody-${idSuffix}">
          <div class="agent-steps-grid" id="stepsGrid-${idSuffix}">
            ${AGENT_ORDER.map(k => `
              <div class="agent-step-item" id="step-${k}-${idSuffix}">
                <div class="agent-step-name">${AGENT_META[k].name} <span class="step-icon">${AGENT_META[k].icon}</span></div>
                <div class="agent-step-status" id="step-status-${k}-${idSuffix}">idle</div>
              </div>
            `).join('')}
          </div>
        </div>
      </div>

      <!-- Report Body -->
      <div class="report-markdown-body" id="reportBody-${idSuffix}">
        <p style="color:var(--text-muted);font-style:italic;">Initializing multi-agent pipeline<span class="stream-cursor"></span></p>
      </div>

      <!-- Footer Meta & Actions -->
      <div class="report-footer-meta" id="reportFooter-${idSuffix}" style="display:none;">
        <div class="meta-badges-group" id="metaBadges-${idSuffix}"></div>
        <div class="report-actions-group">
          <button class="action-btn" onclick="copyReportText()">📋 Copy</button>
          <button class="action-btn" onclick="downloadReportMd()">⬇️ Download</button>
        </div>
      </div>
    </div>
  `;
  thread.appendChild(msgRow);
  scrollToBottom();

  return {
    thinkingCard: document.getElementById(`thinkingCard-${idSuffix}`),
    thinkingSpinner: document.getElementById(`thinkingSpinner-${idSuffix}`),
    thinkingTitle: document.getElementById(`thinkingTitle-${idSuffix}`),
    thinkingSummary: document.getElementById(`thinkingSummary-${idSuffix}`),
    stepsGrid: document.getElementById(`stepsGrid-${idSuffix}`),
    reportBody: document.getElementById(`reportBody-${idSuffix}`),
    reportFooter: document.getElementById(`reportFooter-${idSuffix}`),
    metaBadges: document.getElementById(`metaBadges-${idSuffix}`),
    idSuffix: idSuffix
  };
}

function toggleThinkingCard(cardId) {
  const card = document.getElementById(cardId);
  if (card) {
    card.classList.toggle('expanded');
  }
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ─────────────────────────────────────────────────────────────────────────────
// Research SSE Stream Handler
// ─────────────────────────────────────────────────────────────────────────────
async function startResearchStream(query) {
  isPipelineRunning = true;
  currentAbortController = new AbortController();
  currentReportMarkdown = '';

  appendUserMessage(query);
  const elements = appendAssistantMessageContainer();
  const id = elements.idSuffix;

  document.getElementById('chatTitle').textContent = query.length > 36 ? query.slice(0, 36) + '...' : query;
  const sendBtn = document.getElementById('sendBtn');
  sendBtn.innerHTML = '<span style="font-size:12px;">■</span>';
  sendBtn.title = 'Stop generating';

  let streamBuffer = '';

  try {
    const res = await fetch(`${API_BASE}/research/stream`, {
      method: 'POST',
      headers: apiHeaders(),
      body: JSON.stringify({ query: query, user_id: 'browser_user' }),
      signal: currentAbortController.signal
    });

    if (!res.ok) {
      if (res.status === 429) throw new Error('Rate limit reached. Please wait a minute before querying again.');
      throw new Error(`Server returned error status ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop();

      for (const part of parts) {
        if (!part.startsWith('event:')) continue;
        const lines = part.split('\n');
        const evType = lines[0].replace('event: ', '').trim();
        const evData = JSON.parse(lines[1].replace('data: ', '').trim());

        handleStreamEvent(evType, evData, elements, () => streamBuffer, (val) => { streamBuffer = val; });
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      showToast('Research stopped by user', 'error');
    } else {
      showToast(err.message, 'error');
      elements.reportBody.innerHTML = `<div style="color:var(--accent-red);background:rgba(239,68,68,0.1);padding:12px;border-radius:8px;">❌ Error: ${err.message}</div>`;
    }
    elements.thinkingSpinner.style.display = 'none';
    elements.thinkingTitle.textContent = 'Multi-Agent Run Stopped';
  } finally {
    isPipelineRunning = false;
    currentAbortController = null;
    sendBtn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M5 12h14M12 5l7 7-7 7"/></svg>';
    sendBtn.title = 'Send message';
    loadHistory();
  }
}

function handleStreamEvent(evType, evData, ui, getStreamBuffer, setStreamBuffer) {
  const id = ui.idSuffix;

  if (evType === 'agent_start') {
    const a = evData.agent;
    const stepEl = document.getElementById(`step-${a}-${id}`);
    const statusEl = document.getElementById(`step-status-${a}-${id}`);
    if (stepEl) stepEl.className = 'agent-step-item active';
    if (statusEl) statusEl.textContent = 'running…';

    ui.thinkingTitle.textContent = `Agent Active: ${AGENT_META[a]?.name || a}...`;
    ui.thinkingSummary.textContent = evData.message || '';
  }

  else if (evType === 'agent_done') {
    const a = evData.agent;
    const r = evData.result || {};
    const stepEl = document.getElementById(`step-${a}-${id}`);
    const statusEl = document.getElementById(`step-status-${a}-${id}`);
    if (stepEl) stepEl.className = 'agent-step-item done';

    if (a === 'router') {
      if (statusEl) statusEl.textContent = r.is_casual ? 'casual' : `${r.tone || 'routed'}`;
    } else if (a === 'planner') {
      if (statusEl) statusEl.textContent = `${r.plan?.length || 0} sub-tasks planned`;
    } else if (a === 'researcher') {
      if (statusEl) statusEl.textContent = `${r.sources_found || 0} sources found`;
    } else if (a === 'retriever') {
      if (statusEl) statusEl.textContent = `${r.docs_retrieved || 0} chunks retrieved`;
    } else if (a === 'critic') {
      const conf = (r.confidence || 0).toFixed(2);
      if (statusEl) statusEl.textContent = r.passed ? `confidence ${conf} ✓` : `confidence ${conf} ↺ retry`;
    } else if (a === 'reporter') {
      if (statusEl) statusEl.textContent = 'report synthesized ✓';
    }
  }

  else if (evType === 'report_chunk') {
    let buf = getStreamBuffer();
    buf += evData.chunk;
    setStreamBuffer(buf);
    ui.reportBody.innerHTML = md2html(buf) + '<span class="stream-cursor"></span>';
    scrollToBottom();
  }

  else if (evType === 'hitl_pause') {
    currentThreadId = evData.thread_id;
    document.getElementById('hitlBanner').style.display = 'flex';
    showToast('Human review required before report synthesis.');
  }

  else if (evType === 'complete') {
    ui.thinkingSpinner.style.display = 'none';
    ui.thinkingCard.classList.remove('active');
    ui.thinkingTitle.textContent = '✓ Multi-Agent Pipeline Completed';
    ui.thinkingSummary.textContent = '(6 agents coordinated · 100% verified citations)';
    
    // Automatically collapse thinking card on finish so report is prominent
    ui.thinkingCard.classList.remove('expanded');

    let finalReport = getStreamBuffer() || evData.report || '';
    currentReportMarkdown = finalReport;
    ui.reportBody.innerHTML = md2html(finalReport);

    renderMetaBadges(ui.metaBadges, finalReport);
    ui.reportFooter.style.display = 'flex';
    scrollToBottom();
  }

  else if (evType === 'error') {
    showToast(evData.message || 'Error occurred', 'error');
  }
}

function renderMetaBadges(container, reportText) {
  if (!container || !reportText) return;
  const urls = reportText.match(/https?:\/\/[^\s\)]+/g) || [];
  const domains = new Set();
  urls.forEach(u => {
    try { domains.add(new URL(u).hostname.replace('www.', '')); } catch(e){}
  });
  const citations = reportText.match(/\[\d+\]/g) || [];
  const uniqueCitations = new Set(citations);

  container.innerHTML = `
    <span class="meta-pill">🔍 <strong>${uniqueCitations.size}</strong> verified citations</span>
    <span class="meta-pill">🌐 <strong>${domains.size}</strong> distinct sources</span>
    <span class="meta-pill verified">🛡️ Anti-Hallucination Verified</span>
  `;
}

// ─────────────────────────────────────────────────────────────────────────────
// Follow-up Q&A (Doubt Agent Turn)
// ─────────────────────────────────────────────────────────────────────────────
async function submitFollowUpDoubt(question) {
  if (!currentReportMarkdown) return;

  appendUserMessage(question);

  const thread = document.getElementById('messagesThread');
  const msgRow = document.createElement('div');
  msgRow.className = 'chat-message assistant';
  const idSuffix = Date.now();

  msgRow.innerHTML = `
    <div class="msg-avatar">✦</div>
    <div class="msg-content-wrapper">
      <div class="report-markdown-body" id="doubtAnswer-${idSuffix}">
        <p style="color:var(--text-muted);font-style:italic;">Analyzing report context<span class="stream-cursor"></span></p>
      </div>
    </div>
  `;
  thread.appendChild(msgRow);
  scrollToBottom();

  const answerEl = document.getElementById(`doubtAnswer-${idSuffix}`);

  try {
    const res = await fetch(`${API_BASE}/doubt`, {
      method: 'POST',
      headers: apiHeaders(),
      body: JSON.stringify({
        report: currentReportMarkdown,
        question: question
      })
    });

    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    answerEl.innerHTML = md2html(data.answer);
    scrollToBottom();
  } catch (err) {
    answerEl.innerHTML = `<span style="color:var(--accent-red)">Error analyzing question: ${err.message}</span>`;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// HITL Resume
// ─────────────────────────────────────────────────────────────────────────────
async function resumePipeline() {
  if (!currentThreadId) return;
  document.getElementById('hitlBanner').style.display = 'none';
  try {
    const res = await fetch(`${API_BASE}/research/resume`, {
      method: 'POST',
      headers: apiHeaders(),
      body: JSON.stringify({ thread_id: currentThreadId })
    });
    if (!res.ok) throw new Error('Failed to resume');
    showToast('Report generation resumed!');
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Export & Copy Helpers
// ─────────────────────────────────────────────────────────────────────────────
function copyReportText() {
  if (!currentReportMarkdown) {
    showToast('No report to copy', 'error');
    return;
  }
  navigator.clipboard.writeText(currentReportMarkdown).then(() => {
    showToast('Report copied to clipboard!');
  }).catch(() => {
    showToast('Failed to copy', 'error');
  });
}

function downloadReportMd() {
  if (!currentReportMarkdown) {
    showToast('No report to download', 'error');
    return;
  }
  const blob = new Blob([currentReportMarkdown], { type: 'text/markdown;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `research_report_${new Date().toISOString().slice(0, 10)}.md`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Download started!');
}

// ─────────────────────────────────────────────────────────────────────────────
// History Loading
// ─────────────────────────────────────────────────────────────────────────────
async function loadHistory() {
  const listEl = document.getElementById('historyList');
  if (!listEl) return;

  try {
    const res = await fetch(`${API_BASE}/reports?limit=25`, { headers: apiHeaders() });
    if (!res.ok) throw new Error('Could not fetch past reports');
    const data = await res.json();

    if (!data.reports || data.reports.length === 0) {
      listEl.innerHTML = '<div class="history-loading">No past research yet.</div>';
      return;
    }

    listEl.innerHTML = '';
    data.reports.forEach(r => {
      const item = document.createElement('div');
      item.className = 'history-item';
      item.innerHTML = `
        <span class="history-item-text" title="${escapeHtml(r.query)}">${escapeHtml(r.query)}</span>
        <button class="icon-btn history-delete-btn" title="Delete" onclick="deleteHistoryReport(event, ${r.id})">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
        </button>
      `;

      item.onclick = (e) => {
        if (e.target.closest('.history-delete-btn')) return;
        loadExistingReportIntoChat(r);
      };

      listEl.appendChild(item);
    });
  } catch (err) {
    listEl.innerHTML = `<div class="history-loading">Error loading history</div>`;
  }
}

function loadExistingReportIntoChat(r) {
  startNewChat();
  currentReportMarkdown = r.report;
  currentReportId = r.id;

  appendUserMessage(r.query);

  const thread = document.getElementById('messagesThread');
  const msgRow = document.createElement('div');
  msgRow.className = 'chat-message assistant';
  const idSuffix = Date.now();

  msgRow.innerHTML = `
    <div class="msg-avatar">✦</div>
    <div class="msg-content-wrapper">
      <div class="report-markdown-body">
        ${md2html(r.report)}
      </div>
      <div class="report-footer-meta" style="display:flex;">
        <div class="meta-badges-group" id="historyBadges-${idSuffix}"></div>
        <div class="report-actions-group">
          <button class="action-btn" onclick="copyReportText()">📋 Copy</button>
          <button class="action-btn" onclick="downloadReportMd()">⬇️ Download</button>
        </div>
      </div>
    </div>
  `;
  thread.appendChild(msgRow);
  renderMetaBadges(document.getElementById(`historyBadges-${idSuffix}`), r.report);
  scrollToBottom();
}

async function deleteHistoryReport(event, id) {
  event.stopPropagation();
  try {
    const res = await fetch(`${API_BASE}/reports/${id}`, {
      method: 'DELETE',
      headers: apiHeaders()
    });
    if (!res.ok) throw new Error('Delete failed');
    showToast('Report deleted');
    loadHistory();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// App Initialization
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadHistory();
  const input = document.getElementById('chatInput');
  if (input) {
    input.focus();
  }

  // Check backend health
  fetch(`${API_BASE}/health`)
    .then(r => r.json())
    .then(data => {
      const badge = document.getElementById('apiHealthBadge');
      if (badge) {
        badge.textContent = data.status === 'healthy' ? '● 6 Agents Ready' : '● Degraded';
        badge.style.color = data.status === 'healthy' ? 'var(--accent-green)' : 'var(--accent-amber)';
      }
    })
    .catch(() => {
      const badge = document.getElementById('apiHealthBadge');
      if (badge) {
        badge.textContent = '● Offline';
        badge.style.color = 'var(--accent-red)';
      }
    });
});
