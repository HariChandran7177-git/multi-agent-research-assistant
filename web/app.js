const API_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? 'http://localhost:8000'
  : window.location.origin;

const AGENT_ORDER = ['router', 'planner', 'researcher', 'retriever', 'critic', 'reporter'];

const AGENT_META = {
  'router':     { icon: '⌁', name: 'Router',     tech: 'Groq · Model' },
  'planner':    { icon: '◫', name: 'Planner',    tech: 'LangGraph' },
  'researcher': { icon: '⌕', name: 'Researcher', tech: 'Tavily API' },
  'retriever':  { icon: '▤', name: 'Retriever',  tech: 'Qdrant' },
  'critic':     { icon: '◎', name: 'Critic',     tech: 'Hybrid Scoring' },
  'reporter':   { icon: '▧', name: 'Reporter',   tech: 'Markdown' },
};

// Markdown Parser
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
  h = h.replace(/(\|.+\|\n)(\|[-:| ]+\|\n)((?:\|.+\|\n?)*)/g, (_, header, sep, body) => {
    const heads = header.trim().split('|').filter(Boolean).map(x => `<th>${x.trim()}</th>`).join('');
    const rows = body.trim().split('\n').map(row =>
      `<tr>${row.trim().split('|').filter(Boolean).map(c => `<td>${c.trim()}</td>`).join('')}</tr>`
    ).join('');
    return `<table><thead><tr>${heads}</tr></thead><tbody>${rows}</tbody></table>`;
  });
  h = h.replace(/((?:^[ \t]*[-*+] .+\n?)+)/gm, match => {
    const items = match.trim().split('\n').map(l => `<li>${l.replace(/^[ \t]*[-*+] /, '')}</li>`).join('');
    return `<ul>${items}</ul>`;
  });
  h = h.replace(/((?:^\d+\. .+\n?)+)/gm, match => {
    const items = match.trim().split('\n').map(l => `<li>${l.replace(/^\d+\. /, '')}</li>`).join('');
    return `<ol>${items}</ol>`;
  });
  h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
  h = h.split('\n\n').map(block => {
    block = block.trim();
    if (!block) return '';
    if (/^<(h[1-6]|ul|ol|pre|table|blockquote|hr)/.test(block)) return block;
    return `<p>${block.replace(/\n/g, '<br/>')}</p>`;
  }).join('\n');
  return h;
}

function showToast(msg, type = 'success') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast' + (type === 'error' ? ' error' : '');
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 3500);
}

// ─────────────────────────────────────────────────────────
// PIPELINE RENDERING
// ─────────────────────────────────────────────────────────
const pipelineEl = document.getElementById('pipeline');
AGENT_ORDER.forEach((key, i) => {
  const a = AGENT_META[key];
  const wrap = document.createElement('div');
  wrap.className = 'node-wrap';
  wrap.innerHTML = `
    <div class="node" id="node-${key}" onclick="openDrawer('${key}')">
      <div class="node-icon">
        <div class="spinner"></div>
        ${a.icon}
        <div class="node-badge" id="badge-${key}">✓</div>
      </div>
      <div class="node-name dim" id="name-${key}">${a.name}</div>
      <div class="node-status" id="status-${key}">idle</div>
    </div>
    ${i < AGENT_ORDER.length-1 ? '<div class="connector"><div class="fill" id="conn-'+key+'"></div></div>' : ''}
  `;
  pipelineEl.appendChild(wrap);
});

let agentLogs = {}; // stores latest logs per agent for the drawer
function openDrawer(key) {
  const a = AGENT_META[key];
  const drawer = document.getElementById('drawer');
  document.getElementById('drawerTitle').textContent = a.name;
  document.getElementById('drawerTech').textContent = a.tech;
  
  if (agentLogs[key]) {
    document.getElementById('drawerBody').innerHTML = agentLogs[key].desc || '—';
    document.getElementById('drawerLog').textContent = agentLogs[key].log || '—';
  } else {
    document.getElementById('drawerBody').textContent = 'Waiting for data...';
    document.getElementById('drawerLog').textContent = '—';
  }
  drawer.classList.add('open');
}

function resetPipeline() {
  AGENT_ORDER.forEach(key => {
    const node = document.getElementById('node-' + key);
    node.className = 'node';
    document.getElementById('name-' + key).classList.add('dim');
    document.getElementById('status-' + key).textContent = 'idle';
    if(document.getElementById('conn-' + key)) {
      document.getElementById('conn-' + key).style.width = '0%';
    }
  });
  agentLogs = {};
  document.getElementById('drawer').classList.remove('open');
  document.getElementById('conf-svg').style.strokeDashoffset = '238.7';
  document.getElementById('conf-val').textContent = '0.00';
  document.getElementById('trend-bars').innerHTML = '<div style="color:var(--muted-2); font-size:12px; height:100%; display:flex; align-items:center;">Waiting for critic loops...</div>';
  
  document.getElementById('report-viewer-section').style.display = 'none';
  document.getElementById('report-body').innerHTML = '<p style="color:var(--muted);font-style:italic;">Writing report<span class="stream-cursor"></span></p>';
  document.getElementById('resume-btn').style.display = 'none';
}

function setNodeState(key, state, statusText) {
  const node = document.getElementById('node-' + key);
  if(!node) return;
  node.className = 'node ' + state;
  document.getElementById('name-' + key).classList.remove('dim');
  if(statusText) document.getElementById('status-' + key).textContent = statusText;
  
  // Update connector if done
  if(state === 'done') {
    const conn = document.getElementById('conn-' + key);
    if(conn) conn.style.width = '100%';
  }
}

// ─────────────────────────────────────────────────────────
// SSE PIPELINE EXECUTION
// ─────────────────────────────────────────────────────────
let eventSource = null;
let currentThreadId = null;
let _streamBuffer = '';
let isPipelineRunning = false;
let currentAbortController = null;

function fillQuery(el) { 
  document.getElementById('queryInput').value = el.textContent; 
  startRealPipeline();
}

async function startRealPipeline() {
  if (isPipelineRunning) {
    if (currentAbortController) currentAbortController.abort();
    await new Promise(r => setTimeout(r, 100)); // wait for cleanup
  }
  
  const query = document.getElementById('queryInput').value.trim();
  if (!query) { showToast('Please enter a query', 'error'); return; }

  isPipelineRunning = true;
  currentAbortController = new AbortController();
  document.getElementById('run-btn').disabled = true;
  document.getElementById('status-ring').style.animationPlayState = 'running';
  document.getElementById('run-status').textContent = 'RESEARCHING...';
  
  resetPipeline();
  document.getElementById('pipeline-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
  
  const payload = { query: query, user_id: 'browser_user' };

  try {
    const res = await fetch(`${API_BASE}/research/stream`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      signal: currentAbortController.signal
    });

    if (!res.ok) {
      if (res.status === 429) throw new Error('Too many requests. Please wait a minute.');
      throw new Error(`Server error: ${res.status}`);
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
        handleSSEEvent(evType, evData);
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') return;
    showToast(err.message, 'error');
    document.getElementById('run-status').textContent = 'ERROR / ABORTED';
  } finally {
    if (currentAbortController && !currentAbortController.signal.aborted) {
      isPipelineRunning = false;
      document.getElementById('run-btn').disabled = false;
      document.getElementById('status-ring').style.animationPlayState = 'paused';
      if(document.getElementById('run-status').textContent === 'RESEARCHING...') {
        document.getElementById('run-status').textContent = 'IDLE';
      }
    }
  }
}

function handleSSEEvent(evType, evData) {
  if (evType === 'agent_start') {
    const agent = evData.agent;
    setNodeState(agent, 'active', 'working…');
    agentLogs[agent] = { desc: evData.message, log: '…' };
    openDrawer(agent);
  }
  
  else if (evType === 'agent_done') {
    const agent = evData.agent;
    const r = evData.result || {};
    let desc = agentLogs[agent] ? agentLogs[agent].desc : '';
    let log = '';

    if (agent === 'router') log = `→ tone: ${r.tone}, casual: ${r.is_casual}`;
    else if (agent === 'planner') log = `→ ${r.plan?.length || 0} sub-tasks planned`;
    else if (agent === 'researcher') log = `→ found ${r.sources_found || 0} sources (pass ${r.iteration || 1})`;
    else if (agent === 'retriever') log = `→ retrieved ${r.docs_retrieved || 0} vectors`;
    else if (agent === 'critic') {
      const score = (r.confidence || 0).toFixed(2);
      if (r.passed) {
        log = `→ confidence ${score} ✓`;
        setNodeState(agent, 'done', `confidence ${score} ✓`);
      } else {
        log = `→ confidence ${score} ↺ retrying`;
        setNodeState(agent, 'loop', `confidence ${score} ↺`);
        // Animate connector backwards if going from critic to researcher
        const prevConn = document.getElementById('conn-retriever');
        if(prevConn) prevConn.style.width = '0%';
        const prevConn2 = document.getElementById('conn-researcher');
        if(prevConn2) prevConn2.style.width = '0%';
      }
      updateConfidenceUI(score, r.iteration);
    }
    else if (agent === 'reporter') log = `→ report compiled (${r.report_length} chars)`;

    if (agent !== 'critic' || r.passed) {
      setNodeState(agent, 'done', r.cached ? 'cached ✓' : 'done');
    }
    
    agentLogs[agent] = { desc: desc, log: log };
    
    // update drawer if it's currently open for this agent
    const drawerTitle = document.getElementById('drawerTitle').textContent;
    if (drawerTitle.toLowerCase() === AGENT_META[agent].name.toLowerCase()) {
      openDrawer(agent);
    }
  }
  
  else if (evType === 'hitl_pause') {
    currentThreadId = evData.thread_id;
    document.getElementById('run-status').textContent = 'WAITING FOR APPROVAL';
    document.getElementById('resume-btn').style.display = 'inline-block';
    showToast('Human review required. Check drawer and approve.', 'success');
  }
  
  else if (evType === 'report_chunk') {
    if (document.getElementById('report-viewer-section').style.display === 'none') {
      document.getElementById('report-viewer-section').style.display = 'block';
      _streamBuffer = '';
    }
    _streamBuffer += evData.chunk;
    document.getElementById('report-body').innerHTML = md2html(_streamBuffer) + '<span class="stream-cursor"></span>';
  }
  
  else if (evType === 'complete') {
    document.getElementById('run-status').textContent = 'COMPLETE';
    document.getElementById('resume-btn').style.display = 'none';
    if(_streamBuffer) {
      document.getElementById('report-body').innerHTML = md2html(_streamBuffer); // remove cursor
    } else if (evData.report) {
      document.getElementById('report-viewer-section').style.display = 'block';
      document.getElementById('report-body').innerHTML = md2html(evData.report);
    }
    loadArchive();
  }
  
  else if (evType === 'error') {
    showToast(evData.message || 'Error occurred', 'error');
    document.getElementById('run-status').textContent = 'ERROR';
  }
}

async function resumePipeline() {
  if (!currentThreadId) return;
  document.getElementById('resume-btn').style.display = 'none';
  document.getElementById('run-status').textContent = 'RESEARCHING...';
  document.getElementById('report-viewer-section').scrollIntoView({ behavior: 'smooth', block: 'start' });
  try {
    const res = await fetch(`${API_BASE}/research/resume`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: currentThreadId })
    });
    if (!res.ok) throw new Error('Failed to resume');
    showToast('Generating final report...');
  } catch (err) {
    showToast(err.message, 'error');
  }
}

let trendTimeline = [];
function updateConfidenceUI(scoreStr, iteration) {
  const score = parseFloat(scoreStr);
  document.getElementById('conf-val').textContent = score.toFixed(2);
  
  // update radial gauge
  const dash = 238.7 * score; 
  document.getElementById('conf-svg').style.strokeDashoffset = 238.7 - dash;
  
  // update color based on score
  let color = '#E38080'; // red
  if (score > 0.75) color = '#7DD3A8'; // green
  else if (score > 0.5) color = '#FBBF6B'; // amber
  document.getElementById('conf-svg').style.stroke = color;
  
  // update trend bars
  if (iteration === 1) {
    trendTimeline = [];
    document.getElementById('trend-bars').innerHTML = '';
  }
  trendTimeline.push({ score, color });
  
  const bars = trendTimeline.map(t => {
    const h = Math.max(10, t.score * 80); // scale height up to 80px
    return `<div class="trend-bar" style="height:${h}px; background:${t.color};"><span>${t.score.toFixed(2)}</span></div>`;
  }).join('');
  document.getElementById('trend-bars').innerHTML = bars;
}

// ─────────────────────────────────────────────────────────
// ARCHIVE
// ─────────────────────────────────────────────────────────
function timeAgo(ts) {
  const diff = Date.now()/1000 - ts;
  if(diff < 60) return 'Just now';
  if(diff < 3600) return Math.floor(diff/60) + 'm ago';
  if(diff < 86400) return Math.floor(diff/3600) + 'h ago';
  return Math.floor(diff/86400) + 'd ago';
}

async function loadArchive() {
  const grid = document.getElementById('archiveGrid');
  grid.innerHTML = '<div class="empty-state">Loading history...</div>';
  try {
    const res = await fetch(`${API_BASE}/reports?limit=12`);
    if (!res.ok) throw new Error('Failed to load reports');
    const data = await res.json();
    
    if(!data.reports || data.reports.length === 0) {
      grid.innerHTML = '<div class="empty-state">No past runs found.</div>';
      return;
    }
    
    let reportsToRender = data.reports;
    if (currentArchiveFilter === 'high') {
      reportsToRender = reportsToRender.filter(r => r.confidence > 0.75);
    }
    
    if(reportsToRender.length === 0) {
      grid.innerHTML = '<div class="empty-state">No high confidence runs found.</div>';
      return;
    }
    
    grid.innerHTML = '';
    reportsToRender.forEach(r => {
      const color = r.confidence > 0.75 ? '#7DD3A8' : r.confidence > 0.5 ? '#FBBF6B' : '#E38080';
      const dash = 100.5 * r.confidence;
      const card = document.createElement('div');
      card.className = 'card';
      
      // Provide a clean excerpt of the markdown report text
      let rawText = (r.report || '').replace(/#|\*|`|>|\[|\]/g, ' ').substring(0, 120);
      
      card.innerHTML = `
        <div class="card-top">
          <div class="card-title">${r.query}</div>
          <svg class="mini-gauge" viewBox="0 0 40 40">
            <circle cx="20" cy="20" r="16" fill="none" stroke="#1F2833" stroke-width="5"/>
            <circle cx="20" cy="20" r="16" fill="none" stroke="${color}" stroke-width="5" stroke-linecap="round"
              stroke-dasharray="100.5" stroke-dashoffset="${100.5 - dash}" transform="rotate(-90 20 20)"/>
            <text x="20" y="24" text-anchor="middle" font-family="JetBrains Mono" font-size="9" fill="${color}">${r.confidence.toFixed(2)}</text>
          </svg>
        </div>
        <div class="card-excerpt">${rawText}...</div>
        <div class="card-meta"><span>${r.tone}</span><span>${timeAgo(r.created_at)}</span></div>
      `;
      // allow clicking card to load report in the viewer
      card.onclick = () => {
        document.getElementById('report-viewer-section').style.display = 'block';
        document.getElementById('report-body').innerHTML = md2html(r.report);
        document.getElementById('report-viewer-section').scrollIntoView();
      };
      grid.appendChild(card);
    });
  } catch(e) {
    grid.innerHTML = `<div class="empty-state">Could not load history: ${e.message}</div>`;
  }
}

// init
let currentArchiveFilter = 'all';

async function filterArchive(filterType, element) {
  currentArchiveFilter = filterType;
  document.querySelectorAll('.archive-controls .chip').forEach(c => c.classList.remove('active'));
  element.classList.add('active');
  await loadArchive();
}

document.addEventListener('DOMContentLoaded', () => {
  const chips = document.querySelectorAll('.archive-controls .chip');
  if(chips.length >= 2) {
    chips[0].onclick = (e) => filterArchive('all', e.target);
    chips[1].onclick = (e) => filterArchive('high', e.target);
  }
});

setTimeout(() => loadArchive(), 500);

document.getElementById('queryInput').addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.keyCode === 13) {
    e.preventDefault();
    startRealPipeline();
  }
});
