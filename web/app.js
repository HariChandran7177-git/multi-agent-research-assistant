/* ─────────────────────────────────────────────────────────
   NEURALDESK — app.js
   Multi-Agent Research Assistant Frontend Logic
───────────────────────────────────────────────────────── */

// Auto-detect API base: same origin on Render, localhost:8000 in local dev
const API_BASE = (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? 'http://localhost:8000'
  : window.location.origin;

// ─────────────────────────────────────────────────────────
//  PARTICLE CANVAS
// ─────────────────────────────────────────────────────────
(function initParticles() {
  const canvas = document.getElementById('particle-canvas');
  const ctx = canvas.getContext('2d');
  let W, H, particles = [];
  let mouse = { x: null, y: null };
  const COUNT = 80;
  const MAX_DIST = 140;
  const COLORS = ['194,65,12', '217,119,6', '101,163,13'];

  function resize() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function mkParticle() {
    return {
      x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - .5) * .4, vy: (Math.random() - .5) * .4,
      r: Math.random() * 1.5 + .5,
      color: COLORS[Math.floor(Math.random() * COLORS.length)],
      alpha: Math.random() * .4 + .1,
    };
  }

  function init() { resize(); particles = []; for (let i = 0; i < COUNT; i++) particles.push(mkParticle()); }

  let lastScrollY = window.scrollY;
  let scrollVelocity = 0;
  window.addEventListener('scroll', () => {
    const currentScrollY = window.scrollY;
    scrollVelocity = (currentScrollY - lastScrollY) * 0.15;
    lastScrollY = currentScrollY;
  });

  function draw() {
    ctx.clearRect(0, 0, W, H);
    scrollVelocity *= 0.92; // decay scroll physics
    
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy - (scrollVelocity * p.r * 0.6); // 3D parallax scroll effect
      
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
      
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.color},${p.alpha})`; ctx.fill();

      for (let j = i + 1; j < particles.length; j++) {
        const q = particles[j];
        const dx = p.x - q.x, dy = p.y - q.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < MAX_DIST) {
          ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y);
          ctx.strokeStyle = `rgba(139,92,246,${(1 - d / MAX_DIST) * .15})`;
          ctx.lineWidth = .8; ctx.stroke();
        }
      }

      if (mouse.x !== null) {
        const dx = p.x - mouse.x, dy = p.y - mouse.y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < 80) {
          p.vx += dx / d * .3; p.vy += dy / d * .3;
          const sp = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
          if (sp > 2) { p.vx /= sp; p.vy /= sp; }
        }
      }
    }
    requestAnimationFrame(draw);
  }

  window.addEventListener('resize', resize);
  window.addEventListener('mousemove', e => { mouse.x = e.clientX; mouse.y = e.clientY; });
  window.addEventListener('mouseleave', () => { mouse.x = null; mouse.y = null; });
  init(); draw();
})();

// ─────────────────────────────────────────────────────────
//  SCROLL REVEAL (GSAP + ScrollTrigger)
// ─────────────────────────────────────────────────────────
(function initScrollReveal() {
  if (typeof gsap === 'undefined' || typeof ScrollTrigger === 'undefined') {
    console.warn("GSAP or ScrollTrigger not loaded.");
    return;
  }
  
  gsap.registerPlugin(ScrollTrigger);

  // Helper to split text into chars
  function splitTextToChars(element) {
    if (!element) return;
    const text = element.innerText.trim();
    element.innerHTML = text.split('').map(char => {
      if (char === ' ') return '<span>&nbsp;</span>';
      return `<span>${char}</span>`;
    }).join('');
  }

  // Do NOT split hero title lines — gradient -webkit-background-clip breaks when innerHTML is replaced

  // Hero animations — animate LINES as whole units, not split chars
  // (char-splitting breaks the CSS gradient clip on .line-1 / .line-2)
  gsap.from('.hero-eyebrow', { opacity: 0, y: 15, duration: 0.6, ease: 'power2.out' });
  gsap.from('.hero-title .line-1', { opacity: 0, y: 30, duration: 0.7, ease: 'power3.out', delay: 0.15 });
  gsap.from('.hero-title .line-2', { opacity: 0, y: 30, duration: 0.7, ease: 'power3.out', delay: 0.3, clearProps: 'transform,opacity' });
  gsap.from('.hero-sub', { opacity: 0, y: 15, duration: 0.8, ease: 'power2.out', delay: 0.45 });
  gsap.from('.search-container', { opacity: 0, y: 15, duration: 0.8, ease: 'power2.out', delay: 0.55 });

  // Section titles — fade whole element, no splitting
  document.querySelectorAll('.section-title').forEach(title => {
    gsap.from(title, {
      scrollTrigger: { trigger: title, start: 'top 88%', toggleActions: 'play none none none' },
      duration: 0.5,
      ease: 'power2.out'
    });
  });

  // Scroll triggers for eyebrows and subtitles
  document.querySelectorAll('.section-eyebrow, .section-subtitle').forEach(el => {
    gsap.from(el, {
      scrollTrigger: {
        trigger: el,
        start: 'top 88%',
        toggleActions: 'play none none none'
      },
      opacity: 0,
      y: 15,
      duration: 0.6,
      ease: 'power2.out'
    });
  });

  // Agent Cards reveal
  if (document.querySelector('.pipeline-grid')) {
    gsap.from('.agent-card', {
      scrollTrigger: {
        trigger: '.pipeline-grid',
        start: 'top 80%',
        toggleActions: 'play none none none'
      },
      opacity: 0,
      y: 25,
      stagger: 0.06,
      duration: 0.7,
      ease: 'power3.out',
      clearProps: 'transform,opacity'
    });
  }
  
  // Tech Cards reveal
  if (document.querySelector('.tech-grid')) {
    gsap.from('.tech-card', {
      scrollTrigger: {
        trigger: '.tech-grid',
        start: 'top 85%',
        toggleActions: 'play none none none'
      },
      opacity: 0,
      y: 20,
      stagger: 0.04,
      duration: 0.6,
      ease: 'power2.out',
      clearProps: 'transform,opacity'
    });
  }
})();

// ─────────────────────────────────────────────────────────
//  TEXTAREA AUTO-RESIZE
// ─────────────────────────────────────────────────────────
(function() {
  const ta = document.getElementById('query-input');
  ta.addEventListener('input', () => { ta.style.height = 'auto'; ta.style.height = ta.scrollHeight + 'px'; });
  ta.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); startResearch(); }
  });
})();

// ─────────────────────────────────────────────────────────
//  UTILITY
// ─────────────────────────────────────────────────────────
function setQuery(text) {
  const ta = document.getElementById('query-input');
  ta.value = text;
  ta.style.height = 'auto'; ta.style.height = ta.scrollHeight + 'px'; ta.focus();
}

function showToast(msg, type = 'success') {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'toast' + (type === 'error' ? ' error' : '');
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 3500);
}

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// ─────────────────────────────────────────────────────────
//  PIPELINE NODE STATE
// ─────────────────────────────────────────────────────────
function setNodeActive(name) {
  document.querySelectorAll('.pipe-node').forEach(n => n.classList.remove('active'));
  const n = document.getElementById('pnode-' + name);
  if (n) n.classList.add('active');
}

function setNodeDone(name) {
  const n = document.getElementById('pnode-' + name);
  if (!n) return;
  n.classList.remove('active'); n.classList.add('done');
  const wrap = n.querySelector('.pipe-icon-wrap');
  if (wrap) {
    const orig = wrap.textContent;
    wrap.textContent = '✓';
    setTimeout(() => { wrap.textContent = orig; }, 700);
  }
}

// ─────────────────────────────────────────────────────────
//  ACTIVITY LOG
// ─────────────────────────────────────────────────────────
function logStart(data) {
  const log = document.getElementById('activity-log');
  const item = document.createElement('div');
  item.className = 'activity-item active-item';
  item.id = 'alog-' + data.agent + '-' + Date.now();
  item.dataset.agent = data.agent;
  item.innerHTML = `
    <div class="activity-icon">${data.icon || '⚙️'}</div>
    <div class="activity-content">
      <div class="activity-agent">${(data.label || data.agent).toUpperCase()}</div>
      <div class="activity-msg">${data.message}</div>
    </div>
    <div class="activity-time">${fmtTime(data.timestamp || Date.now() / 1000)}</div>
  `;
  log.appendChild(item);
  log.scrollTop = log.scrollHeight;
  // Tag as "last active" for this agent
  log.querySelectorAll(`[data-agent="${data.agent}"].active-item`).forEach(el => {
    if (el !== item) { el.classList.remove('active-item'); }
  });
  setNodeActive(data.agent);
}

function logDone(data) {
  // Find most recent active item for this agent
  const log = document.getElementById('activity-log');
  const items = log.querySelectorAll(`[data-agent="${data.agent}"].active-item`);
  const item = items[items.length - 1];
  if (item) {
    item.classList.remove('active-item'); item.classList.add('done-item');
    const r = data.result || {};
    let txt = '';
    if (r.tone) txt = `Tone: <strong>${r.tone}</strong>`;
    if (r.plan && r.plan.length) {
      txt = `<strong>${r.plan.length} sub-tasks planned:</strong><br><ul style="margin: 4px 0 0 16px; padding: 0;">` + r.plan.map(t => `<li style="margin-bottom: 2px;">${t}</li>`).join('') + `</ul>`;
    }
    if (r.sources_found !== undefined) txt = `<strong>${r.sources_found} sources retrieved</strong> via parallel web search`;
    if (r.docs_retrieved !== undefined) txt = `<strong>${r.docs_retrieved} docs</strong> retrieved from Qdrant vector store`;
    if (r.confidence !== undefined) {
      txt = `<strong>Score: ${r.confidence}</strong> — ${r.passed ? '<span style="color:var(--emerald)">✓ passed threshold</span>' : '<span style="color:var(--rose)">↻ looping for depth</span>'}<br><em>Critique: ${r.critique}</em>`;
      updateConfidence(r.confidence);
    }
    if (r.report_length) txt = `<strong>${r.report_length.toLocaleString()} chars generated</strong>`;
    
    if (txt) {
      const content = item.querySelector('.activity-content');
      const res = document.createElement('div');
      res.className = 'activity-result'; 
      res.innerHTML = txt;
      content.appendChild(res);
    }
  }
  setNodeDone(data.agent);
}

function updateConfidence(score) {
  document.getElementById('conf-value').textContent = score.toFixed(2);
  setTimeout(() => { document.getElementById('conf-fill').style.width = (score * 100) + '%'; }, 50);
}

// ─────────────────────────────────────────────────────────
//  MARKDOWN PARSER (lightweight, no deps)
// ─────────────────────────────────────────────────────────
function md2html(md) {
  if (!md) return '';
  let h = md;

  // Fenced code blocks
  h = h.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
    const esc = code.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    return `<pre><code class="lang-${lang}">${esc.trim()}</code></pre>`;
  });

  // Inline code
  h = h.replace(/`([^`]+)`/g, (_, c) => `<code>${c.replace(/</g,'&lt;')}</code>`);

  // Headers
  h = h.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  h = h.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  h = h.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Horizontal rules
  h = h.replace(/^---$/gm, '<hr>');

  // Bold + italic combinations
  h = h.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Blockquotes
  h = h.replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>');

  // Tables (header | sep | rows)
  h = h.replace(/(\|.+\|\n)(\|[-:| ]+\|\n)((?:\|.+\|\n?)*)/g, (_, header, sep, body) => {
    const heads = header.trim().split('|').filter(Boolean).map(x => `<th>${x.trim()}</th>`).join('');
    const rows = body.trim().split('\n').map(row =>
      `<tr>${row.trim().split('|').filter(Boolean).map(c => `<td>${c.trim()}</td>`).join('')}</tr>`
    ).join('');
    return `<table><thead><tr>${heads}</tr></thead><tbody>${rows}</tbody></table>`;
  });

  // Unordered lists
  h = h.replace(/((?:^[ \t]*[-*+] .+\n?)+)/gm, match => {
    const items = match.trim().split('\n').map(l => `<li>${l.replace(/^[ \t]*[-*+] /, '')}</li>`).join('');
    return `<ul>${items}</ul>`;
  });

  // Ordered lists
  h = h.replace(/((?:^\d+\. .+\n?)+)/gm, match => {
    const items = match.trim().split('\n').map(l => `<li>${l.replace(/^\d+\. /, '')}</li>`).join('');
    return `<ol>${items}</ol>`;
  });

  // Links
  h = h.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

  // Paragraphs
  h = h.split('\n\n').map(block => {
    block = block.trim();
    if (!block) return '';
    if (/^<(h[1-6]|ul|ol|pre|table|blockquote|hr)/.test(block)) return block;
    return `<p>${block.replace(/\n/g, '<br/>')}</p>`;
  }).join('\n');

  return h;
}

// ─────────────────────────────────────────────────────────
//  SHOW REPORT
// ─────────────────────────────────────────────────────────
let _currentReport = '';

function showReport(data) {
  _currentReport = data.report || '';
  document.getElementById('report-placeholder').style.display = 'none';
  const content = document.getElementById('report-content');
  content.classList.add('visible');

  const bd = data.score_breakdown || {};
  document.getElementById('report-meta').innerHTML = `
    <div class="meta-chip emerald">Confidence: ${(data.confidence * 100).toFixed(0)}%</div>
    <div class="meta-chip violet">Iterations: ${data.iterations}</div>
    <div class="meta-chip cyan">Tone: ${data.tone}</div>
    ${bd.llm_score ? `<div class="meta-chip">LLM: ${bd.llm_score}</div>` : ''}
    ${bd.objective_score ? `<div class="meta-chip">Objective: ${bd.objective_score.toFixed(3)}</div>` : ''}
  `;

  document.getElementById('report-body').innerHTML = md2html(_currentReport);
  document.getElementById('copy-btn').style.display = 'block';
  document.getElementById('download-btn').style.display = 'block';
  updateConfidence(data.confidence);
}

function copyReport() {
  navigator.clipboard.writeText(_currentReport).then(() => showToast('Report copied!'));
}

function downloadReport() {
  const blob = new Blob([_currentReport], { type: 'text/markdown' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'research-report.md'; a.click();
  showToast('Downloading...');
}

// ─────────────────────────────────────────────────────────
//  RESET DASHBOARD
// ─────────────────────────────────────────────────────────
function resetDashboard() {
  document.getElementById('dashboard').classList.remove('active');
  document.getElementById('activity-log').innerHTML = '';
  document.getElementById('conf-value').textContent = '--';
  document.getElementById('conf-fill').style.width = '0%';
  document.getElementById('report-placeholder').style.display = 'flex';
  document.getElementById('report-content').classList.remove('visible');
  document.getElementById('report-meta').innerHTML = '';
  document.getElementById('report-body').innerHTML = '';
  document.getElementById('copy-btn').style.display = 'none';
  document.getElementById('download-btn').style.display = 'none';
  document.querySelectorAll('.pipe-node').forEach(n => n.classList.remove('active','done'));
  document.getElementById('live-label').textContent = 'LIVE';
  document.getElementById('live-dot').style.cssText = '';
  const btn = document.getElementById('run-btn');
  btn.disabled = false; btn.classList.remove('loading');
  _currentReport = '';
  document.getElementById('hero').scrollIntoView({ behavior: 'smooth' });
}

// ─────────────────────────────────────────────────────────
//  SSE EVENT HANDLER
// ─────────────────────────────────────────────────────────
function handleSSE(event, data) {
  switch (event) {
    case 'start':
      // Kicked off — router is first
      logStart({
        agent: 'router', icon: '🔀', label: 'Router',
        message: 'Analyzing query intent and detecting tone...',
        timestamp: data.timestamp
      });
      break;
    case 'agent_start':
      logStart(data);
      break;
    case 'agent_done':
      logDone(data);
      break;
    case 'complete':
      onComplete(data);
      break;
    case 'error':
      onError(data);
      break;
    case 'plain_llm_done':
      showPlainLLM(data);
      break;
  }
}

function showPlainLLM(data) {
  const container = document.getElementById('plain-llm-body');
  if (container) {
    container.innerHTML = md2html(data.response || '');
  }
}

function onComplete(data) {
  const dot = document.getElementById('live-dot');
  dot.style.background = 'var(--emerald)';
  dot.style.boxShadow = '0 0 6px var(--emerald)';
  document.getElementById('live-label').textContent = 'DONE';
  showReport(data);
  const btn = document.getElementById('run-btn');
  btn.disabled = false; btn.classList.remove('loading');
  showToast('Research complete! 🎉');
  document.getElementById('report-content').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function onError(data) {
  const log = document.getElementById('activity-log');
  const item = document.createElement('div');
  item.className = 'activity-item';
  item.style.cssText = 'border-color:rgba(244,63,94,.3);background:rgba(244,63,94,.05)';
  item.innerHTML = `
    <div class="activity-icon">❌</div>
    <div class="activity-content">
      <div class="activity-agent" style="color:var(--rose)">ERROR</div>
      <div class="activity-msg">${data.message}</div>
    </div>`;
  log.appendChild(item);
  const btn = document.getElementById('run-btn');
  btn.disabled = false; btn.classList.remove('loading');
  showToast(data.message, 'error');
}

// ─────────────────────────────────────────────────────────
//  MAIN: START RESEARCH
// ─────────────────────────────────────────────────────────
async function startResearch() {
  const query = document.getElementById('query-input').value.trim();
  if (!query) { showToast('Please enter a research query', 'error'); return; }

  // UI loading
  const btn = document.getElementById('run-btn');
  btn.disabled = true; btn.classList.add('loading');

  // Show dashboard
  document.getElementById('query-display-text').textContent = query;
  document.getElementById('dashboard').classList.add('active');
  document.getElementById('dashboard').scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Reset
  document.querySelectorAll('.pipe-node').forEach(n => n.classList.remove('active','done'));
  document.getElementById('activity-log').innerHTML = '';
  document.getElementById('conf-value').textContent = '--';
  document.getElementById('conf-fill').style.width = '0%';
  document.getElementById('report-placeholder').style.display = 'flex';
  document.getElementById('report-content').classList.remove('visible');
  
  const plainBody = document.getElementById('plain-llm-body');
  if (plainBody) plainBody.innerHTML = '<div class="loading-pulse">Thinking...</div>';
  const reportBody = document.getElementById('report-body');
  if (reportBody) reportBody.innerHTML = '<div class="loading-pulse">Agents are researching...</div>';

  document.getElementById('copy-btn').style.display = 'none';
  document.getElementById('download-btn').style.display = 'none';
  document.getElementById('live-label').textContent = 'LIVE';
  document.getElementById('live-dot').style.cssText = '';
  _currentReport = '';

  try {
    const resp = await fetch(`${API_BASE}/research/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop();
      let evType = null;
      for (const line of lines) {
        if (line.startsWith('event: ')) evType = line.slice(7).trim();
        else if (line.startsWith('data: ') && evType) {
          try { handleSSE(evType, JSON.parse(line.slice(6))); } catch (_) {}
          evType = null;
        }
      }
    }
  } catch (err) {
    console.warn('API unavailable, using demo mode:', err.message);
    showToast('API offline — running demo mode', 'error');
    runDemoMode(query);
  }
}

// ─────────────────────────────────────────────────────────
//  DEMO MODE (no backend needed)
// ─────────────────────────────────────────────────────────
const DEMO_REPORT = `# AWS vs GCP for Startups in 2025: A Senior Engineer's Breakdown

The honest answer? **It depends on your workload** — but the decision is far less symmetric than AWS's market dominance implies.

---

## Market Reality

**AWS holds ~31% of the global cloud market** vs. GCP's ~12% (Synergy Research, Q1 2025). That gap matters — it translates directly into a larger ecosystem, more StackOverflow answers, and a deeper talent pool of engineers who already know the platform.

**GCP's stronghold** is data and ML workloads. If your startup is building anything that touches large-scale data pipelines, ML training, or analytics, GCP's native BigQuery + Vertex AI + TPU stack is genuinely superior and often meaningfully cheaper.

---

## Comparison: What Actually Matters

| Criteria | AWS | GCP |
|---|---|---|
| **Market share** | Dominant (31%) | Smaller (12%) |
| **Startup credits** | $5K-$100K via Activate | $200K via Google for Startups |
| **Managed Kubernetes** | EKS (complex config) | GKE (superior managed K8s) |
| **ML/AI tooling** | SageMaker (verbose) | Vertex AI + TPUs |
| **Data warehouse** | Redshift (complex tuning) | BigQuery (serverless, pay-per-query) |
| **Networking pricing** | Expensive egress ($0.09/GB) | Cheaper egress, free between GCP services |

---

## The Real Decision Framework

**Choose AWS if:**
- You're building a general SaaS product with no specific ML/data angle
- Your team already has AWS experience
- You need the widest range of third-party integrations
- You're in a compliance-heavy industry

**Choose GCP if:**
- ML model training, fine-tuning, or inference is core to your product
- You need a managed data warehouse without a dedicated data engineer
- You're building on Kubernetes and want GKE
- You're cost-sensitive on compute + networking

---

## The Senior Engineer's Take

> AWS is the safe default. GCP is the smart choice if data or ML is core to your product.

The worst decision is agonizing over this for weeks. Pick one, use Terraform from day one so switching is possible later, and focus on shipping. The infrastructure difference will not be your startup's bottleneck.

---

### Sources

- [Synergy Research Group - Cloud Market Share Q1 2025](https://www.srgresearch.com)
- [AWS Activate for Startups](https://aws.amazon.com/activate/)
- [Google for Startups Cloud Program](https://cloud.google.com/startup)
- [BigQuery Pricing Overview](https://cloud.google.com/bigquery/pricing)

---

*Generated end-to-end by the 6-agent LangGraph pipeline in ~42 seconds.*`;

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function runDemoMode(query) {
  const steps = [
    { agent: 'router',     icon: '🔀', label: 'Router',     message: 'Analyzing query intent & detecting tone...', delay: 1200, result: { tone: 'professional and technical', is_casual: false } },
    { agent: 'planner',    icon: '📋', label: 'Planner',    message: 'Breaking into 4 targeted research sub-tasks...', delay: 1400, result: { plan: ['Market share analysis', 'Feature comparison', 'Cost model analysis', 'Startup ecosystem review'] } },
    { agent: 'researcher', icon: '⚡', label: 'Researcher', message: 'Parallel Tavily web search — pass 1 (ThreadPoolExecutor)...', delay: 2000, result: { sources_found: 12 } },
    { agent: 'retriever',  icon: '🧠', label: 'Retriever',  message: 'Gemini embedding (3072-dim) → Qdrant cosine retrieval...', delay: 1600, result: { docs_retrieved: 8 } },
    { agent: 'critic',     icon: '🧐', label: 'Critic',     message: 'Hybrid scoring: 60% LLM + 40% objective signals...', delay: 1800, result: { confidence: 0.67, passed: false, critique: 'Missing financial metrics and cost benchmarks. Looping for deeper research.' } },
    { agent: 'researcher', icon: '⚡', label: 'Researcher', message: 'Parallel web search — pass 2 (deeper research pass)...', delay: 1800, result: { sources_found: 9 } },
    { agent: 'retriever',  icon: '🧠', label: 'Retriever',  message: 'Re-embedding and retrieving enriched context...', delay: 1400, result: { docs_retrieved: 11 } },
    { agent: 'critic',     icon: '🧐', label: 'Critic',     message: 'Re-evaluating with enriched data...', delay: 1600, result: { confidence: 0.84, passed: true, critique: 'Comprehensive coverage with verified source citations and financial data. Threshold passed.' } },
    { agent: 'reporter',   icon: '📝', label: 'Reporter',   message: 'Writing tone-aware markdown report...', delay: 2000, result: { report_length: DEMO_REPORT.length } },
  ];

  for (const step of steps) {
    logStart({ ...step, timestamp: Date.now() / 1000 });
    await sleep(step.delay);
    logDone({ ...step, timestamp: Date.now() / 1000 });
    await sleep(250);
  }

  await sleep(500);
  onComplete({
    report: DEMO_REPORT,
    confidence: 0.84,
    iterations: 2,
    tone: 'professional and technical',
    is_casual: false,
    score_breakdown: { llm_score: 0.87, objective_score: 0.79 },
  });
}

// ─────────────────────────────────────────────────────────
//  PREMIUM HOVER MICRO-INTERACTIONS (Scramble Effect)
// ─────────────────────────────────────────────────────────
(function initNavHover() {
  // Select navigation links and primary search button
  document.querySelectorAll('.nav-links a, #run-btn').forEach(link => {
    const target = link.querySelector('.btn-text') || link;
    if (!target) return;
    
    // Avoid double splitting
    if (target.classList.contains('split-done')) return;
    target.classList.add('split-done');
    
    const text = target.innerText.trim();
    if (!text) return;
    
    target.innerHTML = text.split('').map(char => {
      if (char === ' ') return '<span>&nbsp;</span>';
      return `<span class="hover-char" data-orig="${char}" style="display:inline-block; transition:transform 0.15s var(--ease);">${char}</span>`;
    }).join('');
    
    const chars = target.querySelectorAll('.hover-char');
    
    // Character scramble effect on hover!
    link.addEventListener('mouseenter', () => {
      chars.forEach((span, idx) => {
        // Shift letter up slightly
        gsap.to(span, {
          y: -2,
          color: 'var(--cyan-glow)',
          duration: 0.15,
          delay: idx * 0.015,
          ease: 'power1.out',
          overwrite: 'auto'
        });
        
        const orig = span.dataset.orig;
        const randoms = ['A','B','C','X','Y','Z','0','1','*','#','@','!','%','&'];
        
        // Staggered character scramble
        setTimeout(() => {
          span.innerText = randoms[Math.floor(Math.random() * randoms.length)];
          setTimeout(() => {
            span.innerText = randoms[Math.floor(Math.random() * randoms.length)];
            setTimeout(() => {
              span.innerText = orig;
            }, 80);
          }, 60);
        }, idx * 30);
      });
    });
    
    link.addEventListener('mouseleave', () => {
      chars.forEach((span, idx) => {
        gsap.to(span, {
          y: 0,
          color: '',
          duration: 0.15,
          delay: idx * 0.01,
          ease: 'power1.in',
          overwrite: 'auto'
        });
      });
    });
  });
})();

// ─────────────────────────────────────────────────────────
//  EASTER EGG — Konami Code: ↑↑↓↓←→←→BA
// ─────────────────────────────────────────────────────────
(function initKonamiCode() {
  const CODE = ['ArrowUp','ArrowUp','ArrowDown','ArrowDown','ArrowLeft','ArrowRight','ArrowLeft','ArrowRight','b','a'];
  let pos = 0;
  document.addEventListener('keydown', function(e) {
    if (e.key === CODE[pos]) {
      pos++;
      if (pos === CODE.length) {
        pos = 0;
        document.getElementById('konami-overlay').classList.add('active');
      }
    } else {
      pos = (e.key === CODE[0]) ? 1 : 0;
    }
  });
})();
