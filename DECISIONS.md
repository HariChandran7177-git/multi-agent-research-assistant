# Technical Decisions & Design Explanations (DECISIONS.md)

This document outlines the core architecture and UI implementation decisions made during the redesign of the Multi-Agent Research Assistant landing page.

---

### 1. Ingestion Strategy: Local Semantic RAG vs. Direct Raw Injection
During the pipeline execution, research findings gathered from Tavily are embedded using Google Gemini (3072-dim) and ingested dynamically into Qdrant (falling back to in-memory if remote is unreachable) with a session filter. 

* **The Obvious Alternative Rejected:** Feeding the raw search results (entire HTML contents or full scraped texts) directly into the LLM context.
* **Why We Rejected It:** 
  1. **Prompt Bloat & Latency:** Raw search responses contain a massive amount of noise (boilerplate text, repeating nav headers). Directly feeding this to the LLM increases latency (longer token parsing) and significantly raises API costs.
  2. **The "Lost in the Middle" Phenomenon:** Piling long, unstructured texts into Groq context decreases reasoning accuracy, causing the Critic and Reporter to overlook critical details.
  3. **Context / Rate Limits:** Parallel Tavily queries return up to 20-30 separate page snippets. Injecting all of them directly easily hits Groq's RPM/TPM rate limits.
  * **Our Ingestion Strategy:** Running semantic retrieval scoped to the current `session_id` ensures that only the top 5 most relevant content chunks are fed to the LLM, keeping the prompt focused and response time under 10 seconds.

---

### 2. Time Limit Trade-off & Future Work
* **The Trade-off Made:** Due to the time constraint, we implemented a static client-side "Demo Mode" fallback in the JavaScript client. When the backend is offline (such as on static hosting on GitHub Pages), the interface simulates the full 6-agent LangGraph workflow step-by-step using actual sample payloads from a real run.
* **What I'd Do With a Real Week:** 
  1. **Live SSE Server Deployment:** Deploy the FastAPI Python server on a service like Render or AWS ECS with a persistent Qdrant cloud collection.
  2. **WebSocket / SSE Tracing:** Integrate LangSmith tracing natively so users can click on any agent card in the UI and inspect the raw LLM prompts, input tokens, and execution times in real-time.
  3. **Interactive Graph Navigation:** Turn the static pipeline visualization into an interactive D3.js or React Flow canvas, allowing users to zoom in, branch off sub-tasks manually, or edit the plan mid-execution.

---

### 3. AI Tool Utilization & Personal Verification
* **Where AI Was Used:**
  * Drafted the boilerplate structure for the GSAP ScrollTrigger timeline reveals.
  * Generated raw RGB colors matching the target "Aloe green" and "Rust brown" hexadecimal values.
* **What I Personally Verified & Modified:**
  * **MIME-Type Fix:** Personally resolved a critical Windows registry MIME-type issue. FastAPI's static server was serving `.js` files as `text/plain` on Windows, causing modern browsers to reject them due to `X-Content-Type-Options: nosniff`. I added explicit MIME registration (`mimetypes.add_type`) at the top of `api/research_api.py`.
  * **CSS Transition Conflicts:** Identified and resolved a conflict where CSS transition classes clashed with GSAP’s inline frame-by-frame transforms. I added `clearProps: 'transform,opacity'` to the GSAP callbacks to clean up the inline styles after entry, ensuring smooth hover transitions.
  * **Responsive Grid Border Trick:** Replaced complex media query rules with a clean CSS Grid gap border strategy (`gap: 1px; background: var(--grid-border);`), guaranteeing perfect 1px cell borders on all screen sizes (390px to 1440px) without double-line overlaps.
  * **Remote API Fallbacks:** Swapped deprecated Llama/Groq model IDs in `agents/router.py` and `core/config.py` with active ones (`groq/compound` and `groq/compound-mini`) after querying the live endpoint models list.
