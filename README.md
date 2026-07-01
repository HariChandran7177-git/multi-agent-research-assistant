# 🔬 Multi-Agent Research Assistant

> An AI-powered research assistant that orchestrates multiple specialized agents to plan, search, retrieve, critique, and synthesize comprehensive research reports — all from a single query.

---

## 🛠️ Tech Stack

![LangGraph](https://img.shields.io/badge/LangGraph-1.2-blue?style=for-the-badge&logo=data:image/svg+xml;base64,&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.3-green?style=for-the-badge&logo=chainlink&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.138-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-1.18-red?style=for-the-badge&logo=data:image/svg+xml;base64,&logoColor=white)
![Tavily](https://img.shields.io/badge/Tavily-0.7-orange?style=for-the-badge&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)

---

## 🏗️ Architecture

> 🚧 **Diagram coming soon.**
>
> The system uses a multi-agent graph powered by LangGraph. Each agent has a specialized role:
>
> | Agent | Role |
> |---|---|
> | **Planner** | Breaks down the research query into sub-tasks |
> | **Researcher** | Searches the web for relevant information |
> | **Retriever** | Fetches and indexes documents in the vector store |
> | **Critic** | Evaluates quality and identifies gaps |
> | **Reporter** | Synthesizes findings into a final report |

---

## 🚀 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/multi-agent-research-assistant.git
cd multi-agent-research-assistant
```

### 2. Create & activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 5. Run the application

```bash
python main.py
```

---

## 📁 Project Structure

```
multi-agent-research-assistant/
├── agents/             # Specialized AI agents
│   ├── planner.py      # Query decomposition
│   ├── researcher.py   # Web search & gathering
│   ├── retriever.py    # Vector store retrieval
│   ├── critic.py       # Quality evaluation
│   └── reporter.py     # Report synthesis
├── core/               # Core orchestration
│   ├── state.py        # Shared state definition
│   └── graph.py        # LangGraph workflow
├── api/                # FastAPI endpoints
├── frontend/           # UI (coming soon)
├── tests/              # Test suite
└── main.py             # Entry point
```

---

## 📄 License

MIT
