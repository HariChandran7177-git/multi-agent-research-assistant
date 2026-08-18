# Contributing to Multi-Agent Research Assistant

First off — thank you for taking the time to contribute! 🎉

This project is a LangGraph-based multi-agent AI pipeline. Contributions of all kinds are welcome: bug fixes, new agents, performance improvements, documentation, and tests.

---

## 🚀 Getting Started

1. **Fork** the repository and clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/multi-agent-research-assistant.git
   cd multi-agent-research-assistant
   ```

2. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS/Linux
   pip install -r requirements.txt
   ```

3. **Set up your `.env`** by copying `.env.example`:
   ```bash
   cp .env.example .env
   # Fill in your API keys
   ```

4. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## 📐 Code Style

- Follow **PEP 8**. We use `ruff` and `black` for formatting.
- Run before committing:
  ```bash
  ruff check .
  black .
  ```
- All agents must accept a `ResearchState` and return a `ResearchState`.
- Use `get_logger(__name__)` from `core.logger` — no bare `print()` statements in agent code.

---

## 🧪 Running Tests

```bash
pytest tests/
```

All PRs must pass existing tests. Adding new tests for new functionality is strongly encouraged.

---

## 🛠️ Adding a New Agent

1. Create `agents/your_agent.py` — implement a `your_agent_node(state: ResearchState) -> ResearchState` function.
2. Register the node in [`core/graph.py`](core/graph.py).
3. Update `core/state.py` if the agent needs new state fields.
4. Add tests in `tests/`.
5. Document the agent in the `README.md` architecture table.

---

## 📬 Pull Request Guidelines

- Keep PRs **focused** — one feature or fix per PR.
- Write a clear PR description explaining *what* and *why*.
- Reference any related issue: `Closes #123`.
- Ensure CI passes before requesting review.

---

## 🐛 Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.md). Please include:
- Python version
- Full error traceback
- Steps to reproduce

---

## 💡 Suggesting Features

Use the [Feature Request template](.github/ISSUE_TEMPLATE/feature_request.md).

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
