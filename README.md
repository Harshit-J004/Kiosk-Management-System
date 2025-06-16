# 🧠 Multi-Store Kiosk Management System

An **agentic AI-powered platform** for managing kiosks across multiple stores. This system integrates **CrewAI**, **LangChain**, and modular tools to automate kiosk diagnostics, order operations, alert monitoring, and performance analytics — all driven by LLMs.

---

## 🚀 Features

- ✅ Full multi-store agentic AI architecture
- 📦 Intelligent order processing and revenue analysis
- 🖥️ Kiosk health checks, restarts, and maintenance
- 🚨 Real-time alert monitoring and resolution
- ⏱️ Latency reporting and performance metrics
- 🤖 OpenAI + Ollama-compatible LLM setup
- 📊 CLI interface for managing all stores

---

## 🧠 Architecture Flow

```

User Input (main menu)
│
└──> EnhancedSupervisor (L0)
└── OperatorApp for each store (L1)
├── Kiosk Specialist (L2)
├── Order Specialist (L2)
└── Alert Specialist (L2)
└── Tools: \_run() via EnhancedTool (L3)

```

Each store's agents are executed via sequential `Crew` with `Task` + `expected_output`.

---

## 🧩 Core Components

### 🎯 Supervisor (`EnhancedSupervisor`)
- Manages all stores and initializes operator apps
- Supports full AI ops, quick status, restarts, and monitoring

### 🛠️ Operator App (`EnhancedOperatorApp`)
- Instantiated per store
- Composed of specialized agents
- Three main crews:
  - Diagnostics Crew
  - Operations Crew
  - Maintenance Crew

### 🤖 Agents
- **Multi-Store Kiosk Specialist**
- **Multi-Store Order Specialist**
- **Multi-Store Alert Specialist**

Each uses a tool to execute based on task prompts.

### 🧰 Tools
- `EnhancedKioskTool`: Diagnostics, restart, health
- `EnhancedOrderTool`: Process orders, revenue, priority
- `EnhancedAlertTool`: Monitor, resolve, and log alerts

---

## 📂 Example CLI Options

```

1. View Supervisor Dashboard
2. Run Monitoring Cycle
3. Run Full AI Operations (CrewAI)
4. Restart Store Systems
5. Quick Status
6. Emergency Response
7. Start Continuous Monitoring
8. Stop Continuous Monitoring
9. View Latency Report
10. Exit

````

---

## ⚙️ Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
````

Or manually:

```bash
pip install crewai langchain langchain-openai python-dotenv
```

### 2. LLM Configuration

#### 🔑 OpenAI (Recommended)

Create a `.env` file:

```
OPENAI_API_KEY=sk-xxxxxx
```

Use in Python:

```python
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
```

#### 💡 Ollama (Alternative - Local)

Install Ollama and pull:

```bash
ollama pull llama3
ollama serve
```

In code:

```python
from langchain_community.llms import Ollama
llm = Ollama(model="llama3", base_url="http://localhost:11434")
```

---

## 📈 Performance Tracking

The system logs every operation's:

* Start/End time
* Duration
* Success/failure
* Store ID

Command `View Latency Report` displays these logs in real time.

---

## 📊 Dashboard Overview

Shows:

* Total stores, kiosks, orders
* Online kiosk counts
* Active alerts
* Monitoring status
* Recent latency metrics

---

## 🔮 Future Enhancements

* ✅ Vector DB integration (for memory/log search)
* 🔁 Slack/email alerting
* 📡 Web dashboard (Streamlit or Flask)
* 📊 Real-time charting of performance

---

## 🏁 Author

Built by [Harshit Joshi](https://github.com/yourusername)
A modular, production-ready project demonstrating **agentic AI operations for real-world IoT kiosks**.

---
