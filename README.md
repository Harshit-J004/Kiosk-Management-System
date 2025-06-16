# Multi-Store Kiosk Management System
🚀 Overview
This system is an agentic AI-based kiosk manager designed to monitor, diagnose, maintain, and operate kiosks across multiple retail stores. It uses CrewAI agents, LangChain LLMs, and custom store-aware tools for intelligent decision-making and task execution.

🧠 Architecture Overview
sql
Copy
Edit
[ User ]
   │
   ▼
[ Main CLI App (main) ]         ← L0: Supervisor
   │
   ▼
[ EnhancedSupervisor ]          ← L0
   ├─ OperatorApp for store001  ← L1
   ├─ OperatorApp for store002
   ├─ OperatorApp for store003
   └─ OperatorApp (global)

Each OperatorApp:
   ├─ Kiosk Specialist Agent     ← L2
   ├─ Order Specialist Agent     ← L2
   └─ Alert Specialist Agent     ← L2

Each Agent uses:
   └─ Enhanced Tool (_run logic) ← L3 (Tool level)
⚙️ Key Components
🔹 EnhancedSupervisor (L0)
Controls all operator apps per store

Runs monitoring, emergency response, dashboards

Manages crew orchestration for each store

🔹 EnhancedOperatorApp (L1)
Each store gets its own app instance

Builds diagnostic, operations, and maintenance crews

Each crew has 2 agents and 2 tasks run sequentially

🔹 Agents (L2)
Multi-Store Kiosk Specialist

Multi-Store Order Specialist

Multi-Store Alert Specialist

Each agent has:

Role, goal, backstory

Custom llm (e.g., GPT)

Tools: EnhancedKioskTool, EnhancedOrderTool, EnhancedAlertTool

🔹 Tools (L3)
Tool logic triggers directly from Task descriptions:

diagnostic, status, restart → EnhancedKioskTool

process orders, metrics, priority → EnhancedOrderTool

alerts, critical, resolve → EnhancedAlertTool

🧩 Crew + Task Flow
Each Crew is created with:

python
Copy
Edit
Crew(
  agents=[Agent1, Agent2],
  tasks=[
    Task(description="...", expected_output="...", agent=Agent1),
    Task(description="...", expected_output="...", agent=Agent2)
  ],
  process=Process.sequential
)
✅ Agents read the task → understand intent → call tool’s _run(query) method.

🔄 Trigger Flow (Example: Option 3 — Full AI Ops)
text
Copy
Edit
User selects: Run Full AI Operations (Option 3)
↓
main() → EnhancedSupervisor.run_full_operations()
↓
For each store:
  → operator_app.run_diagnostics()
    ↳ crew.kickoff() → agent → tool._run("diagnostic store001")

  → operator_app.run_operations()
    ↳ crew.kickoff() → agent → tool._run("process orders store001")
📊 Built-In Tools & Use Cases
Tool	Sample Queries	Description
EnhancedKioskTool	diagnostic store001, status	Kiosk health checks and restart
EnhancedOrderTool	process orders, revenue report	Order analytics and fulfillment
EnhancedAlertTool	critical alerts, resolve alerts	System-wide alert tracking

🛠️ Setup Instructions
📦 Dependencies
python-dotenv

langchain

langchain-openai

crewai

openai (if using GPT)

ollama (if using local models like mistral)

🔐 API Setup (OpenAI)
env
Copy
Edit
# .env
OPENAI_API_KEY=sk-xxxxxx
In code:

python
Copy
Edit
from dotenv import load_dotenv
load_dotenv()
🧠 LLM Setup Recommendation
For best performance:

python
Copy
Edit
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # or "gpt-4"
    temperature=0.3
)
Avoid Ollama unless you're doing tool-only flow with expected_output disabled.

✅ Feature Summary
🔄 Restart Systems

🧠 AI Diagnostics + Operations

🛠️ Maintenance Scheduling

📊 Quick Status + Revenue Analytics

🚨 Critical Alerts Management

📈 Latency Tracking

👥 Fully Agentic CrewAI Architecture

🧭 Sample Execution Flow
text
Copy
Edit
main()
├── Option 3: Run Full AI Operations
│   └── EnhancedSupervisor.run_full_operations()
│       ├── operator_app.run_diagnostics()
│       │   └── crew.kickoff() → agent → tool._run("diagnostic store001")
│       └── operator_app.run_operations()
│           └── crew.kickoff() → agent → tool._run("process store001")
│
└── Option 5: Quick Status
    └── operator_app.get_quick_status()
        └── calls tool._run() directly (no LLM)
📌 Final Notes
✅ Your code is perfectly modular and scalable

✅ Agent reasoning works best when using OpenAI models + proper expected_output

✅ Tools are powerful — they handle 100% of the store-specific logic

🧠 Future scope: Add memory, database logging, Slack alerts, or vector search
