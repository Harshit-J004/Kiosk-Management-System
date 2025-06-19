import os
import time
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_packages():
    """Install required packages with error handling"""
    packages = [
        "crewai>=0.28.8",
        "crewai-tools",
        "langchain-community",
        "langchain-openai", 
        "transformers",
        "pydantic>=2.0.0",
        "requests",
        "openai",
        "litellm"
    ]
    
    for package in packages:
        try:
            pkg_name = package.split('>=')[0].split('==')[0].replace('-', '_')
            if pkg_name == 'langchain_community':
                pkg_name = 'langchain_community'
            elif pkg_name == 'langchain_openai':
                pkg_name = 'langchain_openai'
            elif pkg_name == 'crewai_tools':
                pkg_name = 'crewai_tools'
            __import__(pkg_name)
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}: {e}")

install_packages()

try:
    from crewai import Agent, Task, Crew, Process, LLM
    from crewai.tools import BaseTool
    from langchain_community.llms import Ollama
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel, Field
    import requests
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all packages are installed correctly.")
    sys.exit(1)

@dataclass
class KioskInfo:
    store_id: str
    version: str
    battery_level: int
    status: str = "online"
    last_restart: str = "Never"
    issues: List[str] = None
    uptime: float = 0.0

    def __post_init__(self):
        if self.issues is None:
            self.issues = []

@dataclass
class OperationResult:
    success: bool
    message: str
    latency: float
    timestamp: str
    operation_type: str

# Global reference to KMS instance
kms_instance = None

# Tool Input Models
class KioskOperationInput(BaseModel):
    store_id: str = Field(description="Store ID (store_1, store_2, store_3, store_4, store_5)")

class RestartInput(BaseModel):
    store_id: str = Field(description="Store ID to restart")
    confirm: bool = Field(default=True, description="Confirmation for restart")

# Core Tools
class GetKioskStatusTool(BaseTool):
    name: str = "get_kiosk_status"
    description: str = "Get current status and details of a specific kiosk store"
    args_schema: type[BaseModel] = KioskOperationInput
    
    def _run(self, store_id: str) -> str:
        global kms_instance
        if kms_instance is None:
            return "❌ KMS instance not available"
        return kms_instance.get_kiosk_status_impl(store_id)

class RestartKioskTool(BaseTool):
    name: str = "restart_kiosk"
    description: str = "Restart a specific kiosk with pre-checks and validation"
    args_schema: type[BaseModel] = RestartInput
    
    def _run(self, store_id: str, confirm: bool = True) -> str:
        global kms_instance
        if kms_instance is None:
            return "❌ KMS instance not available"
        return kms_instance.restart_kiosk_impl(store_id, confirm)

class CheckAllStoresStatusTool(BaseTool):
    name: str = "check_all_stores_status"
    description: str = "Get comprehensive status of all stores in the network"
    
    def _run(self) -> str:
        global kms_instance
        if kms_instance is None:
            return "❌ KMS instance not available"
        return kms_instance.check_all_stores_status_impl()

class AgenticKioskManagementSystem:
    def __init__(self):
        global kms_instance
        kms_instance = self
        
        # Initialize 5 stores with realistic data
        self.kiosks = {
            "store_1": KioskInfo("Store 1 - Downtown", "v2.1.0", 85, "online", uptime=156.5),
            "store_2": KioskInfo("Store 2 - Mall", "v2.1.0", 67, "online", uptime=89.2, issues=["low_battery_warning"]),
            "store_3": KioskInfo("Store 3 - Airport", "v2.0.9", 92, "online", uptime=234.1),
            "store_4": KioskInfo("Store 4 - Campus", "v2.1.0", 43, "warning", uptime=12.3, issues=["critical_battery"]),
            "store_5": KioskInfo("Store 5 - Highway", "v2.0.8", 78, "maintenance", uptime=0.0, issues=["scheduled_maintenance"])
        }
        
        self.operation_logs = []
        self.llm_available = False
        self.llm = None
        
        # Initialize LLM with robust fallback
        self.llm = self._initialize_robust_llm()
        
        # Setup tools
        self._setup_tools()
        
        # Setup agents only if LLM is available
        if self.llm_available:
            self._setup_agents()
        
        print("🚀 Agentic Kiosk Management System initialized!")
        if self.llm_available:
            print(f"🧠 Using LLM model successfully")
            print("🤖 AI Agents ready for autonomous operations")
        else:
            print("💻 Running in Direct Mode (no AI agents)")
        print("💬 Chat interface ready - Type 'help' for commands")

    def _initialize_robust_llm(self):
        """Initialize LLM with multiple fallback options"""
        
        # Trying with OpenAI first (most reliable with CrewAI)
        try:
            openai_key = os.getenv('OPENAI_API_KEY')
            if openai_key and openai_key.startswith('sk-'):
                print("🔄 Connecting to OpenAI...")
                # Use CrewAI's LLM wrapper for better compatibility
                llm = LLM(
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                    api_key=openai_key,
                    max_tokens=1000,
                    timeout=30
                )
                # Test with a simple call
                try:
                    test_response = llm.call("Hello")
                    if test_response and len(str(test_response)) > 0:
                        print("✅ OpenAI connected successfully")
                        self.llm_available = True
                        return llm
                except Exception as test_e:
                    print(f"⚠️ OpenAI test failed: {test_e}")
        except Exception as e:
            print(f"⚠️ OpenAI setup failed: {e}")

        # Try Ollama with CrewAI's LLM wrapper
        print("🔄 Attempting to connect to Ollama...")
        ollama_models = ["llama3.2", "llama3.1", "mistral", "llama2"]
        
        try:
            # Test if Ollama is running
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code != 200:
                print("⚠️ Ollama server not running")
                raise Exception("Ollama not available")
            
            # Check available models with proper error handling
            try:
                models_data = response.json()
                models_list = models_data.get('models', [])
                
                if not models_list:
                    print("⚠️ No models found in Ollama")
                    raise Exception("No Ollama models available")
                
                model_names = [m.get('name', '') for m in models_list if isinstance(m, dict) and 'name' in m]
                
                if not model_names:
                    print("⚠️ No valid model names found in Ollama response")
                    raise Exception("Invalid Ollama models data")
                
                print(f"📋 Available Ollama models: {model_names}")
                
            except (ValueError, KeyError, TypeError) as e:
                print(f"⚠️ Error parsing Ollama models response: {e}")
                raise Exception("Failed to parse Ollama models")
            
            for model in ollama_models:
                # Find matching model
                matching_model = None
                for available_model in model_names:
                    if model in available_model:
                        matching_model = available_model
                        break
                
                if not matching_model:
                    print(f"⚠️ Model {model} not found in Ollama")
                    continue
                
                try:
                    print(f"🔄 Testing CrewAI LLM with: {matching_model}")
                    
                    # Use CrewAI's LLM wrapper with proper Ollama format
                    llm = LLM(
                        model=f"ollama/{matching_model}",
                        base_url="http://localhost:11434",
                        temperature=0.1,
                        timeout=30
                    )
                    
                    # Test the connection
                    test_response = llm.call("Hello")
                    if test_response and len(str(test_response)) > 2:
                        print(f"✅ Ollama connected with {matching_model}")
                        self.llm_available = True
                        return llm
                        
                except Exception as e:
                    print(f"⚠️ Ollama {matching_model} test failed: {e}")
                    continue
                    
        except Exception as e:
            print(f"⚠️ Ollama connection failed: {e}")
        
        # Try using langchain wrapper as fallback with better error handling
        print("🔄 Trying LangChain Ollama wrapper...")
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                try:
                    models_data = response.json()
                    models_list = models_data.get('models', [])
                    
                    if models_list and len(models_list) > 0:
                        # Safely get the first model
                        first_model = models_list[0]
                        if isinstance(first_model, dict) and 'name' in first_model:
                            model_name = first_model['name']
                            print(f"🔄 Using LangChain with {model_name}")
                            
                            langchain_llm = Ollama(
                                model=model_name,
                                base_url="http://localhost:11434",
                                timeout=30,
                                temperature=0.1
                            )
                            
                            # Test the connection
                            test_response = langchain_llm.invoke("Hello")
                            if test_response and len(str(test_response)) > 2:
                                print(f"✅ LangChain Ollama connected with {model_name}")
                                self.llm_available = True
                                return langchain_llm
                        else:
                            print("⚠️ Invalid model data structure in Ollama response")
                    else:
                        print("⚠️ No models available in Ollama")
                        
                except (ValueError, KeyError, TypeError, IndexError) as e:
                    print(f"⚠️ Error processing Ollama models list: {e}")
                    
        except Exception as e:
            print(f"⚠️ LangChain Ollama failed: {e}")
        
        # No LLM available - use direct mode
        print("🔄 No LLM available - running in Direct Mode")
        self.llm_available = False
        return None

    def _setup_tools(self):
        """Setup core tools"""
        self.tools = [
            GetKioskStatusTool(),
            RestartKioskTool(),
            CheckAllStoresStatusTool()
        ]

    def _setup_agents(self):
        """Setup hierarchical agent structure"""
        if not self.llm_available or not self.llm:
            return
            
        try:
            # L0 - Supervisor Agent
            self.supervisor_agent = Agent(
                role='Operations Supervisor',
                goal='Coordinate and oversee all kiosk operations across 5 stores with strategic decision making',
                backstory="""You are the strategic supervisor managing 5 retail kiosk locations:
                - Store 1 (Downtown): High-traffic business district
                - Store 2 (Mall): Retail center with peak hours  
                - Store 3 (Airport): 24/7 critical operations
                - Store 4 (Campus): University location
                - Store 5 (Highway): Rest stop location
                
                You make strategic decisions, coordinate operations, and ensure business continuity.""",
                verbose=True,
                llm=self.llm,
                allow_delegation=True,
                max_iter=2,
                max_execution_time=30
            )
            
            # L1 - Operator Agent  
            self.operator_agent = Agent(
                role='Operations Manager',
                goal='Execute operational tasks and coordinate with kiosk systems for optimal performance',
                backstory="""You are the operational manager who bridges strategic decisions with 
                system execution. You understand both business requirements and technical constraints.
                You validate operations, ensure safety protocols, and coordinate with field systems.""",
                verbose=True,
                llm=self.llm,
                allow_delegation=True,
                max_iter=2,
                max_execution_time=30
            )
            
            # L2 - Kiosk Agent
            self.kiosk_agent = Agent(
                role='Kiosk System Controller',
                goal='Execute direct system operations, diagnostics, and maintenance tasks',
                backstory="""You are the hands-on system controller with direct access to kiosk hardware 
                and software. You perform actual system operations, run diagnostics, execute restarts,
                and monitor system health in real-time.""",
                tools=self.tools,
                verbose=True,
                llm=self.llm,
                max_iter=3,
                max_execution_time=45
            )
            
            print("✅ AI Agents initialized successfully")
            
        except Exception as e:
            print(f"⚠️ Agent setup failed: {e}")
            self.llm_available = False

    # Implementation methods
    def get_kiosk_status_impl(self, store_id: str) -> str:
        """Get detailed kiosk status"""
        start_time = time.time()
        
        store_key = store_id.lower().replace(" ", "_")
        if store_key not in self.kiosks:
            return f"❌ Store {store_id} not found. Available stores: store_1, store_2, store_3, store_4, store_5"
        
        kiosk = self.kiosks[store_key]
        latency = time.time() - start_time
        
        # Health score calculation
        health_score = 100
        if kiosk.battery_level < 50:
            health_score -= 30
        if kiosk.issues:
            health_score -= len(kiosk.issues) * 15
        if kiosk.status != "online":
            health_score -= 20
        
        health_score = max(0, health_score)
        
        result = f"""
📊 STATUS REPORT: {kiosk.store_id}
{'='*40}
🔋 Battery: {kiosk.battery_level}%
📱 Version: {kiosk.version}
⚡ Status: {kiosk.status.upper()}
⏰ Uptime: {kiosk.uptime:.1f} hours
🏥 Health Score: {health_score}/100

⚠️ Issues ({len(kiosk.issues)}):
{chr(10).join([f'  • {issue}' for issue in kiosk.issues]) if kiosk.issues else '  ✅ No issues detected'}

🕐 Last Restart: {kiosk.last_restart}
⚡ Query Time: {latency:.3f}s
        """
        
        self._log_operation("status_check", store_id, True, "Status retrieved", latency)
        return result

    def restart_kiosk_impl(self, store_id: str, confirm: bool = True) -> str:
        """Execute kiosk restart with validation"""
        start_time = time.time()
        
        store_key = store_id.lower().replace(" ", "_")
        if store_key not in self.kiosks:
            return f"❌ Store {store_id} not found. Available stores: store_1, store_2, store_3, store_4, store_5"
        
        kiosk = self.kiosks[store_key]
        
        print(f"🔄 Initiating restart sequence for {kiosk.store_id}...")
        
        # Restart sequence with timing
        steps = [
            ("🔍 Pre-restart validation", 0.5),
            ("💾 Saving system state", 0.3),
            ("🛑 Graceful shutdown", 0.4),
            ("⚡ Hardware restart", 1.2),
            ("🚀 System initialization", 0.8),
            ("✅ Post-restart validation", 0.4)
        ]
        
        for step, duration in steps:
            print(f"  {step}...")
            time.sleep(duration)
        
        # Update kiosk state
        kiosk.status = "online"
        kiosk.last_restart = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        kiosk.uptime = 0.0
        kiosk.issues = [] 
        
        latency = time.time() - start_time
        
        result = f"""
🚀 RESTART COMPLETED: {kiosk.store_id}
{'='*40}
✅ Status: ONLINE
🔋 Battery: {kiosk.battery_level}%
📅 Restart Time: {kiosk.last_restart}
⏱️ Total Time: {latency:.2f}s
🎯 Success: All systems operational
        """
        
        self._log_operation("restart", store_id, True, "Restart completed successfully", latency)
        return result

    def check_all_stores_status_impl(self) -> str:
        """Get status of all stores"""
        start_time = time.time()
        
        print("📊 Gathering status from all stores...")
        
        results = []
        online_count = 0
        
        for store_key, kiosk in self.kiosks.items():
            if kiosk.status == "online":
                online_count += 1
            
            status_icon = "🟢" if kiosk.status == "online" else "🟡" if kiosk.status == "warning" else "🔴"
            results.append(f"{status_icon} {kiosk.store_id}: {kiosk.status.upper()} | Battery: {kiosk.battery_level}% | Issues: {len(kiosk.issues)}")
        
        latency = time.time() - start_time
        
        result = f"""
🏪 ALL STORES STATUS REPORT
{'='*50}
📊 Network Overview: {online_count}/5 stores online ({online_count/5*100:.0f}%)

{chr(10).join(results)}

⏱️ Report generated in {latency:.3f}s
🕐 Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """
        
        self._log_operation("all_status", "all_stores", True, "Network status retrieved", latency)
        return result

    def _log_operation(self, operation_type: str, target: str, success: bool, message: str, latency: float):
        """Log operation for tracking"""
        log_entry = OperationResult(
            success=success,
            message=message,
            latency=latency,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            operation_type=operation_type
        )
        self.operation_logs.append(log_entry)

    def create_crew_for_task(self, task_description: str, operation_type: str):
        """Create a crew dynamically for specific tasks"""
        if not self.llm_available or not self.llm:
            return None
            
        try:
            # Creating task based on operation type
            if operation_type == "restart":
                task = Task(
                    description=task_description,
                    expected_output="Detailed restart completion report with timing and status",
                    agent=self.kiosk_agent
                )
            elif operation_type == "status":
                task = Task(
                    description=task_description,
                    expected_output="Detailed status report with health metrics",
                    agent=self.kiosk_agent
                )
            elif operation_type == "all_status":
                task = Task(
                    description=task_description,
                    expected_output="Complete network overview with all store statuses",
                    agent=self.supervisor_agent
                )
            else:
                return None
            
            # Createing crew with proper configuration
            crew = Crew(
                agents=[self.supervisor_agent, self.operator_agent, self.kiosk_agent],
                tasks=[task],
                process=Process.hierarchical,
                manager_llm=self.llm,
                verbose=True,
                max_execution_time=90,
                memory=False,  
                cache=False   
            )
            
            return crew
            
        except Exception as e:
            print(f"⚠️ Crew creation failed: {e}")
            logger.error(f"Crew creation error: {e}", exc_info=True)
            return None

    def chat_interface(self):
        """Interactive chat interface"""
        mode = "🤖 AI Agents Mode" if self.llm_available else "💻 Direct Mode"
        
        print(f"""
🤖 AGENTIC KIOSK MANAGEMENT SYSTEM
=====================================
Mode: {mode}
Available Commands:
• "restart store_1" - Restart specific store
• "status store_2" - Get store status  
• "check all stores" - Network overview
• "show logs" - View operation history
• "help" - Show this help
• "quit" - Exit system

Stores: store_1, store_2, store_3, store_4, store_5
        """)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print(f"""
Available Commands:
• restart [store_id] - Restart kiosk
• status [store_id] - Get kiosk status
• check all stores - All stores status
• show logs - View operation history
• quit - Exit

Current Mode: {mode}
                    """)
                    continue
                
                if user_input.lower() == 'show logs':
                    self._show_operation_logs()
                    continue
                
                # Processingg user input
                response = self.process_user_request(user_input)
                print(f"\n🤖 System: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logger.error(f"Chat interface error: {e}", exc_info=True)

    def _show_operation_logs(self):
        """Show recent operation logs"""
        if not self.operation_logs:
            print("📋 No operations logged yet")
            return
        
        print("\n📋 Recent Operations:")
        print("=" * 50)
        
        for log in self.operation_logs[-10:]:  # Show last 10 operations
            status = "✅" if log.success else "❌"
            print(f"{status} {log.timestamp} | {log.operation_type.upper()} | {log.message} | {log.latency:.3f}s")

    def process_user_request(self, user_input: str) -> str:
        """Process user request through agentic system or direct execution"""
        start_time = time.time()
        
        print(f"\n🧠 Processing: '{user_input}'")
        
        user_input_lower = user_input.lower()
        
        # Routing to appropriate handler
        if "restart" in user_input_lower:
            return self._handle_restart_request(user_input)
        elif "status" in user_input_lower and "all" not in user_input_lower:
            return self._handle_status_request(user_input)
        elif "all" in user_input_lower or "network" in user_input_lower:
            return self._handle_all_status_request()
        else:
            return self._show_help_message()

    def _handle_restart_request(self, user_input: str) -> str:
        """Handle restart requests"""
        store_id = self._extract_store_id(user_input)
        if not store_id:
            return "❌ Please specify which store to restart (e.g., 'restart store_1')"
        
        print(f"🎯 Operation: RESTART {store_id}")
        
        if self.llm_available:
            try:
                print("🤖 Using AI Agents for autonomous execution...")
                crew = self.create_crew_for_task(
                    f"Execute restart operation for {store_id} with full validation and monitoring",
                    "restart"
                )
                
                if crew:
                    result = crew.kickoff()
                    print("✅ AI Agent task completed")
                    return str(result)
                else:
                    print("🔄 Falling back to direct execution...")
                    return self.restart_kiosk_impl(store_id)
                    
            except Exception as e:
                print(f"🔄 AI execution failed, using direct mode: {e}")
                logger.error(f"AI restart execution error: {e}", exc_info=True)
                return self.restart_kiosk_impl(store_id)
        else:
            print("💻 Using direct execution...")
            return self.restart_kiosk_impl(store_id)

    def _handle_status_request(self, user_input: str) -> str:
        """Handle status requests"""
        store_id = self._extract_store_id(user_input)
        if not store_id:
            return "❌ Please specify which store to check (e.g., 'status store_1')"
        
        print(f"🎯 Operation: STATUS CHECK {store_id}")
        
        if self.llm_available:
            try:
                print("🤖 Using AI Agents for system analysis...")
                crew = self.create_crew_for_task(
                    f"Get comprehensive status report for {store_id}",
                    "status"
                )
                
                if crew:
                    result = crew.kickoff()
                    print("✅ AI Agent analysis completed")
                    return str(result)
                else:
                    print("🔄 Falling back to direct execution...")
                    return self.get_kiosk_status_impl(store_id)
                    
            except Exception as e:
                print(f"🔄 AI execution failed, using direct mode: {e}")
                logger.error(f"AI status execution error: {e}", exc_info=True)
                return self.get_kiosk_status_impl(store_id)
        else:
            print("💻 Using direct execution...")
            return self.get_kiosk_status_impl(store_id)

    def _handle_all_status_request(self) -> str:
        """Handle all stores status requests"""
        print("🎯 Operation: NETWORK STATUS CHECK")
        
        if self.llm_available:
            try:
                print("🤖 Using AI Agents for network analysis...")
                crew = self.create_crew_for_task(
                    "Generate comprehensive network status report for all stores",
                    "all_status"
                )
                
                if crew:
                    result = crew.kickoff()
                    print("✅ AI Agent network analysis completed")
                    return str(result)
                else:
                    print("🔄 Falling back to direct execution...")
                    return self.check_all_stores_status_impl()
                    
            except Exception as e:
                print(f"🔄 AI execution failed, using direct mode: {e}")
                logger.error(f"AI all status execution error: {e}", exc_info=True)
                return self.check_all_stores_status_impl()
        else:
            print("💻 Using direct execution...")
            return self.check_all_stores_status_impl()

    def _show_help_message(self) -> str:
        """Show help message"""
        return """❓ I understand you want to manage kiosks. Please specify:
• "restart store_X" - to restart a specific store
• "status store_X" - to check store status
• "check all stores" - for network overview
• "show logs" - to view operation history

Available stores: store_1, store_2, store_3, store_4, store_5"""

    def _extract_store_id(self, text: str) -> Optional[str]:
        """Extract store ID from user input"""
        text_lower = text.lower()
        for i in range(1, 6):
            if f"store_{i}" in text_lower or f"store {i}" in text_lower:
                return f"store_{i}"
        return None

if __name__ == "__main__":
    try:
        print("🚀 Starting Agentic Kiosk Management System...")
        kms = AgenticKioskManagementSystem()
        kms.chat_interface()
    except KeyboardInterrupt:
        print("\n👋 System shutdown")
    except Exception as e:
        print(f"❌ System error: {e}")
        logger.error(f"System error: {e}", exc_info=True)