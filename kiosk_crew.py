import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import re
from langchain_community.llms import Ollama

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

# Set up Ollama as the LLM (free alternative)
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_MODEL_NAME"] = "llama3:instruct"  
os.environ["OPENAI_API_KEY"] = "ollama" 
os.environ["OPENAI_TIMEOUT"] = "60"

llm = Ollama(
    model="llama3:instruct",
    base_url="http://localhost:11434",
    temperature=0.2
)

# ============================================================================
# DATA MODELS
# ============================================================================

class KioskStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"

class OrderStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class OperationType(Enum):
    RESTART = "restart"
    DIAGNOSTIC = "diagnostic"
    MAINTENANCE = "maintenance"
    STATUS_CHECK = "status_check"
    ORDER_PROCESS = "order_process"

@dataclass
class Store:
    id: str
    name: str
    location: str
    manager: str
    phone: str
    status: str = "active"

@dataclass
class Kiosk:
    id: str
    name: str
    location: str
    store_id: str
    status: KioskStatus
    last_seen: str
    error_count: int = 0
    uptime: str = "99.5%"

@dataclass
class Order:
    id: str
    kiosk_id: str
    store_id: str
    customer: str
    items: str
    total: float
    status: OrderStatus
    priority: Priority = Priority.MEDIUM
    created_at: str = None
    estimated_completion: str = None

@dataclass
class Alert:
    id: str
    level: Priority
    message: str
    source: str
    store_id: str
    timestamp: str
    resolved: bool = False

@dataclass
class OperationReport:
    timestamp: str
    operation_type: str
    store_id: str
    duration_ms: float
    success: bool
    details: str
    recommendations: List[str] = field(default_factory=list)

@dataclass
class LatencyMetrics:
    operation_type: str
    start_time: float
    end_time: float
    duration_ms: float
    store_id: str
    success: bool

# ============================================================================
# GLOBAL DATA STORAGE - MULTI-STORE DATA
# ============================================================================

# Store definitions
STORES_DATA = {
    "store001": Store("store001", "Downtown Mall", "123 Main St", "John Manager", "+1-555-0101"),
    "store002": Store("store002", "Airport Terminal", "Airport Blvd", "Sarah Johnson", "+1-555-0102"),
    "store003": Store("store003", "Shopping Center", "456 Oak Ave", "Mike Wilson", "+1-555-0103"),
    "store004": Store("store004", "University Campus", "789 College Rd", "Lisa Chen", "+1-555-0104"),
    "store005": Store("store005", "Business District", "321 Corporate Way", "David Kim", "+1-555-0105")
}

# Kiosks per store
KIOSKS_DATA = {
    # Store 1 - Downtown Mall
    "k001": Kiosk("k001", "Main Entrance", "Front Lobby", "store001", KioskStatus.ONLINE, "2 min ago", 0),
    "k002": Kiosk("k002", "Electronics Section", "Aisle 5", "store001", KioskStatus.ONLINE, "1 min ago", 1),
    "k003": Kiosk("k003", "Customer Service", "Help Desk", "store001", KioskStatus.MAINTENANCE, "30 min ago", 3),
    
    # Store 2 - Airport Terminal
    "k004": Kiosk("k004", "Gate A", "Terminal A", "store002", KioskStatus.ONLINE, "5 min ago", 0),
    "k005": Kiosk("k005", "Gate B", "Terminal B", "store002", KioskStatus.OFFLINE, "45 min ago", 5),
    
    # Store 3 - Shopping Center
    "k006": Kiosk("k006", "Food Court", "Level 2", "store003", KioskStatus.ONLINE, "3 min ago", 1),
    "k007": Kiosk("k007", "Parking Level", "Ground Floor", "store003", KioskStatus.ERROR, "15 min ago", 7),
    
    # Store 4 - University Campus
    "k008": Kiosk("k008", "Library", "Main Building", "store004", KioskStatus.ONLINE, "1 min ago", 0),
    "k009": Kiosk("k009", "Student Center", "Campus Center", "store004", KioskStatus.ONLINE, "4 min ago", 2),
    
    # Store 5 - Business District
    "k010": Kiosk("k010", "Lobby", "Building Entrance", "store005", KioskStatus.MAINTENANCE, "20 min ago", 4),
    "k011": Kiosk("k011", "Cafeteria", "Floor 15", "store005", KioskStatus.ONLINE, "2 min ago", 1)
}

# Orders per store
ORDERS_DATA = {
    "ord001": Order("ord001", "k001", "store001", "John Doe", "Laptop, Wireless Mouse", 1250.99, OrderStatus.READY, Priority.HIGH, "10:30 AM", "11:00 AM"),
    "ord002": Order("ord002", "k002", "store001", "Jane Smith", "Phone Case, Screen Protector", 29.99, OrderStatus.PROCESSING, Priority.MEDIUM, "10:45 AM", "11:15 AM"),
    "ord003": Order("ord003", "k004", "store002", "Bob Wilson", "Bluetooth Headphones", 199.99, OrderStatus.COMPLETED, Priority.LOW, "09:30 AM", "10:00 AM"),
    "ord004": Order("ord004", "k006", "store003", "Alice Brown", "Tablet, Stylus", 499.99, OrderStatus.PENDING, Priority.HIGH, "11:00 AM", "11:30 AM"),
    "ord005": Order("ord005", "k008", "store004", "Tom Green", "Smart Watch", 299.99, OrderStatus.PROCESSING, Priority.MEDIUM, "10:15 AM", "10:45 AM")
}

# Alerts per store
ALERTS_DATA = [
    Alert("alert001", Priority.HIGH, "Kiosk k005 has been offline for 45 minutes", "KioskMonitor", "store002", datetime.now().strftime("%H:%M")),
    Alert("alert002", Priority.CRITICAL, "Kiosk k007 experiencing critical errors", "KioskMonitor", "store003", datetime.now().strftime("%H:%M")),
    Alert("alert003", Priority.MEDIUM, "High order volume detected at Downtown Mall", "OrderMonitor", "store001", (datetime.now() - timedelta(minutes=10)).strftime("%H:%M")),
    Alert("alert004", Priority.LOW, "Scheduled maintenance due for Business District lobby", "MaintenanceScheduler", "store005", (datetime.now() - timedelta(minutes=30)).strftime("%H:%M"))
]

# Latency tracking
LATENCY_HISTORY = []

# ============================================================================
# LATENCY TRACKING UTILITIES
# ============================================================================

class LatencyTracker:
    def __init__(self, operation_type: str, store_id: str = "unknown"):
        self.operation_type = operation_type
        self.store_id = store_id
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time() * 1000  # Convert to milliseconds
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time() * 1000
        duration_ms = end_time - self.start_time
        success = exc_type is None
        
        metrics = LatencyMetrics(
            operation_type=self.operation_type,
            start_time=self.start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            store_id=self.store_id,
            success=success
        )
        
        LATENCY_HISTORY.append(metrics)
        
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"⏱️  LATENCY: {self.operation_type} | {duration_ms:.2f}ms | {status} | Store: {self.store_id}")

def get_latency_report(store_id: str = None, limit: int = 10) -> str:
    """Generate latency report for operations"""
    relevant_metrics = LATENCY_HISTORY
    if store_id:
        relevant_metrics = [m for m in LATENCY_HISTORY if m.store_id == store_id]
    
    if not relevant_metrics:
        return "📊 No latency data available"
    
    recent_metrics = relevant_metrics[-limit:]
    
    report = f"📊 LATENCY REPORT (Last {len(recent_metrics)} operations)\n"
    report += "=" * 50 + "\n"
    
    total_time = sum(m.duration_ms for m in recent_metrics)
    avg_time = total_time / len(recent_metrics)
    success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics) * 100
    
    report += f"Average Latency: {avg_time:.2f}ms\n"
    report += f"Success Rate: {success_rate:.1f}%\n"
    report += f"Total Operations: {len(recent_metrics)}\n\n"
    
    report += "Recent Operations:\n"
    for metric in recent_metrics[-5:]:
        status = "✅" if metric.success else "❌"
        report += f"  {status} {metric.operation_type}: {metric.duration_ms:.2f}ms (Store: {metric.store_id})\n"
    
    return report

# ============================================================================
# STORE IDENTIFICATION UTILITIES
# ============================================================================

def extract_store_from_command(command: str) -> Optional[str]:
    """Extract store information from user command"""
    command_lower = command.lower()
    
    # Check for store ID patterns
    store_id_match = re.search(r'store(\d{3})', command_lower)
    if store_id_match:
        store_id = f"store{store_id_match.group(1)}"
        if store_id in STORES_DATA:
            return store_id
    
    # Check for store names
    for store_id, store in STORES_DATA.items():
        store_keywords = [
            store.name.lower(),
            store.location.lower(),
            store_id.lower()
        ]
        
        # Additional keywords for easier identification
        if "downtown" in store.name.lower() and ("downtown" in command_lower or "mall" in command_lower):
            return store_id
        elif "airport" in store.name.lower() and "airport" in command_lower:
            return store_id
        elif "shopping" in store.name.lower() and ("shopping" in command_lower or "center" in command_lower):
            return store_id
        elif "university" in store.name.lower() and ("university" in command_lower or "campus" in command_lower):
            return store_id
        elif "business" in store.name.lower() and ("business" in command_lower or "corporate" in command_lower):
            return store_id
    
    return None

def list_available_stores() -> str:
    """Return formatted list of available stores"""
    store_list = "🏪 AVAILABLE STORES:\n"
    store_list += "=" * 30 + "\n"
    
    for store_id, store in STORES_DATA.items():
        kiosk_count = len([k for k in KIOSKS_DATA.values() if k.store_id == store_id])
        store_list += f"{store_id}: {store.name}\n"
        store_list += f"   📍 Location: {store.location}\n"
        store_list += f"   👤 Manager: {store.manager}\n"
        store_list += f"   🖥️  Kiosks: {kiosk_count}\n"
        store_list += f"   📞 Phone: {store.phone}\n\n"
    
    return store_list

# ============================================================================
# L2 LEVEL - ENHANCED SPECIALIZED TOOLS
# ============================================================================

class EnhancedKioskTool(BaseTool):
    name: str = "enhanced_kiosk_manager"
    description: str = "Advanced multi-store kiosk operations and diagnostics"

    def _run(self, query: str) -> str:
        with LatencyTracker("kiosk_operation", self._extract_store_from_query(query)):
            return self._process_kiosk_query(query)
    
    def _extract_store_from_query(self, query: str) -> str:
        store_id = extract_store_from_command(query)
        return store_id if store_id else "unknown"
    
    def _process_kiosk_query(self, query: str) -> str:
        query_lower = query.lower()
        store_id = extract_store_from_command(query)
        
        if "restart" in query_lower:
            return self._restart_kiosk(query, store_id)
        elif "diagnostic" in query_lower or "health" in query_lower:
            return self._run_diagnostics(store_id)
        elif "maintenance" in query_lower:
            return self._schedule_maintenance(store_id)
        elif "status" in query_lower:
            return self._get_detailed_status(store_id)
        elif "performance" in query_lower:
            return self._get_performance_metrics(store_id)
        else:
            return self._get_detailed_status(store_id)

    def _restart_kiosk(self, query: str, store_id: str) -> str:
        if not store_id:
            return ("🔄 RESTART REQUEST RECEIVED\n"
                   "❓ Which store would you like to restart systems for?\n\n" + 
                   list_available_stores() + 
                   "\nPlease specify the store (e.g., 'restart store001' or 'restart downtown mall')")
        
        # Find kiosks in the specified store
        store_kiosks = [k for k in KIOSKS_DATA.values() if k.store_id == store_id]
        
        if not store_kiosks:
            return f"❌ No kiosks found for store {store_id}"
        
        store_name = STORES_DATA[store_id].name
        restarted_kiosks = []
        
        # Restart all kiosks in the store or specific kiosk if mentioned
        for kiosk in store_kiosks:
            if kiosk.status in [KioskStatus.OFFLINE, KioskStatus.ERROR, KioskStatus.MAINTENANCE]:
                kiosk.status = KioskStatus.ONLINE
                kiosk.last_seen = "now"
                kiosk.error_count = 0
                restarted_kiosks.append(kiosk.name)
        
        if restarted_kiosks:
            return (f"✅ RESTART COMPLETED - {store_name}\n"
                   f"🔄 Restarted kiosks: {', '.join(restarted_kiosks)}\n"
                   f"📊 All kiosks now ONLINE")
        else:
            return f"ℹ️  All kiosks at {store_name} are already running normally"

    def _run_diagnostics(self, store_id: str) -> str:
        if not store_id:
            return ("🔍 DIAGNOSTIC REQUEST RECEIVED\n"
                   "❓ Which store would you like to run diagnostics for?\n\n" + 
                   list_available_stores())
        
        store_kiosks = [k for k in KIOSKS_DATA.values() if k.store_id == store_id]
        store_name = STORES_DATA[store_id].name
        
        report = f"🔍 DIAGNOSTIC REPORT - {store_name}\n"
        report += "=" * 50 + "\n"
        
        issues = []
        healthy_count = 0
        
        for kiosk in store_kiosks:
            if kiosk.status == KioskStatus.OFFLINE:
                issues.append(f"❌ {kiosk.name}: OFFLINE for {kiosk.last_seen}")
            elif kiosk.error_count > 2:
                issues.append(f"⚠️ {kiosk.name}: High error count ({kiosk.error_count})")
            elif kiosk.status == KioskStatus.MAINTENANCE:
                issues.append(f"🔧 {kiosk.name}: Under maintenance")
            elif kiosk.status == KioskStatus.ERROR:
                issues.append(f"🔴 {kiosk.name}: System error detected")
            else:
                healthy_count += 1
        
        report += f"📊 SUMMARY: {healthy_count}/{len(store_kiosks)} kiosks healthy\n\n"
        
        if issues:
            report += "🚨 ISSUES DETECTED:\n"
            for issue in issues:
                report += f"   {issue}\n"
        else:
            report += "✅ All kiosks running normally\n"
        
        report += f"\n📈 RECOMMENDATIONS:\n"
        if issues:
            report += "   • Address offline/error kiosks immediately\n"
            report += "   • Schedule maintenance for high-error kiosks\n"
        report += "   • Monitor performance during peak hours\n"
        report += f"   • Contact store manager: {STORES_DATA[store_id].manager}\n"
        
        return report

    def _get_detailed_status(self, store_id: str) -> str:
        if not store_id:
            # Return status for all stores
            report = "🖥️ MULTI-STORE KIOSK STATUS:\n"
            report += "=" * 50 + "\n"
            
            for sid, store in STORES_DATA.items():
                store_kiosks = [k for k in KIOSKS_DATA.values() if k.store_id == sid]
                online_count = sum(1 for k in store_kiosks if k.status == KioskStatus.ONLINE)
                
                report += f"🏪 {store.name} ({sid})\n"
                report += f"   Status: {online_count}/{len(store_kiosks)} online\n"
                report += f"   Manager: {store.manager}\n\n"
            
            return report
        
        # Return detailed status for specific store
        store_kiosks = [k for k in KIOSKS_DATA.values() if k.store_id == store_id]
        store_name = STORES_DATA[store_id].name
        
        report = f"🖥️ DETAILED STATUS - {store_name}\n"
        report += "=" * 50 + "\n"
        
        online_count = sum(1 for k in store_kiosks if k.status == KioskStatus.ONLINE)
        
        report += f"📊 OVERVIEW: {online_count}/{len(store_kiosks)} kiosks operational\n\n"
        
        for kiosk in store_kiosks:
            icon = self._get_status_icon(kiosk.status)
            report += f"{icon} {kiosk.name} ({kiosk.id})\n"
            report += f"   Location: {kiosk.location}\n"
            report += f"   Status: {kiosk.status.value.upper()}\n"
            report += f"   Last Seen: {kiosk.last_seen}\n"
            report += f"   Uptime: {kiosk.uptime}\n"
            report += f"   Errors: {kiosk.error_count}\n\n"
        
        return report

    def _schedule_maintenance(self, store_id: str) -> str:
        if not store_id:
            return ("🔧 MAINTENANCE REQUEST RECEIVED\n"
                   "❓ Which store would you like to schedule maintenance for?\n\n" + 
                   list_available_stores())
        
        store_kiosks = [k for k in KIOSKS_DATA.values() if k.store_id == store_id]
        store_name = STORES_DATA[store_id].name
        
        maintenance_scheduled = []
        for kiosk in store_kiosks:
            if kiosk.error_count > 2 or kiosk.status == KioskStatus.ERROR:
                kiosk.status = KioskStatus.MAINTENANCE
                maintenance_scheduled.append(kiosk.name)
        
        if maintenance_scheduled:
            return (f"🔧 MAINTENANCE SCHEDULED - {store_name}\n"
                   f"🛠️ Kiosks: {', '.join(maintenance_scheduled)}\n"
                   f"👤 Contact: {STORES_DATA[store_id].manager}")
        return f"✅ No kiosks require immediate maintenance at {store_name}"

    def _get_performance_metrics(self, store_id: str) -> str:
        if not store_id:
            return ("📊 PERFORMANCE REQUEST RECEIVED\n"
                   "❓ Which store would you like performance metrics for?\n\n" + 
                   list_available_stores())
        
        store_kiosks = [k for k in KIOSKS_DATA.values() if k.store_id == store_id]
        store_name = STORES_DATA[store_id].name
        
        if not store_kiosks:
            return f"❌ No kiosks found for {store_name}"
        
        report = f"📊 PERFORMANCE METRICS - {store_name}\n"
        report += "=" * 40 + "\n"
        
        total_uptime = sum(float(k.uptime.rstrip('%')) for k in store_kiosks) / len(store_kiosks)
        total_errors = sum(k.error_count for k in store_kiosks)
        
        best_kiosk = min(store_kiosks, key=lambda k: k.error_count)
        worst_kiosk = max(store_kiosks, key=lambda k: k.error_count)
        
        report += f"Average Uptime: {total_uptime:.1f}%\n"
        report += f"Total Errors: {total_errors}\n"
        report += f"Best Performer: {best_kiosk.name} ({best_kiosk.error_count} errors)\n"
        report += f"Needs Attention: {worst_kiosk.name} ({worst_kiosk.error_count} errors)\n"
        report += f"Store Manager: {STORES_DATA[store_id].manager}\n"
        
        return report

    def _get_status_icon(self, status: KioskStatus) -> str:
        icons = {
            KioskStatus.ONLINE: "🟢",
            KioskStatus.OFFLINE: "🔴", 
            KioskStatus.MAINTENANCE: "🟡",
            KioskStatus.ERROR: "🔴"
        }
        return icons.get(status, "⚪")

class EnhancedOrderTool(BaseTool):
    name: str = "enhanced_order_manager"
    description: str = "Advanced multi-store order processing and customer management"

    def _run(self, query: str) -> str:
        with LatencyTracker("order_operation", self._extract_store_from_query(query)):
            return self._process_order_query(query)
    
    def _extract_store_from_query(self, query: str) -> str:
        store_id = extract_store_from_command(query)
        return store_id if store_id else "unknown"
    
    def _process_order_query(self, query: str) -> str:
        query_lower = query.lower()
        store_id = extract_store_from_command(query)
        
        if "analytics" in query_lower or "metrics" in query_lower:
            return self._get_order_analytics(store_id)
        elif "priority" in query_lower or "urgent" in query_lower:
            return self._get_priority_orders(store_id)
        elif "process" in query_lower:
            return self._process_orders(store_id)
        elif "pending" in query_lower:
            return self._get_pending_orders(store_id)
        elif "revenue" in query_lower:
            return self._get_revenue_report(store_id)
        else:
            return self._get_comprehensive_report(store_id)

    def _get_comprehensive_report(self, store_id: str) -> str:
        if not store_id:
            # Multi-store summary
            report = "📦 MULTI-STORE ORDER SUMMARY:\n"
            report += "=" * 50 + "\n"
            
            for sid, store in STORES_DATA.items():
                store_orders = [o for o in ORDERS_DATA.values() if o.store_id == sid]
                store_revenue = sum(o.total for o in store_orders if o.status != OrderStatus.CANCELLED)
                
                report += f"🏪 {store.name}:\n"
                report += f"   Orders: {len(store_orders)} | Revenue: ${store_revenue:.2f}\n\n"
            
            return report
        
        # Single store detailed report
        store_orders = [o for o in ORDERS_DATA.values() if o.store_id == store_id]
        store_name = STORES_DATA[store_id].name
        
        report = f"📦 ORDER REPORT - {store_name}\n"
        report += "=" * 50 + "\n"
        
        # Summary stats
        status_counts = {}
        total_revenue = 0
        
        for order in store_orders:
            status_counts[order.status] = status_counts.get(order.status, 0) + 1
            if order.status != OrderStatus.CANCELLED:
                total_revenue += order.total
        
        report += f"📊 SUMMARY:\n"
        report += f"   Total Orders: {len(store_orders)}\n"
        report += f"   Total Revenue: ${total_revenue:.2f}\n"
        report += f"   Pending: {status_counts.get(OrderStatus.PENDING, 0)}\n"
        report += f"   Processing: {status_counts.get(OrderStatus.PROCESSING, 0)}\n"
        report += f"   Ready: {status_counts.get(OrderStatus.READY, 0)}\n"
        report += f"   Completed: {status_counts.get(OrderStatus.COMPLETED, 0)}\n\n"
        
        # Recent orders
        report += "📋 RECENT ORDERS:\n"
        for order in store_orders[-3:]:  # Last 3 orders
            icon = self._get_order_icon(order.status)
            priority_icon = self._get_priority_icon(order.priority)
            
            report += f"{icon} {priority_icon} {order.id}: {order.customer}\n"
            report += f"   Items: {order.items}\n"
            report += f"   Total: ${order.total} | Status: {order.status.value.upper()}\n\n"
        
        return report

    def _get_priority_orders(self, store_id: str) -> str:
        if not store_id:
            return ("🚨 PRIORITY ORDER REQUEST\n"
                   "❓ Which store would you like to check priority orders for?\n\n" + 
                   list_available_stores())
        
        store_orders = [o for o in ORDERS_DATA.values() if o.store_id == store_id]
        high_priority = [o for o in store_orders if o.priority == Priority.HIGH]
        critical_priority = [o for o in store_orders if o.priority == Priority.CRITICAL]
        
        store_name = STORES_DATA[store_id].name
        
        report = f"🚨 PRIORITY ORDERS - {store_name}\n"
        report += "=" * 30 + "\n"
        
        if critical_priority:
            report += "🔴 CRITICAL:\n"
            for order in critical_priority:
                report += f"   • {order.id}: {order.customer} - ${order.total} - {order.status.value}\n"
        
        if high_priority:
            report += "🟡 HIGH:\n"
            for order in high_priority:
                report += f"   • {order.id}: {order.customer} - ${order.total} - {order.status.value}\n"
        
        if not high_priority and not critical_priority:
            report += "✅ No high priority orders at this time\n"
        
        return report

    def _process_orders(self, store_id: str) -> str:
        if not store_id:
            return ("🔄 ORDER PROCESSING REQUEST\n"
                   "❓ Which store would you like to process orders for?\n\n" + 
                   list_available_stores())
        
        store_orders = [o for o in ORDERS_DATA.values() if o.store_id == store_id]
        store_name = STORES_DATA[store_id].name
        
        processed = []
        for order in store_orders:
            if order.status == OrderStatus.PENDING:
                order.status = OrderStatus.PROCESSING
                processed.append(order.id)
            elif order.status == OrderStatus.PROCESSING:
                order.status = OrderStatus.READY
                processed.append(order.id)
        
        if processed:
            return (f"✅ ORDERS PROCESSED - {store_name}\n"
                   f"🔄 Updated orders: {', '.join(processed)}")
        return f"ℹ️ No orders to process at {store_name}"

    def _get_order_icon(self, status: OrderStatus) -> str:
        icons = {
            OrderStatus.PENDING: "⏳",
            OrderStatus.PROCESSING: "🔄",
            OrderStatus.READY: "✅",
            OrderStatus.COMPLETED: "🏁",
            OrderStatus.CANCELLED: "❌"
        }
        return icons.get(status, "⚪")

    def _get_priority_icon(self, priority: Priority) -> str:
        icons = {
            Priority.LOW: "🟢",
            Priority.MEDIUM: "🟡",
            Priority.HIGH: "🟠",
            Priority.CRITICAL: "🔴"
        }
        return icons.get(priority, "⚪")

    def _get_pending_orders(self, store_id: str) -> str:
        if not store_id:
            return ("⏳ PENDING ORDER REQUEST\n"
                   "❓ Which store would you like to check pending orders for?\n\n" + 
                   list_available_stores())
        
        store_orders = [o for o in ORDERS_DATA.values() if o.store_id == store_id]
        pending = [o for o in store_orders if o.status == OrderStatus.PENDING]
        store_name = STORES_DATA[store_id].name
        
        if not pending:
            return f"✅ No pending orders at {store_name}"
        
        report = f"⏳ PENDING ORDERS - {store_name}\n"
        for order in pending:
            report += f"   • {order.id}: {order.customer} - ${order.total}\n"
        
        return report

    def _get_revenue_report(self, store_id: str) -> str:
        if not store_id:
            # Multi-store revenue comparison
            report = "💰 MULTI-STORE REVENUE REPORT:\n"
            report += "=" * 50 + "\n"
            
            store_revenues = []
            for sid, store in STORES_DATA.items():
                store_orders = [o for o in ORDERS_DATA.values() if o.store_id == sid]
                total = sum(o.total for o in store_orders if o.status != OrderStatus.CANCELLED)
                completed = sum(o.total for o in store_orders if o.status == OrderStatus.COMPLETED)
                store_revenues.append((store.name, total, completed))
            
            # Sort by total revenue
            store_revenues.sort(key=lambda x: x[1], reverse=True)
            
            for name, total, completed in store_revenues:
                report += f"🏪 {name}:\n"
                report += f"   Total: ${total:.2f} | Completed: ${completed:.2f}\n"
                report += f"   Pending: ${total - completed:.2f}\n\n"
            
            return report
        
        # Single store detailed revenue
        store_orders = [o for o in ORDERS_DATA.values() if o.store_id == store_id]
        store_name = STORES_DATA[store_id].name
        
        total = sum(o.total for o in store_orders if o.status != OrderStatus.CANCELLED)
        completed = sum(o.total for o in store_orders if o.status == OrderStatus.COMPLETED)
        
        report = f"💰 REVENUE REPORT - {store_name}\n"
        report += "=" * 50 + "\n"
        report += f"Total Revenue: ${total:.2f}\n"
        report += f"Completed Revenue: ${completed:.2f}\n"
        report += f"Pending Revenue: ${total - completed:.2f}\n"
        
        return report

    def _get_order_analytics(self, store_id: str) -> str:
        if not store_id:
            return ("📈 ORDER ANALYTICS REQUEST\n"
                   "❓ Which store would you like analytics for?\n\n" + 
                   list_available_stores())
        
        store_orders = [o for o in ORDERS_DATA.values() if o.store_id == store_id]
        store_name = STORES_DATA[store_id].name
        
        if not store_orders:
            return f"❌ No order data available for {store_name}"
        
        report = f"📈 ORDER ANALYTICS - {store_name}\n"
        report += "=" * 50 + "\n"
        
        total = sum(o.total for o in store_orders if o.status != OrderStatus.CANCELLED)
        avg = total / len(store_orders) if store_orders else 0
        
        # Kiosk performance
        kiosk_orders = {}
        for order in store_orders:
            kiosk_orders[order.kiosk_id] = kiosk_orders.get(order.kiosk_id, 0) + 1
        
        top_kiosk = max(kiosk_orders.items(), key=lambda x: x[1]) if kiosk_orders else ("None", 0)
        
        report += f"Total Revenue: ${total:.2f}\n"
        report += f"Avg Order Value: ${avg:.2f}\n"
        report += f"Top Kiosk: {top_kiosk[0]} ({top_kiosk[1]} orders)\n"
        
        return report

class EnhancedAlertTool(BaseTool):
    name: str = "enhanced_alert_manager"
    description: str = "Multi-store system monitoring and alert management"

    def _run(self, query: str) -> str:
        with LatencyTracker("alert_operation", self._extract_store_from_query(query)):
            return self._process_alert_query(query)
    
    def _extract_store_from_query(self, query: str) -> str:
        store_id = extract_store_from_command(query)
        return store_id if store_id else "unknown"
    
    def _process_alert_query(self, query: str) -> str:
        query_lower = query.lower()
        store_id = extract_store_from_command(query)
        
        if "critical" in query_lower or "urgent" in query_lower:
            return self._get_critical_alerts(store_id)
        elif "resolve" in query_lower:
            return self._resolve_alerts(store_id)
        elif "history" in query_lower:
            return self._get_alert_history(store_id)
        else:
            return self._get_active_alerts(store_id)

    def _get_active_alerts(self, store_id: str) -> str:
        active_alerts = [a for a in ALERTS_DATA if not a.resolved]
        if store_id:
            active_alerts = [a for a in active_alerts if a.store_id == store_id]
        
        report = "🚨 ACTIVE ALERTS:\n" if store_id else "🚨 ALL ACTIVE ALERTS:\n"
        report += "=" * 50 + "\n"
        
        if not active_alerts:
            location = STORES_DATA[store_id].name if store_id else "system"
            report += f"✅ No active alerts for {location}\n"
            return report
        
        for alert in active_alerts:
            icon = self._get_priority_icon(alert.level)
            store_name = STORES_DATA[alert.store_id].name if alert.store_id in STORES_DATA else "Unknown"
            report += f"{icon} [{alert.timestamp}] {store_name}\n"
            report += f"   {alert.message}\n"
            report += f"   Source: {alert.source}\n\n"
        
        return report

    def _get_critical_alerts(self, store_id: str) -> str:
        critical = [a for a in ALERTS_DATA 
                   if a.level in [Priority.HIGH, Priority.CRITICAL] 
                   and not a.resolved]
        
        if store_id:
            critical = [a for a in critical if a.store_id == store_id]
        
        if not critical:
            location = STORES_DATA[store_id].name if store_id else "system"
            return f"✅ No critical alerts for {location}"
        
        report = "🔴 CRITICAL ALERTS:\n"
        for alert in critical:
            store_name = STORES_DATA[alert.store_id].name if alert.store_id in STORES_DATA else "Unknown"
            report += f"   • {store_name}: {alert.message} ({alert.timestamp})\n"
        
        return report

    def _resolve_alerts(self, store_id: str) -> str:
        alerts_to_resolve = [a for a in ALERTS_DATA if not a.resolved]
        if store_id:
            alerts_to_resolve = [a for a in alerts_to_resolve if a.store_id == store_id]
        
        resolved_count = 0
        for alert in alerts_to_resolve:
            if alert.level != Priority.CRITICAL:
                alert.resolved = True
                resolved_count += 1
        
        location = STORES_DATA[store_id].name if store_id else "all stores"
        return f"✅ Resolved {resolved_count} alerts at {location}"

    def _get_alert_history(self, store_id: str) -> str:
        alert_history = ALERTS_DATA
        if store_id:
            alert_history = [a for a in alert_history if a.store_id == store_id]
        
        report = "📜 ALERT HISTORY:\n" if store_id else "📜 FULL ALERT HISTORY:\n"
        report += "=" * 50 + "\n"
        
        for alert in alert_history[-10:]:  # Last 10 alerts
            status = "✅ RESOLVED" if alert.resolved else "🔴 ACTIVE"
            store_name = STORES_DATA[alert.store_id].name if alert.store_id in STORES_DATA else "Unknown"
            report += f"{alert.timestamp} - {store_name}\n"
            report += f"   {alert.message} [{status}]\n\n"
        
        return report

    def _get_priority_icon(self, priority: Priority) -> str:
        icons = {
            Priority.LOW: "🟢",
            Priority.MEDIUM: "🟡",
            Priority.HIGH: "🟠",
            Priority.CRITICAL: "🔴"
        }
        return icons.get(priority, "⚪")

# ============================================================================
# L2 LEVEL - SPECIALIZED AGENTS (ENHANCED)
# ============================================================================

def create_enhanced_kiosk_specialist():
    return Agent(
        role='Multi-Store Kiosk Specialist',
        goal='Maintain optimal kiosk performance across all stores',
        backstory=(
            "You are an expert in retail kiosk management with experience "
            "managing systems across multiple locations. You specialize in "
            "diagnostics, maintenance scheduling, and performance optimization."
        ),
        tools=[EnhancedKioskTool()],
        verbose=True,
        llm=llm
    )

def create_enhanced_order_specialist():
    return Agent(
        role='Multi-Store Order Specialist', 
        goal='Optimize order flow and ensure customer satisfaction across all locations',
        backstory=(
            "You specialize in order management systems for retail chains. "
            "Your expertise includes revenue optimization, priority processing, "
            "and multi-location coordination."
        ),
        tools=[EnhancedOrderTool()],
        verbose=True,
        llm=llm
    )

def create_enhanced_alert_specialist():
    return Agent(
        role='Multi-Store Alert Specialist',
        goal='Monitor system health and manage critical alerts across all stores',
        backstory=(
            "You are responsible for system-wide monitoring and incident response. "
            "Your focus is on minimizing downtime and quickly resolving issues "
            "across all retail locations."
        ),
        tools=[EnhancedAlertTool()],
        verbose=True,
        llm=llm
    )

# ============================================================================
# L1 LEVEL - ENHANCED OPERATOR APP (MULTI-STORE)
# ============================================================================

class EnhancedOperatorApp:
    """L1 Level - Operator Application for multi-store management"""
    
    def __init__(self, store_id: str = None):
        self.store_id = store_id
        self.store_name = STORES_DATA[store_id].name if store_id else "Multi-Store"
        self.name = f"Operator App - {self.store_name}"
        
        print(f"🎯 Initializing {self.name} (L1 Level)...")
        
        # Initialize L2 specialists
        self.kiosk_specialist = create_enhanced_kiosk_specialist()
        self.order_specialist = create_enhanced_order_specialist()
        self.alert_specialist = create_enhanced_alert_specialist()
        
        # Create specialized crews
        self.diagnostic_crew = self._create_diagnostic_crew()
        self.operations_crew = self._create_operations_crew()
        self.maintenance_crew = self._create_maintenance_crew()
        
        print(f"✅ {self.name} initialized with specialized crews")

    def _create_diagnostic_crew(self) -> Crew:
        store_context = f" for {self.store_name}" if self.store_id else " across all stores"
        
        return Crew(
            agents=[self.kiosk_specialist, self.alert_specialist],
            tasks=[
                Task(
                    description=f"Perform comprehensive kiosk diagnostics{store_context}",
                    expected_output="Detailed diagnostic report with recommendations",
                    agent=self.kiosk_specialist
                ),
                Task(
                    description=f"Check for critical alerts{store_context}",
                    expected_output="Alert summary with priority actions",
                    agent=self.alert_specialist
                )
            ],
            process=Process.sequential,
            verbose=True
        )

    def _create_operations_crew(self) -> Crew:
        store_context = f" for {self.store_name}" if self.store_id else " across all stores"
        
        return Crew(
            agents=[self.order_specialist, self.kiosk_specialist],
            tasks=[
                Task(
                    description=f"Process pending orders{store_context}",
                    expected_output="Order processing report with status updates",
                    agent=self.order_specialist
                ),
                Task(
                    description=f"Ensure kiosk availability{store_context}",
                    expected_output="Kiosk readiness report",
                    agent=self.kiosk_specialist
                )
            ],
            process=Process.sequential,
            verbose=True
        )

    def _create_maintenance_crew(self) -> Crew:
        store_context = f" for {self.store_name}" if self.store_id else " across all stores"
        
        return Crew(
            agents=[self.kiosk_specialist, self.alert_specialist],
            tasks=[
                Task(
                    description=f"Schedule maintenance{store_context}",
                    expected_output="Maintenance schedule with affected kiosks",
                    agent=self.kiosk_specialist
                ),
                Task(
                    description=f"Update alert status{store_context}",
                    expected_output="Updated alert status report",
                    agent=self.alert_specialist
                )
            ],
            process=Process.sequential,
            verbose=True
        )

    def run_diagnostics(self) -> OperationReport:
        """Run comprehensive system diagnostics"""
        print(f"\n🔍 Running Diagnostics for {self.store_name}...")
        
        try:
            with LatencyTracker("full_diagnostics", self.store_id):
                result = self.diagnostic_crew.kickoff()
                
                report = OperationReport(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    operation_type=OperationType.DIAGNOSTIC.value,
                    store_id=self.store_id or "all",
                    duration_ms=LATENCY_HISTORY[-1].duration_ms if LATENCY_HISTORY else 0,
                    success=True,
                    details=result,
                    recommendations=[
                        "Review high-error kiosks",
                        "Schedule preventive maintenance"
                    ]
                )
                
                return report
                
        except Exception as e:
            print(f"❌ Diagnostic error: {str(e)}")
            return OperationReport(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                operation_type=OperationType.DIAGNOSTIC.value,
                store_id=self.store_id or "all",
                duration_ms=0,
                success=False,
                details=f"Error: {str(e)}",
                recommendations=["Check system connectivity"]
            )

    def run_operations(self) -> OperationReport:
        """Run standard operations cycle"""
        print(f"\n⚙️ Running Operations for {self.store_name}...")
        
        try:
            with LatencyTracker("standard_operations", self.store_id):
                result = self.operations_crew.kickoff()
                
                report = OperationReport(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    operation_type=OperationType.ORDER_PROCESS.value,
                    store_id=self.store_id or "all",
                    duration_ms=LATENCY_HISTORY[-1].duration_ms if LATENCY_HISTORY else 0,
                    success=True,
                    details=result,
                    recommendations=[
                        "Monitor high-priority orders",
                        "Optimize kiosk allocation"
                    ]
                )
                
                return report
                
        except Exception as e:
            print(f"❌ Operations error: {str(e)}")
            return OperationReport(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                operation_type=OperationType.ORDER_PROCESS.value,
                store_id=self.store_id or "all",
                duration_ms=0,
                success=False,
                details=f"Error: {str(e)}",
                recommendations=["Check order processing system"]
            )

    def run_maintenance(self) -> OperationReport:
        """Run maintenance operations"""
        print(f"\n🛠️ Running Maintenance for {self.store_name}...")
        
        try:
            with LatencyTracker("maintenance_operations", self.store_id):
                result = self.maintenance_crew.kickoff()
                
                report = OperationReport(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    operation_type=OperationType.MAINTENANCE.value,
                    store_id=self.store_id or "all",
                    duration_ms=LATENCY_HISTORY[-1].duration_ms if LATENCY_HISTORY else 0,
                    success=True,
                    details=result,
                    recommendations=[
                        "Verify maintenance completion",
                        "Update maintenance records"
                    ]
                )
                
                return report
                
        except Exception as e:
            print(f"❌ Maintenance error: {str(e)}")
            return OperationReport(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                operation_type=OperationType.MAINTENANCE.value,
                store_id=self.store_id or "all",
                duration_ms=0,
                success=False,
                details=f"Error: {str(e)}",
                recommendations=["Check maintenance scheduling system"]
            )

    def restart_systems(self) -> OperationReport:
        """Restart kiosk systems"""
        print(f"\n🔄 Restarting Systems for {self.store_name}...")
        
        try:
            with LatencyTracker("system_restart", self.store_id):
                tool = EnhancedKioskTool()
                result = tool._run(f"restart {self.store_id}" if self.store_id else "restart")
                
                report = OperationReport(
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    operation_type=OperationType.RESTART.value,
                    store_id=self.store_id or "all",
                    duration_ms=LATENCY_HISTORY[-1].duration_ms if LATENCY_HISTORY else 0,
                    success="success" in result.lower() or "restarted" in result.lower(),
                    details=result,
                    recommendations=[
                        "Verify system status after restart",
                        "Check for recurring issues"
                    ]
                )
                
                return report
                
        except Exception as e:
            print(f"❌ Restart error: {str(e)}")
            return OperationReport(
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                operation_type=OperationType.RESTART.value,
                store_id=self.store_id or "all",
                duration_ms=0,
                success=False,
                details=f"Error: {str(e)}",
                recommendations=["Check system connectivity"]
            )

    def get_quick_status(self) -> str:
        """Get quick status without AI processing"""
        with LatencyTracker("quick_status", self.store_id):
            kiosk_tool = EnhancedKioskTool()
            order_tool = EnhancedOrderTool()
            alert_tool = EnhancedAlertTool()
            
            report = f"\n📊 QUICK STATUS - {self.store_name}\n"
            report += "=" * 60 + "\n"
            
            if self.store_id:
                report += kiosk_tool._run(f"status {self.store_id}") + "\n"
                report += order_tool._run(f"priority {self.store_id}") + "\n"
                report += alert_tool._run(f"critical {self.store_id}") + "\n"
            else:
                report += kiosk_tool._run("status") + "\n"
                report += order_tool._run("priority") + "\n"
                report += alert_tool._run("critical") + "\n"
            
            return report

# ============================================================================
# L0 LEVEL - ENHANCED SUPERVISOR (MULTI-STORE)
# ============================================================================

class EnhancedSupervisor:
    """L0 Level - Supervisor for multi-store management"""
    
    def __init__(self):
        self.name = "Multi-Store Kiosk Supervisor"
        print(f"🎯 Initializing {self.name} (L0 Level)...")
        
        # Initialize Operator Apps for each store
        self.operator_apps = {
            store_id: EnhancedOperatorApp(store_id)
            for store_id in STORES_DATA.keys()
        }
        
        # Add global operator app
        self.operator_apps["global"] = EnhancedOperatorApp()
        
        self.operation_history = []
        self.is_running = False
        self.monitoring_thread = None
        
        print("✅ Supervisor initialized with operator apps for all stores")

    def start_monitoring(self, interval_sec: int = 300):
        """Start continuous monitoring of all stores"""
        if self.is_running:
            print("⚠️ Monitoring is already running")
            return
        
        self.is_running = True
        print(f"🚀 Starting Supervisor Monitoring (interval: {interval_sec}s)...")
        
        def monitoring_loop():
            while self.is_running:
                print(f"\n🔄 Monitoring Cycle - {datetime.now().strftime('%H:%M:%S')}")
                self.run_supervision_cycle()
                time.sleep(interval_sec)
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        if not self.is_running:
            print("⚠️ Monitoring is not running")
            return
        
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        print("🛑 Stopped Supervisor Monitoring")

    def run_supervision_cycle(self):
        """Run one supervision cycle across all stores"""
        print("\n" + "="*60)
        print("🎯 SUPERVISOR CYCLE STARTING")
        print("="*60)
        
        for app_name, app in self.operator_apps.items():
            print(f"\n📋 Processing {app_name}...")
            
            # Get quick status
            start_time = time.time()
            status = app.get_quick_status()
            duration = (time.time() - start_time) * 1000
            
            print(status)
            print(f"⏱️  Status check took {duration:.2f}ms")
            
            # Store in history
            self.operation_history.append({
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'app': app_name,
                'duration_ms': duration,
                'status': 'completed'
            })

    def run_full_operations(self, store_id: str = None) -> Dict[str, OperationReport]:
        """Run full AI-powered operations for specific store or all stores"""
        print("\n🤖 Running Full AI Operations...")
        
        target_apps = {}
        if store_id and store_id in self.operator_apps:
            target_apps = {store_id: self.operator_apps[store_id]}
        else:
            target_apps = self.operator_apps
        
        results = {}
        for app_name, app in target_apps.items():
            print(f"\n🎯 Running operations for {app_name}...")
            
            # Run both diagnostics and operations
            diagnostic_result = app.run_diagnostics()
            operation_result = app.run_operations()
            
            results[app_name] = {
                'diagnostics': diagnostic_result,
                'operations': operation_result
            }
        
        return results

    def restart_store_systems(self, store_id: str = None) -> OperationReport:
        """Restart systems for specific store or prompt for store selection"""
        if not store_id:
            print("🔄 RESTART REQUEST RECEIVED")
            print("Available stores:")
            print(list_available_stores())
            return None
        
        if store_id not in self.operator_apps:
            print(f"❌ Store {store_id} not found")
            return None
        
        return self.operator_apps[store_id].restart_systems()

    def get_supervisor_dashboard(self) -> str:
        """Get comprehensive supervisor dashboard"""
        dashboard = f"\n🎯 {self.name} - DASHBOARD\n"
        dashboard += "="*80 + "\n"
        
        # System overview
        dashboard += f"📊 SYSTEM OVERVIEW:\n"
        dashboard += f"   Active Stores: {len(STORES_DATA)}\n"
        dashboard += f"   Total Kiosks: {len(KIOSKS_DATA)}\n"
        dashboard += f"   Active Orders: {len([o for o in ORDERS_DATA.values() if o.status != OrderStatus.COMPLETED])}\n"
        dashboard += f"   Active Alerts: {len([a for a in ALERTS_DATA if not a.resolved])}\n"
        dashboard += f"   Monitoring: {'🟢 ACTIVE' if self.is_running else '🔴 INACTIVE'}\n\n"
        
        # Store status summary
        dashboard += "🏪 STORE STATUS SUMMARY:\n"
        for store_id, store in STORES_DATA.items():
            kiosks = [k for k in KIOSKS_DATA.values() if k.store_id == store_id]
            online_kiosks = len([k for k in kiosks if k.status == KioskStatus.ONLINE])
            alerts = len([a for a in ALERTS_DATA if a.store_id == store_id and not a.resolved])
            
            dashboard += f"   {store.name} ({store_id}):\n"
            dashboard += f"      Kiosks: {online_kiosks}/{len(kiosks)} online\n"
            dashboard += f"      Alerts: {alerts} active\n"
            dashboard += f"      Manager: {store.manager}\n\n"
        
        # Latency metrics
        dashboard += get_latency_report(limit=5)
        
        return dashboard

    def emergency_response(self, store_id: str = None):
        """Handle emergency situations for specific store or all stores"""
        print("\n🚨 EMERGENCY RESPONSE ACTIVATED")
        print("="*40)
        
        target_apps = {}
        if store_id and store_id in self.operator_apps:
            target_apps = {store_id: self.operator_apps[store_id]}
        else:
            target_apps = self.operator_apps
        
        for app_name, app in target_apps.items():
            print(f"\n🔍 Emergency check for {app_name}...")
            status = app.get_quick_status()
            print(status)
        
        print("\n✅ Emergency response completed")

# ============================================================================
# MAIN APPLICATION (ENHANCED)
# ============================================================================

def main():
    """Main function - Entry point for the enhanced hierarchical system"""
    print("🏪 Welcome to Multi-Store Kiosk Management System!")
    print("Architecture: L0 (Supervisor) -> L1 (Operator Apps) -> L2 (Specialists)")
    print("\n⚠️  SETUP NOTES:")
    print("1. Install Ollama: https://ollama.ai")
    print("2. Run: ollama pull deepseek-r1:7b")
    print("3. Start Ollama server: ollama serve")
    
    # Initialize L0 Supervisor
    supervisor = EnhancedSupervisor()
    
    while True:
        print("\n" + "="*60)
        print("🎯 MULTI-STORE MANAGEMENT - MAIN MENU")
        print("="*60)
        print("1. 🎯 View Supervisor Dashboard")
        print("2. 🔄 Run Monitoring Cycle")
        print("3. 🤖 Run Full AI Operations")
        print("4. 🛠️  Restart Store Systems")
        print("5. 📊 Quick System Status")
        print("6. 🚨 Emergency Response")
        print("7. 📈 Start Continuous Monitoring")
        print("8. 🛑 Stop Continuous Monitoring")
        print("9. ⏱️  View Latency Report")
        print("10. 🚪 Exit")
        
        choice = input("\nSelect option (1-10): ").strip()
        
        if choice == '1':
            print(supervisor.get_supervisor_dashboard())
            
        elif choice == '2':
            supervisor.run_supervision_cycle()
            
        elif choice == '3':
            store_id = input("Enter store ID (or leave blank for all stores): ").strip()
            if store_id and store_id not in STORES_DATA and store_id != "global":
                print(f"❌ Invalid store ID: {store_id}")
                continue
                
            print("\n🤖 Initiating AI-powered operations...")
            results = supervisor.run_full_operations(store_id if store_id else None)
            print("✅ AI operations completed!")
            
        elif choice == '4':
            store_id = input("Enter store ID to restart: ").strip()
            if not store_id:
                print("Available stores:")
                print(list_available_stores())
                continue
                
            if store_id not in STORES_DATA:
                print(f"❌ Invalid store ID: {store_id}")
                continue
                
            result = supervisor.restart_store_systems(store_id)
            if result:
                print(f"\n🔄 Restart Results for {STORES_DATA[store_id].name}:")
                print(f"Status: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
                print(f"Details: {result.details}")
            
        elif choice == '5':
            store_id = input("Enter store ID (or leave blank for all stores): ").strip()
            if store_id and store_id not in STORES_DATA and store_id != "global":
                print(f"❌ Invalid store ID: {store_id}")
                continue
                
            if store_id:
                print(supervisor.operator_apps[store_id].get_quick_status())
            else:
                print(supervisor.operator_apps["global"].get_quick_status())
            
        elif choice == '6':
            store_id = input("Enter store ID (or leave blank for all stores): ").strip()
            if store_id and store_id not in STORES_DATA:
                print(f"❌ Invalid store ID: {store_id}")
                continue
                
            supervisor.emergency_response(store_id if store_id else None)
            
        elif choice == '7':
            interval = input("Enter monitoring interval in seconds (default 300): ").strip()
            try:
                interval = int(interval) if interval else 300
                supervisor.start_monitoring(interval)
            except ValueError:
                print("❌ Invalid interval, using default 300s")
                supervisor.start_monitoring()
            
        elif choice == '8':
            supervisor.stop_monitoring()
            
        elif choice == '9':
            store_id = input("Enter store ID (or leave blank for all operations): ").strip()
            limit = input("Enter number of operations to show (default 10): ").strip()
            
            try:
                limit = int(limit) if limit else 10
            except ValueError:
                limit = 10
                
            print(get_latency_report(store_id if store_id else None, limit))
            
        elif choice == '10':
            supervisor.stop_monitoring()
            print("\n👋 Shutting down Kiosk Management System...")
            print("Thank you for using the Multi-Store Kiosk Management System!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-10.")

if __name__ == "__main__":
    main()