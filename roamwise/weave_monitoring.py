"""
Enhanced Weave Monitoring and Analytics for RoamWise
Comprehensive observability, performance tracking, and error monitoring
"""

import weave
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque
import threading
import functools

from .config import config


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations"""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class WeaveMonitor:
    """Enhanced Weave monitoring with real-time analytics and alerting"""
    
    def __init__(self, max_metrics_history: int = 1000):
        self.metrics_history: deque = deque(maxlen=max_metrics_history)
        self.operation_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.performance_stats = defaultdict(list)
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def record_operation(self, metrics: PerformanceMetrics):
        """Record operation metrics"""
        with self.lock:
            self.metrics_history.append(metrics)
            self.operation_counts[metrics.operation_name] += 1
            
            if not metrics.success:
                self.error_counts[metrics.error_type or "unknown"] += 1
            
            self.performance_stats[metrics.operation_name].append(metrics.duration)
            
            # Keep only recent performance stats (last 100 operations per type)
            if len(self.performance_stats[metrics.operation_name]) > 100:
                self.performance_stats[metrics.operation_name] = \
                    self.performance_stats[metrics.operation_name][-100:]
    
    @weave.op()
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self.lock:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Calculate performance statistics
            perf_summary = {}
            for op_name, durations in self.performance_stats.items():
                if durations:
                    perf_summary[op_name] = {
                        "count": len(durations),
                        "avg_duration": sum(durations) / len(durations),
                        "min_duration": min(durations),
                        "max_duration": max(durations),
                        "total_time": sum(durations)
                    }
            
            # Recent activity (last 5 minutes)
            recent_cutoff = current_time - 300  # 5 minutes
            recent_metrics = [m for m in self.metrics_history 
                            if m.start_time >= recent_cutoff]
            
            return {
                "system_uptime_seconds": uptime,
                "total_operations": sum(self.operation_counts.values()),
                "total_errors": sum(self.error_counts.values()),
                "error_rate": sum(self.error_counts.values()) / max(sum(self.operation_counts.values()), 1),
                "operation_counts": dict(self.operation_counts),
                "error_counts": dict(self.error_counts),
                "performance_summary": perf_summary,
                "recent_activity": {
                    "operations_last_5min": len(recent_metrics),
                    "errors_last_5min": len([m for m in recent_metrics if not m.success]),
                    "avg_duration_last_5min": sum(m.duration for m in recent_metrics) / max(len(recent_metrics), 1)
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    @weave.op()
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status with alerts"""
        summary = self.get_performance_summary()
        
        # Health checks
        health_status = "healthy"
        alerts = []
        
        # Check error rate
        if summary["error_rate"] > 0.1:  # More than 10% errors
            health_status = "warning"
            alerts.append(f"High error rate: {summary['error_rate']:.2%}")
        
        # Check recent activity
        recent_errors = summary["recent_activity"]["errors_last_5min"]
        if recent_errors > 5:
            health_status = "critical"
            alerts.append(f"High recent error count: {recent_errors}")
        
        # Check performance
        avg_duration = summary["recent_activity"]["avg_duration_last_5min"]
        if avg_duration > 30:  # More than 30 seconds average
            health_status = "warning"
            alerts.append(f"Slow performance: {avg_duration:.2f}s average")
        
        return {
            "status": health_status,
            "alerts": alerts,
            "summary": summary,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global monitor instance
monitor = WeaveMonitor()


def weave_monitor(operation_name: str = None):
    """Enhanced monitoring decorator with automatic metrics collection"""
    def decorator(func):
        @functools.wraps(func)
        @weave.op()
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                
                # Record successful operation
                metrics = PerformanceMetrics(
                    operation_name=op_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    success=True,
                    metadata={
                        "function_name": func.__name__,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    }
                )
                monitor.record_operation(metrics)
                
                return result
                
            except Exception as e:
                end_time = time.time()
                
                # Record failed operation
                metrics = PerformanceMetrics(
                    operation_name=op_name,
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    success=False,
                    error_type=type(e).__name__,
                    metadata={
                        "function_name": func.__name__,
                        "error_message": str(e),
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    }
                )
                monitor.record_operation(metrics)
                
                # Re-raise the exception
                raise
        
        return wrapper
    return decorator


@weave.op()
def log_user_interaction(interaction_type: str, details: Dict[str, Any]) -> Dict[str, Any]:
    """Log user interactions for analytics"""
    interaction_log = {
        "interaction_type": interaction_type,
        "details": details,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": getattr(log_user_interaction, 'session_id', 'unknown')
    }
    
    return interaction_log


@weave.op()
def log_api_call(api_name: str, endpoint: str, method: str, 
                 response_time: float, status_code: int, 
                 request_size: int = 0, response_size: int = 0) -> Dict[str, Any]:
    """Log external API calls for monitoring"""
    api_log = {
        "api_name": api_name,
        "endpoint": endpoint,
        "method": method,
        "response_time_seconds": response_time,
        "status_code": status_code,
        "request_size_bytes": request_size,
        "response_size_bytes": response_size,
        "success": 200 <= status_code < 400,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    return api_log


@weave.op()
def create_performance_report(time_range_hours: int = 24) -> Dict[str, Any]:
    """Create comprehensive performance report"""
    summary = monitor.get_performance_summary()
    health = monitor.get_health_status()
    
    # Calculate time range
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=time_range_hours)
    
    report = {
        "report_period": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_hours": time_range_hours
        },
        "performance_summary": summary,
        "health_status": health,
        "recommendations": _generate_recommendations(summary, health),
        "generated_at": datetime.now(timezone.utc).isoformat()
    }
    
    return report


def _generate_recommendations(summary: Dict[str, Any], health: Dict[str, Any]) -> List[str]:
    """Generate performance recommendations based on metrics"""
    recommendations = []
    
    # Error rate recommendations
    if summary["error_rate"] > 0.05:
        recommendations.append("Consider implementing retry logic for failed operations")
    
    # Performance recommendations
    for op_name, perf in summary["performance_summary"].items():
        if perf["avg_duration"] > 10:
            recommendations.append(f"Optimize {op_name} - average duration is {perf['avg_duration']:.2f}s")
    
    # Activity recommendations
    if summary["recent_activity"]["operations_last_5min"] == 0:
        recommendations.append("No recent activity detected - system may be idle")
    
    return recommendations
