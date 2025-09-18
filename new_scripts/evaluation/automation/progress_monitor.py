#!/usr/bin/env python3
"""
Progress Monitoring System
=========================
Real-time monitoring for the MIMIC-Eye evaluation pipeline.

This module provides:
- Real-time progress tracking
- Memory usage monitoring
- Performance metrics tracking
- Resource utilization monitoring

Author: Medical Report Evaluation System
Version: 1.0
"""

import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import threading
import json
import os


@dataclass
class PerformanceMetrics:
    """Container for performance monitoring data."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read: int
    disk_io_write: int
    processing_rate: float  # reports per second
    estimated_completion: Optional[datetime] = None


class EvaluationMonitor:
    """
    Real-time monitoring system for evaluation pipeline.
    
    Tracks performance, resource usage, and progress.
    """
    
    def __init__(self, 
                 update_interval: float = 1.0,
                 log_file: str = "logs/evaluation_monitor.log"):
        """
        Initialize the evaluation monitor.
        
        Args:
            update_interval (float): Seconds between monitoring updates
            log_file (str): Path to monitoring log file
        """
        self.update_interval = update_interval
        self.log_file = log_file
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self.is_monitoring = False
        self.start_time = None
        self.end_time = None
        self.monitor_thread = None
        
        # Progress tracking
        self.total_items = 0
        self.completed_items = 0
        self.current_item = ""
        self.progress_callback = None
        
        # Performance data
        self.metrics_history: List[PerformanceMetrics] = []
        self.peak_memory = 0
        self.avg_processing_rate = 0
        
        # Resource baselines
        self.initial_memory = 0
        self.initial_disk_io = {"read": 0, "write": 0}
        
        # Create logs directory
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        self.logger.info("Evaluation monitor initialized")
    
    def start_monitoring(self, 
                        total_items: int,
                        progress_callback: Optional[Callable] = None):
        """
        Start the monitoring system.
        
        Args:
            total_items (int): Total number of items to process
            progress_callback (Optional[Callable]): Callback for progress updates
        """
        if self.is_monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self.total_items = total_items
        self.completed_items = 0
        self.progress_callback = progress_callback
        self.start_time = datetime.now()
        self.is_monitoring = True
        
        # Record initial resource state
        self._record_initial_state()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"Started monitoring {total_items} items")
    
    def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.end_time = datetime.now()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        # Generate final report
        self._generate_final_report()
        
        self.logger.info("Monitoring stopped")
    
    def update_progress(self, 
                       completed_items: int,
                       current_item: str = ""):
        """
        Update progress tracking.
        
        Args:
            completed_items (int): Number of completed items
            current_item (str): Current item being processed
        """
        self.completed_items = completed_items
        self.current_item = current_item
        
        # Calculate processing rate
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.avg_processing_rate = completed_items / elapsed if elapsed > 0 else 0
        
        # Call progress callback if provided
        if self.progress_callback:
            self.progress_callback(completed_items, self.total_items, current_item)
    
    def _record_initial_state(self):
        """Record initial system resource state."""
        self.initial_memory = psutil.virtual_memory().used
        disk_io = psutil.disk_io_counters()
        if disk_io:
            self.initial_disk_io = {
                "read": disk_io.read_bytes,
                "write": disk_io.write_bytes
            }
    
    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        while self.is_monitoring:
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Update peak memory
                if metrics.memory_used_mb > self.peak_memory:
                    self.peak_memory = metrics.memory_used_mb
                
                # Log current status
                self._log_status(metrics)
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_used_mb = (memory.used - self.initial_memory) / (1024 * 1024)
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read = 0
        disk_write = 0
        if disk_io:
            disk_read = disk_io.read_bytes - self.initial_disk_io["read"]
            disk_write = disk_io.write_bytes - self.initial_disk_io["write"]
        
        # Processing rate
        processing_rate = self.avg_processing_rate
        
        # Estimated completion
        estimated_completion = None
        if processing_rate > 0 and self.total_items > self.completed_items:
            remaining_items = self.total_items - self.completed_items
            remaining_time = remaining_items / processing_rate
            estimated_completion = datetime.now() + timedelta(seconds=remaining_time)
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory_used_mb,
            disk_io_read=disk_read,
            disk_io_write=disk_write,
            processing_rate=processing_rate,
            estimated_completion=estimated_completion
        )
    
    def _log_status(self, metrics: PerformanceMetrics):
        """Log current monitoring status."""
        progress_pct = (self.completed_items / self.total_items * 100) if self.total_items > 0 else 0
        
        status_msg = (
            f"Progress: {self.completed_items}/{self.total_items} ({progress_pct:.1f}%) | "
            f"Rate: {metrics.processing_rate:.2f} items/s | "
            f"CPU: {metrics.cpu_percent:.1f}% | "
            f"Memory: {metrics.memory_used_mb:.1f}MB | "
            f"Current: {self.current_item}"
        )
        
        self.logger.info(status_msg)
        
        # Write to monitoring log file
        self._write_to_log_file(metrics, status_msg)
    
    def _write_to_log_file(self, metrics: PerformanceMetrics, status_msg: str):
        """Write metrics to log file."""
        try:
            log_entry = {
                "timestamp": metrics.timestamp.isoformat(),
                "progress": {
                    "completed": self.completed_items,
                    "total": self.total_items,
                    "percentage": (self.completed_items / self.total_items * 100) if self.total_items > 0 else 0,
                    "current_item": self.current_item
                },
                "performance": {
                    "cpu_percent": metrics.cpu_percent,
                    "memory_percent": metrics.memory_percent,
                    "memory_used_mb": metrics.memory_used_mb,
                    "processing_rate": metrics.processing_rate
                },
                "resources": {
                    "disk_read_bytes": metrics.disk_io_read,
                    "disk_write_bytes": metrics.disk_io_write
                },
                "status_message": status_msg
            }
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            self.logger.error(f"Error writing to log file: {e}")
    
    def _generate_final_report(self):
        """Generate final monitoring report."""
        if not self.start_time or not self.end_time:
            return
        
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        # Calculate summary statistics
        if self.metrics_history:
            avg_cpu = sum(m.cpu_percent for m in self.metrics_history) / len(self.metrics_history)
            avg_memory = sum(m.memory_used_mb for m in self.metrics_history) / len(self.metrics_history)
            max_cpu = max(m.cpu_percent for m in self.metrics_history)
            max_memory = max(m.memory_used_mb for m in self.metrics_history)
        else:
            avg_cpu = avg_memory = max_cpu = max_memory = 0
        
        final_report = {
            "monitoring_summary": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "total_duration_seconds": total_duration,
                "total_items": self.total_items,
                "completed_items": self.completed_items,
                "success_rate": (self.completed_items / self.total_items * 100) if self.total_items > 0 else 0,
                "average_processing_rate": self.avg_processing_rate
            },
            "resource_usage": {
                "average_cpu_percent": avg_cpu,
                "average_memory_mb": avg_memory,
                "peak_cpu_percent": max_cpu,
                "peak_memory_mb": max_memory,
                "total_memory_used_mb": self.peak_memory
            },
            "performance_metrics": {
                "data_points_collected": len(self.metrics_history),
                "monitoring_interval_seconds": self.update_interval,
                "efficiency_score": self._calculate_efficiency_score()
            }
        }
        
        # Save final report
        report_file = self.log_file.replace('.log', '_final_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2)
        
        self.logger.info(f"Final monitoring report saved to: {report_file}")
        self.logger.info(f"Processing completed: {self.completed_items}/{self.total_items} items in {total_duration:.2f}s")
        self.logger.info(f"Average rate: {self.avg_processing_rate:.2f} items/s, Peak memory: {self.peak_memory:.1f}MB")
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall efficiency score (0-100)."""
        if not self.metrics_history:
            return 0.0
        
        # Factors: processing rate, resource usage, completion rate
        rate_score = min(self.avg_processing_rate * 10, 50)  # Max 50 points for rate
        
        avg_cpu = sum(m.cpu_percent for m in self.metrics_history) / len(self.metrics_history)
        cpu_efficiency = max(0, 100 - avg_cpu) * 0.3  # Max 30 points for CPU efficiency
        
        completion_rate = (self.completed_items / self.total_items * 100) if self.total_items > 0 else 0
        completion_score = completion_rate * 0.2  # Max 20 points for completion
        
        return min(100, rate_score + cpu_efficiency + completion_score)
    
    def get_current_status(self) -> Dict:
        """Get current monitoring status."""
        if not self.is_monitoring:
            return {"status": "not_monitoring"}
        
        progress_pct = (self.completed_items / self.total_items * 100) if self.total_items > 0 else 0
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            "status": "monitoring",
            "progress": {
                "completed": self.completed_items,
                "total": self.total_items,
                "percentage": progress_pct,
                "current_item": self.current_item
            },
            "timing": {
                "elapsed_seconds": elapsed,
                "processing_rate": self.avg_processing_rate,
                "estimated_completion": self.metrics_history[-1].estimated_completion.isoformat() if self.metrics_history and self.metrics_history[-1].estimated_completion else None
            },
            "resources": {
                "peak_memory_mb": self.peak_memory,
                "metrics_collected": len(self.metrics_history)
            }
        }


def create_progress_monitor(total_items: int,
                          update_interval: float = 1.0,
                          log_file: str = "logs/evaluation_monitor.log") -> EvaluationMonitor:
    """
    Convenience function to create and start a progress monitor.
    
    Args:
        total_items (int): Total number of items to process
        update_interval (float): Seconds between updates
        log_file (str): Path to log file
        
    Returns:
        EvaluationMonitor: Configured monitor instance
    """
    monitor = EvaluationMonitor(update_interval, log_file)
    monitor.start_monitoring(total_items)
    return monitor


