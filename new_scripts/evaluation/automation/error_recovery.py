#!/usr/bin/env python3
"""
Error Recovery System
====================
Robust error handling and recovery for the MIMIC-Eye evaluation pipeline.

This module provides:
- Retry logic for transient failures
- Memory management for large batches
- Graceful handling of corrupted files
- Partial results saving during failures

Author: Medical Report Evaluation System
Version: 1.0
"""

import time
import os
import json
import logging
import traceback
import psutil
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from functools import wraps
import gc


@dataclass
class ErrorInfo:
    """Container for error information."""
    timestamp: datetime
    error_type: str
    error_message: str
    traceback_str: str
    context: Dict[str, Any]
    retry_attempt: int
    severity: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class RecoveryConfig:
    """Configuration for error recovery behavior."""
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    memory_threshold_mb: int = 8192  # 8GB
    max_consecutive_errors: int = 10
    save_interval: int = 50  # Save partial results every N items
    checkpoint_dir: str = "checkpoints"
    

class ErrorRecoveryManager:
    """
    Comprehensive error recovery and resilience manager.
    
    Handles various failure scenarios and provides recovery mechanisms.
    """
    
    def __init__(self, config: RecoveryConfig = None):
        """
        Initialize the error recovery manager.
        
        Args:
            config (RecoveryConfig): Recovery configuration
        """
        self.config = config or RecoveryConfig()
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.errors: List[ErrorInfo] = []
        self.consecutive_errors = 0
        self.total_retries = 0
        
        # Recovery state
        self.is_recovery_mode = False
        self.last_checkpoint = None
        self.partial_results = []
        
        # Memory management
        self.memory_warnings = 0
        self.gc_collections = 0
        
        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        self.logger.info("Error recovery manager initialized")
    
    def with_retry(self, 
                   max_retries: Optional[int] = None,
                   retry_delay: Optional[float] = None,
                   exceptions: Tuple = (Exception,)):
        """
        Decorator for adding retry logic to functions.
        
        Args:
            max_retries (Optional[int]): Override default max retries
            retry_delay (Optional[float]): Override default retry delay
            exceptions (Tuple): Exception types to retry on
            
        Returns:
            Decorated function with retry logic
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self._execute_with_retry(
                    func, args, kwargs, 
                    max_retries or self.config.max_retries,
                    retry_delay or self.config.retry_delay,
                    exceptions
                )
            return wrapper
        return decorator
    
    def _execute_with_retry(self, 
                           func: Callable,
                           args: tuple,
                           kwargs: dict,
                           max_retries: int,
                           retry_delay: float,
                           exceptions: Tuple) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                # Check memory before execution
                self._check_memory_usage()
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Reset consecutive error counter on success
                if self.consecutive_errors > 0:
                    self.consecutive_errors = 0
                    self.logger.info("Recovery successful, error counter reset")
                
                return result
                
            except exceptions as e:
                last_exception = e
                self.consecutive_errors += 1
                self.total_retries += 1
                
                # Record error
                error_info = ErrorInfo(
                    timestamp=datetime.now(),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    traceback_str=traceback.format_exc(),
                    context={
                        "function": func.__name__,
                        "attempt": attempt + 1,
                        "args_len": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    },
                    retry_attempt=attempt,
                    severity=self._determine_error_severity(e)
                )
                
                self.errors.append(error_info)
                
                # Check if we should abort
                if self._should_abort(error_info):
                    self.logger.critical(f"Aborting due to critical error: {e}")
                    raise
                
                # Log retry attempt
                if attempt < max_retries:
                    delay = self._calculate_retry_delay(retry_delay, attempt)
                    self.logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {max_retries + 1} attempts failed: {e}")
        
        # All retries exhausted
        raise last_exception
    
    def _calculate_retry_delay(self, base_delay: float, attempt: int) -> float:
        """Calculate retry delay with optional exponential backoff."""
        if self.config.exponential_backoff:
            return base_delay * (2 ** attempt)
        return base_delay
    
    def _determine_error_severity(self, error: Exception) -> str:
        """Determine error severity level."""
        error_type = type(error).__name__
        
        if error_type in ['MemoryError', 'SystemExit', 'KeyboardInterrupt']:
            return 'critical'
        elif error_type in ['FileNotFoundError', 'PermissionError', 'ConnectionError']:
            return 'high'
        elif error_type in ['ValueError', 'TypeError', 'AttributeError']:
            return 'medium'
        else:
            return 'low'
    
    def _should_abort(self, error_info: ErrorInfo) -> bool:
        """Determine if processing should be aborted."""
        # Critical errors always abort
        if error_info.severity == 'critical':
            return True
        
        # Too many consecutive errors
        if self.consecutive_errors >= self.config.max_consecutive_errors:
            self.logger.critical(f"Too many consecutive errors: {self.consecutive_errors}")
            return True
        
        return False
    
    def _check_memory_usage(self):
        """Check and manage memory usage."""
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024 * 1024)
        
        if memory_used_mb > self.config.memory_threshold_mb:
            self.memory_warnings += 1
            self.logger.warning(f"High memory usage: {memory_used_mb:.1f}MB")
            
            # Force garbage collection
            self._force_garbage_collection()
            
            # Check again after GC
            memory = psutil.virtual_memory()
            memory_used_mb = memory.used / (1024 * 1024)
            
            if memory_used_mb > self.config.memory_threshold_mb:
                self.logger.error(f"Memory usage still high after GC: {memory_used_mb:.1f}MB")
                raise MemoryError(f"Memory usage exceeded threshold: {memory_used_mb:.1f}MB")
    
    def _force_garbage_collection(self):
        """Force garbage collection to free memory."""
        self.gc_collections += 1
        self.logger.info("Forcing garbage collection...")
        
        # Multiple GC passes for thoroughness
        for i in range(3):
            collected = gc.collect()
            self.logger.debug(f"GC pass {i+1}: collected {collected} objects")
        
        # Report memory after GC
        memory = psutil.virtual_memory()
        self.logger.info(f"Memory after GC: {memory.used / (1024 * 1024):.1f}MB")
    
    def save_checkpoint(self, 
                       data: Any,
                       checkpoint_id: str,
                       metadata: Dict = None):
        """
        Save a recovery checkpoint.
        
        Args:
            data (Any): Data to save
            checkpoint_id (str): Unique checkpoint identifier
            metadata (Dict): Additional metadata
        """
        try:
            checkpoint_file = os.path.join(
                self.config.checkpoint_dir, 
                f"checkpoint_{checkpoint_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "checkpoint_id": checkpoint_id,
                "data": data,
                "metadata": metadata or {},
                "error_count": len(self.errors),
                "consecutive_errors": self.consecutive_errors
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            self.last_checkpoint = checkpoint_file
            self.logger.info(f"Checkpoint saved: {checkpoint_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_file: str) -> Dict:
        """
        Load a recovery checkpoint.
        
        Args:
            checkpoint_file (str): Path to checkpoint file
            
        Returns:
            Dict: Loaded checkpoint data
        """
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            self.logger.info(f"Checkpoint loaded: {checkpoint_file}")
            return checkpoint_data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint file."""
        try:
            checkpoint_files = [
                f for f in os.listdir(self.config.checkpoint_dir)
                if f.startswith('checkpoint_') and f.endswith('.json')
            ]
            
            if not checkpoint_files:
                return None
            
            # Sort by modification time
            checkpoint_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(self.config.checkpoint_dir, f)),
                reverse=True
            )
            
            latest = os.path.join(self.config.checkpoint_dir, checkpoint_files[0])
            self.logger.info(f"Latest checkpoint: {latest}")
            return latest
            
        except Exception as e:
            self.logger.error(f"Error finding latest checkpoint: {e}")
            return None
    
    def handle_corrupted_file(self, 
                             file_path: str,
                             error: Exception) -> bool:
        """
        Handle corrupted or problematic files.
        
        Args:
            file_path (str): Path to problematic file
            error (Exception): Error that occurred
            
        Returns:
            bool: True if file should be skipped, False if should retry
        """
        self.logger.warning(f"Corrupted file detected: {file_path} - {error}")
        
        # Record in error log
        error_info = ErrorInfo(
            timestamp=datetime.now(),
            error_type="CorruptedFile",
            error_message=f"File: {file_path}, Error: {str(error)}",
            traceback_str=traceback.format_exc(),
            context={"file_path": file_path},
            retry_attempt=0,
            severity="medium"
        )
        
        self.errors.append(error_info)
        
        # Move corrupted file to quarantine
        self._quarantine_file(file_path)
        
        # Skip this file
        return True
    
    def _quarantine_file(self, file_path: str):
        """Move corrupted file to quarantine directory."""
        try:
            quarantine_dir = os.path.join(self.config.checkpoint_dir, "quarantine")
            os.makedirs(quarantine_dir, exist_ok=True)
            
            filename = os.path.basename(file_path)
            quarantine_path = os.path.join(
                quarantine_dir, 
                f"quarantine_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
            )
            
            if os.path.exists(file_path):
                os.rename(file_path, quarantine_path)
                self.logger.info(f"File quarantined: {file_path} -> {quarantine_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to quarantine file {file_path}: {e}")
    
    def generate_error_report(self) -> Dict:
        """Generate comprehensive error report."""
        if not self.errors:
            return {"status": "no_errors", "total_errors": 0}
        
        # Group errors by type
        error_by_type = {}
        error_by_severity = {}
        
        for error in self.errors:
            error_by_type[error.error_type] = error_by_type.get(error.error_type, 0) + 1
            error_by_severity[error.severity] = error_by_severity.get(error.severity, 0) + 1
        
        # Calculate error rates
        total_errors = len(self.errors)
        time_span = (self.errors[-1].timestamp - self.errors[0].timestamp).total_seconds()
        error_rate = total_errors / max(time_span / 3600, 1)  # errors per hour
        
        return {
            "status": "errors_detected",
            "summary": {
                "total_errors": total_errors,
                "consecutive_errors": self.consecutive_errors,
                "total_retries": self.total_retries,
                "memory_warnings": self.memory_warnings,
                "gc_collections": self.gc_collections,
                "error_rate_per_hour": error_rate
            },
            "breakdown": {
                "by_type": error_by_type,
                "by_severity": error_by_severity
            },
            "recent_errors": [
                {
                    "timestamp": error.timestamp.isoformat(),
                    "type": error.error_type,
                    "message": error.error_message,
                    "severity": error.severity
                }
                for error in self.errors[-10:]  # Last 10 errors
            ],
            "recovery_info": {
                "checkpoint_dir": self.config.checkpoint_dir,
                "last_checkpoint": self.last_checkpoint,
                "recovery_mode": self.is_recovery_mode
            }
        }
    
    def clear_error_history(self):
        """Clear error history (use with caution)."""
        self.errors.clear()
        self.consecutive_errors = 0
        self.logger.info("Error history cleared")
    
    def get_recovery_recommendations(self) -> List[str]:
        """Get recovery recommendations based on error patterns."""
        recommendations = []
        
        if self.memory_warnings > 5:
            recommendations.append("Consider reducing batch size to manage memory usage")
        
        if self.consecutive_errors > 3:
            recommendations.append("Consider pausing and investigating persistent errors")
        
        if self.gc_collections > 10:
            recommendations.append("Memory management may need optimization")
        
        # Analyze error patterns
        if self.errors:
            recent_errors = self.errors[-10:]
            file_errors = [e for e in recent_errors if "File" in e.error_type]
            if len(file_errors) > 5:
                recommendations.append("Multiple file-related errors detected - check data integrity")
        
        return recommendations


def create_recovery_manager(checkpoint_dir: str = "checkpoints",
                          memory_threshold_mb: int = 8192,
                          max_retries: int = 3) -> ErrorRecoveryManager:
    """
    Convenience function to create an error recovery manager.
    
    Args:
        checkpoint_dir (str): Directory for checkpoints
        memory_threshold_mb (int): Memory threshold in MB
        max_retries (int): Maximum retry attempts
        
    Returns:
        ErrorRecoveryManager: Configured recovery manager
    """
    config = RecoveryConfig(
        checkpoint_dir=checkpoint_dir,
        memory_threshold_mb=memory_threshold_mb,
        max_retries=max_retries
    )
    
    return ErrorRecoveryManager(config)


