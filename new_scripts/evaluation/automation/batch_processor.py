#!/usr/bin/env python3
"""
Batch Processing Engine
======================
Efficient batch processing for large-scale MIMIC-Eye evaluation.

This module provides:
- Incremental progress saving
- Configurable batch sizes
- Efficient handling of large datasets
- Parallel processing support

Author: Medical Report Evaluation System
Version: 1.0
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from pathlib import Path


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    batch_size: int = 50
    max_workers: int = 4
    use_multiprocessing: bool = False
    save_interval: int = 10  # Save progress every N batches
    checkpoint_interval: int = 100  # Create checkpoint every N items
    output_dir: str = "output/batch_processing"
    temp_dir: str = "temp/batch_processing"


class BatchProcessor:
    """
    Efficient batch processing engine for evaluation pipeline.
    
    Handles large datasets with configurable batching and parallel processing.
    """
    
    def __init__(self, config: BatchConfig = None):
        """
        Initialize the batch processor.
        
        Args:
            config (BatchConfig): Batch processing configuration
        """
        self.config = config or BatchConfig()
        self.logger = logging.getLogger(__name__)
        
        # Processing state
        self.total_items = 0
        self.processed_items = 0
        self.current_batch = 0
        self.results = []
        self.failed_items = []
        
        # Threading
        self.executor = None
        self.processing_lock = threading.Lock()
        
        # Create directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.temp_dir, exist_ok=True)
        
        self.logger.info("Batch processor initialized")
    
    def process_items(self,
                     items: List[Any],
                     process_function: Callable,
                     progress_callback: Optional[Callable] = None,
                     batch_callback: Optional[Callable] = None) -> Dict:
        """
        Process items in batches with parallel execution.
        
        Args:
            items (List[Any]): Items to process
            process_function (Callable): Function to process each item
            progress_callback (Optional[Callable]): Progress update callback
            batch_callback (Optional[Callable]): Batch completion callback
            
        Returns:
            Dict: Processing results summary
        """
        self.total_items = len(items)
        self.processed_items = 0
        self.current_batch = 0
        self.results = []
        self.failed_items = []
        
        start_time = time.time()
        
        self.logger.info(f"Starting batch processing: {self.total_items} items, batch size: {self.config.batch_size}")
        
        try:
            # Create executor based on configuration
            executor_class = ProcessPoolExecutor if self.config.use_multiprocessing else ThreadPoolExecutor
            
            with executor_class(max_workers=self.config.max_workers) as executor:
                self.executor = executor
                
                # Process items in batches
                for batch_items in self._create_batches(items):
                    self.current_batch += 1
                    
                    batch_results = self._process_batch(
                        batch_items, 
                        process_function, 
                        executor
                    )
                    
                    # Update results
                    self.results.extend(batch_results['successful'])
                    self.failed_items.extend(batch_results['failed'])
                    self.processed_items += len(batch_items)
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback(self.processed_items, self.total_items)
                    
                    # Batch callback
                    if batch_callback:
                        batch_callback(self.current_batch, batch_results)
                    
                    # Save progress periodically
                    if self.current_batch % self.config.save_interval == 0:
                        self._save_progress()
                    
                    # Create checkpoint periodically
                    if self.processed_items % self.config.checkpoint_interval == 0:
                        self._create_checkpoint()
                    
                    self.logger.info(
                        f"Batch {self.current_batch} completed: "
                        f"{len(batch_results['successful'])} successful, "
                        f"{len(batch_results['failed'])} failed"
                    )
        
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise
        
        finally:
            self.executor = None
        
        # Final results
        total_time = time.time() - start_time
        
        summary = {
            "processing_summary": {
                "total_items": self.total_items,
                "processed_items": self.processed_items,
                "successful_items": len(self.results),
                "failed_items": len(self.failed_items),
                "total_batches": self.current_batch,
                "processing_time": total_time,
                "items_per_second": self.processed_items / total_time if total_time > 0 else 0
            },
            "results": self.results,
            "failed_items": self.failed_items,
            "config": {
                "batch_size": self.config.batch_size,
                "max_workers": self.config.max_workers,
                "use_multiprocessing": self.config.use_multiprocessing
            }
        }
        
        # Save final results
        self._save_final_results(summary)
        
        self.logger.info(
            f"Batch processing completed: {len(self.results)}/{self.total_items} successful "
            f"in {total_time:.2f}s ({self.processed_items/total_time:.2f} items/s)"
        )
        
        return summary
    
    def _create_batches(self, items: List[Any]) -> Iterator[List[Any]]:
        """Create batches from items list."""
        for i in range(0, len(items), self.config.batch_size):
            yield items[i:i + self.config.batch_size]
    
    def _process_batch(self,
                      batch_items: List[Any],
                      process_function: Callable,
                      executor) -> Dict:
        """Process a single batch of items."""
        batch_start = time.time()
        successful = []
        failed = []
        
        # Submit all items in batch to executor
        future_to_item = {
            executor.submit(self._safe_process_item, process_function, item, idx): (item, idx)
            for idx, item in enumerate(batch_items)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_item):
            item, idx = future_to_item[future]
            
            try:
                result = future.result()
                if result['success']:
                    successful.append(result)
                else:
                    failed.append({
                        'item': item,
                        'index': idx,
                        'error': result['error']
                    })
            
            except Exception as e:
                self.logger.error(f"Future execution failed for item {idx}: {e}")
                failed.append({
                    'item': item,
                    'index': idx,
                    'error': f"Future execution failed: {str(e)}"
                })
        
        batch_time = time.time() - batch_start
        
        return {
            'successful': successful,
            'failed': failed,
            'batch_time': batch_time,
            'items_per_second': len(batch_items) / batch_time if batch_time > 0 else 0
        }
    
    def _safe_process_item(self,
                          process_function: Callable,
                          item: Any,
                          index: int) -> Dict:
        """Safely process a single item with error handling."""
        try:
            start_time = time.time()
            result = process_function(item)
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'item_index': index,
                'result': result,
                'processing_time': processing_time
            }
        
        except Exception as e:
            self.logger.error(f"Error processing item {index}: {e}")
            return {
                'success': False,
                'item_index': index,
                'error': str(e),
                'processing_time': 0
            }
    
    def _save_progress(self):
        """Save current processing progress."""
        try:
            progress_file = os.path.join(
                self.config.temp_dir,
                f"progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            progress_data = {
                "timestamp": datetime.now().isoformat(),
                "total_items": self.total_items,
                "processed_items": self.processed_items,
                "current_batch": self.current_batch,
                "successful_items": len(self.results),
                "failed_items": len(self.failed_items)
            }
            
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2)
            
            self.logger.debug(f"Progress saved: {progress_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")
    
    def _create_checkpoint(self):
        """Create a processing checkpoint."""
        try:
            checkpoint_file = os.path.join(
                self.config.temp_dir,
                f"checkpoint_{self.processed_items}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            checkpoint_data = {
                "timestamp": datetime.now().isoformat(),
                "processed_items": self.processed_items,
                "current_results": self.results[-100:],  # Last 100 results
                "recent_failures": self.failed_items[-50:],  # Last 50 failures
                "batch_config": {
                    "batch_size": self.config.batch_size,
                    "max_workers": self.config.max_workers
                }
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.logger.info(f"Checkpoint created: {checkpoint_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
    
    def _save_final_results(self, summary: Dict):
        """Save final processing results."""
        try:
            results_file = os.path.join(
                self.config.output_dir,
                f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Final results saved: {results_file}")
            
            # Also save a summary file
            summary_file = os.path.join(
                self.config.output_dir,
                f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            summary_only = {
                "processing_summary": summary["processing_summary"],
                "config": summary["config"],
                "timestamp": datetime.now().isoformat()
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_only, f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save final results: {e}")
    
    def get_processing_status(self) -> Dict:
        """Get current processing status."""
        return {
            "total_items": self.total_items,
            "processed_items": self.processed_items,
            "current_batch": self.current_batch,
            "successful_items": len(self.results),
            "failed_items": len(self.failed_items),
            "progress_percentage": (self.processed_items / self.total_items * 100) if self.total_items > 0 else 0,
            "is_processing": self.executor is not None
        }


def create_batch_processor(batch_size: int = 50,
                          max_workers: int = 4,
                          use_multiprocessing: bool = False) -> BatchProcessor:
    """
    Convenience function to create a batch processor.
    
    Args:
        batch_size (int): Number of items per batch
        max_workers (int): Maximum number of worker threads/processes
        use_multiprocessing (bool): Use processes instead of threads
        
    Returns:
        BatchProcessor: Configured batch processor
    """
    config = BatchConfig(
        batch_size=batch_size,
        max_workers=max_workers,
        use_multiprocessing=use_multiprocessing
    )
    
    return BatchProcessor(config)


