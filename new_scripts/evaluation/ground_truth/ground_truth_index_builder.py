#!/usr/bin/env python3
"""
Ground Truth Index Builder
==========================
Creates and manages the ground truth index mapping image IDs to report paths.

This module handles:
- Building comprehensive index of all available reports
- Mapping image IDs to transcript file paths
- Verification of image-report pairs
- Index validation and completeness checking
- Export and import of index files

Author: Medical Report Evaluation System
Version: 1.0
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime

# Import our utility modules
import sys
sys.path.append(str(Path(__file__).parents[1]))

from utils.ground_truth_loader import GroundTruthLoader
from utils.text_extractor import ReportTextExtractor
from utils.ground_truth_validator import GroundTruthValidator


class GroundTruthIndexBuilder:
    """
    Builds and manages ground truth index files.
    
    Creates comprehensive mapping between image IDs and report paths,
    validates completeness, and provides index management utilities.
    """
    
    def __init__(self, mimic_base_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize the Ground Truth Index Builder.
        
        Args:
            mimic_base_path (str): Base path to MIMIC dataset
            logger (Optional[logging.Logger]): Logger instance
        """
        self.mimic_base_path = Path(mimic_base_path)
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize utility classes
        self.loader = GroundTruthLoader(str(self.mimic_base_path), self.logger)
        self.text_extractor = ReportTextExtractor(self.logger)
        self.validator = GroundTruthValidator(self.logger)
        
        # Index storage
        self.index_data = {}
        self.index_metadata = {}
    
    def build_comprehensive_index(self, source_types: List[str] = ["eyegaze", "reflacx"]) -> Dict:
        """
        Build a comprehensive index of all available ground truth reports.
        
        Args:
            source_types (List[str]): List of source types to include in index
            
        Returns:
            Dict: Comprehensive index data
        """
        self.logger.info(f"Building comprehensive index for sources: {source_types}")
        
        # Initialize index structure
        index = {
            "metadata": {
                "creation_date": datetime.now().isoformat(),
                "mimic_base_path": str(self.mimic_base_path),
                "source_types": source_types,
                "total_reports": 0,
                "total_images": 0,
                "sources": {}
            },
            "image_to_reports": {},  # image_id -> list of report info
            "report_to_image": {},   # report_path -> image_id
            "source_statistics": {},
            "validation_summary": {}
        }
        
        all_reports = {}
        
        # Process each source type
        for source_type in source_types:
            self.logger.info(f"Processing {source_type} reports...")
            
            # Find transcript files for this source
            transcript_files = self.loader.find_transcript_files(source_type)
            source_files = transcript_files.get(source_type, [])
            
            self.logger.info(f"Found {len(source_files)} {source_type} files")
            
            # Load transcripts
            source_reports = {}
            for file_path in source_files:
                transcript_data = self.loader.load_single_transcript(file_path)
                if transcript_data and transcript_data.get("image_id"):
                    image_id = transcript_data["image_id"]
                    source_reports[image_id] = transcript_data
            
            # Update index with source data
            index["sources"][source_type] = {
                "file_count": len(source_files),
                "loaded_count": len(source_reports),
                "success_rate": len(source_reports) / len(source_files) * 100 if source_files else 0
            }
            
            # Merge with all reports
            all_reports.update(source_reports)
        
        # Build image-to-reports mapping
        for image_id, report_data in all_reports.items():
            report_info = {
                "image_id": image_id,
                "file_path": report_data["file_path"],
                "source_type": report_data["source_type"],
                "text_length": report_data["text_length"],
                "has_timestamps": report_data["has_timestamps"],
                "timestamp_count": report_data["timestamp_count"],
                "loaded_at": report_data["loaded_at"]
            }
            
            # Add to image mapping
            if image_id not in index["image_to_reports"]:
                index["image_to_reports"][image_id] = []
            index["image_to_reports"][image_id].append(report_info)
            
            # Add to report mapping
            index["report_to_image"][report_data["file_path"]] = image_id
        
        # Update metadata
        index["metadata"]["total_reports"] = len(all_reports)
        index["metadata"]["total_images"] = len(index["image_to_reports"])
        
        # Generate source statistics
        index["source_statistics"] = self._generate_source_statistics(all_reports)
        
        # Validate the index
        validation_results = self._validate_index(index, all_reports)
        index["validation_summary"] = validation_results
        
        # Store the index
        self.index_data = index
        self.index_metadata = index["metadata"]
        
        self.logger.info(f"Index building complete: {index['metadata']['total_images']} images, "
                        f"{index['metadata']['total_reports']} reports")
        
        return index
    
    def _generate_source_statistics(self, all_reports: Dict) -> Dict:
        """
        Generate statistics about different sources in the index.
        
        Args:
            all_reports (Dict): Dictionary of all loaded reports
            
        Returns:
            Dict: Source statistics
        """
        stats = {
            "by_source": {},
            "text_length_distribution": {},
            "timestamp_availability": {},
            "quality_scores": {}
        }
        
        # Group by source type
        by_source = {}
        for report_data in all_reports.values():
            source = report_data["source_type"]
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(report_data)
        
        # Calculate statistics for each source
        for source, reports in by_source.items():
            source_stats = {
                "count": len(reports),
                "avg_text_length": sum(r["text_length"] for r in reports) / len(reports),
                "min_text_length": min(r["text_length"] for r in reports),
                "max_text_length": max(r["text_length"] for r in reports),
                "timestamp_percentage": sum(1 for r in reports if r["has_timestamps"]) / len(reports) * 100,
                "avg_timestamp_count": sum(r["timestamp_count"] for r in reports) / len(reports)
            }
            stats["by_source"][source] = source_stats
        
        return stats
    
    def _validate_index(self, index: Dict, all_reports: Dict) -> Dict:
        """
        Validate the completeness and consistency of the index.
        
        Args:
            index (Dict): Built index data
            all_reports (Dict): All loaded reports
            
        Returns:
            Dict: Validation results
        """
        validation = {
            "is_valid": True,
            "issues": [],
            "statistics": {},
            "completeness_score": 0.0
        }
        
        # Check index consistency
        image_count = len(index["image_to_reports"])
        report_count = len(index["report_to_image"])
        expected_reports = len(all_reports)
        
        if report_count != expected_reports:
            validation["is_valid"] = False
            validation["issues"].append(f"Report count mismatch: expected {expected_reports}, got {report_count}")
        
        # Check for missing mappings
        missing_mappings = 0
        duplicate_mappings = 0
        
        for image_id, reports in index["image_to_reports"].items():
            if not reports:
                missing_mappings += 1
            elif len(reports) > 1:
                duplicate_mappings += 1
        
        if missing_mappings > 0:
            validation["issues"].append(f"Missing mappings: {missing_mappings} images")
        
        if duplicate_mappings > 0:
            validation["issues"].append(f"Duplicate mappings: {duplicate_mappings} images")
        
        # Check bidirectional consistency
        inconsistent_mappings = 0
        for report_path, image_id in index["report_to_image"].items():
            if image_id not in index["image_to_reports"]:
                inconsistent_mappings += 1
            else:
                # Check if report path exists in image mapping
                found = False
                for report_info in index["image_to_reports"][image_id]:
                    if report_info["file_path"] == report_path:
                        found = True
                        break
                if not found:
                    inconsistent_mappings += 1
        
        if inconsistent_mappings > 0:
            validation["is_valid"] = False
            validation["issues"].append(f"Inconsistent mappings: {inconsistent_mappings} reports")
        
        # Calculate completeness score
        completeness_factors = []
        
        # Coverage factor
        if expected_reports > 0:
            coverage = report_count / expected_reports
            completeness_factors.append(coverage)
        
        # Consistency factor
        consistency = 1.0 - (inconsistent_mappings / max(report_count, 1))
        completeness_factors.append(consistency)
        
        # Mapping quality factor
        mapping_quality = 1.0 - (missing_mappings / max(image_count, 1))
        completeness_factors.append(mapping_quality)
        
        validation["completeness_score"] = sum(completeness_factors) / len(completeness_factors)
        
        # Generate statistics
        validation["statistics"] = {
            "total_images": image_count,
            "total_reports": report_count,
            "expected_reports": expected_reports,
            "missing_mappings": missing_mappings,
            "duplicate_mappings": duplicate_mappings,
            "inconsistent_mappings": inconsistent_mappings,
            "completeness_score": validation["completeness_score"]
        }
        
        return validation
    
    def export_index(self, output_path: str, include_validation: bool = True) -> None:
        """
        Export the index to a JSON file.
        
        Args:
            output_path (str): Path to save the index file
            include_validation (bool): Whether to include validation results
        """
        if not self.index_data:
            self.logger.error("No index data to export. Build index first.")
            return
        
        try:
            export_data = self.index_data.copy()
            
            if not include_validation:
                export_data.pop("validation_summary", None)
            
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Index exported to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export index: {e}")
    
    def load_index(self, index_path: str) -> Dict:
        """
        Load an existing index from a JSON file.
        
        Args:
            index_path (str): Path to the index file
            
        Returns:
            Dict: Loaded index data
        """
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)
            
            self.index_data = index_data
            self.index_metadata = index_data.get("metadata", {})
            
            self.logger.info(f"Index loaded from: {index_path}")
            self.logger.info(f"Index contains {len(index_data.get('image_to_reports', {}))} images")
            
            return index_data
            
        except Exception as e:
            self.logger.error(f"Failed to load index: {e}")
            return {}
    
    def get_reports_for_image(self, image_id: str) -> List[Dict]:
        """
        Get all reports associated with an image ID.
        
        Args:
            image_id (str): Image ID to look up
            
        Returns:
            List[Dict]: List of report information
        """
        if not self.index_data:
            return []
        
        return self.index_data.get("image_to_reports", {}).get(image_id, [])
    
    def get_image_for_report(self, report_path: str) -> Optional[str]:
        """
        Get the image ID associated with a report path.
        
        Args:
            report_path (str): Report file path
            
        Returns:
            Optional[str]: Image ID if found, None otherwise
        """
        if not self.index_data:
            return None
        
        return self.index_data.get("report_to_image", {}).get(report_path)
    
    def get_index_statistics(self) -> Dict:
        """
        Get comprehensive statistics about the current index.
        
        Returns:
            Dict: Index statistics
        """
        if not self.index_data:
            return {"error": "No index data available"}
        
        return {
            "metadata": self.index_metadata,
            "source_statistics": self.index_data.get("source_statistics", {}),
            "validation_summary": self.index_data.get("validation_summary", {}),
            "current_status": {
                "total_images": len(self.index_data.get("image_to_reports", {})),
                "total_reports": len(self.index_data.get("report_to_image", {})),
                "is_loaded": bool(self.index_data)
            }
        }


def test_ground_truth_index_builder():
    """Test function for the Ground Truth Index Builder."""
    import logging
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test")
    
    # Use relative path to MIMIC dataset
    mimic_path = "../../mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0/mimic-eye/"
    
    try:
        print("🧪 Testing Ground Truth Index Builder...")
        
        # Initialize builder
        builder = GroundTruthIndexBuilder(mimic_path, logger)
        
        # Test index building (limit to small subset for testing)
        print("Building test index...")
        index = builder.build_comprehensive_index(["eyegaze"])
        
        if index:
            print(f"✅ Index built successfully:")
            print(f"   Total images: {index['metadata']['total_images']}")
            print(f"   Total reports: {index['metadata']['total_reports']}")
            print(f"   Sources: {list(index['metadata']['sources'].keys())}")
            
            # Test validation
            validation = index.get('validation_summary', {})
            print(f"   Validation: {'✅ Valid' if validation.get('is_valid') else '❌ Invalid'}")
            print(f"   Completeness score: {validation.get('completeness_score', 0):.2f}")
            
            # Test index operations
            image_to_reports = index.get('image_to_reports', {})
            if image_to_reports:
                sample_image_id = list(image_to_reports.keys())[0]
                reports = builder.get_reports_for_image(sample_image_id)
                print(f"   Sample lookup: {len(reports)} reports for image {sample_image_id}")
            
            # Test export
            test_export_path = "test_index.json"
            builder.export_index(test_export_path)
            
            # Test load
            loaded_index = builder.load_index(test_export_path)
            if loaded_index:
                print("✅ Index export/load successful")
            
            # Cleanup test file
            if os.path.exists(test_export_path):
                os.remove(test_export_path)
        
        print("\n✅ All index builder tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    """Run tests when script is executed directly."""
    success = test_ground_truth_index_builder()
    
    if success:
        print("\n🎉 Ground Truth Index Builder tests passed!")
    else:
        print("\n❌ Some tests failed!") 