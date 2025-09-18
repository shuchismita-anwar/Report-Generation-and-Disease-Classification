import os
import json
import re
import glob
import logging
from typing import Dict, List, Set
from datetime import datetime
from tqdm import tqdm

class MedicalReportCleaner:
    """
    Cleans medical reports by removing unnecessary content patterns identified 
    through AI analysis, preserving only clinical findings and impressions.
    """
    
    def __init__(self, analysis_file: str = "medical_report_cleanup_analysis.json"):
        self.analysis_file = analysis_file
        self.setup_logging()
        self.load_cleanup_patterns()
        
        # Statistics tracking
        self.stats = {
            "total_reports_found": 0,
            "successfully_cleaned": 0,
            "failed_reports": 0,
            "total_patterns_removed": 0,
            "average_reduction_percent": 0,
            "processing_errors": []
        }
    
    def setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/report_cleanup_final.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
    
    def load_cleanup_patterns(self):
        """Load and compile cleanup patterns from analysis file"""
        try:
            with open(self.analysis_file, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            self.cleanup_patterns = self._compile_patterns(analysis_data)
            self.logger.info(f"Loaded {len(self.cleanup_patterns)} cleanup patterns")
            
        except Exception as e:
            self.logger.error(f"Error loading analysis file: {e}")
            raise
    
    def _compile_patterns(self, analysis_data: Dict) -> List[Dict]:
        """Compile patterns from analysis data into regex patterns for efficient matching"""
        patterns = []
        
        # Extract patterns from each category
        for category, data in analysis_data['unnecessary_content_patterns'].items():
            for example in data['examples']:
                pattern_info = {
                    'category': category,
                    'text': example['text'],
                    'reason': example['reason'],
                    'regex': self._create_regex_pattern(example['text'])
                }
                patterns.append(pattern_info)
        
        # Sort by length (longest first) to avoid partial matches
        patterns.sort(key=lambda x: len(x['text']), reverse=True)
        
        return patterns
    
    def _create_regex_pattern(self, text: str) -> re.Pattern:
        """Create regex pattern from text, handling special cases"""
        # Escape special regex characters
        escaped_text = re.escape(text)
        
        # Handle common placeholders and variations
        escaped_text = escaped_text.replace(r'\_\_\_', r'[^\s]*')  # Match any placeholder
        escaped_text = escaped_text.replace(r'\s+', r'\s+')  # Flexible whitespace
        
        # Make pattern case-insensitive and handle word boundaries
        try:
            return re.compile(escaped_text, re.IGNORECASE | re.MULTILINE)
        except re.error:
            # Fallback to simple string matching for problematic patterns
            return re.compile(re.escape(text), re.IGNORECASE)
    
    def _extract_findings_and_impression(self, content: str) -> str:
        """Extract only FINDINGS and IMPRESSION sections with their headers"""
        
        # Initialize result
        result_parts = []
        
        # Extract FINDINGS section content (without header)
        findings_pattern = r'FINDINGS:\s*(.*?)(?=\n\s*[A-Z]+\s*:|$)'
        findings_match = re.search(findings_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if findings_match:
            findings_content = findings_match.group(1).strip()
            # Format: "FINDINGS:" + newline + content
            formatted_findings = f"FINDINGS:\n{findings_content}"
            result_parts.append(formatted_findings)
        
        # Extract IMPRESSION section content (without header)
        impression_pattern = r'IMPRESSION:\s*(.*?)(?=\n\s*[A-Z]+\s*:|$)'
        impression_match = re.search(impression_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if impression_match:
            impression_content = impression_match.group(1).strip()
            # Format: "IMPRESSION:" + newline + content
            formatted_impression = f"IMPRESSION:\n{impression_content}"
            result_parts.append(formatted_impression)
        
        # Join with double newline separation
        return '\n\n'.join(result_parts) if result_parts else content
    
    def _apply_phrase_removal(self, content: str) -> str:
        """Remove specific phrases and patterns identified in the analysis"""
        
        patterns_removed = 0
        
        for pattern_info in self.cleanup_patterns:
            original_content = content
            try:
                content = pattern_info['regex'].sub('', content)
                if content != original_content:
                    patterns_removed += 1
            except Exception as e:
                self.logger.warning(f"Error applying pattern '{pattern_info['text'][:50]}...': {e}")
        
        return content, patterns_removed
    
    def _apply_conservative_cleanup(self, content: str) -> str:
        """Apply conservative cleanup within FINDINGS and IMPRESSION content"""
        
        # Only remove very specific comparative phrases that don't affect clinical meaning
        conservative_patterns = [
            # Only remove direct comparisons to unspecified prior studies
            r'\s*,?\s*unchanged\s*\.?\s*',
            r'\s*,?\s*stable\s*\.?\s*',
            r'\s*,?\s*similar\s*\.?\s*',
            # Remove phrases referencing unspecified prior studies
            r'\s*compared\s+to\s+prior\s*\.?\s*',
            r'\s*as\s+before\s*\.?\s*',
            r'\s*again\s+noted\s*',
        ]
        
        for pattern in conservative_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content
    
    def _apply_minimal_administrative_cleanup(self, content: str) -> str:
        """Remove only essential administrative placeholders within FINDINGS/IMPRESSION"""
        
        # Only remove clear placeholders
        content = re.sub(r'\b___+\b', '', content)
        
        # Remove excessive spacing but preserve structure
        content = re.sub(r'[ \t]{3,}', ' ', content)    # Multiple spaces to single space
        content = re.sub(r'\n{3,}', '\n\n', content)    # Max 2 consecutive newlines
        
        return content
    
    def _apply_administrative_cleanup(self, content: str) -> str:
        """Remove administrative content like placeholders, timestamps, etc."""
        
        admin_patterns = [
            # Placeholders
            r'\b___+\b',  # Multiple underscores
            r'\b\d{1,3}-year-old\b',  # Age placeholders that weren't caught
            
            # Timestamps and dates
            r'\d{1,2}:\d{2}\s*(AM|PM|a\.m\.|p\.m\.)',
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{4}-\d{2}-\d{2}',
            
            # Formatting artifacts
            r'_{5,}',  # Long underscores
            r'\s{3,}',  # Multiple spaces
            r'\n{3,}',  # Multiple newlines
            
            # List numbers at start of lines
            r'^\s*\d+\.\s*',
            
            # Communication notes
            r'discussed\s+with.*?(?=\.|,|\n)',
            r'called\s+to.*?(?=\.|,|\n)',
            r'paged.*?(?=\.|,|\n)',
        ]
        
        for pattern in admin_patterns:
            content = re.sub(pattern, '', content, flags=re.MULTILINE | re.IGNORECASE)
        
        return content
    
    def _post_process_cleanup(self, content: str) -> str:
        """Final cleanup and formatting while preserving structure"""
        
        # Basic punctuation cleanup first
        content = re.sub(r'\s*,\s*', ', ', content)
        content = re.sub(r'\s*\.\s*', '. ', content)
        
        # Normalize excessive whitespace but preserve structure
        content = re.sub(r'\n{3,}', '\n\n', content)  # Max 2 consecutive newlines
        content = re.sub(r'[ \t]+', ' ', content)      # Multiple spaces to single space
        
        return content.strip()
    
    def clean_report(self, content: str) -> tuple[str, int]:
        """Clean a single report content"""
        if not content or not content.strip():
            return "", 0
        
        original_length = len(content)
        
        # Simple extraction only
        content = self._extract_findings_and_impression(content)
        
        # Only remove clear placeholders
        content = re.sub(r'\b___+\b', '', content)
        
        # Calculate reduction percentage
        final_length = len(content)
        reduction_percent = ((original_length - final_length) / original_length * 100) if original_length > 0 else 0
        
        return content, reduction_percent
    
    def find_all_reports(self, base_path: str) -> List[str]:
        """Find all report files in the MIMIC-Eye dataset structure"""
        pattern = os.path.join(base_path, "mimic-eye", "patient_*", "CXR-DICOM", "s*.txt")
        all_reports = glob.glob(pattern)
        
        self.logger.info(f"Found {len(all_reports)} total reports in {base_path}")
        self.stats["total_reports_found"] = len(all_reports)
        
        return all_reports
    
    def process_all_reports(self, 
                          input_base_path: str = "../../mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0",
                          output_dir: str = "../../cleaned_reports"):
        """Process all reports and save cleaned versions"""
        
        print("🏥 Starting Medical Report Cleanup Process...")
        print(f"📂 Input path: {input_base_path}")
        print(f"💾 Output directory: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all reports
        print("🔍 Finding all report files...")
        all_reports = self.find_all_reports(input_base_path)
        
        if not all_reports:
            self.logger.error("No reports found!")
            return
        
        print(f"📊 Processing {len(all_reports)} reports...")
        
        # Process reports with progress bar
        total_reduction = 0
        
        with tqdm(total=len(all_reports), desc="Cleaning reports", unit="report") as pbar:
            for report_path in all_reports:
                try:
                    # Read original report
                    with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
                        original_content = f.read()
                    
                    # Clean the report
                    cleaned_content, reduction_percent = self.clean_report(original_content)
                    
                    # Generate output path maintaining directory structure
                    # Extract patient folder and maintain structure: patient_XXXXXXXX/CXR-DICOM/sXXXXXXXX.txt
                    relative_path = os.path.relpath(report_path, os.path.join(input_base_path, "mimic-eye"))
                    output_path = os.path.join(output_dir, relative_path)
                    
                    # Create output directory structure if it doesn't exist
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Save cleaned report
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                    
                    # Update statistics
                    self.stats["successfully_cleaned"] += 1
                    total_reduction += reduction_percent
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Success': self.stats["successfully_cleaned"],
                        'Avg Reduction': f"{total_reduction/self.stats['successfully_cleaned']:.1f}%"
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {report_path}: {e}")
                    self.stats["failed_reports"] += 1
                    self.stats["processing_errors"].append({
                        "file": report_path,
                        "error": str(e)
                    })
                    pbar.update(1)
        
        # Calculate final statistics
        if self.stats["successfully_cleaned"] > 0:
            self.stats["average_reduction_percent"] = total_reduction / self.stats["successfully_cleaned"]
        
        # Print final summary
        self.print_summary()
        
        # Save processing log
        self.save_processing_log(output_dir)
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*60)
        print("📊 MEDICAL REPORT CLEANUP SUMMARY")
        print("="*60)
        print(f"📁 Total reports found: {self.stats['total_reports_found']}")
        print(f"✅ Successfully cleaned: {self.stats['successfully_cleaned']}")
        print(f"❌ Failed reports: {self.stats['failed_reports']}")
        print(f"📉 Average content reduction: {self.stats['average_reduction_percent']:.1f}%")
        print(f"🧹 Cleanup patterns available: {len(self.cleanup_patterns)}")
        
        if self.stats['failed_reports'] > 0:
            print(f"\n⚠️  {self.stats['failed_reports']} reports failed to process")
            print("Check the log file for details")
        
        success_rate = (self.stats['successfully_cleaned'] / self.stats['total_reports_found'] * 100) if self.stats['total_reports_found'] > 0 else 0
        print(f"🎯 Success rate: {success_rate:.1f}%")
        print("="*60)
    
    def save_processing_log(self, output_dir: str):
        """Save detailed processing log"""
        log_data = {
            "processing_date": datetime.now().isoformat(),
            "statistics": self.stats,
            "cleanup_patterns_used": len(self.cleanup_patterns),
            "analysis_file": self.analysis_file
        }
        
        log_path = os.path.join(output_dir, "cleanup_processing_log.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Processing log saved to: {log_path}")

def main():
    """Main function to run the cleanup process"""
    try:
        # Initialize cleaner
        cleaner = MedicalReportCleaner()
        
        # Process all reports
        cleaner.process_all_reports()
        
        print("\n🎉 Cleanup process completed successfully!")
        print("📁 Cleaned reports are saved in the 'cleaned_reports' directory")
        print("📋 Original reports remain unchanged")
        
    except Exception as e:
        print(f"❌ Error during cleanup process: {e}")
        logging.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()
