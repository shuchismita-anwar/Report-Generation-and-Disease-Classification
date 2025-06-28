import os
import json
import requests
import time
import logging
import signal
import sys
import glob
import random
import re
from typing import Dict, List, Tuple, Set
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from collections import defaultdict, Counter
from difflib import SequenceMatcher

class MedicalReportCleanupAnalyzer:
    """
    Analyzes medical reports using Gemini 2.5 Pro to identify unnecessary content patterns
    that should be removed for better BLEU/ROUGE/METEOR scoring accuracy.
    """
    
    def __init__(self, api_key: str = None):
        # Load environment variables
        load_dotenv('.env')
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # Setup logging
        self._setup_logging()
        
        # API configuration for Gemini 2.5 Pro
        self.api_config = {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
            "headers": {"Content-Type": "application/json"},
            "max_retries": 5,
            "base_retry_delay": 3,
            "timeout": 120,
            "rate_limit_delay": 2.0
        }
        
        # Processing statistics
        self.stats = {
            "total_batches": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "total_reports_processed": 0,
            "total_patterns_found": 0,
            "unique_patterns_after_dedup": 0,
            "api_errors": 0,
            "start_time": None
        }
        
        # Pattern storage
        self.raw_patterns = []
        self.consolidated_patterns = {
            "comparison_references": {
                "pattern_type": "comparative_phrases",
                "examples": [],
                "frequency": 0,
                "variations": set(),
                "reasons": []
            },
            "examination_headers": {
                "pattern_type": "metadata_sections", 
                "examples": [],
                "frequency": 0,
                "variations": set(),
                "reasons": []
            },
            "indication_sections": {
                "pattern_type": "metadata_sections",
                "examples": [],
                "frequency": 0,
                "variations": set(),
                "reasons": []
            },
            "technique_sections": {
                "pattern_type": "metadata_sections",
                "examples": [],
                "frequency": 0,
                "variations": set(),
                "reasons": []
            },
            "other_unnecessary": {
                "pattern_type": "miscellaneous",
                "examples": [],
                "frequency": 0,
                "variations": set(),
                "reasons": []
            }
        }
        
        # Failed batch tracking
        self.failed_batches_file = "failed_report_batches.json"
        self.progress_file = "cleanup_analysis_progress.json"
        
        # Setup graceful shutdown
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Gemini prompt for identifying unnecessary content
        self.analysis_prompt = """You are a medical report analysis expert. Your task is to identify unnecessary content in chest X-ray radiology reports that should be removed to improve automated evaluation metrics (BLEU, ROUGE, METEOR).

ANALYZE the following medical report and identify content that is:
1. METADATA/PROCEDURAL (examination type, technique, indication, comparison references)
2. ADMINISTRATIVE (patient age placeholders like "___", timestamps, procedure codes)
3. COMPARATIVE REFERENCES (mentions of prior studies, comparisons to previous exams)
4. NON-CLINICAL CONTENT (technical acquisition details, equipment information)

For each unnecessary pattern you find, provide:
- The EXACT text that should be removed
- The REASON why it's unnecessary for clinical evaluation
- The CATEGORY (comparison_reference, examination_header, indication_section, technique_section, or other_unnecessary)

IMPORTANT: Focus on content that adds NO clinical value for diagnosis. Keep ALL clinical findings, impressions, and medical observations.

RESPONSE FORMAT: Return a JSON object with this exact structure:
{
  "unnecessary_patterns": [
    {
      "text": "exact text to remove",
      "reason": "why this should be removed", 
      "category": "category_name"
    }
  ]
}

Return ONLY the JSON object, no additional text.

MEDICAL REPORT TO ANALYZE:
"""

        self.logger.info("MedicalReportCleanupAnalyzer initialized with Gemini 2.5 Pro")

    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown on Ctrl+C"""
        print(f"\n\n🛑 Keyboard interrupt received! Gracefully shutting down...")
        self.interrupted = True
        
        # Save current progress
        self._save_progress()
        
        print(f"📊 Current Progress:")
        print(f"  Successful batches: {self.stats['successful_batches']}")
        print(f"  Failed batches: {self.stats['failed_batches']}")
        print(f"  Reports processed: {self.stats['total_reports_processed']}")
        print(f"  Patterns found: {self.stats['total_patterns_found']}")
        
        print(f"👋 Analysis interrupted. Progress saved. Exiting gracefully...")
        sys.exit(0)

    def _setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/report_cleanup_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)

    def _validate_api_key(self) -> bool:
        """Validate Gemini API key"""
        if not self.api_key:
            self.logger.error("Gemini API key not provided. Set GEMINI_API_KEY environment variable.")
            return False
        return True

    def find_all_reports(self, base_path: str) -> List[str]:
        """Find all .txt report files in the MIMIC-Eye dataset structure"""
        pattern = os.path.join(base_path, "mimic-eye", "patient_*", "CXR-DICOM", "s*.txt")
        all_reports = glob.glob(pattern)
        
        self.logger.info(f"Found {len(all_reports)} total reports in {base_path}")
        return all_reports

    def select_random_reports(self, all_reports: List[str], sample_size: int = 1000) -> List[str]:
        """Randomly select reports for analysis"""
        if len(all_reports) <= sample_size:
            selected = all_reports
        else:
            selected = random.sample(all_reports, sample_size)
        
        self.logger.info(f"Selected {len(selected)} reports for analysis")
        return selected

    def read_report_content(self, file_path: str) -> str:
        """Read and return the content of a medical report"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().strip()
            return content
        except Exception as e:
            self.logger.error(f"Error reading report {file_path}: {e}")
            return ""

    def create_gemini_payload(self, report_content: str) -> Dict:
        """Create API payload for Gemini analysis"""
        full_prompt = self.analysis_prompt + "\n" + report_content
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": full_prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 1,
                "topP": 0.1,
                "maxOutputTokens": 8192,
                "candidateCount": 1
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
        }
        
        return payload

    def call_gemini_api(self, payload: Dict) -> str:
        """Make API call to Gemini with error handling and retry logic"""
        url = f"{self.api_config['base_url']}?key={self.api_key}"
        
        for attempt in range(self.api_config['max_retries']):
            try:
                response = requests.post(
                    url,
                    headers=self.api_config['headers'],
                    json=payload,
                    timeout=self.api_config['timeout']
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if 'candidates' in result and len(result['candidates']) > 0:
                        candidate = result['candidates'][0]
                        
                        if 'content' in candidate and 'parts' in candidate['content']:
                            text_response = candidate['content']['parts'][0]['text']
                            return text_response.strip()
                        elif 'finishReason' in candidate and candidate['finishReason'] == 'MAX_TOKENS':
                            self.logger.error("MAX_TOKENS hit during analysis")
                            return None
                        else:
                            self.logger.warning(f"Unexpected response structure: {result}")
                
                elif response.status_code == 429:  # Rate limit
                    wait_time = self.api_config['base_retry_delay'] * (2 ** attempt)
                    self.logger.warning(f"Rate limit hit (attempt {attempt + 1}), waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                elif response.status_code == 503:  # Service unavailable
                    wait_time = self.api_config['base_retry_delay'] * (2 ** attempt)
                    self.logger.warning(f"Service unavailable (attempt {attempt + 1}), waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    self.logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                    if attempt < self.api_config['max_retries'] - 1:
                        wait_time = self.api_config['base_retry_delay'] * (2 ** attempt)
                        self.logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    return None
                    
            except requests.exceptions.Timeout:
                wait_time = self.api_config['base_retry_delay'] * (2 ** attempt)
                self.logger.warning(f"API request timeout (attempt {attempt + 1})")
                if attempt < self.api_config['max_retries'] - 1:
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return None
                
            except Exception as e:
                wait_time = self.api_config['base_retry_delay'] * (2 ** attempt)
                self.logger.error(f"API request failed (attempt {attempt + 1}): {e}")
                if attempt < self.api_config['max_retries'] - 1:
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return None
        
        self.stats['api_errors'] += 1
        return None

    def parse_gemini_response(self, response_text: str) -> List[Dict]:
        """Parse Gemini's JSON response into pattern list"""
        try:
            # Try to extract JSON from response
            response_text = response_text.strip()
            
            # Remove markdown formatting if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            # Parse JSON
            parsed = json.loads(response_text)
            
            if 'unnecessary_patterns' in parsed:
                return parsed['unnecessary_patterns']
            else:
                self.logger.warning("No 'unnecessary_patterns' key in response")
                return []
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.error(f"Response text: {response_text[:500]}...")
            return []
        except Exception as e:
            self.logger.error(f"Error parsing response: {e}")
            return []

    def similarity_score(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def is_duplicate_pattern(self, new_pattern: str, existing_patterns: List[str], threshold: float = 0.8) -> bool:
        """Check if a pattern is too similar to existing patterns"""
        for existing in existing_patterns:
            if self.similarity_score(new_pattern, existing) >= threshold:
                return True
        return False

    def categorize_pattern(self, pattern_text: str, suggested_category: str) -> str:
        """Categorize pattern into predefined categories with fallback logic"""
        pattern_lower = pattern_text.lower()
        
        # Override category based on content analysis
        if any(word in pattern_lower for word in ['comparison', 'compared', 'prior', 'previous', 'study of']):
            return 'comparison_references'
        elif any(word in pattern_lower for word in ['examination', 'exam:', 'dx chest']):
            return 'examination_headers'
        elif any(word in pattern_lower for word in ['indication', 'history']):
            return 'indication_sections'
        elif any(word in pattern_lower for word in ['technique', 'single-view', 'portable']):
            return 'technique_sections'
        else:
            return suggested_category if suggested_category in self.consolidated_patterns else 'other_unnecessary'

    def add_pattern_to_consolidated(self, pattern_text: str, reason: str, category: str):
        """Add pattern to consolidated storage with deduplication"""
        category = self.categorize_pattern(pattern_text, category)
        
        if category not in self.consolidated_patterns:
            category = 'other_unnecessary'
        
        # Check for duplicates within category
        existing_examples = [ex['text'] for ex in self.consolidated_patterns[category]['examples']]
        
        if not self.is_duplicate_pattern(pattern_text, existing_examples):
            self.consolidated_patterns[category]['examples'].append({
                'text': pattern_text,
                'reason': reason
            })
            self.consolidated_patterns[category]['frequency'] += 1
            self.consolidated_patterns[category]['variations'].add(pattern_text)
            self.consolidated_patterns[category]['reasons'].append(reason)

    def process_batch(self, report_batch: List[str]) -> List[Dict]:
        """Process a batch of reports through Gemini AI"""
        batch_patterns = []
        
        for i, report_path in enumerate(report_batch):
            if self.interrupted:
                break
                
            try:
                # Show individual report progress
                report_name = os.path.basename(report_path)
                print(f"  📄 Processing report {i+1}/{len(report_batch)}: {report_name}")
                
                # Read report content
                content = self.read_report_content(report_path)
                if not content:
                    print(f"    ⚠️  Empty report, skipping...")
                    continue
                
                print(f"    🔍 Analyzing content ({len(content)} characters)...")
                
                # Create API payload
                payload = self.create_gemini_payload(content)
                
                # Call Gemini API
                response_text = self.call_gemini_api(payload)
                if not response_text:
                    print(f"    ❌ Failed to get AI response")
                    self.logger.error(f"Failed to get response for {report_path}")
                    continue
                
                # Parse response
                patterns = self.parse_gemini_response(response_text)
                print(f"    ✅ Found {len(patterns)} unnecessary patterns")
                
                # Add report context to patterns
                for pattern in patterns:
                    pattern['source_report'] = report_path
                    batch_patterns.append(pattern)
                
                self.stats['total_reports_processed'] += 1
                
                # Small delay between reports to avoid overwhelming API
                time.sleep(0.5)
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                self.logger.error(f"Error processing report {report_path}: {e}")
                continue
        
        return batch_patterns

    def consolidate_patterns_incremental(self, new_patterns: List[Dict]):
        """Add new patterns to consolidated storage with deduplication"""
        for pattern in new_patterns:
            if 'text' in pattern and 'reason' in pattern:
                category = pattern.get('category', 'other_unnecessary')
                self.add_pattern_to_consolidated(pattern['text'], pattern['reason'], category)
                self.stats['total_patterns_found'] += 1

    def _save_progress(self):
        """Save current progress to file"""
        try:
            # Convert sets to lists for JSON serialization
            serializable_patterns = {}
            for category, data in self.consolidated_patterns.items():
                serializable_patterns[category] = {
                    'pattern_type': data['pattern_type'],
                    'examples': data['examples'],
                    'frequency': data['frequency'],
                    'variations': list(data['variations']),
                    'reasons': data['reasons']
                }
            
            progress_data = {
                'stats': self.stats,
                'consolidated_patterns': serializable_patterns,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2)
            
            self.logger.info("Progress saved to file")
            
        except Exception as e:
            self.logger.error(f"Error saving progress: {e}")

    def _load_progress(self) -> bool:
        """Load previous progress if available"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                
                self.stats.update(progress_data['stats'])
                
                # Restore consolidated patterns
                for category, data in progress_data['consolidated_patterns'].items():
                    if category in self.consolidated_patterns:
                        self.consolidated_patterns[category]['examples'] = data['examples']
                        self.consolidated_patterns[category]['frequency'] = data['frequency']
                        self.consolidated_patterns[category]['variations'] = set(data['variations'])
                        self.consolidated_patterns[category]['reasons'] = data['reasons']
                
                self.logger.info(f"Loaded previous progress: {self.stats['successful_batches']} batches completed")
                return True
                
        except Exception as e:
            self.logger.error(f"Error loading progress: {e}")
        
        return False

    def generate_final_report(self, output_file: str):
        """Generate comprehensive analysis report"""
        
        # Calculate final statistics
        total_unique_patterns = sum(len(data['examples']) for data in self.consolidated_patterns.values())
        self.stats['unique_patterns_after_dedup'] = total_unique_patterns
        
        # Prepare final report
        report = {
            'analysis_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'model_used': 'gemini-2.5-pro',
                'total_reports_analyzed': self.stats['total_reports_processed'],
                'total_batches_processed': self.stats['successful_batches'],
                'total_patterns_found': self.stats['total_patterns_found'],
                'unique_patterns_after_deduplication': self.stats['unique_patterns_after_dedup'],
                'processing_time_seconds': time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
            },
            'processing_statistics': self.stats,
            'unnecessary_content_patterns': {}
        }
        
        # Add consolidated patterns (convert sets to lists)
        for category, data in self.consolidated_patterns.items():
            if data['examples']:  # Only include categories with patterns
                report['unnecessary_content_patterns'][category] = {
                    'pattern_type': data['pattern_type'],
                    'frequency': data['frequency'],
                    'example_count': len(data['examples']),
                    'examples': data['examples'][:20],  # Limit examples to first 20
                    'common_variations': list(data['variations'])[:10],  # Top 10 variations
                    'removal_reasons': list(set(data['reasons']))  # Unique reasons
                }
        
        # Save final report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Final analysis report saved to: {output_file}")

    def print_analysis_summary(self):
        """Print analysis summary to console"""
        print(f"\n=== MEDICAL REPORT CLEANUP ANALYSIS SUMMARY ===")
        print(f"Total reports processed: {self.stats['total_reports_processed']}")
        print(f"Total batches processed: {self.stats['successful_batches']}")
        print(f"Total patterns found: {self.stats['total_patterns_found']}")
        print(f"Unique patterns after dedup: {self.stats['unique_patterns_after_dedup']}")
        print(f"API errors: {self.stats['api_errors']}")
        
        print(f"\n=== PATTERN BREAKDOWN BY CATEGORY ===")
        for category, data in self.consolidated_patterns.items():
            if data['examples']:
                print(f"{category.replace('_', ' ').title()}: {data['frequency']} patterns")
                print(f"  Examples: {len(data['examples'])} unique")
                print(f"  Variations: {len(data['variations'])}")

    def main(self, 
             base_path: str = "../../mimic-eye-integrating-mimic-datasets-with-reflacx-and-eye-gaze-for-multimodal-deep-learning-applications-1.0.0",
             sample_size: int = 1000,
             batch_size: int = 25,
             output_file: str = "medical_report_cleanup_analysis.json"):
        """Main function to analyze medical reports for unnecessary content patterns"""
        
        if not self._validate_api_key():
            raise ValueError("Invalid API key")
        
        print("🏥 Starting Medical Report Cleanup Analysis with Gemini 2.5 Pro...")
        print(f"📂 Scanning reports in: {base_path}")
        print(f"📊 Target sample size: {sample_size} reports")
        print(f"📦 Batch size: {batch_size} reports per batch")
        print(f"💾 Progress saved after every batch")
        print(f"⚠️  Press Ctrl+C to gracefully interrupt and save progress")
        
        # Load previous progress if exists
        progress_loaded = self._load_progress()
        
        # Find all available reports
        print("🔍 Finding all available reports...")
        all_reports = self.find_all_reports(base_path)
        
        if len(all_reports) == 0:
            raise ValueError(f"No reports found in {base_path}")
        
        # Select random sample
        selected_reports = self.select_random_reports(all_reports, sample_size)
        
        # Create batches
        batches = [selected_reports[i:i + batch_size] for i in range(0, len(selected_reports), batch_size)]
        
        # Start from where we left off if progress was loaded
        start_batch = self.stats['successful_batches'] if progress_loaded else 0
        remaining_batches = batches[start_batch:]
        
        self.stats['total_batches'] = len(batches)
        if not self.stats['start_time']:
            self.stats['start_time'] = time.time()
        
        print(f"📦 Processing {len(remaining_batches)} remaining batches...")
        
        # Process batches with detailed progress
        for batch_idx, batch in enumerate(remaining_batches):
            if self.interrupted:
                break
            
            current_batch_num = start_batch + batch_idx + 1
            print(f"\n📦 Processing Batch {current_batch_num}/{self.stats['total_batches']} ({len(batch)} reports)")
            print(f"📊 Progress: {self.stats['total_reports_processed']}/1000 reports, {self.stats['total_patterns_found']} patterns found")
            
            try:
                # Process batch
                batch_patterns = self.process_batch(batch)
                
                # Consolidate patterns with deduplication
                self.consolidate_patterns_incremental(batch_patterns)
                
                # Update statistics
                self.stats['successful_batches'] += 1
                
                # Save progress after every batch
                self._save_progress()
                
                print(f"✅ Batch {current_batch_num} completed: {len(batch_patterns)} new patterns found")
                print(f"📈 Total progress: {self.stats['total_reports_processed']}/1000 reports ({(self.stats['total_reports_processed']/1000*100):.1f}%)")
                
                # Rate limiting
                time.sleep(self.api_config['rate_limit_delay'])
                
            except KeyboardInterrupt:
                self.interrupted = True
                break
            except Exception as e:
                print(f"❌ Batch {current_batch_num} failed: {e}")
                self.logger.error(f"Error processing batch {start_batch + batch_idx}: {e}")
                self.stats['failed_batches'] += 1
                continue
        
        # Generate final report
        if not self.interrupted:
            print("📊 Generating final analysis report...")
            self.generate_final_report(output_file)
            
            # Clean up progress file
            if os.path.exists(self.progress_file):
                os.remove(self.progress_file)
        
        # Print summary
        self.print_analysis_summary()
        
        if self.interrupted:
            print("🛑 Analysis interrupted. Progress saved. Run again to continue.")
        else:
            print(f"✅ Analysis completed! Results saved to: {output_file}")

if __name__ == "__main__":
    analyzer = MedicalReportCleanupAnalyzer()
    analyzer.main()
