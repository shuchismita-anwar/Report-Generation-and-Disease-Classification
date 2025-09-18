import os
import json
import requests
import time
import logging
import signal
import sys
from typing import Dict, List, Tuple
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm

class GeminiAICleanup:
    """
    AI-powered keyword cleanup using Gemini 2.5 Flash to filter out irrelevant keywords
    from medical condition keyword lists. Preserves confidence levels but tells AI to ignore them.
    """
    
    def __init__(self, api_key: str = None):
        # Load environment variables
        load_dotenv('.env')
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # Setup logging
        self._setup_logging()
        
        # API configuration for Gemini 2.5 Pro with thinking mode
        self.api_config = {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
            "headers": {"Content-Type": "application/json"},
            "max_retries": 5,       # Increased retries
            "base_retry_delay": 3,  # Base delay for exponential backoff
            "timeout": 120,         # Increased timeout to 120 seconds
            "rate_limit_delay": 2.0 # Increased delay for Pro model
        }
        
        # Processing statistics
        self.stats = {
            "total_batches": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "total_keywords_processed": 0,
            "total_keywords_kept": 0,
            "total_keywords_removed": 0,
            "api_errors": 0,
            "recovered_batches": 0,
            "start_time": None
        }
        
        # Failed batch tracking
        self.failed_batches_file = "failed_batches_recovery.json"
        self.failed_batches = []
        
        # Setup graceful shutdown
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Dialed back prompt - less aggressive but still targets real issues
        self.filtering_prompt = """TASK: Medical keyword relevance filtering. Answer ONLY "YES" or "NO" for each keyword-condition pair.

IMPORTANT: You will see confidence levels (high/medium/low) - IGNORE THEM COMPLETELY. Treat all keywords equally regardless of confidence level.

RULES - Answer "NO" if keyword is:

1. CLEARLY VAGUE/UNINFORMATIVE:
   - "evaluate", "eval", "assess", "check for", "r/o"
   - "no evidence of", "cannot be excluded", "rule out"
   - Pure symptoms without diagnostic value: "fever", "chills" (unless condition-specific)
   - "normal", "clear", "unchanged", "stable" (when alone, not condition-specific)

2. UNCERTAINTY PHRASES:
   - Contains "?", "possibly", "may be", "concerning for", "suspicious for"
   - "cannot exclude", "cannot be ruled out", "difficult to exclude"

3. OBVIOUS DUPLICATES/VARIANTS of same concept:
   - Multiple variations of same phrase (e.g. "lung volumes are low", "lung volumes remain low", "lung volumes somewhat low")
   - Keep the most common/standard medical term
   - Minor formatting differences: punctuation, spacing, plurals, casing

4. CLEAR CONDITION MISMATCH:
   - Device terms in disease conditions (e.g. "pacemaker" in Pneumonia)
   - Disease terms in device conditions (e.g. "pneumonia" in Support Devices)
   - Anatomical regions that don't relate to the condition

5. NON-MEDICAL WORDS:
   - "the", "and", "of", "in", "with", "are", "is", "was"
   - Pure procedural language: "demonstrated", "noted", "seen", "identified"

BE LIBERAL with keeping medical terms. When in doubt, KEEP the keyword. Only remove if it clearly falls into above categories.

RESPONSE FORMAT: Only output "YES" or "NO" - one per line. NO explanations. NO additional text. NO numbering.

Examples:
Condition: Pneumonia, Keyword: "consolidation" → YES
Condition: Pneumonia, Keyword: "pneumonia" → YES (keep condition name itself)
Condition: Cardiomegaly, Keyword: "cardiomegaly" → YES (keep condition name)
Condition: Pneumonia, Keyword: "lung volumes are low" → YES (medical finding)
Condition: Pneumonia, Keyword: "lung volumes remain low" → NO (duplicate variant)
Condition: Pneumonia, Keyword: "pacemaker" → NO (irrelevant to condition)
Condition: Support Devices, Keyword: "catheter" → YES

Process these keyword-condition-confidence triplets (IGNORE CONFIDENCE LEVELS):"""

        self.logger.info("GeminiAICleanup initialized with Gemini 2.5 Pro")

    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown on Ctrl+C"""
        print(f"\n\n🛑 Keyboard interrupt received! Gracefully shutting down...")
        self.interrupted = True
        
        # Save any failed batches immediately
        if self.failed_batches:
            print(f"💾 Saving {len(self.failed_batches)} failed batches before exit...")
            self.save_failed_batches(self.failed_batches)
        
        print(f"📊 Current Progress:")
        print(f"  Successful batches: {self.stats['successful_batches']}")
        print(f"  Failed batches: {self.stats['failed_batches']}")
        print(f"  Keywords processed: {self.stats['total_keywords_processed']}")
        print(f"  Keywords kept: {self.stats['total_keywords_kept']}")
        print(f"  Keywords removed: {self.stats['total_keywords_removed']}")
        
        if self.failed_batches:
            print(f"🔄 Run the script again to automatically process the {len(self.failed_batches)} failed batches.")
        
        print(f"👋 Cleanup interrupted. Exiting gracefully...")
        sys.exit(0)

    def _setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/ai_cleanup.log'),
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

    def load_keywords_file(self, file_path: str) -> Dict:
        """Load the keywords JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"Keywords file loaded: {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load keywords file: {e}")
            raise

    def save_keywords_file(self, data: Dict, file_path: str) -> None:
        """Save the cleaned keywords JSON file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Cleaned keywords saved to: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save keywords file: {e}")
            raise

    def create_keyword_batches_with_confidence(self, conditions_data: Dict, batch_size: int = 30) -> List[List[Tuple[str, str, str]]]:
        """Create batches of (condition, confidence, keyword) triplets - reduced batch size for reliability"""
        batches = []
        current_batch = []
        
        for condition_name, condition_data in conditions_data.items():
            for confidence_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
                if confidence_level in condition_data:
                    confidence_label = confidence_level.replace('_confidence', '')  # high, medium, low
                    
                    for keyword in condition_data[confidence_level]:
                        if keyword and keyword.strip():  # Skip empty keywords
                            current_batch.append((condition_name, confidence_label, keyword.strip()))
                            
                            if len(current_batch) >= batch_size:
                                batches.append(current_batch)
                                current_batch = []
        
        # Add remaining keywords
        if current_batch:
            batches.append(current_batch)
        
        self.logger.info(f"Created {len(batches)} batches with {batch_size} keywords each (preserving confidence levels)")
        return batches

    def load_failed_batches(self) -> List:
        """Load any previously failed batches for recovery"""
        try:
            if os.path.exists(self.failed_batches_file):
                with open(self.failed_batches_file, 'r', encoding='utf-8') as f:
                    failed_batches = json.load(f)
                self.logger.info(f"Loaded {len(failed_batches)} failed batches for recovery")
                return failed_batches
            return []
        except Exception as e:
            self.logger.error(f"Error loading failed batches: {e}")
            return []

    def save_failed_batches(self, failed_batches: List) -> None:
        """Save failed batches for later recovery - IMMEDIATELY"""
        try:
            with open(self.failed_batches_file, 'w', encoding='utf-8') as f:
                json.dump(failed_batches, f, indent=2)
            self.logger.info(f"Saved {len(failed_batches)} failed batches for recovery")
        except Exception as e:
            self.logger.error(f"Error saving failed batches: {e}")

    def save_failed_batch_immediately(self, batch: List) -> None:
        """Save a single failed batch immediately to the recovery file"""
        try:
            # Load existing failed batches
            existing_batches = []
            if os.path.exists(self.failed_batches_file):
                try:
                    with open(self.failed_batches_file, 'r', encoding='utf-8') as f:
                        existing_batches = json.load(f)
                except:
                    existing_batches = []
            
            # Add new failed batch
            existing_batches.append(batch)
            
            # Save immediately
            with open(self.failed_batches_file, 'w', encoding='utf-8') as f:
                json.dump(existing_batches, f, indent=2)
            
            self.logger.info(f"Immediately saved failed batch to recovery file (total: {len(existing_batches)})")
            
        except Exception as e:
            self.logger.error(f"Error immediately saving failed batch: {e}")

    def clear_failed_batches(self) -> None:
        """Clear the failed batches file after successful recovery"""
        try:
            if os.path.exists(self.failed_batches_file):
                os.remove(self.failed_batches_file)
                self.logger.info("Cleared failed batches file after successful recovery")
        except Exception as e:
            self.logger.error(f"Error clearing failed batches file: {e}")

    def save_progress_immediately(self, cleaned_conditions: Dict, original_conditions: Dict) -> None:
        """Save current progress immediately after each batch"""
        try:
            # Load the original data structure
            input_file = "extracted_keywords_result_cleaned.json"
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update with current progress
            data['conditions'] = cleaned_conditions
            
            # Update metadata with current stats
            if 'metadata' in data:
                data['metadata']['ai_cleanup_progress'] = {
                    'last_updated': datetime.now().isoformat(),
                    'batches_processed': self.stats['successful_batches'],
                    'total_batches': self.stats['total_batches'],
                    'keywords_processed': self.stats['total_keywords_processed'],
                    'keywords_kept': self.stats['total_keywords_kept'],
                    'keywords_removed': self.stats['total_keywords_removed']
                }
            
            # Save immediately
            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Progress saved after batch {self.stats['successful_batches']}/{self.stats['total_batches']}")
            
        except Exception as e:
            self.logger.error(f"Error saving progress immediately: {e}")
            # Don't fail the entire process if saving fails

    def create_gemini_payload(self, keyword_batch: List[Tuple[str, str, str]]) -> Dict:
        """Create API payload for Gemini with confidence-aware format"""
        
        # Build the prompt with keyword triplets
        prompt_lines = [self.filtering_prompt, ""]
        
        for i, (condition, confidence, keyword) in enumerate(keyword_batch, 1):
            prompt_lines.append(f"{i}. Condition: {condition}, Confidence: {confidence}, Keyword: \"{keyword}\"")
        
        full_prompt = "\n".join(prompt_lines)
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": full_prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,  # Very low temperature for consistent responses
                "topK": 1,
                "topP": 0.1,
                "maxOutputTokens": 65535,  # Max tokens for response
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
        """Make API call to Gemini with improved error handling and exponential backoff"""
        url = f"{self.api_config['base_url']}?key={self.api_key}"
        
        for attempt in range(self.api_config['max_retries']):
            try:
                self.logger.debug(f"Making API call (attempt {attempt + 1}/{self.api_config['max_retries']})")
                
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
                            self.logger.debug("API call successful")
                            return text_response.strip()
                        elif 'finishReason' in candidate and candidate['finishReason'] == 'MAX_TOKENS':
                            self.logger.error(f"MAX_TOKENS hit - Gemini used {result.get('usageMetadata', {}).get('thoughtsTokenCount', 'unknown')} tokens for internal thoughts!")
                            return None
                        else:
                            self.logger.warning(f"Unexpected response structure: {result}")
                
                elif response.status_code == 429:  # Rate limit
                    wait_time = self.api_config['base_retry_delay'] * (2 ** attempt)  # Exponential backoff
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
                self.logger.warning(f"API request timeout (attempt {attempt + 1}) - {self.api_config['timeout']}s")
                if attempt < self.api_config['max_retries'] - 1:
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return None
                
            except requests.exceptions.ConnectionError as e:
                wait_time = self.api_config['base_retry_delay'] * (2 ** attempt)
                self.logger.warning(f"Connection error (attempt {attempt + 1}): {e}")
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
        self.logger.error("All API retry attempts failed")
        return None

    def parse_gemini_responses(self, response_text: str, batch_size: int) -> List[bool]:
        """Parse Gemini's YES/NO responses into boolean list"""
        try:
            lines = [line.strip().upper() for line in response_text.split('\n') if line.strip()]
            decisions = []
            
            for line in lines:
                if line == 'YES':
                    decisions.append(True)
                elif line == 'NO':
                    decisions.append(False)
                else:
                    # Handle unexpected responses - be more flexible
                    if 'YES' in line:
                        decisions.append(True)
                    elif 'NO' in line:
                        decisions.append(False)
                    else:
                        self.logger.warning(f"Unexpected response line: {line}")
                        # Default to keeping the keyword if response is unclear
                        decisions.append(True)
            
            # Ensure we have the right number of responses
            if len(decisions) != batch_size:
                self.logger.warning(f"Expected {batch_size} decisions, got {len(decisions)}")
                # Pad with True if we're missing responses
                while len(decisions) < batch_size:
                    decisions.append(True)
                # Truncate if we have too many
                decisions = decisions[:batch_size]
            
            return decisions
            
        except Exception as e:
            self.logger.error(f"Error parsing Gemini responses: {e}")
            # Return all True to keep keywords if parsing fails
            return [True] * batch_size

    def process_batch(self, keyword_batch: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str, bool]]:
        """Process a batch of (condition, confidence, keyword) triplets through Gemini AI"""
        try:
            # Check for interruption
            if self.interrupted:
                return [(condition, confidence, keyword, True) for condition, confidence, keyword in keyword_batch]
            
            # Create API payload
            payload = self.create_gemini_payload(keyword_batch)
            
            # Call Gemini API
            response_text = self.call_gemini_api(payload)
            if not response_text:
                self.logger.error("Failed to get response from Gemini - saving batch for recovery IMMEDIATELY")
                # Save failed batch immediately
                self.save_failed_batch_immediately(keyword_batch)
                # Return with all keywords kept for now
                return [(condition, confidence, keyword, True) for condition, confidence, keyword in keyword_batch]
            
            # Parse responses
            decisions = self.parse_gemini_responses(response_text, len(keyword_batch))
            
            # Combine with original data
            results = []
            for i, (condition, confidence, keyword) in enumerate(keyword_batch):
                keep_keyword = decisions[i] if i < len(decisions) else True
                results.append((condition, confidence, keyword, keep_keyword))
            
            self.stats['successful_batches'] += 1
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            self.stats['failed_batches'] += 1
            # Save failed batch immediately
            self.save_failed_batch_immediately(keyword_batch)
            # Return with all keywords kept for now
            return [(condition, confidence, keyword, True) for condition, confidence, keyword in keyword_batch]

    def filter_keywords_with_ai(self, conditions_data: Dict, batch_size: int = 30) -> Dict:
        """Filter keywords using AI with batch processing - preserves confidence levels"""
        
        # Create batches with confidence information
        batches = self.create_keyword_batches_with_confidence(conditions_data, batch_size)
        
        # Initialize cleaned data structure - preserve original structure
        cleaned_conditions = {}
        for condition_name, condition_data in conditions_data.items():
            cleaned_conditions[condition_name] = {
                'high_confidence': [],
                'medium_confidence': [],
                'low_confidence': []
            }
            # Preserve extraction stats if they exist
            if 'extraction_stats' in condition_data:
                cleaned_conditions[condition_name]['extraction_stats'] = condition_data['extraction_stats']
        
        self.stats['total_batches'] = len(batches)
        self.stats['start_time'] = time.time()
        
        # Process batches with progress bar
        with tqdm(total=len(batches), desc="AI filtering batches", unit="batch") as pbar:
            for batch_idx, batch in enumerate(batches):
                try:
                    # Check for interruption
                    if self.interrupted:
                        self.logger.info("Processing interrupted by user")
                        break
                    
                    # Process batch
                    results = self.process_batch(batch)
                    
                    # Count statistics
                    batch_kept = 0
                    batch_removed = 0
                    
                    # Apply filtering results - preserve confidence levels
                    for condition, confidence, keyword, keep in results:
                        self.stats['total_keywords_processed'] += 1
                        
                        confidence_key = f"{confidence}_confidence"
                        
                        if keep:
                            cleaned_conditions[condition][confidence_key].append(keyword)
                            batch_kept += 1
                            self.stats['total_keywords_kept'] += 1
                        else:
                            batch_removed += 1
                            self.stats['total_keywords_removed'] += 1
                    
                    # Save progress after every batch
                    self.save_progress_immediately(cleaned_conditions, conditions_data)
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Kept': batch_kept,
                        'Removed': batch_removed,
                        'Success Rate': f"{self.stats['successful_batches']}/{batch_idx + 1}"
                    })
                    pbar.update(1)
                    
                    # Rate limiting - improved delay
                    time.sleep(self.api_config['rate_limit_delay'])
                    
                except KeyboardInterrupt:
                    self.logger.info("Processing interrupted by user")
                    self.interrupted = True
                    break
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_idx}: {e}")
                    pbar.update(1)
                    continue
        
        return cleaned_conditions

    def process_recovery_batches(self, recovery_batches: List) -> Dict:
        """Process any failed batches from previous runs"""
        recovery_results = {}
        
        with tqdm(total=len(recovery_batches), desc="Processing recovery batches", unit="batch") as pbar:
            for batch_idx, batch in enumerate(recovery_batches):
                try:
                    # Check for interruption
                    if self.interrupted:
                        break
                    
                    # Process recovery batch
                    results = self.process_batch(batch)
                    
                    # Apply filtering results - preserve confidence structure
                    for condition, confidence, keyword, keep in results:
                        if condition not in recovery_results:
                            recovery_results[condition] = {
                                'high_confidence': [],
                                'medium_confidence': [],
                                'low_confidence': []
                            }
                        
                        confidence_key = f"{confidence}_confidence"
                        
                        if keep:
                            recovery_results[condition][confidence_key].append(keyword)
                            self.stats['total_keywords_kept'] += 1
                        else:
                            self.stats['total_keywords_removed'] += 1
                    
                    self.stats['recovered_batches'] += 1
                    pbar.update(1)
                    
                    # Rate limiting
                    time.sleep(self.api_config['rate_limit_delay'])
                    
                except KeyboardInterrupt:
                    self.interrupted = True
                    break
                except Exception as e:
                    self.logger.error(f"Error processing recovery batch {batch_idx}: {e}")
                    pbar.update(1)
                    continue
        
        return recovery_results

    def merge_recovery_results(self, cleaned_conditions: Dict, recovery_results: Dict) -> Dict:
        """Merge recovery results with main results"""
        for condition_name, recovery_data in recovery_results.items():
            if condition_name not in cleaned_conditions:
                cleaned_conditions[condition_name] = {
                    'high_confidence': [],
                    'medium_confidence': [],
                    'low_confidence': []
                }
            
            # Merge each confidence level
            for confidence_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
                if confidence_level in recovery_data:
                    existing_keywords = set(cleaned_conditions[condition_name][confidence_level])
                    recovery_keywords = set(recovery_data[confidence_level])
                    merged_keywords = existing_keywords.union(recovery_keywords)
                    cleaned_conditions[condition_name][confidence_level] = sorted(list(merged_keywords))
        
        return cleaned_conditions

    def print_cleanup_stats(self, original_data: Dict, cleaned_data: Dict):
        """Print detailed cleanup statistics"""
        print(f"\n=== AI CLEANUP STATISTICS ===")
        print(f"Total batches processed: {self.stats['total_batches']}")
        print(f"Successful batches: {self.stats['successful_batches']}")
        print(f"Failed batches: {self.stats['failed_batches']}")
        print(f"Recovered batches: {self.stats['recovered_batches']}")
        print(f"API errors: {self.stats['api_errors']}")
        print(f"Total keywords processed: {self.stats['total_keywords_processed']}")
        print(f"Total keywords kept: {self.stats['total_keywords_kept']}")
        print(f"Total keywords removed: {self.stats['total_keywords_removed']}")
        if self.stats['total_keywords_processed'] > 0:
            print(f"Removal rate: {(self.stats['total_keywords_removed']/self.stats['total_keywords_processed']*100):.1f}%")
        
        # Per-condition statistics
        print(f"\n=== PER-CONDITION BREAKDOWN ===")
        for condition_name in original_data.keys():
            original_total = 0
            cleaned_total = 0
            
            # Count original keywords across all confidence levels
            for conf_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
                if conf_level in original_data[condition_name]:
                    original_total += len(original_data[condition_name][conf_level])
            
            # Count cleaned keywords across all confidence levels  
            for conf_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
                if conf_level in cleaned_data[condition_name]:
                    cleaned_total += len(cleaned_data[condition_name][conf_level])
            
            removed = original_total - cleaned_total
            removal_rate = (removed / original_total * 100) if original_total > 0 else 0
            
            print(f"{condition_name}:")
            print(f"  Original: {original_total} keywords")
            print(f"  Kept: {cleaned_total} keywords")
            print(f"  Removed: {removed} keywords ({removal_rate:.1f}%)")

    def main(self, input_file: str = "extracted_keywords_result_cleaned.json", 
             batch_size: int = 30):  # Reduced default batch size for better reliability
        """Main function to perform AI cleanup - saves to same input file"""
        
        if not self._validate_api_key():
            raise ValueError("Invalid API key")
        
        print("🤖 Starting AI-powered keyword cleanup with Gemini 2.5 Flash...")
        print(f"📂 Processing file: {input_file}")
        print(f"📦 Batch size: {batch_size} keywords per batch (reduced for reliability)")
        print(f"🎯 Strategy: Send confidence levels but instruct AI to ignore them")
        print(f"💾 Progress is saved after EVERY batch - no progress will be lost!")
        print(f"⚠️  Press Ctrl+C to gracefully interrupt and save progress")
        
        # Load keywords
        print("📂 Loading keywords file...")
        data = self.load_keywords_file(input_file)
        
        if 'conditions' not in data:
            raise ValueError("No 'conditions' section found in the data!")
        
        original_conditions = data['conditions']
        
        # Check for recovery batches first
        recovery_batches = self.load_failed_batches()
        recovery_results = {}
        
        if recovery_batches:
            print(f"🔄 Found {len(recovery_batches)} failed batches from previous run - processing recovery...")
            recovery_results = self.process_recovery_batches(recovery_batches)
            # Clear failed batches file after successful recovery
            if recovery_results and not self.interrupted:
                self.clear_failed_batches()
        
        # Check if interrupted during recovery
        if self.interrupted:
            print("🛑 Interrupted during recovery - exiting...")
            return None
        
        # Filter keywords with AI - preserving confidence levels
        print("🧠 Filtering keywords with AI (preserving confidence levels)...")
        cleaned_conditions = self.filter_keywords_with_ai(original_conditions, batch_size)
        
        # Check if interrupted during main processing
        if self.interrupted:
            print("🛑 Interrupted during main processing - progress saved...")
            return None
        
        # Merge recovery results if any
        if recovery_results:
            print("🔄 Merging recovery results...")
            cleaned_conditions = self.merge_recovery_results(cleaned_conditions, recovery_results)
        
        # Update data structure
        cleaned_data = data.copy()
        cleaned_data['conditions'] = cleaned_conditions
        
        # Print statistics
        self.print_cleanup_stats(original_conditions, cleaned_conditions)
        
        # Save failed batches if any occurred during this run
        if self.failed_batches:
            print(f"⚠️  Saving {len(self.failed_batches)} failed batches for next run...")
            self.save_failed_batches(self.failed_batches)
        
        # Final save with complete metadata (progress was saved after each batch)
        print(f"💾 Finalizing cleaned keywords in {input_file}...")
        cleaned_data['metadata']['ai_cleanup'] = {
            'cleanup_date': datetime.now().isoformat(),
            'model_used': 'gemini-2.5-pro',
            'strategy': 'confidence_aware_but_ignored',
            'total_keywords_processed': self.stats['total_keywords_processed'],
            'total_keywords_kept': self.stats['total_keywords_kept'],
            'total_keywords_removed': self.stats['total_keywords_removed'],
            'removal_rate': f"{(self.stats['total_keywords_removed']/self.stats['total_keywords_processed']*100):.1f}%" if self.stats['total_keywords_processed'] > 0 else "0.0%",
            'completed': True
        }
        self.save_keywords_file(cleaned_data, input_file)
        
        print("✅ AI cleanup completed! Progress was saved after every batch.")
        return cleaned_data

if __name__ == "__main__":
    # Initialize and run cleanup
    cleanup = GeminiAICleanup()
    cleanup.main()
