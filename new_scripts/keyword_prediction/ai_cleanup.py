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
    from medical condition keyword lists.
    """
    
    def __init__(self, api_key: str = None):
        # Load environment variables
        load_dotenv('.env')
        
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # Setup logging
        self._setup_logging()
        
        # API configuration for Gemini 2.5 Flash
        self.api_config = {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            "headers": {"Content-Type": "application/json"},
            "max_retries": 3,  # Reduced retries for faster failure detection
            "retry_delay": 2,   # Reduced delay
            "timeout": 60       # Reduced timeout from 120 to 60 seconds
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
        
        # Iron-clad prompt for keyword filtering
        self.filtering_prompt = """TASK: Medical keyword relevance filtering. Answer ONLY "YES" or "NO" for each keyword-condition pair.

STRICT RULES - Answer "NO" if keyword is:
1. Generic observations: "are seen", "noted", "present", "demonstrated", "identified", "observed", "apparent", "evident"
2. Anatomical locations alone: "left lower lobe", "right upper lobe", "bilateral", "bases", "left", "right", "upper", "lower"
3. Intensity modifiers alone: "mild", "severe", "moderate", "chronic", "acute", "minimal", "extensive", "marked", "significant"
4. Temporal descriptors: "again", "previously", "currently", "new", "old", "prior", "recent", "unchanged", "stable", "improving"
5. Procedural terms alone: "post-surgical", "post-operative", "status post", "following", "after", "procedure"
6. Vague descriptors: "changes", "findings", "abnormality", "opacity", "process", "condition", "appearance", "pattern"
7. Non-medical words: "the", "and", "of", "in", "with", "associated", "compatible", "consistent", "likely", "possible"
8. Pure directional/positional: "above", "below", "adjacent", "near", "close", "distant", "surrounding"
9. Generic verbs/actions: "shows", "reveals", "suggests", "indicates", "represents", "reflects", "appears"

Answer "YES" ONLY if keyword contains SPECIFIC medical information directly related to the condition.

RESPONSE FORMAT: Only output "YES" or "NO" - one per line. NO explanations. NO additional text. NO numbering.

Examples:
Condition: Atelectasis, Keyword: "collapsed lung" → YES
Condition: Atelectasis, Keyword: "are seen" → NO
Condition: Pneumonia, Keyword: "consolidation" → YES
Condition: Pneumonia, Keyword: "mild" → NO
Condition: Pleural Effusion, Keyword: "pleural fluid" → YES
Condition: Pleural Effusion, Keyword: "bilateral" → NO

Process these keyword-condition pairs:"""

        self.logger.info("GeminiAICleanup initialized with Gemini 2.5 Flash")

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

    def flatten_keywords(self, conditions_data: Dict) -> Dict:
        """Flatten confidence levels and merge all keywords per condition"""
        flattened = {}
        
        for condition_name, condition_data in conditions_data.items():
            all_keywords = set()  # Use set to automatically handle duplicates
            
            # Merge all confidence levels
            for confidence_level in ['high_confidence', 'medium_confidence', 'low_confidence']:
                if confidence_level in condition_data:
                    for keyword in condition_data[confidence_level]:
                        if keyword and keyword.strip():  # Skip empty keywords
                            all_keywords.add(keyword.strip())
            
            # Convert back to sorted list
            flattened[condition_name] = sorted(list(all_keywords))
            
        self.logger.info(f"Flattened keywords from confidence levels for {len(flattened)} conditions")
        return flattened

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

    def create_keyword_batches(self, flattened_conditions: Dict, batch_size: int = 50) -> List[List[Tuple[str, str]]]:
        """
        Create batches of (condition, keyword) tuples for processing.
        
        Args:
            flattened_conditions: The flattened conditions dictionary
            batch_size: Number of keyword-condition pairs per batch (reduced to 50)
            
        Returns:
            List of batches, each containing tuples of (condition, keyword)
        """
        keyword_pairs = []
        
        # Collect all keyword-condition pairs (ignoring confidence levels)
        for condition_name, keywords in flattened_conditions.items():
            for keyword in keywords:
                keyword_pairs.append((condition_name, keyword))
        
        # Split into batches
        batches = []
        for i in range(0, len(keyword_pairs), batch_size):
            batch = keyword_pairs[i:i + batch_size]
            batches.append(batch)
        
        self.logger.info(f"Created {len(batches)} batches from {len(keyword_pairs)} keyword-condition pairs")
        return batches

    def create_gemini_payload(self, keyword_batch: List[Tuple[str, str]]) -> Dict:
        """Create the API payload for Gemini with a batch of keywords"""
        
        # Format the keyword pairs for the prompt
        batch_text = "\n".join([
            f"Condition: {condition}, Keyword: \"{keyword}\""
            for condition, keyword in keyword_batch
        ])
        
        full_prompt = f"{self.filtering_prompt}\n\n{batch_text}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": full_prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 1,
                "topP": 0.8,
                "maxOutputTokens": 4096,  # Reduced from 8192 for faster responses
                "stopSequences": []
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH", 
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        return payload

    def call_gemini_api(self, payload: Dict) -> str:
        """Make API call to Gemini with robust error handling"""
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
                
                elif response.status_code == 429:  # Rate limit
                    wait_time = (attempt + 1) * 5  # Reduced wait time
                    self.logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    self.logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                    if attempt < self.api_config['max_retries'] - 1:
                        time.sleep(self.api_config['retry_delay'])
                        continue
                    return None
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"API request timeout (attempt {attempt + 1}) - {self.api_config['timeout']}s")
                if attempt < self.api_config['max_retries'] - 1:
                    wait_time = self.api_config['retry_delay'] * (attempt + 1)
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return None
            except requests.exceptions.ConnectionError as e:
                self.logger.warning(f"Connection error (attempt {attempt + 1}): {e}")
                if attempt < self.api_config['max_retries'] - 1:
                    wait_time = self.api_config['retry_delay'] * (attempt + 1)
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                return None
            except Exception as e:
                self.logger.error(f"API request failed (attempt {attempt + 1}): {e}")
                if attempt < self.api_config['max_retries'] - 1:
                    time.sleep(self.api_config['retry_delay'])
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
                    # Handle unexpected responses
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

    def process_batch(self, keyword_batch: List[Tuple[str, str]]) -> List[Tuple[str, str, bool]]:
        """Process a batch of keywords through Gemini AI"""
        try:
            # Check for interruption
            if self.interrupted:
                return [(condition, keyword, True) for condition, keyword in keyword_batch]
            
            # Create API payload
            payload = self.create_gemini_payload(keyword_batch)
            
            # Call Gemini API
            response_text = self.call_gemini_api(payload)
            if not response_text:
                self.logger.error("Failed to get response from Gemini - saving batch for recovery IMMEDIATELY")
                # Save failed batch immediately
                self.save_failed_batch_immediately(keyword_batch)
                # Return with all keywords kept for now
                return [(condition, keyword, True) for condition, keyword in keyword_batch]
            
            # Parse responses
            decisions = self.parse_gemini_responses(response_text, len(keyword_batch))
            
            # Combine with original data
            results = []
            for i, (condition, keyword) in enumerate(keyword_batch):
                keep_keyword = decisions[i] if i < len(decisions) else True
                results.append((condition, keyword, keep_keyword))
            
            self.stats['successful_batches'] += 1
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing batch: {e}")
            self.stats['failed_batches'] += 1
            # Save failed batch immediately
            self.save_failed_batch_immediately(keyword_batch)
            # Return with all keywords kept for now
            return [(condition, keyword, True) for condition, keyword in keyword_batch]

    def filter_keywords_with_ai(self, flattened_conditions: Dict, batch_size: int = 50) -> Dict:
        """Filter keywords using AI with batch processing"""
        
        # Create batches
        batches = self.create_keyword_batches(flattened_conditions, batch_size)
        
        # Initialize cleaned data structure
        cleaned_conditions = {}
        for condition_name in flattened_conditions.keys():
            cleaned_conditions[condition_name] = []
        
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
                    
                    # Apply filtering results
                    for condition, keyword, keep in results:
                        self.stats['total_keywords_processed'] += 1
                        
                        if keep:
                            cleaned_conditions[condition].append(keyword)
                            batch_kept += 1
                            self.stats['total_keywords_kept'] += 1
                        else:
                            batch_removed += 1
                            self.stats['total_keywords_removed'] += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Kept': batch_kept,
                        'Removed': batch_removed,
                        'Success Rate': f"{self.stats['successful_batches']}/{batch_idx + 1}"
                    })
                    pbar.update(1)
                    
                    # Rate limiting - reduced to 0.5 seconds
                    time.sleep(0.5)
                    
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
        """Process previously failed batches for recovery"""
        if not recovery_batches:
            return {}
        
        self.logger.info(f"Processing {len(recovery_batches)} recovery batches...")
        
        recovery_results = {}
        
        with tqdm(total=len(recovery_batches), desc="Processing recovery batches", unit="batch") as pbar:
            for batch_idx, batch in enumerate(recovery_batches):
                try:
                    # Check for interruption
                    if self.interrupted:
                        break
                    
                    # Process recovery batch
                    results = self.process_batch(batch)
                    
                    # Apply filtering results
                    for condition, keyword, keep in results:
                        if condition not in recovery_results:
                            recovery_results[condition] = []
                        
                        if keep:
                            recovery_results[condition].append(keyword)
                            self.stats['total_keywords_kept'] += 1
                        else:
                            self.stats['total_keywords_removed'] += 1
                    
                    self.stats['recovered_batches'] += 1
                    pbar.update(1)
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except KeyboardInterrupt:
                    self.interrupted = True
                    break
                except Exception as e:
                    self.logger.error(f"Error processing recovery batch {batch_idx}: {e}")
                    pbar.update(1)
                    continue
        
        return recovery_results

    def restructure_to_original_format(self, flattened_cleaned: Dict, original_structure: Dict) -> Dict:
        """Restructure cleaned keywords back to original confidence level format"""
        restructured = {}
        
        for condition_name in original_structure.keys():
            restructured[condition_name] = {
                'high_confidence': [],
                'medium_confidence': [],
                'low_confidence': []
            }
            
            # Get cleaned keywords for this condition
            cleaned_keywords = flattened_cleaned.get(condition_name, [])
            
            # Distribute cleaned keywords across confidence levels
            # For simplicity, we'll put all cleaned keywords in high_confidence
            # You could implement more sophisticated distribution logic here
            restructured[condition_name]['high_confidence'] = cleaned_keywords
            
            # Preserve extraction stats if they exist
            if 'extraction_stats' in original_structure[condition_name]:
                restructured[condition_name]['extraction_stats'] = original_structure[condition_name]['extraction_stats']
        
        return restructured

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
             batch_size: int = 50):  # Reduced default batch size
        """Main function to perform AI cleanup - saves to same input file"""
        
        if not self._validate_api_key():
            raise ValueError("Invalid API key")
        
        print("🤖 Starting AI-powered keyword cleanup with Gemini 2.5 Flash...")
        print(f"📂 Processing file: {input_file}")
        print(f"📦 Batch size: {batch_size} keywords per batch")
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
        
        # Flatten confidence levels - ignore confidence, merge all keywords
        print("🔧 Flattening confidence levels...")
        flattened_conditions = self.flatten_keywords(original_conditions)
        
        # Filter keywords with AI
        print("🧠 Filtering keywords with AI...")
        cleaned_flattened = self.filter_keywords_with_ai(flattened_conditions, batch_size)
        
        # Check if interrupted during main processing
        if self.interrupted:
            print("🛑 Interrupted during main processing - progress saved...")
            return None
        
        # Merge recovery results if any
        if recovery_results:
            for condition, keywords in recovery_results.items():
                if condition in cleaned_flattened:
                    # Merge and deduplicate
                    all_keywords = set(cleaned_flattened[condition] + keywords)
                    cleaned_flattened[condition] = sorted(list(all_keywords))
                else:
                    cleaned_flattened[condition] = keywords
        
        # Restructure back to original format
        print("🔧 Restructuring to original format...")
        cleaned_conditions = self.restructure_to_original_format(cleaned_flattened, original_conditions)
        
        # Update data structure
        cleaned_data = data.copy()
        cleaned_data['conditions'] = cleaned_conditions
        
        # Update metadata
        if 'metadata' in cleaned_data:
            cleaned_data['metadata']['ai_cleanup'] = {
                'cleanup_date': datetime.now().isoformat(),
                'model_used': 'gemini-2.5-flash',
                'total_keywords_processed': self.stats['total_keywords_processed'],
                'total_keywords_kept': self.stats['total_keywords_kept'],
                'total_keywords_removed': self.stats['total_keywords_removed'],
                'removal_rate': f"{(self.stats['total_keywords_removed']/self.stats['total_keywords_processed']*100):.1f}%" if self.stats['total_keywords_processed'] > 0 else "0.0%"
            }
        
        # Print statistics
        self.print_cleanup_stats(original_conditions, cleaned_conditions)
        
        # Save failed batches if any occurred during this run
        if self.failed_batches:
            print(f"⚠️  Saving {len(self.failed_batches)} failed batches for next run...")
            self.save_failed_batches(self.failed_batches)
        
        # Save cleaned data to same file
        print(f"💾 Saving cleaned keywords back to {input_file}...")
        self.save_keywords_file(cleaned_data, input_file)
        
        print("✅ AI cleanup completed!")
        return cleaned_data

if __name__ == "__main__":
    # Initialize and run cleanup
    cleanup = GeminiAICleanup()
    cleanup.main()
