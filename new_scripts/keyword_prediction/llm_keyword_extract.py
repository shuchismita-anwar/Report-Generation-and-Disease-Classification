import os
import json
import pandas as pd
import requests
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re
from dotenv import load_dotenv
from tqdm import tqdm

class GeminiKeywordExtractor:
    """
    Robust keyword extraction system using Google Gemini 2.5 Pro API to extract
    medical keywords from MIMIC-CXR reports and transcripts.
    
    Features Gemini 2.5 Pro capabilities:
    - Advanced reasoning through internal "thinking" process
    - Enhanced multimodal understanding 
    - Improved accuracy for complex medical text analysis
    - Dynamic thinking budget allocation based on task complexity
    - State-of-the-art performance in medical domain tasks
    
    Note: Gemini 2.5 Pro is a thinking model that takes more time to process
    but provides significantly better reasoning and accuracy.
    """
    
    def __init__(self, 
                 config_path: str = "config/condition_keywords.json",
                 dataset_path: str = "../../final_dataset_fixed.csv",
                 api_key: Optional[str] = None,
                 base_path: str = "../../"):
        
        # Load environment variables from .env file
        load_dotenv('.env')
        
        self.config_path = config_path
        self.dataset_path = dataset_path
        self.base_path = base_path
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        # Setup logging
        self._setup_logging()
        
        # Load configuration
        self.config = self._load_config()
        self.required_conditions = self.config["extraction_prompts"]["validation_rules"]["required_conditions"]
        
        # Initialize API configuration for Gemini 2.5 Flash
        self.api_config = {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            "headers": {
                "Content-Type": "application/json",
            },
            "max_retries": 3,
            "retry_delay": 2,
            "timeout": 30
        }
        
        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "invalid_responses": 0,
            "api_errors": 0,
            "start_time": None
        }
        
        # Keyword aggregation storage
        self.keyword_aggregator = {
            condition: {
                "high_confidence": set(),
                "medium_confidence": set(), 
                "low_confidence": set()
            } for condition in self.required_conditions
        }
        
        self.logger.info("GeminiKeywordExtractor initialized with Gemini 2.5 Flash")
        self.logger.info(f"Config loaded from: {self.config_path}")
        self.logger.info(f"Dataset path: {self.dataset_path}")
        self.logger.info(f"API key configured: {'Yes' if self.api_key else 'No'}")
        self.logger.info("Using Gemini 2.5 Flash for fast keyword extraction")

    def _setup_logging(self):
        """Setup comprehensive logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/keyword_extraction.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)

    def _load_config(self) -> Dict:
        """Load configuration from JSON file with validation"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Validate required sections
            required_sections = ["conditions", "extraction_prompts"]
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section: {section}")
            
            self.logger.info("Configuration loaded and validated successfully")
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def _validate_api_key(self) -> bool:
        """Validate Gemini API key"""
        if not self.api_key:
            self.logger.error("Gemini API key not provided. Set GEMINI_API_KEY environment variable.")
            return False
        return True

    def _load_dataset(self) -> pd.DataFrame:
        """Load and validate the dataset"""
        try:
            df = pd.read_csv(self.dataset_path)
            
            # Validate required columns
            required_columns = ["report_path", "transcript_path"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Remove rows with missing paths
            initial_count = len(df)
            df = df.dropna(subset=required_columns)
            final_count = len(df)
            
            if initial_count != final_count:
                self.logger.warning(f"Dropped {initial_count - final_count} rows with missing paths")
            
            self.logger.info(f"Dataset loaded: {final_count} patients with report and transcript paths")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise

    def _read_medical_text(self, file_path: str) -> Optional[str]:
        """Safely read medical text files"""
        try:
            # Handle different file formats
            full_path = os.path.join(self.base_path, file_path)
            
            if not os.path.exists(full_path):
                self.logger.warning(f"File not found: {full_path}")
                return None
            
            # Read text file
            if full_path.endswith('.txt'):
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
            elif full_path.endswith('.json'):
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Extract text from transcript JSON
                    content = data.get('full_text', '')
            else:
                self.logger.warning(f"Unsupported file format: {full_path}")
                return None
            
            if len(content) < 10:  # Too short to be meaningful
                self.logger.warning(f"Text too short in file: {full_path}")
                return None
                
            return content
            
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return None

    def _create_gemini_payload(self, medical_text: str) -> Dict:
        """Create the API payload for Gemini"""
        system_prompt = self.config["extraction_prompts"]["system_prompt"]
        user_prompt = self.config["extraction_prompts"]["user_prompt_template"].format(
            medical_text=medical_text
        )
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"{system_prompt}\n\n{user_prompt}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 1,
                "topP": 0.8,
                "maxOutputTokens": 8192,
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

    def _call_gemini_api(self, payload: Dict) -> Optional[Dict]:
        """Make API call to Gemini with robust error handling"""
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
                    
                    # Check if we hit max tokens
                    if 'candidates' in result and len(result['candidates']) > 0:
                        candidate = result['candidates'][0]
                        finish_reason = candidate.get('finishReason', '')
                        
                        if finish_reason == 'MAX_TOKENS':
                            self.logger.warning("Response was truncated due to MAX_TOKENS limit")
                            return None
                        
                        if 'content' in candidate and 'parts' in candidate['content'] and len(candidate['content']['parts']) > 0:
                            # Get the last part that contains text (skip thinking parts)
                            text_response = None
                            for part in candidate['content']['parts']:
                                if 'text' in part and part['text'].strip():
                                    text_response = part['text']
                            
                            if not text_response:
                                self.logger.error("No text content found in response")
                                return None
                            
                            # Log token usage for monitoring
                            usage_metadata = result.get('usageMetadata', {})
                            output_tokens = usage_metadata.get('candidatesTokenCount', 0)
                            total_tokens = usage_metadata.get('totalTokenCount', 0)
                            
                            self.logger.debug(f"Token usage - Output: {output_tokens}, Total: {total_tokens}")
                            
                            return {
                                'text': text_response,
                                'output_tokens': output_tokens,
                                'total_tokens': total_tokens
                            }
                    
                    self.logger.error(f"Unexpected Gemini response format: {result}")
                    return None
                
                elif response.status_code == 429:  # Rate limit
                    wait_time = (attempt + 1) * 10  # Exponential backoff
                    self.logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    self.logger.error(f"Gemini API error: {response.status_code} - {response.text}")
                    if attempt < self.api_config['max_retries'] - 1:
                        time.sleep(self.api_config['retry_delay'])
                        continue
                    return None
                    
            except requests.exceptions.RequestException as e:
                self.logger.error(f"API request failed (attempt {attempt + 1}): {e}")
                if attempt < self.api_config['max_retries'] - 1:
                    time.sleep(self.api_config['retry_delay'])
                    continue
                return None
        
        self.stats['api_errors'] += 1
        return None

    def _fix_common_json_issues(self, json_text: str) -> str:
        """Fix common JSON formatting issues from LLM responses"""
        # Remove trailing commas before closing braces/brackets
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)
        
        # Fix missing commas between array elements
        json_text = re.sub(r'"\s*\n\s*"', '",\n"', json_text)
        
        # Fix missing commas between object properties
        json_text = re.sub(r'}\s*\n\s*"', '},\n"', json_text)
        json_text = re.sub(r']\s*\n\s*"', '],\n"', json_text)
        
        return json_text

    def load_existing_progress(self, output_path: str) -> int:
        """Load existing progress and return number of patients to skip"""
        try:
            if not os.path.exists(output_path):
                self.logger.info("No existing progress file found - starting from beginning")
                return 0
            
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
            
            # Get number of already processed patients
            patients_processed = existing_data.get("metadata", {}).get("total_patients_analyzed", 0)
            
            if patients_processed > 0:
                self.logger.info(f"Found existing progress: {patients_processed} patients already processed")
                
                # Load existing keyword aggregator
                if "conditions" in existing_data:
                    for condition in self.required_conditions:
                        if condition in existing_data["conditions"]:
                            condition_data = existing_data["conditions"][condition]
                            
                            # Convert lists back to sets for aggregation
                            self.keyword_aggregator[condition]["high_confidence"] = set(
                                condition_data.get("high_confidence", [])
                            )
                            self.keyword_aggregator[condition]["medium_confidence"] = set(
                                condition_data.get("medium_confidence", [])
                            )
                            self.keyword_aggregator[condition]["low_confidence"] = set(
                                condition_data.get("low_confidence", [])
                            )
                
                # Load existing stats
                if "processing_stats" in existing_data["metadata"]:
                    existing_stats = existing_data["metadata"]["processing_stats"]
                    self.stats["total_processed"] = existing_stats.get("total_processed", 0)
                    self.stats["successful_extractions"] = existing_stats.get("successful_extractions", 0)
                    self.stats["failed_extractions"] = existing_stats.get("failed_extractions", 0)
                    self.stats["invalid_responses"] = existing_stats.get("invalid_responses", 0)
                    self.stats["api_errors"] = existing_stats.get("api_errors", 0)
                
                self.logger.info(f"Loaded existing keywords and stats - resuming from patient {patients_processed + 1}")
                return patients_processed
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Error loading existing progress: {e}")
            self.logger.info("Starting from beginning due to error")
            return 0

    def _validate_gemini_response(self, response_text: str) -> Optional[Dict]:
        """Validate and parse Gemini response with strict checking"""
        try:
            # Clean the response text
            cleaned_text = response_text.strip()
            
            # Find JSON in the response (handle cases where LLM adds explanation)
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            if json_match:
                json_text = json_match.group()
            else:
                json_text = cleaned_text
            
            # Try to fix common JSON issues
            json_text = self._fix_common_json_issues(json_text)
            
            # Parse JSON
            parsed_response = json.loads(json_text)
            
            # Validate structure
            validation_rules = self.config["extraction_prompts"]["validation_rules"]
            
            # Check all required conditions are present
            for condition in validation_rules["required_conditions"]:
                if condition not in parsed_response:
                    self.logger.error(f"Missing condition in response: {condition}")
                    return None
            
            # Check each condition has required confidence levels
            for condition, keywords in parsed_response.items():
                if condition not in validation_rules["required_conditions"]:
                    self.logger.warning(f"Unexpected condition in response: {condition}")
                    continue
                
                for confidence_level in validation_rules["required_confidence_levels"]:
                    if confidence_level not in keywords:
                        self.logger.error(f"Missing confidence level {confidence_level} for {condition}")
                        return None
                    
                    # Check keyword count (should be exactly 3)
                    keyword_list = keywords[confidence_level]
                    if not isinstance(keyword_list, list):
                        self.logger.error(f"Keywords not in list format for {condition}.{confidence_level}")
                        return None
                    
                    if len(keyword_list) != validation_rules["keywords_per_level"]:
                        self.logger.warning(f"Expected {validation_rules['keywords_per_level']} keywords for {condition}.{confidence_level}, got {len(keyword_list)}")
            
            self.logger.debug("Response validation successful")
            return parsed_response
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parsing failed: {e}")
            self.logger.debug(f"Raw response: {response_text}")
            return None
        except Exception as e:
            self.logger.error(f"Response validation failed: {e}")
            return None

    def _aggregate_keywords(self, extracted_keywords: Dict):
        """Aggregate keywords into the main collection"""
        for condition, confidence_levels in extracted_keywords.items():
            if condition in self.keyword_aggregator:
                for confidence_level, keywords in confidence_levels.items():
                    if confidence_level in self.keyword_aggregator[condition]:
                        # Add keywords to set (automatically handles duplicates)
                        for keyword in keywords:
                            if keyword and keyword.strip():  # Skip empty keywords
                                self.keyword_aggregator[condition][confidence_level].add(keyword.strip().lower())

    def extract_keywords_from_text(self, medical_text: str, text_type: str = "report") -> Optional[Dict]:
        """Extract keywords from a single medical text"""
        try:
            # Create API payload
            payload = self._create_gemini_payload(medical_text)
            
            # Call Gemini API
            api_response = self._call_gemini_api(payload)
            if not api_response:
                return None
            
            # Validate response
            validated_keywords = self._validate_gemini_response(api_response['text'])
            if not validated_keywords:
                self.stats['invalid_responses'] += 1
                return None
            
            # Log successful extraction with token usage
            total_tokens = api_response.get('total_tokens', 0)
            self.logger.debug(f"Keywords extracted successfully from {text_type} - Total tokens: {total_tokens}")
            
            return validated_keywords
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords from {text_type}: {e}")
            return None

    def process_single_patient(self, row: pd.Series) -> Tuple[bool, bool]:
        """Process a single patient's report and transcript"""
        report_success = False
        transcript_success = False
        
        try:
            # Process report
            report_text = self._read_medical_text(row['report_path'])
            if report_text:
                report_keywords = self.extract_keywords_from_text(report_text, "report")
                if report_keywords:
                    self._aggregate_keywords(report_keywords)
                    report_success = True
                    self.logger.debug(f"Report processed: {row['report_path']}")
            
            # Process transcript  
            transcript_text = self._read_medical_text(row['transcript_path'])
            if transcript_text:
                transcript_keywords = self.extract_keywords_from_text(transcript_text, "transcript")
                if transcript_keywords:
                    self._aggregate_keywords(transcript_keywords)
                    transcript_success = True
                    self.logger.debug(f"Transcript processed: {row['transcript_path']}")
            
        except Exception as e:
            self.logger.error(f"Error processing patient {row.name}: {e}")
        
        return report_success, transcript_success

    def process_all_patients(self, max_patients: Optional[int] = None, output_path: str = "extracted_keywords_result.json") -> Dict:
        """Process all patients in the dataset with progress tracking and incremental saving"""
        if not self._validate_api_key():
            raise ValueError("Invalid API key")
        
        # Load existing progress
        skip_patients = self.load_existing_progress(output_path)
        
        # Load dataset
        df = self._load_dataset()
        
        if max_patients:
            df = df.head(max_patients)
            self.logger.info(f"Processing limited to {max_patients} patients")
        
        # Initialize statistics
        self.stats['start_time'] = time.time()
        total_patients = len(df)
        remaining_patients = total_patients - skip_patients
        
        if skip_patients > 0:
            self.logger.info(f"Resuming from patient {skip_patients + 1}")
            print(f"🔄 Resuming: {skip_patients} patients already done, {remaining_patients} remaining...")
        else:
            self.logger.info(f"Starting keyword extraction for {total_patients} patients")
            print(f"🔄 Processing {total_patients} patients with incremental saving...")
        
        # Process each patient with progress bar (starting from skip_patients)
        with tqdm(total=total_patients, initial=skip_patients, desc="Processing patients", unit="patient") as pbar:
            for idx, row in df.iterrows():
                try:
                    # Skip already processed patients
                    if idx < skip_patients:
                        continue
                    
                    # Update progress bar description
                    pbar.set_description(f"Patient {idx + 1}/{total_patients}")
                    
                    report_success, transcript_success = self.process_single_patient(row)
                    
                    # Update statistics
                    self.stats['total_processed'] += 1
                    if report_success or transcript_success:
                        self.stats['successful_extractions'] += 1
                    else:
                        self.stats['failed_extractions'] += 1
                    
                    # Save after each patient (incremental saving)
                    try:
                        self.save_extracted_keywords(output_path)
                    except Exception as save_error:
                        self.logger.warning(f"Failed to save after patient {idx + 1}: {save_error}")
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Success': self.stats['successful_extractions'],
                        'Failed': self.stats['failed_extractions'],
                        'Rate': f"{self.stats['successful_extractions']/self.stats['total_processed']*100:.1f}%"
                    })
                    pbar.update(1)
                    
                    # Rate limiting for API calls
                    time.sleep(0.5)
                    
                except KeyboardInterrupt:
                    self.logger.info("Processing interrupted by user")
                    pbar.close()
                    break
                except Exception as e:
                    self.logger.error(f"Unexpected error processing patient {idx}: {e}")
                    self.stats['failed_extractions'] += 1
                    pbar.update(1)
                    continue
        
        # Final statistics
        total_time = time.time() - self.stats['start_time']
        self.logger.info(f"Processing completed in {total_time:.1f}s")
        self.logger.info(f"Total processed: {self.stats['total_processed']}")
        self.logger.info(f"Successful: {self.stats['successful_extractions']}")
        self.logger.info(f"Failed: {self.stats['failed_extractions']}")
        self.logger.info(f"API errors: {self.stats['api_errors']}")
        self.logger.info(f"Invalid responses: {self.stats['invalid_responses']}")
        
        # Print final summary
        self._print_extraction_summary()
        
        return self.stats

    def save_extracted_keywords(self, output_path: str = "extracted_keywords_result.json"):
        """Save the aggregated keywords to JSON file"""
        try:
            # Convert sets to sorted lists for JSON serialization
            output_data = {
                "metadata": {
                    "version": "2.1",
                    "description": "Medical condition keywords extracted from MIMIC-CXR reports and transcripts using Gemini 2.5 Flash",
                    "extraction_method": "gemini_2.5_flash_api",
                    "model_details": {
                        "model": "gemini-2.5-flash",
                        "thinking_enabled": False,
                        "temperature": 0.1
                    },
                    "total_patients_analyzed": self.stats['total_processed'],
                    "successful_extractions": self.stats['successful_extractions'],
                    "last_updated": datetime.now().isoformat(),
                    "processing_stats": self.stats
                },
                "conditions": {}
            }
            
            for condition in self.required_conditions:
                output_data["conditions"][condition] = {
                    "high_confidence": sorted(list(self.keyword_aggregator[condition]["high_confidence"])),
                    "medium_confidence": sorted(list(self.keyword_aggregator[condition]["medium_confidence"])),
                    "low_confidence": sorted(list(self.keyword_aggregator[condition]["low_confidence"])),
                    "extraction_stats": {
                        "total_occurrences": (
                            len(self.keyword_aggregator[condition]["high_confidence"]) +
                            len(self.keyword_aggregator[condition]["medium_confidence"]) +
                            len(self.keyword_aggregator[condition]["low_confidence"])
                        ),
                        "unique_phrases": {
                            "high": len(self.keyword_aggregator[condition]["high_confidence"]),
                            "medium": len(self.keyword_aggregator[condition]["medium_confidence"]), 
                            "low": len(self.keyword_aggregator[condition]["low_confidence"])
                        }
                    }
                }
            
            # Save to file (no directory creation needed for current dir)
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            self.logger.debug(f"Extracted keywords saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving extracted keywords: {e}")
            raise

    def _print_extraction_summary(self):
        """Print a summary of extracted keywords"""
        self.logger.info("=== KEYWORD EXTRACTION SUMMARY ===")
        
        for condition in self.required_conditions:
            high_count = len(self.keyword_aggregator[condition]["high_confidence"])
            medium_count = len(self.keyword_aggregator[condition]["medium_confidence"])
            low_count = len(self.keyword_aggregator[condition]["low_confidence"])
            total_count = high_count + medium_count + low_count
            
            self.logger.info(f"{condition}: {total_count} total keywords "
                           f"(High: {high_count}, Medium: {medium_count}, Low: {low_count})")
            
            # Show some examples
            if high_count > 0:
                examples = list(self.keyword_aggregator[condition]["high_confidence"])[:3]
                self.logger.info(f"  High confidence examples: {examples}")

def process_all_data():
    """Process all 2877 patients - both reports and transcripts"""
    try:
        print("🚀 Starting full dataset processing with Gemini 2.5 Flash...")
        print("📊 Processing 2877 reports and 2877 transcripts...")
        
        # Initialize extractor
        extractor = GeminiKeywordExtractor()
        
        if not extractor._validate_api_key():
            print("❌ API key validation failed!")
            return
        
        # Process all patients with incremental saving
        stats = extractor.process_all_patients(output_path="extracted_keywords_result.json")
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Total patients processed: {stats['total_processed']}")
        print(f"   Successful extractions: {stats['successful_extractions']}")
        print(f"   Failed extractions: {stats['failed_extractions']}")
        print(f"   Success rate: {(stats['successful_extractions']/stats['total_processed']*100):.1f}%")
        print(f"💾 Results saved to: extracted_keywords_result.json")
        print(f"✅ Full dataset processing COMPLETED!")
        
    except Exception as e:
        print(f"❌ Full dataset processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run full dataset processing
    process_all_data()
