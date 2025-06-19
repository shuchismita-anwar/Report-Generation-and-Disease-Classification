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
        
        # Initialize API configuration for Gemini 2.5 Pro
        self.api_config = {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
            "headers": {
                "Content-Type": "application/json",
            },
            "max_retries": 3,
            "retry_delay": 5,  # Increased for thinking model
            "timeout": 60  # Increased timeout for thinking models
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
        
        self.logger.info("GeminiKeywordExtractor initialized with Gemini 2.5 Pro")
        self.logger.info(f"Config loaded from: {self.config_path}")
        self.logger.info(f"Dataset path: {self.dataset_path}")
        self.logger.info(f"API key configured: {'Yes' if self.api_key else 'No'}")
        self.logger.info("Using Gemini 2.5 Pro thinking model for enhanced reasoning")

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
                "temperature": 0.1,  # Low temperature for consistency with thinking models
                "topK": 1,
                "topP": 0.8,
                "maxOutputTokens": 4096,  # Increased for thinking model responses
                "stopSequences": [],
                "thinkingConfig": {
                    "thinkingBudget": -1  # Dynamic thinking - let model decide complexity
                }
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
                    
                    # Extract text from Gemini response
                    if 'candidates' in result and len(result['candidates']) > 0:
                        candidate = result['candidates'][0]
                        if 'content' in candidate and 'parts' in candidate['content']:
                            text_response = candidate['content']['parts'][0]['text']
                            
                            # Log thinking token usage for monitoring
                            usage_metadata = result.get('usageMetadata', {})
                            thinking_tokens = usage_metadata.get('thoughtsTokenCount', 0)
                            output_tokens = usage_metadata.get('candidatesTokenCount', 0)
                            total_tokens = usage_metadata.get('totalTokenCount', 0)
                            
                            if thinking_tokens > 0:
                                self.logger.debug(f"Token usage - Thinking: {thinking_tokens}, Output: {output_tokens}, Total: {total_tokens}")
                            
                            return {
                                'text': text_response,
                                'thinking_tokens': thinking_tokens,
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
            thinking_tokens = api_response.get('thinking_tokens', 0)
            total_tokens = api_response.get('total_tokens', 0)
            
            if thinking_tokens > 0:
                self.logger.debug(f"Keywords extracted successfully from {text_type} - Thinking tokens: {thinking_tokens}, Total: {total_tokens}")
            else:
                self.logger.debug(f"Keywords extracted successfully from {text_type}")
            
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

    def process_all_patients(self, max_patients: Optional[int] = None) -> Dict:
        """Process all patients in the dataset"""
        if not self._validate_api_key():
            raise ValueError("Invalid API key")
        
        # Load dataset
        df = self._load_dataset()
        
        if max_patients:
            df = df.head(max_patients)
            self.logger.info(f"Processing limited to {max_patients} patients")
        
        # Initialize statistics
        self.stats['start_time'] = time.time()
        total_patients = len(df)
        
        self.logger.info(f"Starting keyword extraction for {total_patients} patients")
        
        # Process each patient
        for idx, row in df.iterrows():
            try:
                report_success, transcript_success = self.process_single_patient(row)
                
                # Update statistics
                self.stats['total_processed'] += 1
                if report_success or transcript_success:
                    self.stats['successful_extractions'] += 1
                else:
                    self.stats['failed_extractions'] += 1
                
                # Progress logging
                if (idx + 1) % 50 == 0:
                    elapsed_time = time.time() - self.stats['start_time']
                    progress = (idx + 1) / total_patients * 100
                    self.logger.info(f"Progress: {idx + 1}/{total_patients} ({progress:.1f}%) - "
                                   f"Success: {self.stats['successful_extractions']}, "
                                   f"Failed: {self.stats['failed_extractions']}, "
                                   f"Time: {elapsed_time:.1f}s")
                
                # Rate limiting for thinking model (takes more time to process)
                time.sleep(1.0)  # 1 second delay between requests for thinking model
                
            except KeyboardInterrupt:
                self.logger.info("Processing interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error processing patient {idx}: {e}")
                self.stats['failed_extractions'] += 1
                continue
        
        # Final statistics
        total_time = time.time() - self.stats['start_time']
        self.logger.info(f"Processing completed in {total_time:.1f}s")
        self.logger.info(f"Total processed: {self.stats['total_processed']}")
        self.logger.info(f"Successful: {self.stats['successful_extractions']}")
        self.logger.info(f"Failed: {self.stats['failed_extractions']}")
        self.logger.info(f"API errors: {self.stats['api_errors']}")
        self.logger.info(f"Invalid responses: {self.stats['invalid_responses']}")
        
        return self.stats

    def save_extracted_keywords(self, output_path: str = "config/condition_keywords_extracted.json"):
        """Save the aggregated keywords to JSON file"""
        try:
            # Convert sets to sorted lists for JSON serialization
            output_data = {
                "metadata": {
                    "version": "2.1",
                    "description": "Medical condition keywords extracted from MIMIC-CXR reports and transcripts using Gemini 2.5 Pro",
                    "extraction_method": "gemini_2.5_pro_thinking_api",
                    "model_details": {
                        "model": "gemini-2.5-pro",
                        "thinking_enabled": True,
                        "thinking_budget": "dynamic",
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
            
            # Save to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            self.logger.info(f"Extracted keywords saved to: {output_path}")
            
            # Print summary
            self._print_extraction_summary()
            
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

def test_ten_prompts():
    """Test function for 10 prompts with Gemini 2.5 Pro"""
    try:
        print("🚀 Testing 10 prompts extraction with Gemini 2.5 Pro...")
        
        # Initialize extractor
        extractor = GeminiKeywordExtractor()
        
        if not extractor._validate_api_key():
            print("❌ API key validation failed!")
            return
        
        # Load dataset and get first 10 patients
        df = extractor._load_dataset()
        if len(df) == 0:
            print("❌ No data found in dataset!")
            return
        
        # Test with first 10 patients
        test_df = df.head(10)
        print(f"📄 Testing with {len(test_df)} patients...")
        
        test_results = []
        success_count = 0
        
        for idx, row in test_df.iterrows():
            print(f"\n--- Processing Patient {idx + 1}/10 ---")
            print(f"Report: {row['report_path']}")
            
            # Read and test report
            report_text = extractor._read_medical_text(row['report_path'])
            if report_text:
                print(f"📋 Report text preview: {report_text[:100]}...")
                
                print(f"🔧 Sending to Gemini 2.5 Pro...")
                report_keywords = extractor.extract_keywords_from_text(report_text, "report")
                
                if report_keywords:
                    print(f"✅ SUCCESS! Keywords extracted")
                    success_count += 1
                    
                    test_results.append({
                        "patient_id": idx,
                        "report_path": row['report_path'],
                        "extracted_keywords": report_keywords,
                        "status": "success"
                    })
                else:
                    print(f"❌ Failed to extract keywords")
                    test_results.append({
                        "patient_id": idx,
                        "report_path": row['report_path'],
                        "status": "failed"
                    })
            else:
                print(f"❌ Failed to read report text")
                test_results.append({
                    "patient_id": idx,
                    "report_path": row['report_path'],
                    "status": "file_read_failed"
                })
        
        # Save test results
        final_output = {
            "test_date": datetime.now().isoformat(),
            "total_tested": len(test_df),
            "successful_extractions": success_count,
            "success_rate": f"{(success_count/len(test_df)*100):.1f}%",
            "results": test_results
        }
        
        with open("test_10_prompts_result.json", "w") as f:
            json.dump(final_output, f, indent=2)
        
        print(f"\n🎯 FINAL RESULTS:")
        print(f"   Total tested: {len(test_df)}")
        print(f"   Successful: {success_count}")
        print(f"   Success rate: {(success_count/len(test_df)*100):.1f}%")
        print(f"💾 Results saved to: test_10_prompts_result.json")
        print(f"✅ 10 prompts test COMPLETED!")
        
    except Exception as e:
        print(f"❌ 10 prompts test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main execution function"""
    try:
        # Initialize extractor
        extractor = GeminiKeywordExtractor()
        
        # Process all patients (or limit for testing)
        # extractor.process_all_patients(max_patients=10)  # For testing
        extractor.process_all_patients()  # For full processing
        
        # Save results
        extractor.save_extracted_keywords()
        
        print("✅ Keyword extraction completed successfully!")
        
    except Exception as e:
        print(f"❌ Keyword extraction failed: {e}")
        raise

if __name__ == "__main__":
    # Run 10 prompts test
    test_ten_prompts()
