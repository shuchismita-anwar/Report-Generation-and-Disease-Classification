import os
import glob
import re
from typing import List, Tuple
from statistics import mean

def count_words_and_sentences(text: str) -> Tuple[int, int]:
    """
    Count words and sentences in a text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Tuple[int, int]: (word_count, sentence_count)
    """
    if not text or not text.strip():
        return 0, 0
    
    # Count words - split by whitespace and filter out empty strings
    words = [word for word in text.split() if word.strip()]
    word_count = len(words)
    
    # Count sentences - split by periods, exclamation marks, question marks
    # Remove common abbreviations that shouldn't end sentences
    sentence_endings = re.findall(r'[.!?]+', text)
    sentence_count = len(sentence_endings)
    
    # Ensure at least 1 sentence if there's content but no sentence endings
    if word_count > 0 and sentence_count == 0:
        sentence_count = 1
    
    return word_count, sentence_count

def find_all_cleaned_reports(base_path: str = "../../cleaned_reports") -> List[str]:
    """
    Find all cleaned report files maintaining the directory structure.
    
    Args:
        base_path (str): Path to cleaned reports directory
        
    Returns:
        List[str]: List of all report file paths
    """
    pattern = os.path.join(base_path, "patient_*", "CXR-DICOM", "s*.txt")
    all_reports = glob.glob(pattern)
    return all_reports

def analyze_reports():
    """
    Analyze all cleaned reports and calculate word/sentence statistics.
    """
    print("📊 Analyzing Cleaned Medical Reports...")
    print("=" * 60)
    
    # Find all cleaned reports
    print("🔍 Finding cleaned reports...")
    report_files = find_all_cleaned_reports()
    
    if not report_files:
        print("❌ No cleaned reports found!")
        print("Make sure you've run the cleanup script first.")
        return
    
    print(f"📁 Found {len(report_files)} cleaned reports")
    print("\n📈 Analyzing content...")
    
    # Statistics tracking
    all_word_counts = []
    all_sentence_counts = []
    total_reports_processed = 0
    failed_reports = 0
    
    # Process each report
    for report_path in report_files:
        try:
            # Read report content
            with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Count words and sentences
            word_count, sentence_count = count_words_and_sentences(content)
            
            # Store statistics
            all_word_counts.append(word_count)
            all_sentence_counts.append(sentence_count)
            total_reports_processed += 1
            
        except Exception as e:
            print(f"⚠️  Error processing {os.path.basename(report_path)}: {e}")
            failed_reports += 1
    
    # Calculate statistics
    if total_reports_processed > 0:
        avg_words_per_report = mean(all_word_counts)
        avg_sentences_per_report = mean(all_sentence_counts)
        total_words = sum(all_word_counts)
        total_sentences = sum(all_sentence_counts)
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 REPORT ANALYSIS RESULTS")
        print("=" * 60)
        print(f"📁 Total reports processed: {total_reports_processed:,}")
        if failed_reports > 0:
            print(f"❌ Failed reports: {failed_reports}")
        
        print(f"\n📝 WORD STATISTICS:")
        print(f"   • Total words across all reports: {total_words:,}")
        print(f"   • Average words per report: {avg_words_per_report:.1f}")
        print(f"   • Min words in a report: {min(all_word_counts)}")
        print(f"   • Max words in a report: {max(all_word_counts)}")
        
        print(f"\n📖 SENTENCE STATISTICS:")
        print(f"   • Total sentences across all reports: {total_sentences:,}")
        print(f"   • Average sentences per report: {avg_sentences_per_report:.1f}")
        print(f"   • Min sentences in a report: {min(all_sentence_counts)}")
        print(f"   • Max sentences in a report: {max(all_sentence_counts)}")
        
        print(f"\n📊 OVERALL AVERAGES:")
        print(f"   • Average words per sentence: {total_words/total_sentences:.1f}")
        
        print("=" * 60)
        
        # Final summary (what user specifically requested)
        print("\n🎯 FINAL SUMMARY:")
        print(f"   ➤ AVERAGE WORDS PER REPORT: {avg_words_per_report:.1f}")
        print(f"   ➤ AVERAGE SENTENCES PER REPORT: {avg_sentences_per_report:.1f}")
        print("=" * 60)
        
    else:
        print("❌ No reports were successfully processed!")

def main():
    """Main function to run the analysis"""
    try:
        analyze_reports()
    except Exception as e:
        print(f"❌ Error during analysis: {e}")

if __name__ == "__main__":
    main()
