"""
Simple Q&A and Summarization Chatbot
Takes a plain text file and provides summaries and answers questions about it.
Uses text analysis rather than neural network generation for practical results.
"""

import os
import sys
from collections import Counter
import re

# ============================================================================
# Text Processing & Analysis
# ============================================================================

def load_file(filepath):
    """Load and read file content."""
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def extract_sentences(text):
    """Split text into sentences."""
    # Simple sentence splitting on . ! ?
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def extract_keywords(text, num_keywords=10):
    """Extract important keywords from text."""
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how'
    }
    
    # Extract words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter stop words and short words
    filtered = [w for w in words if w not in stop_words and len(w) > 3]
    
    # Get most common
    counter = Counter(filtered)
    return [word for word, _ in counter.most_common(num_keywords)]

def calculate_sentence_scores(sentences, keywords):
    """Score sentences based on keyword density."""
    scores = {}
    for i, sentence in enumerate(sentences):
        score = 0
        words = set(re.findall(r'\b\w+\b', sentence.lower()))
        for keyword in keywords:
            if keyword in words:
                score += 1
        scores[i] = score
    return scores

def generate_summary(text, num_sentences=3):
    """Generate a summary of the text."""
    sentences = extract_sentences(text)
    
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    
    # Extract keywords
    keywords = extract_keywords(text, num_keywords=15)
    
    # Score sentences
    scores = calculate_sentence_scores(sentences, keywords)
    
    # Get top sentences in original order
    top_indices = sorted(
        sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences],
        key=lambda x: x[0]
    )
    
    summary_sentences = [sentences[i] for i, _ in top_indices]
    return ' '.join(summary_sentences).strip()

def answer_question(text, question):
    """Try to find relevant sentences that answer the question."""
    sentences = extract_sentences(text)
    
    # Extract keywords from question
    question_words = set(re.findall(r'\b\w+\b', question.lower()))
    
    # Filter out stop words from question
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'what', 'which', 'who', 'when', 'where', 'why', 'how', 'do', 'does',
        'did', 'can', 'could', 'would', 'should', 'will', 'would'
    }
    question_keywords = [w for w in question_words if w not in stop_words]
    
    if not question_keywords:
        return "I need more specific terms in your question to find relevant information."
    
    # Score sentences by keyword matches
    relevant_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        match_count = sum(1 for keyword in question_keywords if keyword in sentence_lower)
        if match_count > 0:
            relevant_sentences.append((sentence, match_count))
    
    if not relevant_sentences:
        return f"I couldn't find information about '{' '.join(question_keywords)}' in the document."
    
    # Sort by relevance and return top 2
    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    answer = ' '.join([s[0] for s in relevant_sentences[:2]])
    
    return answer.strip()

# ============================================================================
# Main Chatbot Interface
# ============================================================================

def main():
    print("=" * 70)
    print("  Text Summarization & Q&A Chatbot")
    print("=" * 70)
    
    # Get file from user
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = input("\n[?] Enter path to text file: ").strip()
    
    # Load file
    content = load_file(filepath)
    if content is None:
        return
    
    print(f"\n[+] Loaded {len(content)} characters from file")
    
    # Show initial summary
    initial_summary = generate_summary(content, num_sentences=2)
    print(f"\n[*] Document Preview:")
    print(f"    {initial_summary}")
    
    # Extract and show keywords
    keywords = extract_keywords(content, num_keywords=8)
    print(f"\n[*] Key Topics: {', '.join(keywords)}")
    
    # Interactive loop
    print("\n" + "=" * 70)
    print("  Commands:")
    print("  - Type 'summary' for a full summary")
    print("  - Ask any question about the document")
    print("  - Type 'quit' to exit")
    print("=" * 70)
    
    while True:
        user_input = input("\n[You]: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("[*] Goodbye!")
            break
        
        elif user_input.lower() == 'summary':
            summary = generate_summary(content, num_sentences=5)
            print(f"\n[Bot Summary]:\n{summary}")
        
        else:
            # Answer question
            answer = answer_question(content, user_input)
            print(f"\n[Bot]:\n{answer}")

if __name__ == "__main__":
    main()
