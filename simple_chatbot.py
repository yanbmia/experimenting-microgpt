import os
import sys
from collections import Counter
import re
from microgpt import MicroGPT, _chunk_text

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
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def extract_keywords(text, num_keywords=10):
    """Extract important keywords from text."""
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'when', 'where', 'why', 'how'
    }
    
    words = re.findall(r'\b\w+\b', text.lower())
    filtered = [w for w in words if w not in stop_words and len(w) > 3]
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
    
    keywords = extract_keywords(text, num_keywords=15)
    scores = calculate_sentence_scores(sentences, keywords)
    
    top_indices = sorted(
        sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences],
        key=lambda x: x[0]
    )
    
    summary_sentences = [sentences[i] for i, _ in top_indices]
    return ' '.join(summary_sentences).strip()

def answer_question(text, question, gpt=None):
    """
    Answer a question about the document using keyword matching.

    Note: MicroGPT (the pure-Python toy GPT in microgpt.py) is a few-thousand-
    parameter, character-level model with a 16-character context window. It is
    not capable of reading comprehension or open-ended Q&A -- that's not a bug,
    it's just far too small/simple a model for the task. Real document Q&A is
    handled by keyword matching here, and by ollama_chatbot.py (which uses an
    actual local LLM) elsewhere in this repo. `gpt`, if provided, is only used
    for the unrelated "babble" easter egg below.
    """
    sentences = extract_sentences(text)
    question_words = re.findall(r'\b\w+\b', question.lower())
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'what', 'which', 'who', 'when', 'where', 'why', 'how', 'do', 'does',
        'did', 'can', 'could', 'would', 'should', 'will', 'would'
    }
    question_keywords = [w for w in question_words if w not in stop_words and len(w) > 2]

    if not question_keywords:
        return "I need more specific terms in your question to find relevant information."

    relevant_sentences = []
    for sentence in sentences:
        sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
        match_count = sum(1 for keyword in question_keywords if keyword in sentence_words)
        if match_count > 0:
            relevant_sentences.append((sentence, match_count))

    if not relevant_sentences:
        return f"I couldn't find information about '{' '.join(question_keywords)}' in the document."

    relevant_sentences.sort(key=lambda x: x[1], reverse=True)
    answer = ' '.join([s[0] for s in relevant_sentences[:2]])
    return answer.strip()


def babble(gpt, text, num_steps=200):
    """
    Quickly train MicroGPT on chunks of the loaded document and have it
    babble a made-up continuation. This is a novelty/demo feature showing
    what the toy GPT actually does (learn character-level texture of a
    corpus), not a Q&A feature. Triggered by typing 'babble' in the chat.
    """
    chunks = _chunk_text(text, gpt.block_size - 2)
    if len(chunks) < 2:
        return "[Document too short to train a babble model on.]"
    print("[*] Training a tiny GPT on this document's text (this takes a bit)...")
    gpt.train(chunks, num_steps=num_steps, verbose=False)
    samples = [gpt.generate(max_tokens=gpt.block_size, temperature=0.7) for _ in range(5)]
    return "\n    ".join(samples)

def main():
    print("=" * 70)
    print("  Text Summarization & Q&A Chatbot (with MicroGPT)")
    print("=" * 70)

    # Initialize MicroGPT (used only for the 'babble' novelty command --
    # Q&A and summary below are plain keyword matching, see answer_question())
    gpt = None
    try:
        gpt = MicroGPT()
        print("[+] MicroGPT initialized (try the 'babble' command)")
    except Exception as e:
        print(f"[!] MicroGPT not available: {e}")

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
    print("  - Type 'babble' to watch the toy MicroGPT train on this doc & hallucinate")
    print("  - Ask any question about the document (answered via keyword matching)")
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

        elif user_input.lower() == 'babble' and gpt is not None:
            result = babble(gpt, content)
            print(f"\n[MicroGPT babble]:\n    {result}")

        else:
            answer = answer_question(content, user_input, gpt=gpt)
            print(f"\n[Bot]:\n{answer}")

if __name__ == "__main__":
    main()
