## How to Run

### Option 1: Provide file as argument
```bash
python simple_chatbot.py sample_document.txt
```

### Option 2: Run and enter file path interactively
```bash
python simple_chatbot.py
```
Then enter the path to your text file when prompted.

## Usage

Once the chatbot loads a document, you can:
- Type `summary` to get a full summary
- Ask any question about the document
- Type `quit` to exit

## Example
```bash
python simple_chatbot.py sample_document.txt

[+] Loaded 1234 characters from file

[*] Document Preview:
    Your document summary here...

[*] Key Topics: topic1, topic2, topic3

[You]: What is the main idea?

[Bot]:
    The relevant answer from the document...
```
