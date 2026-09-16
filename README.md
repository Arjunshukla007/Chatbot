# Personal Python Question-Answering Chatbot

A rule-based chatbot that answers Python programming questions using a
predefined knowledge base and a custom fuzzy-matching engine — no external
NLP libraries, APIs, or internet connection required.

## Project Structure
Everything lives in a single file for easy submission/grading:

```
python_qa_chatbot.py
├── KNOWLEDGE_BASE   → 65+ predefined Python Q&A pairs (basics, data types,
│                       control flow, functions, OOP, exceptions, modules,
│                       files, strings, popular libraries)
├── PythonQABot      → the matching engine
├── run_cli()        → terminal chat interface
└── run_gui()        → Tkinter desktop chat window
```

## How the matching works
1. **Preprocess** — lowercase, strip punctuation, collapse whitespace.
2. **Expand aliases** — common shorthand like `oop`, `kwargs`, `diff`, `py`
   is expanded to its full form.
3. **Extract keywords** — filler words (what/is/a/the/do/you...) are removed
   so the comparison focuses on meaningful content words.
4. **Score every KB question** against the user's keywords using an
   F1-style score (recall × precision), with a high-confidence typo
   tolerance for longer words (e.g. "decorators" still matches "decorator").
5. The highest-scoring question above a confidence threshold wins; below
   the threshold, the bot honestly says it doesn't know rather than
   guessing.

This means you don't have to type a question exactly as worded — "what's
OOP?", "explain inheritance", and "tell me about decorators" all work.

## Requirements
- Python 3.7+ (standard library only — `re`, `difflib`, `datetime`, and
  `tkinter` for the GUI, which ships with most standard Python installs)

## Running it

**Command line (works everywhere):**
```bash
python python_qa_chatbot.py
```

**Desktop GUI (Tkinter window):**
```bash
python python_qa_chatbot.py --gui
```
> Note: Tkinter ships with the standard Python installer on Windows/macOS.
> On some Linux distros you may need `sudo apt install python3-tk` first.

## Example session (CLI)
```
You: hello
Bot: Hello! I'm your Python Q&A assistant. Ask me anything about Python programming.

You: What is a list comprehension?
Bot: A list comprehension is a concise way to create a list. Example:
squares = [x**2 for x in range(10)] creates a list of squares from 0 to 81.

You: what is oop
Bot: Object-Oriented Programming (OOP) organizes code into classes and
objects. Python supports the four OOP pillars: encapsulation, abstraction,
inheritance, and polymorphism.

You: tell me about decorators
Bot: A decorator is a function that wraps another function to extend or
modify its behavior without changing its source code, applied using the
@decorator_name syntax.

You: What is quantum computing
Bot: I'm not sure about that one yet — my knowledge base doesn't cover it.
Try asking about Python basics, data types, functions, OOP, exceptions, or
modules. (e.g. "What is a list comprehension?")

You: topics       <- lists every question the bot can answer
You: quit
Bot: Goodbye! Happy coding. 🐍
```

## Extending the knowledge base
Just add more `"question": "answer"` entries to the `KNOWLEDGE_BASE`
dictionary at the top of the file — no other code changes needed.

## Possible future improvements
- Load the knowledge base from an external JSON/CSV file instead of hardcoding it
- Log unanswered questions so you know what to add next
- Swap the matcher for TF-IDF / sentence embeddings for even better recall
- Add speech-to-text input for a voice assistant version
