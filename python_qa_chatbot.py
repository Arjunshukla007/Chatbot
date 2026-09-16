"""
Personal Python Question-Answering Chatbot
============================================
A rule-based chatbot that answers Python programming questions using a
predefined knowledge base and a fuzzy string-matching engine (no external
NLP libraries required — pure Python standard library).

Author : (your name here)
Usage  :
    python python_qa_chatbot.py           -> runs in Command Line (CLI) mode
    python python_qa_chatbot.py --gui      -> runs in Tkinter GUI mode

Project structure (all in one file for easy submission):
    1. KNOWLEDGE_BASE   - dictionary of question -> answer pairs
    2. PythonQABot      - the matching engine (preprocessing + scoring)
    3. run_cli()        - terminal chat interface
    4. ChatbotGUI        - Tkinter desktop chat window
"""

import re
import sys
import difflib
from datetime import datetime


# ---------------------------------------------------------------------------
# 1. KNOWLEDGE BASE
# ---------------------------------------------------------------------------
# Each key is a normalized "canonical" question. The matcher will compare
# the user's input against ALL of these using both fuzzy string similarity
# and keyword overlap, so users don't have to type the question verbatim.

KNOWLEDGE_BASE = {
    # --- Basics ---------------------------------------------------------
    "what is python": "Python is a high-level, interpreted, general-purpose "
        "programming language known for its simple, readable syntax. It "
        "supports multiple paradigms (procedural, object-oriented, "
        "functional) and is widely used in web development, data science, "
        "automation, and AI.",
    "who created python": "Python was created by Guido van Rossum and was "
        "first released in 1991.",
    "what are the features of python": "Key features of Python include: "
        "easy-to-read syntax, dynamic typing, automatic memory management, "
        "a huge standard library, cross-platform support, and support for "
        "multiple programming paradigms.",
    "is python interpreted or compiled": "Python is primarily an "
        "interpreted language — code is executed line by line by the "
        "Python interpreter (CPython by default), although it is first "
        "compiled internally into bytecode (.pyc files).",
    "what is pep 8": "PEP 8 is Python's official style guide. It defines "
        "conventions for code layout, naming, indentation (4 spaces), and "
        "formatting to keep Python code consistent and readable.",

    # --- Variables & Data Types ------------------------------------------
    "what is a variable in python": "A variable is a named location in "
        "memory used to store a value. In Python you don't need to declare "
        "a type explicitly — e.g. x = 10 creates an integer variable x.",
    "what are the data types in python": "Python's built-in data types "
        "include: int, float, complex, str, bool, list, tuple, dict, set, "
        "and frozenset, plus NoneType.",
    "what is the difference between list and tuple": "A list is mutable "
        "(can be changed after creation) and defined with square brackets "
        "[ ], while a tuple is immutable (cannot be changed) and defined "
        "with parentheses ( ). Tuples are generally faster and used for "
        "fixed collections of items.",
    "what is the difference between a list and a dictionary": "A list is "
        "an ordered collection accessed by numeric index, e.g. my_list[0]. "
        "A dictionary stores key-value pairs and is accessed by key, e.g. "
        "my_dict['name'].",
    "what is a set in python": "A set is an unordered collection of unique "
        "elements. Sets automatically remove duplicates and support "
        "mathematical operations like union, intersection, and difference.",
    "what is a dictionary in python": "A dictionary is a mutable, unordered "
        "(insertion-ordered since Python 3.7) collection of key-value "
        "pairs, created with curly braces, e.g. {'name': 'Alice', 'age': 30}.",
    "how do you convert data types in python": "You can use built-in "
        "conversion functions such as int(), float(), str(), list(), "
        "tuple(), and dict() to convert between data types, e.g. "
        "int('5') converts the string '5' to the integer 5.",
    "what is type casting": "Type casting (or type conversion) is the "
        "process of converting a value from one data type to another, "
        "using functions like int(), str(), or float().",
    "what is none in python": "None is a special built-in constant that "
        "represents the absence of a value or a null value. It is of type "
        "NoneType.",

    # --- Operators & Control Flow ----------------------------------------
    "what are operators in python": "Operators are symbols used to perform "
        "operations on values and variables. Python supports arithmetic "
        "(+ - * / // % **), comparison (== != > <), logical (and or not), "
        "assignment (= += -=), bitwise, identity (is), and membership "
        "(in) operators.",
    "what is the difference between and and & in python": "'and' is a "
        "logical operator used with boolean expressions, while '&' is a "
        "bitwise operator used to perform bit-level AND on integers (or "
        "element-wise AND on some data structures like pandas Series).",
    "what is the difference between is and ==": "'==' checks if two "
        "objects have the same value, while 'is' checks if two variables "
        "refer to the exact same object in memory.",
    "how do if else statements work in python": "if/elif/else statements "
        "let you run different blocks of code based on conditions. "
        "Example:\nif x > 0:\n    print('positive')\nelif x == 0:\n    "
        "print('zero')\nelse:\n    print('negative')",
    "what is a for loop": "A for loop iterates over a sequence (list, "
        "tuple, string, range, etc.) and executes a block of code once per "
        "item. Example: for i in range(5): print(i)",
    "what is a while loop": "A while loop repeatedly executes a block of "
        "code as long as a given condition remains True. Example: "
        "while x < 5: x += 1",
    "what is the difference between break and continue": "'break' "
        "immediately exits the loop entirely, while 'continue' skips the "
        "rest of the current iteration and moves to the next one.",
    "what does the pass statement do": "'pass' is a null operation — it "
        "does nothing and is used as a placeholder where syntax requires a "
        "statement, e.g. inside an empty function or class body.",

    # --- Functions --------------------------------------------------------
    "how do you define a function in python": "You define a function "
        "using the 'def' keyword, e.g.:\ndef greet(name):\n    return "
        "f'Hello, {name}!'",
    "what is a lambda function": "A lambda function is a small anonymous "
        "function defined with the 'lambda' keyword, limited to a single "
        "expression. Example: square = lambda x: x * x",
    "what are args and kwargs": "*args allows a function to accept any "
        "number of positional arguments (as a tuple), and **kwargs allows "
        "it to accept any number of keyword arguments (as a dictionary).",
    "what is a default argument": "A default argument is a parameter that "
        "assumes a default value if no value is provided when the function "
        "is called, e.g. def greet(name='World'):",
    "what is recursion": "Recursion is when a function calls itself to "
        "solve a smaller instance of the same problem, typically with a "
        "base case to stop the recursion.",
    "what is the difference between a function and a method": "A function "
        "is a standalone block of reusable code defined with 'def', while "
        "a method is a function that belongs to an object/class and is "
        "called using dot notation, e.g. my_list.append(5).",
    "what is a return statement": "The 'return' statement exits a function "
        "and optionally sends a value back to the caller. A function "
        "without a return statement implicitly returns None.",

    # --- Comprehensions & Generators --------------------------------------
    "what is a list comprehension": "A list comprehension is a concise way "
        "to create a list. Example: squares = [x**2 for x in range(10)] "
        "creates a list of squares from 0 to 81.",
    "what is a dictionary comprehension": "A dictionary comprehension "
        "builds a dictionary in one line. Example: "
        "{x: x**2 for x in range(5)} creates {0:0, 1:1, 2:4, 3:9, 4:16}.",
    "what is a generator in python": "A generator is a function that uses "
        "'yield' instead of 'return' to produce a sequence of values "
        "lazily, one at a time, saving memory compared to building a full "
        "list.",
    "what is the difference between yield and return": "'return' exits a "
        "function completely and sends back a single value. 'yield' "
        "pauses the function, returns a value, and resumes execution from "
        "the same point on the next call — used to create generators.",

    # --- OOP ----------------------------------------------------------------
    "what is object oriented programming in python": "Object-Oriented "
        "Programming (OOP) organizes code into classes and objects. "
        "Python supports the four OOP pillars: encapsulation, "
        "abstraction, inheritance, and polymorphism.",
    "what is a class in python": "A class is a blueprint for creating "
        "objects. It defines attributes (data) and methods (behavior). "
        "Example: class Dog:\n    def __init__(self, name):\n        "
        "self.name = name",
    "what is self in python": "'self' refers to the current instance of a "
        "class. It is the first parameter of instance methods and is used "
        "to access the object's attributes and other methods.",
    "what is __init__ in python": "__init__ is a special constructor "
        "method that is automatically called when a new object of a class "
        "is created, used to initialize the object's attributes.",
    "what is inheritance in python": "Inheritance allows a class (child) "
        "to inherit attributes and methods from another class (parent), "
        "promoting code reuse. Example: class Puppy(Dog):",
    "what is polymorphism in python": "Polymorphism means 'many forms' — "
        "it allows objects of different classes to be treated through a "
        "common interface, e.g. different classes implementing the same "
        "method name differently.",
    "what is encapsulation in python": "Encapsulation is the bundling of "
        "data and methods within a class, and restricting direct access to "
        "some components using naming conventions like _protected or "
        "__private.",
    "what is method overriding": "Method overriding occurs when a "
        "subclass provides its own implementation of a method that is "
        "already defined in its parent class.",
    "what is a static method": "A static method (defined with "
        "@staticmethod) belongs to a class rather than an instance and "
        "does not receive 'self' or 'cls' automatically — it behaves like "
        "a regular function placed inside the class.",
    "what is a class method": "A class method (defined with @classmethod) "
        "receives the class itself ('cls') as its first argument instead "
        "of an instance, and is often used for alternative constructors.",

    # --- Error / Exception handling -----------------------------------------
    "how do you handle exceptions in python": "Python uses try/except "
        "blocks to handle exceptions. Example:\ntry:\n    x = 1/0\n"
        "except ZeroDivisionError:\n    print('Cannot divide by zero')",
    "what is the difference between try except and finally": "'try' "
        "contains code that might raise an exception, 'except' handles the "
        "exception if it occurs, and 'finally' contains code that always "
        "runs, whether or not an exception occurred.",
    "what is an exception in python": "An exception is an error detected "
        "during program execution that disrupts the normal flow, e.g. "
        "ZeroDivisionError, ValueError, or TypeError.",
    "how do you raise a custom exception": "You raise an exception using "
        "the 'raise' keyword, and can define custom exceptions by "
        "subclassing the built-in Exception class, e.g. class "
        "MyError(Exception): pass",

    # --- Modules, Packages & Environment -------------------------------------
    "what is a module in python": "A module is simply a .py file "
        "containing Python code (functions, classes, variables) that can "
        "be imported and reused in other programs using the 'import' "
        "keyword.",
    "what is a package in python": "A package is a directory containing "
        "multiple related modules along with an __init__.py file, used to "
        "organize a project's code into a namespace hierarchy.",
    "what is pip": "pip is Python's standard package manager, used to "
        "install and manage third-party libraries from the Python Package "
        "Index (PyPI), e.g. pip install requests.",
    "what is a virtual environment": "A virtual environment is an "
        "isolated Python environment that keeps a project's dependencies "
        "separate from other projects and the global Python installation. "
        "Created with: python -m venv myenv",
    "what is the python standard library": "The Python Standard Library "
        "is a collection of built-in modules (like os, sys, math, random, "
        "datetime, json) that come bundled with every Python installation.",

    # --- Strings & Files -----------------------------------------------------
    "how do you read a file in python": "You can read a file using the "
        "built-in open() function, e.g.:\nwith open('file.txt', 'r') as f:"
        "\n    content = f.read()\nUsing 'with' ensures the file is closed "
        "automatically.",
    "how do you write to a file in python": "You open the file in write "
        "mode ('w') or append mode ('a') and use the write() method, e.g.:"
        "\nwith open('file.txt', 'w') as f:\n    f.write('Hello!')",
    "what are common string methods in python": "Common string methods "
        "include .upper(), .lower(), .strip(), .split(), .join(), "
        ".replace(), .find(), and .format() (or f-strings).",
    "what is an f string": "An f-string (formatted string literal) lets "
        "you embed expressions inside string literals using curly braces, "
        "e.g. name = 'Bob'; print(f'Hello, {name}!')",
    "what is slicing in python": "Slicing extracts a portion of a "
        "sequence (string, list, tuple) using the syntax "
        "sequence[start:stop:step]. Example: 'Hello'[1:4] returns 'ell'.",

    # --- Misc / Tools ---------------------------------------------------------
    "what is the difference between python 2 and python 3": "Python 3 is "
        "the current, actively developed version with improvements like "
        "print as a function, better Unicode string handling, and "
        "integer division using //. Python 2 reached end-of-life in "
        "January 2020 and should not be used for new projects.",
    "what is duck typing": "Duck typing is a concept where an object's "
        "suitability is determined by the presence of certain methods and "
        "properties, rather than the object's actual type — 'if it walks "
        "like a duck and quacks like a duck, it's a duck.'",
    "what is the global interpreter lock": "The Global Interpreter Lock "
        "(GIL) is a mutex in CPython that allows only one thread to "
        "execute Python bytecode at a time, which affects performance in "
        "CPU-bound multi-threaded programs.",
    "what is a decorator in python": "A decorator is a function that "
        "wraps another function to extend or modify its behavior without "
        "changing its source code, applied using the @decorator_name "
        "syntax.",
    "what is the range function": "range() generates a sequence of "
        "numbers, commonly used in for loops. Example: range(5) produces "
        "0, 1, 2, 3, 4.",
    "how do you install a library in python": "Use pip from the command "
        "line, e.g. 'pip install numpy'. It's best practice to do this "
        "inside a virtual environment.",
    "what is numpy": "NumPy is a popular third-party library for "
        "numerical computing in Python, providing fast, multi-dimensional "
        "array objects and mathematical functions.",
    "what is pandas": "pandas is a popular Python library for data "
        "manipulation and analysis, providing the DataFrame and Series "
        "data structures for working with tabular data.",
}


# ---------------------------------------------------------------------------
# 2. MATCHING ENGINE
# ---------------------------------------------------------------------------

class PythonQABot:
    """Rule-based Q&A engine that matches free-text questions against a
    predefined knowledge base using a combination of:
      (a) fuzzy sequence-similarity matching (difflib), and
      (b) keyword-overlap scoring
    so that the user doesn't need to phrase the question exactly."""

    GREETINGS = {"hi", "hello", "hey", "hii", "hola", "yo"}
    THANKS = {"thanks", "thank you", "thanks a lot", "thank you very much"}
    FAREWELLS = {"bye", "goodbye", "exit", "quit", "see you"}

    # very small set of "noise" words ignored during keyword scoring
    # Note: "for" is deliberately NOT a stopword — it's meaningful in this
    # domain (e.g. "for loop"), unlike everyday English use.
    STOPWORDS = {
        "what", "is", "the", "a", "an", "are", "how", "do", "you", "does",
        "in", "of", "to", "and", "or", "with", "can", "i", "your",
        "it", "this", "that", "define", "explain", "tell", "me", "about",
    }

    # common abbreviations / shorthand -> expanded form, applied BEFORE
    # keyword extraction so short slang terms map onto real KB vocabulary
    # instead of being fuzzy-matched by accident (e.g. "oop" ~ "loop").
    ALIASES = {
        "oop": "object oriented programming",
        "oops": "object oriented programming",
        "kwargs": "keyword arguments kwargs",
        "args": "arguments args",
        "diff": "difference",
        "py": "python",
        "func": "function",
        "fn": "function",
        "var": "variable",
        "vars": "variables",
        "err": "error exception",
        "exc": "exception",
        "excep": "exception",
        "listcomp": "list comprehension",
        "dictcomp": "dictionary comprehension",
    }

    def __init__(self, knowledge_base=None):
        self.knowledge_base = knowledge_base or KNOWLEDGE_BASE
        self.history = []  # stores (timestamp, user_input, bot_response)

    # -- text preprocessing -------------------------------------------------
    @staticmethod
    def preprocess(text):
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)   # strip punctuation
        text = re.sub(r"\s+", " ", text)       # collapse whitespace
        return text

    def _expand_aliases(self, text):
        words = text.split()
        expanded = [self.ALIASES.get(w, w) for w in words]
        return " ".join(expanded)

    def _keywords(self, text):
        text = self._expand_aliases(text)
        return {w for w in text.split() if w not in self.STOPWORDS and len(w) > 1}

    @staticmethod
    def _word_similar(a, b):
        """True if two individual words are the same or a likely typo of
        each other. Only words of length >= 4 are allowed a fuzzy (typo)
        match, and the bar is high (>=0.84) — this avoids the classic trap
        of short words coincidentally overlapping as substrings (e.g. the
        3-letter word 'oop' is technically a substring of 'loop', so a
        naive ratio would wrongly call them similar)."""
        if a == b:
            return True
        if len(a) < 4 or len(b) < 4:
            return False
        return difflib.SequenceMatcher(None, a, b).ratio() >= 0.84

    # -- core matching --------------------------------------------------------
    def find_best_match(self, user_question, score_cutoff=0.45):
        """Returns (matched_question, answer, confidence) or (None, None, 0).

        Matching is done word-by-word on CONTENT words only (stop-words like
        what/is/a/the/do/you are stripped, and common abbreviations like
        'oop' or 'kwargs' are expanded first). For every keyword the user
        typed, we look for the closest keyword in each candidate KB question
        (exact match, or a high-confidence typo match for longer words).
        The score is the fraction of the user's keywords that found a hit,
        weighted against how many "extra" keywords the KB question has, so
        that a KB question packed with unrelated words doesn't win just by
        being long.
        """
        processed = self.preprocess(user_question)
        if not processed:
            return None, None, 0.0

        user_kw = self._keywords(processed)
        if not user_kw:
            return None, None, 0.0

        best_q, best_score = None, 0.0
        for q in self.knowledge_base:
            q_kw = self._keywords(q)
            if not q_kw:
                continue

            hits = 0
            for uw in user_kw:
                if any(self._word_similar(uw, qw) for qw in q_kw):
                    hits += 1

            if hits == 0:
                continue

            # F1-style score: recall = how much of what the USER said was
            # found in this KB question; precision = how much of the KB
            # question's own vocabulary was actually addressed. Using the
            # harmonic mean rewards a question that is both fully covered
            # and not diluted by a lot of unrelated extra words, without
            # unfairly penalizing plural/typo hits the way a plain set
            # union (Jaccard) would (e.g. "decorators" vs "decorator").
            recall = hits / len(user_kw)
            precision = hits / len(q_kw)
            score = 2 * recall * precision / (recall + precision)
            if score > best_score:
                best_score, best_q = score, q

        if best_q is not None and best_score >= score_cutoff:
            return best_q, self.knowledge_base[best_q], round(best_score, 2)
        return None, None, round(best_score, 2)

    # -- top-level response generation ----------------------------------------
    def get_response(self, user_input):
        processed = self.preprocess(user_input)

        if not processed:
            return "Please type a question — I'm ready when you are!"

        if processed in self.GREETINGS:
            response = "Hello! I'm your Python Q&A assistant. Ask me anything about Python programming."
        elif processed in self.THANKS or processed.startswith("thank"):
            response = "You're welcome! Ask me anything else about Python."
        elif processed in self.FAREWELLS:
            response = "Goodbye! Happy coding. 🐍"
        else:
            matched_q, answer, confidence = self.find_best_match(user_input)
            if answer:
                response = answer
            else:
                response = (
                    "I'm not sure about that one yet — my knowledge base "
                    "doesn't cover it. Try asking about Python basics, data "
                    "types, functions, OOP, exceptions, or modules. "
                    "(e.g. \"What is a list comprehension?\")"
                )

        self.history.append((datetime.now().strftime("%H:%M:%S"), user_input, response))
        return response

    def topics(self):
        """Returns a readable list of sample questions the bot can answer."""
        return sorted(self.knowledge_base.keys())


# ---------------------------------------------------------------------------
# 3. CLI INTERFACE
# ---------------------------------------------------------------------------

def run_cli():
    bot = PythonQABot()
    print("=" * 62)
    print(" PYTHON Q&A CHATBOT  —  Command Line Interface".center(62))
    print("=" * 62)
    print("Ask me anything about Python! Type 'topics' to see sample")
    print("questions, or 'quit' / 'exit' to leave.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye! Happy coding. 🐍")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Bot: Goodbye! Happy coding. 🐍")
            break

        if user_input.lower() == "topics":
            print("\nBot: Here are some things you can ask me about:")
            for q in bot.topics():
                print(f"   • {q}")
            print()
            continue

        response = bot.get_response(user_input)
        print(f"Bot: {response}\n")


# ---------------------------------------------------------------------------
# 4. GUI INTERFACE (Tkinter)
# ---------------------------------------------------------------------------

def run_gui():
    import tkinter as tk
    from tkinter import scrolledtext, font

    bot = PythonQABot()

    class ChatbotGUI:
        def __init__(self, root):
            self.root = root
            root.title("Python Q&A Chatbot")
            root.geometry("560x640")
            root.configure(bg="#1e1e2e")
            root.minsize(420, 480)

            header_font = font.Font(family="Consolas", size=14, weight="bold")
            chat_font = font.Font(family="Consolas", size=11)

            header = tk.Label(
                root, text="🐍 Python Q&A Chatbot", font=header_font,
                bg="#252535", fg="#4fc3f7", pady=14,
            )
            header.pack(fill=tk.X)

            self.chat_area = scrolledtext.ScrolledText(
                root, wrap=tk.WORD, state="disabled", font=chat_font,
                bg="#151521", fg="#e0e0e8", padx=10, pady=10, borderwidth=0,
            )
            self.chat_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))
            self.chat_area.tag_config("user", foreground="#ffd43b")
            self.chat_area.tag_config("bot", foreground="#6ee7b7")

            entry_frame = tk.Frame(root, bg="#1e1e2e")
            entry_frame.pack(fill=tk.X, padx=10, pady=10)

            self.entry = tk.Entry(
                entry_frame, font=chat_font, bg="#252535", fg="#e0e0e8",
                insertbackground="#e0e0e8", borderwidth=0,
            )
            self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 8))
            self.entry.bind("<Return>", self.send_message)
            self.entry.focus()

            send_btn = tk.Button(
                entry_frame, text="Send", command=self.send_message,
                bg="#4fc3f7", fg="#1e1e2e", activebackground="#3aa9db",
                borderwidth=0, font=("Consolas", 10, "bold"), padx=16,
            )
            send_btn.pack(side=tk.RIGHT)

            self._append("Bot", "Hi! I'm your Python Q&A assistant. Ask me "
                                 "anything about Python programming.")

        def _append(self, sender, text):
            self.chat_area.configure(state="normal")
            tag = "user" if sender == "You" else "bot"
            self.chat_area.insert(tk.END, f"{sender}: ", tag)
            self.chat_area.insert(tk.END, f"{text}\n\n")
            self.chat_area.configure(state="disabled")
            self.chat_area.see(tk.END)

        def send_message(self, event=None):
            user_text = self.entry.get().strip()
            if not user_text:
                return
            self.entry.delete(0, tk.END)
            self._append("You", user_text)
            response = bot.get_response(user_text)
            self._append("Bot", response)

    root = tk.Tk()
    ChatbotGUI(root)
    root.mainloop()


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--gui" in sys.argv:
        run_gui()
    else:
        run_cli()
