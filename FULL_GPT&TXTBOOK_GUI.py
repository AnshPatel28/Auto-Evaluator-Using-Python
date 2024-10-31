import tkinter as tk
from tkinter import ttk, messagebox, Scrollbar, Frame, VERTICAL, RIGHT, Y
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from collections import Counter
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet as wn
from nltk.sentiment import SentimentIntensityAnalyzer

# Set the NLTK data path
nltk.data.path.append("/Users/pcompany/nltk_data")

# Download the Vader lexicon if not already downloaded
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('vader_lexicon')


# Functions for processing student answers
def preprocess_answer(answer):
    answer = answer.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(answer)
    tokens_without_punctuation = [token for token in tokens if token not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens_without_stopwords = [token for token in tokens_without_punctuation if token not in stop_words]
    return tokens_without_stopwords


def extract_keywords(tokens):
    word_counts = Counter(tokens)
    keywords = word_counts.most_common(5)
    return [keyword for keyword, _ in keywords]


def calculate_cosine_similarity(student_answer, reference_answer):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([student_answer, reference_answer])
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity


def calculate_keyword_matching(student_tokens, reference_tokens):
    common_keywords = set(student_tokens) & set(reference_tokens)
    matching_score = len(common_keywords) / len(reference_tokens) if reference_tokens else 0
    return matching_score


def calculate_semantic_similarity(student_tokens, reference_tokens):
    similarity_scores = []
    for token1 in student_tokens:
        max_similarity = -1
        for token2 in reference_tokens:
            synset1 = wn.synsets(token1)
            synset2 = wn.synsets(token2)
            if synset1 and synset2:
                similarity = max((s1.path_similarity(s2) or 0) for s1 in synset1 for s2 in synset2)
                if similarity is not None and similarity > max_similarity:
                    max_similarity = similarity
        if max_similarity != -1:
            similarity_scores.append(max_similarity)
    average_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    return average_similarity


def calculate_sentiment_polarity(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)
    polarity = sentiment_score['compound']
    return polarity


def calculate_marks(scores, grading_criteria):
    max_score = sum(grading_criteria[measure]['weight'] for measure in grading_criteria)
    total_marks = sum(score * grading_criteria[measure]['weight'] for measure, score in scores.items())
    percentage = (total_marks / max_score) * 100
    return percentage


# GUI Application
class QuizApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Welcome to evaluator")
        self.root.geometry("600x400")

        # Center the main window
        self.root.eval('tk::PlaceWindow . center')

        # Initialize data storage
        self.num_students = 0
        self.students = []
        self.num_questions = 0
        self.questions = []
        self.answers = []
        self.gpt_answers = []
        self.textbook_answers = []

        self.current_page = 0
        self.create_initial_page()

    def create_initial_page(self):
        self.clear_frame()
        label = tk.Label(self.root, text="Enter Number of Students:")
        label.pack(pady=10)
        self.student_entry = tk.Entry(self.root)
        self.student_entry.pack(pady=5)
        label = tk.Label(self.root, text="Enter Number of Questions:")
        label.pack(pady=10)
        self.question_entry = tk.Entry(self.root)
        self.question_entry.pack(pady=5)
        next_button = tk.Button(self.root, text="Next", command=self.save_initial_data)
        next_button.pack(pady=20)

    def save_initial_data(self):
        try:
            self.num_students = int(self.student_entry.get())
            self.num_questions = int(self.question_entry.get())
            self.current_page = 1
            self.students = [""] * self.num_students
            self.questions = [""] * self.num_questions
            self.answers = [[""] * self.num_students for _ in range(self.num_questions)]
            self.gpt_answers = [""] * self.num_questions
            self.textbook_answers = [""] * self.num_questions
            self.create_student_names_page()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for students and questions.")

    def create_student_names_page(self):
        self.clear_frame()
        label = tk.Label(self.root, text="Enter Student Names:")
        label.pack(pady=10)
        self.student_entries = []
        for i in range(self.num_students):
            entry = tk.Entry(self.root)
            entry.pack(pady=5)
            self.student_entries.append(entry)
        next_button = tk.Button(self.root, text="Next", command=self.save_student_names)
        next_button.pack(pady=20)

    def save_student_names(self):
        self.students = [entry.get() for entry in self.student_entries]
        self.current_page = 2
        self.create_question_answers_page()

    def create_question_answers_page(self):
        self.clear_frame()

        container = Frame(self.root)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container)
        scrollbar = Scrollbar(container, orient=VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollable_frame = Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="n")

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        label = tk.Label(scrollable_frame, text="Enter Questions and Student Answers:")
        label.pack(pady=10)

        self.question_entries = []
        self.answer_entries = []

        for i in range(self.num_questions):
            q_frame = Frame(scrollable_frame)
            q_frame.pack(pady=10, fill="x", expand=True)

            q_label = tk.Label(q_frame, text=f"Question {i + 1}:")
            q_label.pack(pady=5, anchor="center")
            q_entry = tk.Entry(q_frame)
            q_entry.pack(pady=5, anchor="center", fill="x", expand=True)
            self.question_entries.append(q_entry)

            student_answers = []
            for j in range(self.num_students):
                a_frame = Frame(scrollable_frame)
                a_frame.pack(pady=5, fill="x", expand=True)

                a_label = tk.Label(a_frame, text=f"Answer by {self.students[j]}:")
                a_label.pack(pady=2, anchor="center")
                a_entry = tk.Entry(a_frame)
                a_entry.pack(pady=2, anchor="center", fill="x", expand=True)
                student_answers.append(a_entry)
            self.answer_entries.append(student_answers)

        next_button = tk.Button(scrollable_frame, text="Next", command=self.save_question_answers)
        next_button.pack(pady=20)

        # Centering the scrollable frame within the canvas
        scrollable_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

        # Adjust the canvas width to ensure centering
        canvas_width = canvas.winfo_width()
        scrollable_frame_width = scrollable_frame.winfo_reqwidth()

        if scrollable_frame_width < canvas_width:
            canvas.create_window(
                (canvas_width // 2 - scrollable_frame_width // 2, 0),
                window=scrollable_frame,
                anchor="n"
            )
        else:
            canvas.create_window((0, 0), window=scrollable_frame, anchor="n")

    def save_question_answers(self):
        self.questions = [entry.get() for entry in self.question_entries]
        self.answers = [[answer_entry.get() for answer_entry in student_answers] for student_answers in
                        self.answer_entries]
        self.current_page = 3
        self.create_gpt_answers_page()

    def create_gpt_answers_page(self):
        self.clear_frame()
        label = tk.Label(self.root, text="Enter GPT Answers and Textbook Answers:")
        label.pack(pady=10)

        self.gpt_entries = []
        self.textbook_entries = []

        for i in range(self.num_questions):
            gpt_label = tk.Label(self.root, text=f"GPT Answer for Question {i + 1}:")
            gpt_label.pack(pady=5)
            gpt_entry = tk.Entry(self.root)
            gpt_entry.pack(pady=5)
            self.gpt_entries.append(gpt_entry)

            textbook_label = tk.Label(self.root, text=f"Textbook Answer for Question {i + 1}:")
            textbook_label.pack(pady=5)
            textbook_entry = tk.Entry(self.root)
            textbook_entry.pack(pady=5)
            self.textbook_entries.append(textbook_entry)

        next_button = tk.Button(self.root, text="Submit", command=self.calculate_grades)
        next_button.pack(pady=20)

    def calculate_grades(self):
        self.gpt_answers = [entry.get() for entry in self.gpt_entries]
        self.textbook_answers = [entry.get() for entry in self.textbook_entries]

        graded_answers = {}
        grading_criteria = {
            'Cosine Similarity': {'weight': 1},
            'Keyword Matching': {'weight': 1},
            'Semantic Similarity': {'weight': 1},
            'Contradiction Detection': {'weight': 1}
        }

        for question_index, (question, question_answers) in enumerate(zip(self.questions, self.answers)):
            gpt_answer = self.gpt_answers[question_index]
            gpt_tokens = preprocess_answer(gpt_answer)
            gpt_keywords = extract_keywords(gpt_tokens)

            textbook_answer = self.textbook_answers[question_index]
            textbook_tokens = preprocess_answer(textbook_answer)
            textbook_keywords = extract_keywords(textbook_tokens)

            answer_scores = {}
            for answer_index, answer in enumerate(question_answers):
                tokens = preprocess_answer(answer)
                keywords = extract_keywords(tokens)

                # GPT Scores
                gpt_cosine_similarity_score = calculate_cosine_similarity(answer, gpt_answer)
                gpt_keyword_matching_score = calculate_keyword_matching(tokens, gpt_tokens)
                gpt_semantic_similarity_score = calculate_semantic_similarity(tokens, gpt_tokens)
                gpt_contradiction_detection_score = 1 if calculate_sentiment_polarity(
                    answer) * calculate_sentiment_polarity(gpt_answer) >= 0 else 0

                # Textbook Scores
                textbook_cosine_similarity_score = calculate_cosine_similarity(answer, textbook_answer)
                textbook_keyword_matching_score = calculate_keyword_matching(tokens, textbook_tokens)
                textbook_semantic_similarity_score = calculate_semantic_similarity(tokens, textbook_tokens)
                textbook_contradiction_detection_score = 1 if calculate_sentiment_polarity(
                    answer) * calculate_sentiment_polarity(textbook_answer) >= 0 else 0

                gpt_scores = {
                    'Cosine Similarity': gpt_cosine_similarity_score,
                    'Keyword Matching': gpt_keyword_matching_score,
                    'Semantic Similarity': gpt_semantic_similarity_score,
                    'Contradiction Detection': gpt_contradiction_detection_score
                }

                textbook_scores = {
                    'Cosine Similarity': textbook_cosine_similarity_score,
                    'Keyword Matching': textbook_keyword_matching_score,
                    'Semantic Similarity': textbook_semantic_similarity_score,
                    'Contradiction Detection': textbook_contradiction_detection_score
                }

                gpt_marks = calculate_marks(gpt_scores, grading_criteria)
                textbook_marks = calculate_marks(textbook_scores, grading_criteria)

                answer_scores[f'Student {self.students[answer_index]}'] = {
                    'GPT': gpt_marks,
                    'TextBook': textbook_marks
                }

            graded_answers[question] = answer_scores

        self.show_grades_page(graded_answers)

    def show_grades_page(self, graded_answers):
        self.clear_frame()
        label = tk.Label(self.root, text="Graded Answers:")
        label.pack(pady=10)

        for question, answers in graded_answers.items():
            q_label = tk.Label(self.root, text=question)
            q_label.pack(pady=5)
            for student_answer, marks in answers.items():
                a_label_gpt = tk.Label(self.root, text=f"{student_answer} (GPT): {marks['GPT']:.2f}%")
                a_label_gpt.pack(pady=2)
                a_label_textbook = tk.Label(self.root, text=f"{student_answer} (TextBook): {marks['TextBook']:.2f}%")
                a_label_textbook.pack(pady=2)

        finish_button = tk.Button(self.root, text="Finish", command=self.root.quit)
        finish_button.pack(pady=20)

    def clear_frame(self):
        for widget in self.root.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = QuizApp(root)
    root.mainloop()
  