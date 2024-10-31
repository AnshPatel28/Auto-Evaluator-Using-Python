import ssl
import nltk
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

# Download necessary NLTK data if not already downloaded
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    # Workaround for SSL certificate verification issue
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('vader_lexicon')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')

# Function to preprocess and extract keywords
def preprocess_answer(answer):
    # Convert text to lowercase
    answer = answer.lower()

    # Tokenization
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(answer)

    # Punctuation removal
    tokens_without_punctuation = [token for token in tokens if token not in string.punctuation]

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens_without_stopwords = [token for token in tokens_without_punctuation if token not in stop_words]

    return tokens_without_stopwords

def extract_keywords(tokens):
    # Use Counter to count occurrences of each word
    word_counts = Counter(tokens)

    # Extract the top 5 most common words as keywords
    keywords = word_counts.most_common(5)

    return keywords

def calculate_cosine_similarity(student_answer, reference_answer):
    # Convert answers to TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([student_answer, reference_answer])

    # Calculate cosine similarity between the two vectors
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity

def calculate_keyword_matching(student_tokens, reference_tokens):
    # Calculate keyword matching score
    common_keywords = set(student_tokens) & set(reference_tokens)
    matching_score = len(common_keywords) / len(reference_tokens)
    return matching_score

def calculate_semantic_similarity(student_tokens, reference_tokens):
    # Calculate semantic similarity using WordNet
    similarity_scores = []
    for token1 in student_tokens:
        max_similarity = -1
        for token2 in reference_tokens:
            synset1 = wn.synsets(token1)
            synset2 = wn.synsets(token2)
            if synset1 and synset2:  # Check if both tokens have synsets
                similarity = max(s1.path_similarity(s2) for s1 in synset1 for s2 in synset2)
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

# Sample questions and answers data
questions = [
    "What is data mining?",
    "What is Artificial Narrow Intelligence (ANI)?",
    "What is Deep Learning based on?",
    "What is the full form of LSTM?",
    "What is a data cube?"
]

# Sample student answers
student_answers = [
    [
        "Data mining involves extracting patterns, trends, and valuable insights from large datasets through the application of various algorithms, statistical techniques, and machine learning methods. Its importance has grown alongside the increase in data generation and storage capacities, making it essential for informed decision-making.",
        "Data mining is the process of sorting through large data sets to identify patterns and relationships that can help solve business problems through data analysis.",
        "Data mining has improved organizational decision-making through insightful data analyses.",
        "Data mining is essential for extracting valuable insights and knowledge from large datasets."
    ],
    [
        "Artificial Narrow Intelligence (ANI), also known as Weak AI, refers to AI systems that are designed and trained to perform a specific task or a narrow range of tasks.",
        "Artificial Narrow Intelligence (ANI) is the computer’s ability to perform a single task extremely well.",
        "Artificial narrow intelligence refers to the intelligence system obligated to function on a specific task.",
        "Artificial Narrow Intelligence (ANI) refers to a type of artificial intelligence that is designed and trained to perform specific tasks or solve particular problems within a limited domain."
    ],
    [
        "Deep learning is a subfield of machine learning that focuses on the development of artificial neural networks with multiple layers.",
        "Deep learning is a method in artificial intelligence (AI) that teaches computers to process data in a way that is inspired by the human brain.",
        "Deep learning can be considered as a subset of machine learning. It is a field that is based on learning and improving on its own by examining computer algorithms.",
        "Deep learning is based on artificial neural networks (ANNs), which are inspired by the structure and function of the human brain."
    ],
    [
        "LSTM stands for Long Short-Term Memory, and it is a type of recurrent neural network (RNN) architecture that is widely used in artificial intelligence and natural language processing.",
        "Long Short-Term Memory Networks is a deep learning, sequential neural network that allows information to persist.",
        "Long Short-Term Memory is an improved version of recurrent neural network designed by Hochreiter & Schmidhuber.",
        "The full form of LSTM is Long Short-Term Memory. LSTM is a type of recurrent neural network (RNN) architecture designed to overcome the limitations of traditional RNNs in capturing long-term dependencies in sequential data."
    ],
    [
        "A data cube is a multidimensional (3D) representation of data that can be used to support various types of analysis and modeling.",
        "A data cube is a multidimensional representation of data intended to facilitate effortless retrieval and structured analysis of the underlying data.",
        "A data cube refers to a multi-dimensional array of values, commonly used to describe a time series of image data.",
        "A data cube is a multidimensional representation of data used in data warehousing and OLAP (Online Analytical Processing) systems."
    ]
]

# Sample GPT answers
gpt_answers = [
    "Data mining is essential for extracting valuable insights and knowledge from large datasets. It enables organizations to uncover hidden patterns, correlations, and trends that can't be easily discerned through traditional methods. By analyzing data, businesses can make informed decisions, predict future trends, and optimize processes. Data mining plays a crucial role in various fields such as business intelligence, predictive analysis, risk management, customer relationship management, and healthcare. It empowers organizations to enhance their competitiveness, improve efficiency, and drive innovation by leveraging the wealth of information contained within their data assets. Overall, data mining is indispensable for unlocking the full potential of data and turning it into actionable intelligence for strategic decision-making.",
    "Artificial Narrow Intelligence (ANI) refers to a type of artificial intelligence that is designed and trained to perform specific tasks or solve particular problems within a limited domain. ANI systems excel at executing predefined tasks with high precision and efficiency, but they lack the ability to generalize their knowledge or skills beyond the scope of their training data. Examples of ANI include virtual personal assistants like Siri and Alexa, recommendation systems, spam filters, and image recognition algorithms. While ANI has made significant advancements and is prevalent in various applications today, it remains constrained to narrow domains and cannot exhibit human-like intelligence or adaptability across different tasks or contexts.",
    "Deep learning is based on artificial neural networks (ANNs), which are inspired by the structure and function of the human brain. ANNs consist of interconnected layers of nodes (neurons) that process information in a hierarchical manner. Deep learning algorithms, such as deep neural networks (DNNs), utilize multiple layers of neurons to learn intricate patterns and representations from data. Through a process called backpropagation, these networks adjust their parameters iteratively to minimize errors and optimize performance. Deep learning excels at handling complex, unstructured data such as images, text, and audio, and has achieved remarkable success in various fields including computer vision, natural language processing, speech recognition, and autonomous driving. Its ability to automatically learn from large datasets without relying on handcrafted features makes it a powerful tool for solving challenging problems in artificial intelligence.",
    "The full form of LSTM is Long Short-Term Memory. LSTM is a type of recurrent neural network (RNN) architecture designed to overcome the limitations of traditional RNNs in capturing long-term dependencies in sequential data. It achieves this by introducing specialized memory cells that can store information for extended periods, preventing the vanishing or exploding gradient problem often encountered in standard RNNs. The architecture includes gates that regulate the flow of information, such as the input gate, forget gate, and output gate, enabling the network to selectively retain or discard information over time. LSTM networks have proven to be highly effective in various tasks involving sequential data, such as natural language processing, speech recognition, time series forecasting, and more, making them a popular choice in machine learning and artificial intelligence applications.",
    "A data cube is a multidimensional representation of data used in data warehousing and OLAP (Online Analytical Processing) systems. It organizes data into multiple dimensions, such as time, geography, product, or customer, allowing for complex analysis and exploration. Each dimension represents a different attribute or aspect of the data, and the intersections of these dimensions form cells in the cube. These cells contain aggregated data measures, such as sum, count, average, etc., providing a comprehensive view of the data from different perspectives. Data cubes enable users to perform multidimensional analysis, drill down into specific subsets of data, and gain insights into trends, patterns, and relationships that might not be apparent in traditional tabular representations. They are valuable tools for decision-makers seeking to extract actionable insights from large and complex datasets."
]

# Sample textbook answers
textbook_answers = [
    "Data mining is the process of discovering patterns, trends, and useful information from large datasets using various algorithms, statistical methods, and machine learning techniques. It has gained significant importance due to the growth of data generation and storage capabilities. The need for data mining arises from several aspects, including decision-making.",
    "Artificial Narrow Intelligence (ANI), also known as Weak AI, refers to AI systems that are designed and trained to perform a specific task or a narrow range of tasks. These systems are highly specialized and can perform their designated task with a high degree of accuracy and efficiency. This type of technology is also known as Weak AI.",
    "Deep learning is a subfield of machine learning that focuses on the development of artificial neural networks with multiple layers, also known as deep neural networks. These networks are particularly effective in modeling complex, hierarchical patterns and representations in data. Deep learning is inspired by the structure and function of the human brain, specifically the biological neural networks that make up the brain.",
    "LSTM stands for Long Short-Term Memory, and it is a type of recurrent neural network (RNN) architecture that is widely used in artificial intelligence and natural language processing. LSTM networks have been successfully used in a wide range of applications, including speech recognition, language translation, and video analysis, among others.",
    "A data cube is a multidimensional (3D) representation of data that can be used to support various types of analysis and modeling. Data cubes are often used in machine learning and data mining applications to help identify patterns, trends, and correlations in complex datasets."
]

# Function to grade the answers
def grade_answers(reference_answers, student_answers):
    graded_results = []

    for question_index, reference_answer in enumerate(reference_answers):
        question_results = []
        reference_tokens = preprocess_answer(reference_answer)

        for student_answer in student_answers[question_index]:
            student_tokens = preprocess_answer(student_answer)
            cosine_sim = calculate_cosine_similarity(student_answer, reference_answer)
            keyword_matching_score = calculate_keyword_matching(student_tokens, reference_tokens)
            semantic_similarity_score = calculate_semantic_similarity(student_tokens, reference_tokens)
            sentiment_polarity_student = calculate_sentiment_polarity(student_answer)
            sentiment_polarity_reference = calculate_sentiment_polarity(reference_answer)

            # Calculate overall grade based on the criteria from GPT_Code
            overall_grade = (0.4 * cosine_sim) + \
                            (0.2 * keyword_matching_score) + \
                            (0.2 * semantic_similarity_score) + \
                            (0.2 * (1 - abs(sentiment_polarity_student - sentiment_polarity_reference)))

            # Convert overall grade to percentage
            overall_percentage = overall_grade * 100
            question_results.append(overall_percentage)

        graded_results.append(question_results)

    return graded_results

# Calculate total marks for each student
def calculate_total_marks(graded_results):
    num_students = len(graded_results[0])
    total_marks = [0] * num_students

    for results in graded_results:
        for i, grade in enumerate(results):
            total_marks[i] += grade

    return total_marks

# Main execution
def main(questions, student_answers, gpt_answers, textbook_answers):
    graded_results_gpt = grade_answers(gpt_answers, student_answers)
    graded_results_textbook = grade_answers(textbook_answers, student_answers)

    total_marks_gpt = calculate_total_marks(graded_results_gpt)
    total_marks_textbook = calculate_total_marks(graded_results_textbook)

    # Print the graded results for GPT answers
    print("Grades based on GPT answers:")
    for question_index, question in enumerate(questions):
        print(f"Question: {question}")
        for student_index, grade in enumerate(graded_results_gpt[question_index]):
            print(f"Student Answer {student_index + 1}: {grade:.2f}% marks")
        print(f"Total marks for each student: {graded_results_gpt[question_index]}\n")

    # Print the graded results for Textbook answers
    print("Grades based on Textbook answers:")
    for question_index, question in enumerate(questions):
        print(f"Question: {question}")
        for student_index, grade in enumerate(graded_results_textbook[question_index]):
            print(f"Student Answer {student_index + 1}: {grade:.2f}% marks")
        print(f"Total marks for each student: {graded_results_textbook[question_index]}\n")

    # Print total marks for each student across all questions
    print("Total Marks for Each Student (GPT):")
    for student_index, total in enumerate(total_marks_gpt):
        print(f"Student {student_index + 1}: {total / len(questions):.2f}%")

    print("Total Marks for Each Student (Textbook):")
    for student_index, total in enumerate(total_marks_textbook):
        print(f"Student {student_index + 1}: {total / len(questions):.2f}%")

# Run the main function
main(questions, student_answers, gpt_answers, textbook_answers)
