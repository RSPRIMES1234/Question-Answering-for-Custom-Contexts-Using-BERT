# Question-Answering-for-Custom-Contexts-Using-BERT  
This project demonstrates the use of a BERT model for question answering tasks on any given context. It showcases how to process long texts, handle multiple questions, and extract answers with confidence scores, making it adaptable for various domains and use cases.

## Features

- Flexible context input: Use any text as your knowledge base
- BERT-powered question answering
- Long text processing through sentence chunking
- Multiple question handling  
- Answer confidence scoring


## Requirements

- Python 3.7+  
- torch  
- transformers  
- scipy  
- numpy


## Installation

1. Clone this repository:
```python
git clone https://github.com/RSPRIMES1234/Question-Answering-for-Custom-Contexts-Using-BERT\
cd Question-Answering-for-Custom-Contexts-Using-BERT
```  
2. Install the required dependencies:
```python
pip install torch transformers scipy numpy
```
## Usage

1. Open the main script and modify the `context` variable with your desired text.  

2. Update the `questions` list with the questions you want to ask about your context.  

3. Run the script:
```python
python main.py
```

4. The script will process your context and answer the provided questions, printing the answers along with their confidence scores.


## Customization

- Context: Replace the existing text in the `context` variable with any text you want to query.
- Questions: Modify the `questions` list to include any questions relevant to your context.
- Model: You can experiment with different BERT models by changing the `model_name` variable.


## How it works

1. The script loads a pre-trained BERT model for question answering.  
2. It splits the context into manageable chunks for processing long texts.  
3. For each chunk and question, it predicts an answer and calculates a confidence score.  
4. It retains the answer with the highest confidence score for each question.  
5. Finally, it outputs the best answers for all questions.  


## Applications

This tool can be used for various applications, including but not limited to:
- Automated FAQ systems  
- Information extraction from documents  
- Educational tools for text comprehension  
- Research assistance for quick information retrieval


## Limitations

- The model's performance depends on the quality and relevance of the provided context.  
- It may not always provide accurate answers, especially for questions not directly addressed in the context.  
- The current implementation is not optimized for very large texts or real-time applications.  













