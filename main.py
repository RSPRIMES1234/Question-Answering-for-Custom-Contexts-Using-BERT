# Import necessary libraries for data processing, modeling, and visualization
import torch
from transformers import BertForQuestionAnswering, BertTokenizerFast
from scipy.special import softmax
import numpy as np

# Define the context (can be changed according to user's task)
context = """
Frédéric François Chopin[n 1] (born Fryderyk Franciszek Chopin;[n 2][n 3] 1 March 1810 – 17 October 1849) was a Polish composer and virtuoso pianist of the Romantic period, who wrote primarily for solo piano. He has maintained worldwide renown as a leading musician of his era, one whose "poetic genius was based on a professional technique that was without equal in his generation".[5]

Chopin was born in Żelazowa Wola and grew up in Warsaw, which in 1815 became part of Congress Poland. A child prodigy, he completed his musical education and composed his earlier works in Warsaw before leaving Poland at the age of 20, less than a month before the outbreak of the November 1830 Uprising. At 21, he settled in Paris. Thereafter he gave only 30 public performances, preferring the more intimate atmosphere of the salon. He supported himself by selling his compositions and by giving piano lessons, for which he was in high demand. Chopin formed a friendship with Franz Liszt and was admired by many of his musical contemporaries, including Robert Schumann. After a failed engagement to Maria Wodzińska from 1836 to 1837, he maintained an often troubled relationship with the French writer Aurore Dupin (known by her pen name George Sand). A brief and unhappy visit to Mallorca with Sand in 1838–39 would prove one of his most productive periods of composition. In his final years, he was supported financially by his admirer Jane Stirling. For most of his life, Chopin was in poor health. He died in Paris in 1849 at the age of 39.

All of Chopin's compositions feature the piano. Most are for solo piano, though he also wrote two piano concertos, some chamber music, and 19 songs set to Polish lyrics. His piano pieces are technically demanding and expanded the limits of the instrument; his own performances were noted for their nuance and sensitivity. Chopin's major piano works include mazurkas, waltzes, nocturnes, polonaises, the instrumental ballade (which Chopin created as an instrumental genre), études, impromptus, scherzi, preludes, and sonatas, some published only posthumously. Among the influences on his style of composition were Polish folk music, the classical tradition of Mozart and Schubert, and the atmosphere of the Paris salons, of which he was a frequent guest. His innovations in style, harmony, and musical form, and his association of music with nationalism, were influential throughout and after the late Romantic period.

Chopin's music, his status as one of music's earliest celebrities, his indirect association with political insurrection, his high-profile love life, and his early death have made him a leading symbol of the Romantic era. His works remain popular, and he has been the subject of numerous films and biographies of varying historical fidelity. Among his many memorials is the Fryderyk Chopin Institute, which was created by the Parliament of Poland to research and promote his life and works. It hosts the International Chopin Piano Competition, a prestigious competition devoted entirely to his works.
Frédéric Chopin was born in Żelazowa Wola, 46 kilometres (29 miles) west of Warsaw, in what was then the Duchy of Warsaw, a Polish state established by Napoleon. The parish baptismal record, which is dated 23 April 1810, gives his birthday as 22 February 1810, and cites his given names in the Latin form Fridericus Franciscus (in Polish, he was Fryderyk Franciszek).[6][7][8] The composer and his family used the birthdate 1 March,[n 4][7] which is now generally accepted as the correct date.[8]

His father, Nicolas Chopin, was a Frenchman from Lorraine who had emigrated to Poland in 1787 at the age of sixteen.[10][11] He married Justyna Krzyżanowska, a poor relative of the Skarbeks, one of the families for whom he worked.[12] Chopin was baptised in the same church where his parents had married, in Brochów. His eighteen-year-old godfather, for whom he was named, was Fryderyk Skarbek, a pupil of Nicolas Chopin.[7] Chopin was the second child of Nicolas and Justyna and their only son; he had an elder sister, Ludwika, and two younger sisters, Izabela and Emilia, whose death at the age of 14 was probably from tuberculosis.[13][14] Nicolas Chopin was devoted to his adopted homeland, and insisted on the use of the Polish language in the household.[7]
In October 1810, six months after Chopin's birth, the family moved to Warsaw, where his father acquired a post teaching French at the Warsaw Lyceum, then housed in the Saxon Palace. Chopin lived with his family on the Palace grounds. The father played the flute and violin;[15] the mother played the piano and gave lessons to boys in the boarding house that the Chopins kept.[16] Chopin was of slight build, and even in early childhood was prone to illnesses.[15]

Chopin may have had some piano instruction from his mother, but his first professional music tutor, from 1816 to 1821, was the Czech pianist Wojciech Żywny.[17] His elder sister Ludwika also took lessons from Żywny, and occasionally played duets with her brother.[18] It quickly became apparent that he was a child prodigy. By the age of seven he had begun giving public concerts, and in 1817 he composed two polonaises, in G minor and B-flat major.[19] His next work, a polonaise in A-flat major of 1821, dedicated to Żywny, is his earliest surviving musical manuscript.[17]

In 1817 the Saxon Palace was requisitioned by Warsaw's Russian governor for military use, and the Warsaw Lyceum was reestablished in the Kazimierz Palace (today the rectorate of Warsaw University). Chopin and his family moved to a building, which still survives, adjacent to the Kazimierz Palace. During this period, he was sometimes invited to the Belweder Palace as playmate to the son of the ruler of Russian Poland, Grand Duke Konstantin Pavlovich of Russia; he played the piano for Konstantin Pavlovich and composed a march for him. Julian Ursyn Niemcewicz, in his dramatic eclogue, "Nasze Przebiegi" ("Our Discourses", 1818), attested to "little Chopin's" popularity.[20]
"""

# Define questions (can be customized based on the context)
questions = [
    "Who was Frédéric François Chopin?",
    "Where was Chopin born?",
    #added a questions who's answer is not in context
    "who is Robert Downey Jr.?",
    "when did Chopin die?",
    "where did Chopin die?"
]

# Load the pre-trained BERT model and tokenizer for question answering tasks
model_name = "deepset/bert-base-cased-squad2"
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)


# Function to predict the answer to a question given a context
def predict_answer(context, question):
    # Tokenize input and generate model outputs
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    # Process model outputs to get the answer
    start_scores, end_scores = softmax(outputs.start_logits)[0], softmax(outputs.end_logits)[0]
    start_idx, end_idx = np.argmax(start_scores), np.argmax(end_scores)
    confidence_score = (start_scores[start_idx] + end_scores[end_idx]) / 2

    # Extract and format the answer
    answer_ids = inputs.input_ids[0][start_idx: end_idx + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(answer_ids)
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    # Return the answer if it's not just the CLS token
    if answer != tokenizer.cls_token:
        return answer, confidence_score
    return None, confidence_score


# Function to chunk sentences for processing long texts
def chunk_sentences(sentences, chunk_size, stride):
    chunks = []
    num_sentences = len(sentences)
    for i in range(0, num_sentences, chunk_size - stride):
        chunk = sentences[i: i + chunk_size]
        chunks.append(chunk)
    return chunks


# Split the context into sentences and create chunks
sentences = context.split("\n")
chunked_sentences = chunk_sentences(sentences, chunk_size=3, stride=1)

# Process each chunk and question to find the best answers
answers = {}
for chunk in chunked_sentences:
    sub_context = "\n".join(chunk)
    for question in questions:
        answer, score = predict_answer(sub_context, question)

        # Store the answer with the highest confidence score
        if answer:
            if question not in answers or score > answers[question][1]:
                answers[question] = (answer, score)

# Print the final answers
for answerr in answers:
    print(answerr,answers[answerr][0],sep=" ")
    print("Probablity Score :",answers[answerr][1])
