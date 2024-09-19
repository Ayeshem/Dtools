import re
import random
from sentence_transformers import util, SentenceTransformer
import pandas as pd
import numpy as np
import torch
import textwrap
import ast

from transformers import AutoModelForCausalLM, AutoTokenizer
from time import perf_counter as timer

# Set device to CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# Helper function to print wrapped text
def print_wrapped(text, wrap_length=80):
    wrapped_text = textwrap.fill(text, wrap_length)
    print(wrapped_text)

# Cosine similarity function for embedding comparisons
def cosine_similarity(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)
    norm_vector1 = torch.sqrt(torch.sum(vector1**2))
    norm_vector2 = torch.sqrt(torch.sum(vector2**2))
    return dot_product / (norm_vector1 * norm_vector2)

# Retrieve relevant resources based on similarity scores
def retrieve_relevant_resources(query: str, embeddings: torch.tensor, model, n_resources_to_return: int=5, print_time: bool=True):
    query_embedding = model.encode(query, convert_to_tensor=True)
    start_time = timer()
    # dot_scores = util.dot_score(query_embedding, embeddings)[0]
    dot_product = torch.matmul(query_embedding, embeddings.T)
    end_time = timer()

    if print_time:
        print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time - start_time:.5f} seconds.")

    scores, indices = torch.topk(dot_product, k=n_resources_to_return)
    return scores, indices

    # end_time = timer()

    # if print_time:
    #     print(f"[INFO] Time taken to get scores on {len(embeddings)} embeddings: {end_time - start_time:.5f} seconds.")

    # scores, indices = torch.topk(input=dot_scores, k=n_resources_to_return)
    # return scores, indices

# Print top results based on similarity
def print_top_results_and_scores(query: str, embeddings: torch.tensor, pages_and_chunks, n_resources_to_return: int=5):
    scores, indices = retrieve_relevant_resources(query=query, model=embedding_model, embeddings=embeddings, n_resources_to_return=n_resources_to_return)
    for score, index in zip(scores, indices):
        print_wrapped(pages_and_chunks[index]["sentence_chunks"])

# Clean the text by removing unwanted characters and formatting
def clean_text(text):
    text = re.sub(r'[-|!]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\.{2,}', '.', text)
    sentences = re.split(r'(?<=\w\.\s)', text)
    cleaned_text = '\n\n'.join(sentences)
    cleaned_text = re.sub(r'\[.*?\]\(.*?\)', '', cleaned_text)
    return cleaned_text

# Retrieve relevant context based on query
def getting_context(query, embedding_model, embeddings, pages_and_chunks):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    dot_scores = util.dot_score(a=query_embedding, b=embeddings)[0]
    top_results_dot_product = torch.topk(dot_scores, k=3)
    scores, indices = retrieve_relevant_resources(query=query, model=embedding_model, embeddings=embeddings)
    print_top_results_and_scores(query=query, pages_and_chunks=pages_and_chunks, embeddings=embeddings)
    context_items = [pages_and_chunks[i] for i in indices]
    context = "- " + "\n- ".join([item["sentence_chunks"] for item in context_items])

    return context

from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from transformers import pipeline
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

model = AutoModelForQuestionAnswering.from_pretrained("./roberta_qa_model")
tokenizer = AutoTokenizer.from_pretrained("./roberta_qa_tokenizer")

# Create the pipeline with the locally loaded model and tokenizer
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Load RoBERTa QA model using Hugging Face pipeline
# qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Function to call the RoBERTa model for QA
def answer_question_with_roberta(question, context):
    result = qa_pipeline(question=question, context=context)
    return result['answer']

def response4(embedding_model, embeddings, pages_and_chunks):
    response_history = []
    is_message_ended = True
    while is_message_ended:
        prompt = input("\nUser: ")
        if prompt.lower() in ['goodbye', 'exit']:
            print("Exiting chat...")
            break

        if len(response_history) == 1:
            historical_prompt = response_history[0] + prompt
            context = getting_context(historical_prompt, embedding_model, embeddings, pages_and_chunks)
            response_history.clear()  # Clear the history after using it
        else:
            context = getting_context(prompt, embedding_model, embeddings, pages_and_chunks)
        
        # Answer using the RoBERTa QA model
        print("\nAI Response:")
        try:
            answer = answer_question_with_roberta(prompt, context)
            response_history.append(answer)
            print(f"\nResponse: {answer}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    model_name = "intfloat/multilingual-e5-large"
    embedding_model = SentenceTransformer(model_name_or_path=model_name, device="cpu")

    # Load embeddings from CSV
    text_chunks_and_embedding_df = pd.read_csv("sentence_chunks_emb.csv")
    text_chunks_and_embedding_df["embedding"] = text_chunks_and_embedding_df["embedding"].apply(eval)  # Use eval if stored as lists
    embedding_array = np.array(text_chunks_and_embedding_df["embedding"])

    # Process embeddings
    max_seq_length = max(len(seq) for seq in embedding_array)
    processed_arr = np.array([seq + [0.0] * (max_seq_length - len(seq)) for seq in embedding_array])
    embeddings = torch.tensor(processed_arr, dtype=torch.float32)

    # Get pages and chunks for context
    pages_and_chunks = text_chunks_and_embedding_df.to_dict(orient="records")

    print("To exit the chat, type 'goodbye' or 'exit'")
    response4(embedding_model, embeddings, pages_and_chunks)
