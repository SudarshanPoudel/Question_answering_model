'''
This scripts loads question answering model from hugging face and it's function getAnswer returns answer 
text and score (0-1) that tells how confident model is for that answer.
'''

# Import libraries
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")


# Main function that returns answer given a context
def getAnswer(question:str, context:str)-> str: 
    # Tokenize the input
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")

    # Get the input IDs and attention mask
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Get the start and end scores for the answer
    outputs = model(input_ids, attention_mask=attention_mask)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Apply softmax to get probabilities
    start_probs = F.softmax(start_scores, dim=1).tolist()[0]
    end_probs = F.softmax(end_scores, dim=1).tolist()[0]

    # Get the most likely beginning and end of the answer span
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    # Convert token ids to string
    tokens = input_ids[0][start_index : end_index + 1]
    answer = tokenizer.decode(tokens, skip_special_tokens=True)

    # Certainty of answer
    prob_score = (start_probs[start_index] + end_probs[end_index]) / 2

    return {
        'answer': answer,
        'score' : prob_score
    }