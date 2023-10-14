# import sys
# sys.path.insert(1,'D:\\Dev\\onlinexamination\\teacher')
# from views import pdf_path

# from transformers import pipeline
# qa_model = pipeline("question-answering")

# import wandb
# wandb.Api="ee0fce62d4003d1c0b6c5ed1458fe690dedb22bf"
# from huggingface_hub.hf_api import HfFolder
import os
import torch

import random

# from datasets import load_dataset, load_metric, list_metrics

# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,T5Tokenizer,Trainer, DataCollator,EvalPrediction,TrainingArguments, T5ForConditionalGeneration, T5TokenizerFast

# from tqdm import tqdm

# from typing import Dict, List, Optional

# import dataclasses
from dataclasses import dataclass, field

import logging
import os
import sys
import pdfplumber
import re
import numpy as np

# from huggingface_hub import notebook_login
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import json

with open('data.json', 'r') as file:
    serialized_data = file.read()
    data = json.loads(serialized_data)

# print('data >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>',data['pdf_path'])

def process_sentences(sentences_list):
    processed_sentences = []

    for sentence in sentences_list:
        # Split the sentence into words
        words = sentence.split(" ")

        # Initialize lists to store positions
        nlist = []
        position_colonlist = []

        # Iterate through the words
        for i, word in enumerate(words):
            if ":" in word:
                position_colonlist.append(i)
            if "\n" in word:
                nlist.append(i)

        pairs = []

        while position_colonlist:
            min_pos = min(position_colonlist)
            max_n = min(nlist, default=None)  # Initialize max_n with a default value

            if max_n is not None and min_pos - max_n > 5:
                pairs.append((max_n, min_pos))

            position_colonlist.remove(min_pos)
            #  \n adffsad :
            nlist = [n for n in nlist if n >= min_pos]
            if len(nlist) == 0:
                break

        if not pairs:
            # If pairs are empty, return the original sentence
            processed_sentence = " ".join(words)
        else:
            # If pairs are not empty, modify the sentence
            modified_words = []

            for pair in pairs:
                if pair:
                    # Use slicing to extract the words outside the specified range
                    #words_to_modify = words[:pair[0]] + words[pair[1] + 1:]
                    words_to_modify = words[pair[1]:]

                    modified_words.extend(words_to_modify)

            # Set the processed sentence to the modified version
            processed_sentence = " ".join(modified_words)

        processed_sentences.append(processed_sentence)

    return processed_sentences

def extract_text_after_newline(input_text):
    d = ""
    storeof = False
    for y in input_text:
        if not storeof and y == "\n":
            storeof = True
        else:
            d += y
    return d

def extract_substring(input_text):
    # Find the index of "concepts:\n"
    substring_start = input_text.find("\n")
    # Check if "concepts:\n" was found
    if substring_start != -1:
        # Extract the substring from "concepts:\n" to the end
        extracted_substring = input_text[substring_start:]
        return extracted_substring
      # Return None if the substring is not found

def extract_text_after_colons(text):
    # Split the text into words using space as a delimiter
     # : "my name is anas asdsas :"
    "follow example we have  :data security: "
    words = text.split(" ")

    # Initialize variables to keep track of colons and their distances
    first_colon_index = -1
    second_colon_index = -1

    # Find the first colon and its index (considering an uppercase letter before it)
    for i, word in enumerate(words):
        if ":" in word or (word and word[0].isupper() and ":" in word[1:]):
            first_colon_index = i
            break

    # Find the second colon and its index
    for i, word in enumerate(words[first_colon_index + 1:], start=first_colon_index + 1):
        if ":" in word:
            second_colon_index = i
            break

    # Calculate the distance between the first and second colons
    distance = second_colon_index - first_colon_index

    # Check if the colons are within the specified distance and remove everything before them
    if 4 < distance <= 8:
    # 4 < distance <= 9:
        # Find the index of the last occurrence of "\n" before the first colon
        #newline_index = max([i for i, word in enumerate(words[:first_colon_index]) if "\n" in word], default=-1)

        # Include the text from the first colon to the "\n" before it
        d=""
        #for i,y in enumerate(words[second_colon_index+1]):
          #if y=="\n":
            #d=words[second_colon_index+1][y+1:-1]
            #print(d)
        d = extract_text_after_newline(words[second_colon_index])
        result_words = ""
        for i in words[second_colon_index:]:
          result_words += " " + i
        result_words=extract_substring(result_words)

        return result_words
    else:
        return text

function_words = [
    "is", "the", "are", "an", "a","in","to","of","and"
]
def clean_cluster_data(cluster_data):
    cleaned_data = {}  # Initialize a new dictionary for cleaned data

    # Iterate through pages and clean key-value pairs
    for page_num, page_data in cluster_data.items():
        cleaned_page_data = {}
        for key, value in page_data.items():
            cleaned_key = key.strip('\n■ ')
            cleaned_value = value.replace("■","").replace("\n"," ")
            #cleaned_value=cleaned_value.replace("")
            cleaned_page_data[cleaned_key] = cleaned_value
        cleaned_data[page_num] = cleaned_page_data

    return cleaned_data

# Sample input dictionary


# Clean the dictionary using the function

allextract_key_values={}
clustringbypage={}
textdict={}
key_value={}
# Function to extract unique sentences containing function words
def filter_dict_by_page(clustringbypage):
    """
    Filters the dictionary based on the specified logic.

    Args:
    clustringbypage (dict): Dictionary with page numbers as keys and key-value pairs as values.

    Returns:
    dict: Filtered dictionary with keys not starting with '\n' followed by a lowercase letter and non-empty keys.
    """
    return {key: value for key, value in clustringbypage.items() if key and not (str(key).startswith('\n') and len(str(key)) > 1 and str(key)[1].islower())}
def extract_keys_values(sentences):
    testdict = {}

    for sentence in sentences:
      if type(sentence)==str:
        newline_index = sentence.find("\n")
        modifysentences = sentence[newline_index:]
        colon_index = modifysentences.find(":")

        # Check if colon ':' was found in the sentence
        if colon_index == -1:
            continue  # Skip this sentence if ':' is not found

        key = modifysentences[:colon_index]

        # Split the key into words
        key_words = key.split(" ")

        # Check if the key has more than 5 words
        #if len(key_words) > 5:
            #continue  # Skip this key if it has more than 5 words

        value = modifysentences[colon_index + 1:]
        testdict[key] = value

    return testdict

def extract_key_values(input_strings):
    result_list = []

    for input_string in input_strings:
        # Split the text into lines
      if type(input_string)==str:
        lines = input_string.split('\n')

        # Initialize variables to store key and value
        key = ""
        value = ""

        # Loop through each line to find the key and value
        for line in lines:
            if ':' in line:
                parts = line.split(':')
                key = parts[0].strip()
                value = parts[1].strip()

        # If a key-value pair is found, create and append a dictionary to the result list
        if key:
            result = {key: value}
            result_list.append(result)

    return result_list

def extract_unique_sentences_with_function_words(text, function_words):
    # Split the text into sentences using the specified regular expression
    #sentences = re.split(r'(?<=[a-zA-Z0-9])\.(?=\s)', text)
    sentences = re.split(r'(?<=[a-zA-Z0-9\)])\.(?=\s)', text)

    # Initialize a set to store unique sentences containing function words
    relevant_sentences = set()

    # Iterate through each sentence and check for function words
    for sentence in sentences:
        # Convert the sentence to lowercase for case-insensitive matching
        lowercase_sentence = sentence.lower()
        words = lowercase_sentence.split(" ")

        # Check if any function word is present in the sentence
        if any(word in words for word in function_words):
            relevant_sentences.add(sentence)

    return list(relevant_sentences)


#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
pdf_file_path = data['pdf_path']

#################################################################################################
#################################################################################################
#################################################################################################
with pdfplumber.open(pdf_file_path) as pdf:
    page_relevant_sentences = {}
    filtered_sentencesdict={}
    for page_number, page in enumerate(pdf.pages, start=1):
        key_value={}
        # Extract text from the current page
        page_text = page.extract_text()
        # print(page_text)
        page_text=page_text.replace("-\n", "")
        #output = query({
	#"inputs": page_text,
 #})
        #page_text=extract_text_after_colons(page_text)
        textdict[page_number]=page.extract_text()
        # Extract unique sentences containing function words
        relevant_sentences = extract_unique_sentences_with_function_words(page_text, function_words)

        # Filter the relevant sentences to include only those with at least one word in uppercase
        #sentences_with_uppercase = [i for i in relevant_sentences if any(word and word[0].isupper() for word in i.split(" "))]
        # Filter sentences to include only those with no more than 35 words
        sentences_with_uppercase = [
             sentence for sentence in relevant_sentences if any((word and word[0].isupper()) or (word.startswith('\n') and len(word) > 1 and word[1].isupper())
             for word in sentence.split(" ")
          )
        ]
        filtered_sentences = [sentence for sentence in sentences_with_uppercase if len(sentence.split()) <= 60]

        # Store the filtered sentences in the dictionary
        filtered_sentences=process_sentences(filtered_sentences)
        for i,y in enumerate(filtered_sentences):
          filtered_sentences[i]=extract_text_after_colons(filtered_sentences[i])
        filtered_sentencesdict[page_number]=filtered_sentences
        #for i,y in enumerate(filtered_sentences):
          #filtered_sentences[i]=extract_text_after_colons(filtered_sentences[i])
        key_value= extract_keys_values(filtered_sentences)
        if len(key_value)>0:

         #for item in key_value:
               #all_extract_key_values.update(item)

         clustringbypage[page_number]=key_value
         clustringbypage[page_number] =filter_dict_by_page(clustringbypage[page_number])
         clustringbypage[page_number] = {key: value for key, value in clustringbypage[page_number].items() if key != '' and key is not None}
         clustringbypage=clean_cluster_data(clustringbypage)

         page_relevant_sentences[page_number] = filtered_sentences

        

# After the loop, you will have a dictionary 'page_relevant_sentences' containing filtered sentences for each page in the PDF.

cryptographic_terms={}

for i in clustringbypage:
  for j in clustringbypage[i]:
    cryptographic_terms[j]=clustringbypage[i][j]

sd=dict()
# Extract the definitions from the cryptographic_terms dictionary
definitions = list(cryptographic_terms.values())

preprocessed_definitions = []
for definition in definitions:
    tokens = nltk.word_tokenize(definition.lower())  # Tokenize and convert to lowercase
    tokens = [stemmer.stem(word) for word in tokens if word.isalpha()]  # Apply stemming
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    preprocessed_definitions.append(" ".join(tokens))
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_definitions)

# Create the TF-IDF vectorizer and fit it on the definitions
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(definitions)

# Calculate the cosine similarity between each term and every other term
cosine_similarities = cosine_similarity(tfidf_matrix)

# Get the terms from the cryptographic_terms dictionary
terms = list(cryptographic_terms.keys())

# Print the most similar terms for each term
for i, term in enumerate(terms):
    # Get the cosine similarity scores for the current term
    similarity_scores = cosine_similarities[i]

    # Sort the scores in descending order and get the indices of the top 3 terms
    top_3_indices = similarity_scores.argsort()[::-1][1:10]

    # Print the current term and its top 3 most similar terms
    similar_terms = [terms[j] for j in top_3_indices]
    #print(f"{term}: {similar_terms}")
    sd[term]=similar_terms
# Define the input dictionary

# Iterate through the dictionary and modify the value lists
for key, value_list in sd.items():
    # Create a copy of the original value list
    #print(key,value_list)
    # Iterate through the original values and remove items if they match keys
    for item in value_list:
      if item==key:
        value_list.remove(item)

multichoicequestiondict={}
questionsdicts = []  # Create a list to store dictionaries of questions
answerlist=[]
questionlist=[]
choiceslist=[]
for i in sd:
    choices = []


    #reate a dictionary to represent a question
    question_dict = {}
    question_dict["answer"] = i
    answerlist.append(i)
    choices.append(i)
    #definition_extraction = cryptographic_terms.get(i)


    fill_in_the_gap = cryptographic_terms.get(i)
    question_dict["question"] = "...... " + str(fill_in_the_gap)
    d="...... " + str(fill_in_the_gap)
    questionlist.append(d)
    for j in sd.get(i):
        choices.append(j)
        # print(j)
        if len(choices) == 4:
            break

    question_dict["multichoice"] = choices
    shuffled_list=choices.copy()
    random.shuffle(shuffled_list)
    choiceslist.append(shuffled_list)
    # Append the question_dict to the questionsdicts list
multichoicequestiondict={"question":questionlist,"multichoice":choiceslist,"answer":answerlist}
# sd = {term.capitalize(): value for term, value in sd.items()}
#####################################################3
#################################################### this lines below for ranked lists

choicesrank2 = [[key, value[0], value[1], value[3]] for key, value in sd.items()]
choicesrank3 = [[key, value[1], value[3], value[4]] for key, value in sd.items()]
choicesrank4 = [[key, value[3], value[4], value[5]] for key, value in sd.items()]

import random

multichoicequestiondict={}
multichoicequestiondict1={}
multichoicequestiondict2={}
multichoicequestiondict3={}
questionsdicts = []  # Create a list to store dictionaries of questions
answerlist=[]
questionlist=[]
choiceslist=[]
choiceslist2=[]
choiceslist3=[]
choiceslist4=[]
for i in sd:
    choices = []


    #reate a dictionary to represent a question
    question_dict = {}
    question_dict["answer"] = i
    answerlist.append(i)
    choices.append(i)
    #definition_extraction = cryptographic_terms.get(i)


    fill_in_the_gap = cryptographic_terms.get(i)
    question_dict["question"] = "...... " + str(fill_in_the_gap)
    d="...... " + str(fill_in_the_gap)
    questionlist.append(d)
    for j in sd.get(i):
        choices.append(j)
        # print(j)
        if len(choices) == 4:
            break

    question_dict["multichoice"] = choices
    shuffled_list=choices.copy()
    random.shuffle(shuffled_list)
    choiceslist.append(shuffled_list)
    # Append the question_dict to the questionsdicts list
def shuffle_and_append(original_list, target_list):
    for item in original_list:
        item_copy = item.copy()
        random.shuffle(item_copy)
        target_list.append(item_copy)

# New lists to store shuffled items
choiceslist2 = []
choiceslist3 = []
choiceslist4 = []

# Shuffle and append for choicesrank2
shuffle_and_append(choicesrank2, choiceslist2)

# Shuffle and append for choicesrank3
shuffle_and_append(choicesrank3, choiceslist3)

# Shuffle and append for choicesrank4
shuffle_and_append(choicesrank4, choiceslist4)

multichoicequestiondict={"question":questionlist,"multichoice":choiceslist,"answer":answerlist}

multichoicequestiondict1={"question":questionlist,"multichoice":choiceslist2,"answer":answerlist}
multichoicequestiondict2={"question":questionlist,"multichoice":choiceslist3,"answer":answerlist}
multichoicequestiondict3={"question":questionlist,"multichoice":choiceslist4,"answer":answerlist}



# print(''*15)
# print('multichoicequestiondict1>>>>>>>>>',multichoicequestiondict['question'][:5])

answerdict=[]
multichoicelist=[]
questionsdicts = {}
questionslist=[]
for term, description in cryptographic_terms.items():
    multichoicelist.append([True,False])
    # Generate a random number to determine if the statement is true or false
    is_true = random.choice([True, False])
    answerdict.append(is_true)
    # Create the statement based on whether it's true or false
    if is_true:
        # Randomly pick one of the similar terms and use its description
        similar_term = random.choice(sd.get(term, []))
        statement = f"{term} is {cryptographic_terms.get(similar_term, '')}."
        questionslist.append(statement)
    else:
        # Randomly pick a different term that is not similar
        non_similar_terms = [t for t in cryptographic_terms if t != term and t not in sd.get(term, [])]
        if non_similar_terms:
            false_term = random.choice(non_similar_terms)
        else:
            # Handle the case where there are no non-similar terms
            false_term = term
        statement = f"{term} is {cryptographic_terms.get(false_term, '')}."
        questionslist.append(statement)

    


    # Append the question dictionary to the list
    questionsdicts.update(question_dict)

d={"question":questionslist,"answer":answerdict,"multichoice":multichoicelist}


### end of ranked lists code
# Commented out IPython magic to ensure Python compatibility.
# %env WANDB_PROJECT=t5-end-to-end-questions-generation
##############
########################################################
##########################################################################################
##########################################################################################
####################################################################################################################################################################################
# wandb.Api="ee0fce62d4003d1c0b6c5ed1458fe690dedb22bf"

# HfFolder.save_token('$hf_api')
# os.environ['WANDB_PROJECT'] = 't5-end-to-end-questions-generation'
# # !pip install huggingface_hub
# hf_api = "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"
# # !python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('$hf_api')"

# raw_dataset = load_dataset("squad_modified_for_t5_qg.py")

# checkpoint = "t5-base"
# model = T5ForConditionalGeneration.from_pretrained(checkpoint)
# tokenizer = T5TokenizerFast.from_pretrained(checkpoint)

# tokenizer.add_tokens(['<sep>'])
# model.resize_token_embeddings(len(tokenizer))

# max_input_length =  512
# max_target_length = 64

# # tokenize the examples
# def convert_to_features(example_batch):

#     input_encodings = tokenizer.batch_encode_plus(example_batch['context'],
#                                                   max_length=max_input_length,
#                                                   add_special_tokens=True,
#                                                   truncation=True,
#                                                   pad_to_max_length=True)

#     target_encodings = tokenizer.batch_encode_plus(example_batch['questions'],
#                                                    max_length=max_target_length,
#                                                    add_special_tokens=True,
#                                                    truncation=True, pad_to_max_length=True)

#     encodings = {
#         'input_ids': input_encodings['input_ids'],
#         'attention_mask': input_encodings['attention_mask'],
#         'decoder_input_ids': target_encodings['input_ids']
#         ,'decoder_attention_mask': target_encodings['attention_mask']
#     }

#     return encodings

# def add_eos_examples(example):
#   example['context'] = example['context'] + " </s>"
#   example['questions'] = example['questions'] + " </s>"
#   return example


# def add_special_tokens(example):
#   example['questions'] = example['questions'].replace("{sep_token}", '<sep>')
#   return example

# tokenized_dataset  = raw_dataset.map(add_eos_examples)
# tokenized_dataset = tokenized_dataset.map(add_special_tokens)
# tokenized_dataset  = tokenized_dataset.map(convert_to_features,  batched=True)

# tokenized_dataset = tokenized_dataset.remove_columns(
#     ["context", "questions"]
# )

# train_dataset = tokenized_dataset["train"]
# valid_dataset = tokenized_dataset["validation"]

# columns = ['input_ids', 'decoder_input_ids', 'attention_mask', 'decoder_attention_mask']
# train_dataset.set_format(type='torch', columns=columns)
# valid_dataset.set_format(type='torch', columns=columns)

# torch.save(train_dataset, 'train_data.pt')
# torch.save(valid_dataset, 'valid_data.pt')

# class T2TDataCollator():
#   def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
    

#     input_ids = torch.stack([example['input_ids'] for example in batch])
#     lm_labels = torch.stack([example['decoder_input_ids'] for example in batch])
#     lm_labels[lm_labels[:, :] == 0] = -100
#     attention_mask = torch.stack([example['attention_mask'] for example in batch])
#     decoder_attention_mask = torch.stack([example['decoder_attention_mask'] for example in batch])

#     return {
#         'input_ids': input_ids,
#         'attention_mask': attention_mask,
#         'labels': lm_labels,
#         'decoder_attention_mask': decoder_attention_mask
#     }

# # from transformers import T5ForConditionalGeneration, T5TokenizerFast

# hfmodel = T5ForConditionalGeneration.from_pretrained("ThomasSimonini/t5-end2end-question-generation")

# def hf_run_model(input_string, **generator_args):
#   generator_args = {
#   "max_length": 256,
#   "num_beams": 4,
#   "length_penalty": 1.5,
#   "no_repeat_ngram_size": 3,
#   "early_stopping": True,
#   }
#   input_string = "generate questions: " + input_string + " </s>"
#   input_ids = tokenizer.encode(input_string, return_tensors="pt")
#   res = hfmodel.generate(input_ids, **generator_args)
#   output = tokenizer.batch_decode(res, skip_special_tokens=True)
#   output = [item.split("<sep>") for item in output]
#   return output

# # Assuming you have flat_questions and listofanswers for each term in the loop
# listofquestions=[]
# questionenumerator={}
# descrtivemultichoicequestiondict={}
# for i, y in enumerate(cryptographic_terms):
#     listofanswers=[]
#     modifyquestion = []
#     questions = hf_run_model(y + ":" + cryptographic_terms.get(y))
#     flat_questions = [q for sublist in questions for q in sublist if q]
#     listofquestions.append(flat_questions)
#     context = y + ":" + cryptographic_terms.get(y)

#     for j in flat_questions:
#         d = str(j).replace("?", "")  # Corrected line to remove question marks

#         answer = qa_model(question=d, context=context)
#         listofanswers.append(answer)

#     # Store questions and answers in dictionaries
#     questionenumerator[i] = {'questions': flat_questions, 'answers': listofanswers}
#     descrtivemultichoicequestiondict.update(questionenumerator)

# newdictexample={}
# for key, value in descrtivemultichoicequestiondict.items():
#     question_answer_pairs = {}
#     for i, question in enumerate(value['questions']):
#         answer = value['answers'][i]['answer']
#         question_answer_pairs[question] = answer
#     newdictexample[key] = dict(question_answer_pairs)

# questionlist = []
# answerlist = []
# multichoicelist = []

# # Iterate through the newdictexample
# for key, value in newdictexample.items():
#     questions = list(value.keys())  # Extract the questions

#     for question in questions:
#         answer = value[question]

#         # Check if the answer is a keyword in the 'sd' dictionary
#         if answer in sd:
#             # Create a list of choices containing the keyword and top 3 similar words
#             choices = [answer] + sd[answer][:3]
#             random.shuffle(choices)  # Shuffle the choices randomly

#             # Append values to the respective lists
#             questionlist.append(question)
#             answerlist.append(answer)
#             multichoicelist.append(choices)

# # Create a dictionary with keys and their corresponding lists
# result_dict = {
#     "question": questionlist,
#     "answer": answerlist,
#     "multichoice": multichoicelist
# }
