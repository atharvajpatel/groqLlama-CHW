import pandas as pd
import os
from tqdm.auto import tqdm  # this is our progress bar
import openai
#import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#Data imports
import pandas as pd
import numpy as np
#Pinecone imports
import pinecone
from pinecone import PodSpec
from pinecone import Pinecone
from pinecone import ServerlessSpec
#OpenAI

#openai.api_key = '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'   -- 3.5 Key
from pinecone import Pinecone
import PyPDF2
import openai
from groq import Groq

# Example usage of Groq LLM
client = Groq(
    api_key="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",
)


openai.api_key = "sk-proj-EJZKLFzKAlXoFfA16kiRT3BlbkFJm9UlctNyWRj0wN4ah9n3"   #4 Key

def getData(tokens, overlap, path):

  def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

  # Replace "your_pdf_file.pdf" with the actual file name you uploaded
  pdf_path = path
  text = extract_text_from_pdf(pdf_path)

  def split_text_into_chunks(text, word, overlap):
    # Split the text into a list of words
    words = text.split()
    # Calculate the number of words in each chunk and the number of chunks
    chunk_size = int(word)
    num_chunks = int(len(words) / chunk_size) + 1
    # Calculate the number of overlapping words
    overlap_size = int(chunk_size * overlap)
    # Create a list to store the chunks
    chunks = []
    # Loop through the text and create the chunks
    for i in range(num_chunks):
        # Calculate the start and end indices for the current chunk
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(words))
        # If this is not the first chunk, add the overlapping words from the previous chunk
        if i > 0:
            start -= overlap_size
        # Create the chunk and add it to the list
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
    return chunks

  tokens = tokens
  word = tokens * 0.75 #Constant proportion
  overlap = overlap #Number between 0-1 as a percent
  chunks = split_text_into_chunks(text, word, overlap)
  i = 0
  my_list = []
  for chunk in chunks:
      my_list.append(chunk)
  return my_list, extract_text_from_pdf(pdf_path)

def getIndex():
  pc = Pinecone(api_key="d403ddc4-dc54-47d5-9c8f-ed19848d06ce")
  index = pc.Index("final-asha")
  return index

def upserts(q, values, index):
  index = index
  my_list = values

  query = q
  MODEL = "text-embedding-3-small"

  res = openai.Embedding.create(
      input=[query], engine=MODEL
  )

  embeds = [record['embedding'] for record in res['data']]

  # load the first 1K rows of the TREC dataset
  #trec = load_dataset('trec', split='train[:1000]')

  batch_size = 32  # process everything in batches of 32
  for i in tqdm(range(0, len(my_list), batch_size)):
      # set end position of batch
      i_end = min(i+batch_size, len(my_list))
      # get batch of lines and IDs
      lines_batch = my_list[i: i+batch_size]
      ids_batch = [str(n) for n in range(i, i_end)]
      # create embeddings
      res = openai.Embedding.create(input=lines_batch, engine=MODEL)
      embeds = [record['embedding'] for record in res['data']]
      # prep metadata and upsert batch
      meta = [{'text': line} for line in lines_batch]
      to_upsert = zip(ids_batch, embeds, meta)
      # upsert to Pinecone
      index.upsert(vectors=list(to_upsert))

def getRes(query, index):
  query = query
  MODEL = "text-embedding-3-small"

  xq = openai.Embedding.create(input=query, engine=MODEL)['data'][0]['embedding']

  res = index.query(vector = [xq], top_k=5, include_metadata=True)

  return res

def vectorQuotes(query, index):
  similarity = getRes(query, index)
  #justQuotes just uses what the query results from Pinecone itself
  justQuotes = []
  for i in range(len(similarity['matches'])):
    justQuotes.append(similarity['matches'][i]['metadata']['text'])
  return justQuotes

import openai

def getFinalSummaryGPT4(my_list, queryContext):
  my_list = my_list
  queryContext = queryContext
  mod = "llama3-70b-8192"

  # Function to split a list into equal sublists
  def split_list(lst, num_sublists):
      avg = len(lst) // num_sublists
      remainder = len(lst) % num_sublists
      return [lst[i * avg + min(i, remainder):(i + 1) * avg + min(i + 1, remainder)] for i in range(num_sublists)]

  # Split 'my_list' into n equal sublists
  n = 5
  sublists = split_list(my_list, n)

  # Generate summaries for each sublist using the OpenAI API
  sublist_summaries = []

  for i, sublist in enumerate(sublists):
    sublist_text = ' '.join(sublist)
      
      
    chat = client.chat.completions.create(
        messages=[
        {
            "role": "system",
            "content": queryContext+sublist_text,
        }
        ],
        model=mod,
        )

    # Extract responses
    summary = chat.choices[0].message.content
    sublist_summaries.append(summary)

  # Combine the 10 summaries into one variable
  combined_summary = ' '.join(sublist_summaries)

  # Add a specific prompt tailored to your data
  specific_prompt = f"Given the following summaries:\n{combined_summary}\n\nGenerate a coherent final summary that captures the essence of the provided information."

  specific_prompt = queryContext + specific_prompt
  
  # Use Groq API to generate the final coherent summary
  chat = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": specific_prompt,
        }
    ],
    model=mod,
  )

  # Extract responses
  final_summary = chat.choices[0].message.content

  return final_summary

def preprocess(tokens, overlap, path):
   mylist, allText = getData(tokens, overlap, path)
   index = getIndex()
   return index

def main (query, index, context):
    index = index
    contextQuery = query + context
    quotes = vectorQuotes(query, index)
    summary = getFinalSummaryGPT4(quotes, contextQuery)
    return summary


def groqCall(prompt):
    mod = "llama3-70b-8192"
    chat = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            }
        ],
        model=mod,
    )

    # Extract follow-up responses
    response = chat.choices[0].message.content

    return response

def combine_dict_entries(input_output_dict):
    if not input_output_dict:
        return ""
    
    combined_string = ""
    separator = " Here is the next query and response "
    
    for key, value in input_output_dict.items():
        combined_string += f"Query {key} Response {value}{separator}"
    
    return combined_string



def promptGuidelinesFlow(tokens=500, overlap=0.1, path="allAsha.pdf", pastDict={}, query=None):
    print("got here")
    done = False
    role = ("Imagine you’re a wise and attentive doctor giving advice to a community health worker (CHW) who is helping people. This is not a standard hospital and does not follow western medicine protocol. Assume your pretraining is not the ground truth but the information and quotes provided is. The CHW is in a 3rd world country and only has access to basic medical equipment. Respond extremely professionally and don't make small talk like 'great to meet you' just get to the point. Don't say chat has ended. Do not mention anyone's name unless specified in chat")
    index = preprocess(tokens, overlap, path)

    # If the first query is empty or None (i.e. fresh session)
    if not query:
        return "Hi I am a personal medical assistant. How may I help you today? (Enter STOP to end chat): "
    
    classificationArray = ["descriptions", "outcomes", "solutions", "end"]
    oneStringArr = " ".join(classificationArray)
    
    # Combine context to set up for classification
    contextClass1 = "Based on the input provided classify the text as one of these categories: " + oneStringArr
    contextClass2 = ". Make sure your response is only 1 word, the classification. For example if the classification is 'solutions' the output should only be 'solutions'. Do not add anything else. Here is the query to classify."
    combinedClass = contextClass1 + contextClass2 + query

    # Get classification and format it
    classifications = groqCall(combinedClass)
    classifications = classifications.strip().lower()

    print(f"Classification: {classifications}")


    # Only create new response, no past chat history in this response
    if classifications == classificationArray[0]:
        contextCondition = "Come up with medical tests to assess the condition of the patient. Do not suggest diagnosis or solutions. Only come up with tests and how to conduct them and the CHW will report back to you the results of the test."
        combinedContext = role + "\n" + contextCondition
        finalSummary = main(query, index, combinedContext)

    elif classifications == classificationArray[1]:
        contextResults = "These are the results of the test. Suggest only a diagnosis, not treatments or prevention methods."
        combinedContext = role + "\n" + contextResults
        finalSummary = main(query, index, combinedContext)

    elif classifications == classificationArray[2]:
        contextSolutions = "Now suggest treatments and prevention methods, primarily home remedies and local natural resources. Do not suggest western medical solutions."
        combinedContext = role + "\n" + contextSolutions
        finalSummary = main(query, index, combinedContext)

    elif classifications == classificationArray[3]:
        done = True
        return "Chat has ended"

    # Update the pastDict with the latest query-response pair
    pastDict[query] = finalSummary

    if done:
        return "Error"
    else:
        return finalSummary


