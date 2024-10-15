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
import os
from dotenv import load_dotenv


# Example usage of Groq LLM
client = Groq(
    api_key="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",
)

load_dotenv()

openai.api_key = os.getenv("APIKEY")  #4 Key

def getIndex():
  pc = Pinecone(api_key="d403ddc4-dc54-47d5-9c8f-ed19848d06ce")
  index = pc.Index("final-asha")
  return index


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


def main (query, index, context):
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
    index = getIndex()

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
    print("Should return")
    
    if done:
        return "Error"
    else:
        return finalSummary


