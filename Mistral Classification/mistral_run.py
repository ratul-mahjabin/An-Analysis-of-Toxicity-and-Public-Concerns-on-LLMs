import pandas as pd
from langchain.llms import LlamaCpp
from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load data from CSV file
data = pd.read_csv('test_toxic_dataset_without_label.csv')

# Prepare the Callback Manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Define the template for prompting the model
template = """

Determine if the following text is toxic (very bad, unpleasant, harmful, extremely bad, and offensive) or not. Consider a comment toxic if it fits the definition of being very bad, unpleasant, harmful, or extremely offensive. Please don't explain your answers.

Text: "{text}"

Is the text toxic? (Yes or No):

"""

prompt_template = PromptTemplate(
    input_variables=["text"], 
    template=template
)

# Initialize the model
llm = LlamaCpp(
    model_path="/home/kamruzzaman1/mistral-model/mistral_demo/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_gpu_layers=-1,
    callback_manager=callback_manager,
    verbose=False,
    n_ctx=2048,
    f16_kv=True,
)

# Process each row to detect toxicity
results = []
for index, row in data.iterrows():
    prompt = prompt_template.format(text=row['processed_text'])
    
    try:
        llm_response = llm(prompt)
        response = llm_response.strip().lower()
        
        # Save the actual model response
        row['actual_toxic'] = llm_response
        
        # Interpret the model's response for toxicity
        is_toxic = 1 if response.startswith('yes') else 0
        row['toxic'] = is_toxic
        results.append(row)
    except Exception as e:
        print('Exception occurred:', e)
        # Save the actual model response as 'Error' if an exception occurred
        row['actual_toxic'] = 'Error'
        row['toxic'] = 0  # Default to 0 in case of an error
        results.append(row)

# Compile results into a DataFrame
result_df = pd.DataFrame(results)

# Save the result to a new CSV file
result_df.to_csv('processed_with_labels1.csv', index=False, encoding='utf-8')


###################################################################################################################################################################################################

import pandas as pd
from langchain.llms import LlamaCpp
from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load data from CSV file
data = pd.read_csv('processed_with_labels1.csv')

# Prepare the Callback Manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Define the template for prompting the model
template = """

Determine if the following text is Severe toxic (extremely bad and offensive) or not. Consider a comment severe toxic if it fits the definition of being extremely bad and offensive. Please don't explain your answers.
Text: "{text}"

Is the text severe toxic? (Yes or No):

"""

prompt_template = PromptTemplate(
    input_variables=["text"], 
    template=template
)

# Initialize the model
llm = LlamaCpp(
    model_path="/home/kamruzzaman1/mistral-model/mistral_demo/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    #model_path="/home/kamruzzaman1/llama-model/llama_demo/llama-2-13b.Q4_K_M.gguf",
    n_gpu_layers=-1,
    callback_manager=callback_manager,
    verbose=False,
    n_ctx=2048,
    f16_kv=True,
)

# Process each row to detect toxicity
results = []
for index, row in data.iterrows():
    prompt = prompt_template.format(text=row['processed_text'])
    
    try:
        llm_response = llm(prompt)
        response = llm_response.strip().lower()
        
        # Save the actual model response
        row['actual_severe_toxic'] = llm_response
        
        # Interpret the model's response for toxicity
        is_toxic = 1 if response.startswith('yes') else 0
        row['severe_toxic'] = is_toxic
        results.append(row)
    except Exception as e:
        print('Exception occurred:', e)
        # Save the actual model response as 'Error' if an exception occurred
        row['actual_severe_toxic'] = 'Error'
        row['severe_toxic'] = 0  # Default to 0 in case of an error
        results.append(row)

# Compile results into a DataFrame
result_df = pd.DataFrame(results)

# Save the result to a new CSV file
result_df.to_csv('processed_with_labels2.csv', index=False, encoding='utf-8')


###################################################################################################################################################################################################

import pandas as pd
from langchain.llms import LlamaCpp
from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load data from CSV file
data = pd.read_csv('processed_with_labels2.csv')

# Prepare the Callback Manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Define the template for prompting the model
template = """

Determine if the following text is Obscene (of the portrayal or description of sexual matters) offensive or disgusting by accepted standards of morality and decency or not. Consider a comment obscene if it fits the definition of being of the portrayal or description of sexual matters) offensive or disgusting by accepted standards of morality and decency. Please don't explain your answers.
Text: "{text}"

Is the text obscene? (Yes or No):

"""

prompt_template = PromptTemplate(
    input_variables=["text"], 
    template=template
)

# Initialize the model
llm = LlamaCpp(
    model_path="/home/kamruzzaman1/mistral-model/mistral_demo/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    #model_path="/home/kamruzzaman1/llama-model/llama_demo/llama-2-13b.Q4_K_M.gguf",
    n_gpu_layers=-1,
    callback_manager=callback_manager,
    verbose=False,
    n_ctx=2048,
    f16_kv=True,
)

# Process each row to detect toxicity
results = []
for index, row in data.iterrows():
    prompt = prompt_template.format(text=row['processed_text'])
    
    try:
        llm_response = llm(prompt)
        response = llm_response.strip().lower()
        
        # Save the actual model response
        row['actual_obscene'] = llm_response
        
        # Interpret the model's response for toxicity
        is_toxic = 1 if response.startswith('yes') else 0
        row['obscene'] = is_toxic
        results.append(row)
    except Exception as e:
        print('Exception occurred:', e)
        # Save the actual model response as 'Error' if an exception occurred
        row['actual_obscene'] = 'Error'
        row['obscene'] = 0  # Default to 0 in case of an error
        results.append(row)

# Compile results into a DataFrame
result_df = pd.DataFrame(results)

# Save the result to a new CSV file
result_df.to_csv('processed_with_labels3.csv', index=False, encoding='utf-8')



###################################################################################################################################################################################################

import pandas as pd
from langchain.llms import LlamaCpp
from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load data from CSV file
data = pd.read_csv('processed_with_labels3.csv')

# Prepare the Callback Manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Define the template for prompting the model
template = """

Determine if the following text is Threat (a statement of an intention to inflict pain, injury, damage, or other hostile action on someone in retribution for something done or not done) or not. Consider a comment threat if it fits the definition of being a statement of an intention to inflict pain, injury, damage, or other hostile action on someone in retribution for something done or not done. Please don't explain your answers.
Text: "{text}"

Is the text threat? (Yes or No):

"""

prompt_template = PromptTemplate(
    input_variables=["text"], 
    template=template
)

# Initialize the model
llm = LlamaCpp(
    model_path="/home/kamruzzaman1/mistral-model/mistral_demo/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    #model_path="/home/kamruzzaman1/llama-model/llama_demo/llama-2-13b.Q4_K_M.gguf",
    n_gpu_layers=-1,
    callback_manager=callback_manager,
    verbose=False,
    n_ctx=2048,
    f16_kv=True,
)

# Process each row to detect toxicity
results = []
for index, row in data.iterrows():
    prompt = prompt_template.format(text=row['processed_text'])
    
    try:
        llm_response = llm(prompt)
        response = llm_response.strip().lower()
        
        # Save the actual model response
        row['actual_threat'] = llm_response
        
        # Interpret the model's response for toxicity
        is_toxic = 1 if response.startswith('yes') else 0
        row['threat'] = is_toxic
        results.append(row)
    except Exception as e:
        print('Exception occurred:', e)
        # Save the actual model response as 'Error' if an exception occurred
        row['actual_threat'] = 'Error'
        row['threat'] = 0  # Default to 0 in case of an error
        results.append(row)

# Compile results into a DataFrame
result_df = pd.DataFrame(results)

# Save the result to a new CSV file
result_df.to_csv('processed_with_labels4.csv', index=False, encoding='utf-8')


###################################################################################################################################################################################################

import pandas as pd
from langchain.llms import LlamaCpp
from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load data from CSV file
data = pd.read_csv('processed_with_labels4.csv')

# Prepare the Callback Manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Define the template for prompting the model
template = """

Determine if the following text is Insult (speak to or treat with disrespect or scornful abuse) or not. Consider a comment insult if it fits the definition of being speak to or treat with disrespect or scornful abuse. Please don't explain your answers.
Text: "{text}"

Is the text insult? (Yes or No):

"""

prompt_template = PromptTemplate(
    input_variables=["text"], 
    template=template
)

# Initialize the model
llm = LlamaCpp(
    model_path="/home/kamruzzaman1/mistral-model/mistral_demo/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    #model_path="/home/kamruzzaman1/llama-model/llama_demo/llama-2-13b.Q4_K_M.gguf",
    n_gpu_layers=-1,
    callback_manager=callback_manager,
    verbose=False,
    n_ctx=2048,
    f16_kv=True,
)

# Process each row to detect toxicity
results = []
for index, row in data.iterrows():
    prompt = prompt_template.format(text=row['processed_text'])
    
    try:
        llm_response = llm(prompt)
        response = llm_response.strip().lower()
        
        # Save the actual model response
        row['actual_insult'] = llm_response
        
        # Interpret the model's response for toxicity
        is_toxic = 1 if response.startswith('yes') else 0
        row['insult'] = is_toxic
        results.append(row)
    except Exception as e:
        print('Exception occurred:', e)
        # Save the actual model response as 'Error' if an exception occurred
        row['actual_insult'] = 'Error'
        row['insult'] = 0  # Default to 0 in case of an error
        results.append(row)

# Compile results into a DataFrame
result_df = pd.DataFrame(results)

# Save the result to a new CSV file
result_df.to_csv('processed_with_labels5.csv', index=False, encoding='utf-8')


###################################################################################################################################################################################################

import pandas as pd
from langchain.llms import LlamaCpp
from langchain import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load data from CSV file
data = pd.read_csv('processed_with_labels5.csv')

# Prepare the Callback Manager
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Define the template for prompting the model
template = """

Determine if the following text is Identity hate (hatred, hostility, or violence towards members of a race, ethnicity, nation, religion, gender, gender identity, sexual orientation or any other designated sector of society) or not. Consider a comment identity hate if it fits the definition of being hatred, hostility, or violence towards members of a race, ethnicity, nation, religion, gender, gender identity, sexual orientation or any other designated sector of society. Please don't explain your answers.
Text: "{text}"

Is the text identity hate? (Yes or No):

"""

prompt_template = PromptTemplate(
    input_variables=["text"], 
    template=template
)

# Initialize the model
llm = LlamaCpp(
    model_path="/home/kamruzzaman1/mistral-model/mistral_demo/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    #model_path="/home/kamruzzaman1/llama-model/llama_demo/llama-2-13b.Q4_K_M.gguf",
    n_gpu_layers=-1,
    callback_manager=callback_manager,
    verbose=False,
    n_ctx=2048,
    f16_kv=True,
)

# Process each row to detect toxicity
results = []
for index, row in data.iterrows():
    prompt = prompt_template.format(text=row['processed_text'])
    
    try:
        llm_response = llm(prompt)
        response = llm_response.strip().lower()
        
        # Save the actual model response
        row['actual_identity_hate'] = llm_response
        
        # Interpret the model's response for toxicity
        is_toxic = 1 if response.startswith('yes') else 0
        row['identity_hate'] = is_toxic
        results.append(row)
    except Exception as e:
        print('Exception occurred:', e)
        # Save the actual model response as 'Error' if an exception occurred
        row['actual_identity_hate'] = 'Error'
        row['identity_hate'] = 0  # Default to 0 in case of an error
        results.append(row)

# Compile results into a DataFrame
result_df = pd.DataFrame(results)

# Save the result to a new CSV file
result_df.to_csv('final_toxic_dataset.csv', index=False, encoding='utf-8')










