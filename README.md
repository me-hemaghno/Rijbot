# Rijbot
#A baby chatbot developed using nlp
nlp_library_choice = "Hugging Face Transformers"
print(f"Selected NLP library/framework: {nlp_library_choice}")
print("Reasoning:")
print("- Access to state-of-the-art transformer models.")
print("- Excellent capabilities for language understanding and generation.")
print("- Suitable for building sophisticated and intelligent chatbots.")
import spacy
import pandas as pd

# Load English tokenizer, tagger, parser, NER and word vectors
# Use a smaller model for efficiency if a large one is not required
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Sample conversational data (replace with your actual data loading)
data = [
    {"text": "Hello, how are you today?"},
    {"text": "I am doing well, thank you for asking."},
    {"text": "That is good to hear. What have you been up to?"},
    {"text": "Not much, just working on some projects."},
    {"text": "Interesting. Tell me more about them."}, # Added a comma here
    {"text": "Tell me a joke."}, # Added this entry and a comma
    {"text": "Why don't scientists trust atoms? Because they make up everything."}, # Added this entry and a comma
    {"text": "What is the capital of France?"}, # Added this entry and a comma
    {"text": "The capital of France is Paris."}, # Added this entry and a comma
    {"text": "What is the largest planet in our solar system?"}, # Added this entry and a comma
    {"text": "Jupiter is the largest planet in our solar system."}, # Added this entry and a comma
    {"text": "What is the square root of 144?"}, # Added this entry and a comma
    {"text": "The square root of 144 is 12."}, # Added this entry and a comma
    {"text": "Where is Burj Khalifa ?"}, # Added this entry and a comma
    {"text": "Burj Khalifa is in Dubai"}, # Added this entry and a comma
    {"text": "Tell me about yourself ?"}, # Added this entry and a comma
    {"text": "Hi I am Rij the chatbot who is baby in the field but a chatbot in need, a chatbot indeed !"}, # Added this entry and a comma
    {"text": "What is your Birthday?"}, # Added this entry and a comma
    {"text": "Let me check my birthcertificate"} # Added this entry
]


preprocessed_data = []

for entry in data:
    text = entry["text"]
    doc = nlp(text)
    # Tokenization, Lowercasing, Removing stop words and punctuation using spaCy
    filtered_tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    preprocessed_data.append({"original_text": text, "preprocessed_tokens": filtered_tokens})

# Display the preprocessed data
preprocessed_df = pd.DataFrame(preprocessed_data)
display(preprocessed_df)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer("This is an example sentence.", padding=True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")
inputs = tokenizer("This is an example sentence.", padding=True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForCausalLM.from_pretrained("bert-base-uncased")
inputs = tokenizer("This is an example sentence.", padding=True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
outputs = model(**inputs) # Where 'inputs' is the dictionary returned by tokenizer()
# Assuming input_ids and attention_mask are already prepared
outputs = model(input_ids=input_ids, attention_mask=attention_mask)
# Load DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-small" # Using the small version for faster training on limited data
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add a padding token to the tokenizer and resize the model's token embeddings
# This is often necessary for batching in training
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

print(f"Model and tokenizer loaded: {model_name}")
# Concatenate the preprocessed tokens into a single string for encoding
# For a simple example, treat the sequence of sentences as one continuous conversation
# In a real scenario, you would structure pairs of turns (user input, model response)
# and potentially add special tokens to delineate turns.

# For this small example, we'll create a simple sequence of the original texts.
# A more robust approach for conversational fine-tuning would involve
# formatting the data into turn-by-turn examples with appropriate separators.
# However, given the limited data and for demonstration, we'll use the original text sequence.

conversation_text = " ".join(preprocessed_df['original_text'].tolist())

# Encode the conversation text
# The max_length and truncation/padding strategy should be chosen based on the model's
# context window and the nature of the data.
encoded_conversation = tokenizer.encode(conversation_text, return_tensors="pt", padding=True, truncation=True, max_length=512)

print("Conversation text prepared and encoded.")
print("Shape of encoded conversation tensor:", encoded_conversation.shape)
from torch.optim import AdamW

# Define training parameters
epochs = 3
learning_rate = 5e-5

# Define optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training loop
model.train() # Set the model to training mode

for epoch in range(epochs):
    outputs = model(encoded_conversation, labels=encoded_conversation)
    loss = outputs.loss

    loss.backward() # Backpropagate the error
    optimizer.step() # Update model parameters
    optimizer.zero_grad() # Clear gradients

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("Fine-tuning complete.")
# Define output directory
output_dir = "./fine_tuned_dialogpt"

# Save the fine-tuned model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Fine-tuned model and tokenizer saved to {output_dir}")
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Load the fine-tuned model and tokenizer from the saved directory.
output_dir = "./fine_tuned_dialogpt"
loaded_model = AutoModelForCausalLM.from_pretrained(output_dir)
loaded_tokenizer = AutoTokenizer.from_pretrained(output_dir)

print("Fine-tuned model and tokenizer loaded.")

# 2. Define a function that takes user input as a string.
# 3. Inside the function, encode the user input using the loaded tokenizer.
# 4. Use the loaded model to generate a response based on the encoded input.
# 5. Decode the generated response token IDs back into a human-readable string using the tokenizer.
# 6. Return the decoded response.
def chatbot_response(user_input, chat_history_ids=None):
    """Generates a chatbot response based on user input."""
    # Encode the new user input
    new_input_ids = loaded_tokenizer.encode(user_input + loaded_tokenizer.eos_token, return_tensors='pt')

    # Append the new input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # Generate a response
    # `max_length` should be appropriate for the conversation length.
    # `pad_token_id` is set to the tokenizer's eos_token_id for DialoGPT
    chat_history_ids = loaded_model.generate(
        bot_input_ids, max_length=1000,
        pad_token_id=loaded_tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True,
        top_k=100,
        top_p=0.7,
        temperature=0.8
    )

    # Decode the generated response, excluding the input
    response = loaded_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return response, chat_history_ids

# 7. Implement a simple loop to allow for continuous conversation with the chatbot
print("Start chatting with the bot (type 'quit' to exit):")
chat_history_ids = None

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break

    # Incorporate preprocessed_df data into the user input for the next turn
    if user_input == "Please use preprocessed_df in my query in the next sentence. Burj Khalifa is in Dubai":
      user_input = "Burj Khalifa is in Dubai" + " " + " ".join(preprocessed_df['original_text'].tolist())

    response, chat_history_ids = chatbot_response(user_input, chat_history_ids)
    print(f"Bot: {response}")

print("Chat session ended.")
# The interactive loop for testing the chatbot is already implemented in the previous cell.
# Execute the previous cell to interact with the chatbot and perform manual evaluation.

# Example of manual evaluation points during the interaction:
# - Provide input: "Hello" - Expected: A greeting or similar response.
# - Provide input: "What have you been up to?" (related to training data) - Expected: A response related to the training conversation.
# - Provide input: "Tell me about the capital of France." (outside training data) - Expected: A potentially less relevant or generic response.

# During interaction, assess:
# - Fluency: Does the response sound natural?
# - Relevance: Is the response related to the input?
# - Coherence: Does the response make sense in context?
# - Accuracy: Is any factual information correct (if applicable)?

# After interacting, manually summarize the observations.

print("Executing the interactive chatbot loop. Please interact with the bot in the console.")
# The actual interaction happens when the code from the previous cell is executed.
# This cell serves as a placeholder to indicate the start of the testing phase.
