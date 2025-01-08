import os
from dotenv import load_dotenv

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

import streamlit as st
import google.generativeai as genai


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = 'models/embedding-001'
        title = "Custom query"
        return genai.embed_content(model=model,
                                   content=input,
                                   task_type="retrieval_document",
                                   title=title)["embedding"]


def get_relevant_passages(query, db, n_results=10):
    """
    Query the database to get relevant documents based on the user's query.
    """
    try:
        result = db.query(query_texts=[query], n_results=n_results)
        if result and result['documents']:
            return result['documents'][0]
        else:
            return None
    except Exception as e:
        st.error(f"Error querying database: {e}")
        return None


def make_prompt(query, relevant_passage):
    """
    Generate a conversational prompt for the chatbot.
    """
    escaped = relevant_passage.replace("'", "").replace('"', "")
    prompt = f"""
    You are a chatbot specializing in helping users find deals for tourist attractions in Dubai. 

    User's question: {query}

    Context information about available deals:
    {escaped}

    Your response must be detailed, friendly, and helpful. 
    Provide recommendations for deals related to the user's query, including:
    - Attraction name
    - Deal description
    - Price (in AED and USD, assuming a fixed exchange rate)
    - Relevant tips (e.g., operating hours, best visit times, nearby attractions)

    Ensure the response is conversational and encourages the user to ask follow-up questions. 
    Also note that for Burj Khalifa only return Burj Khalifa deals and for burj al arab only return burj al arab deals. 
    """
    return prompt


def convert_passages_to_string(passages):
    """
    Convert the list of passages into a single string for the prompt.
    """
    context = "\n".join(passages)
    return context


def setUpGoogleAPI():
    """
    Configure the Google API key for Generative AI.
    """
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    genai.configure(api_key=api_key)


def loadVectorDataBase():
    """
    Load the vector database with the name `sme_db`.
    """
    chroma_client = chromadb.PersistentClient(path="../database/")
    db = chroma_client.get_or_create_collection(
        name="sme_db", embedding_function=GeminiEmbeddingFunction())
    return db


def main():
    # Set Streamlit page configuration
    st.set_page_config(page_title="DuneGuide: Dubai AI Tourist Guide", page_icon="🇦🇪", layout="wide")
    
    # Initialize API and database
    setUpGoogleAPI()
    db = loadVectorDataBase()
    
    # Initialize session state for conversation history
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []

    # Title and introduction
    st.title("🏙️ Dubai's First Ever AI Tourist Guide 🇦🇪")
    st.write(
        "🌟 **Welcome to your personalized Dubai guide!** 🌟\n\n"
        "I’m here to inform you about Dubai’s most stunning attractions, exclusive deals, and hidden gems. 🏙️✨\n\n"
        "From the breathtaking **Burj Khalifa** to adventurous desert safaris🏜️ 🌀, "
        "Dubai has something for everyone! 🌴\n\n"
        "Let’s explore together and make your Dubai experience unforgettable! Feel free to ask me about "
        "tourist attractions, deals, and more. 🗺️😊"
    )

    # Input for user's query
    user_query = st.text_input("Your question:")

    if user_query:
        # Append user query to the conversation history
        st.session_state["conversation_history"].append({"role": "user", "content": user_query})

        # Fetch relevant passages from the database
        passages = get_relevant_passages(user_query, db, 5)

        if passages:
            # Convert passages to string and generate a prompt
            context = convert_passages_to_string(passages)
            prompt = make_prompt(user_query, context)

            # Generate AI response
            model = genai.GenerativeModel(model_name="gemini-pro")
            response = model.generate_content(prompt)

            # Append AI response to the conversation history
            st.session_state["conversation_history"].append({"role": "assistant", "content": response.text})
        else:
            response_text = "Sorry, I couldn't find any relevant deals. Please try another query!"
            st.session_state["conversation_history"].append({"role": "assistant", "content": response_text})

    # Display the conversation history
    for message in st.session_state["conversation_history"]:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**DuneGuide:** {message['content']}")


if __name__ == "__main__":
    main()
