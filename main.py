import os
import random
import pandas as pd
import streamlit as st
from typing import Dict, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables.graph import MermaidDrawMethod
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from io import BytesIO

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Check if API key is set
if not os.environ.get("OPENAI_API_KEY"):
    st.error("Please set your OPENAI_API_KEY in the .env file.")
    st.stop()

# Define State Structure
class State(TypedDict):
    query: str
    category: str
    sentiment: str
    response: str

# Define Node Functions
def categorize(state: State) -> State:
    """Categorize the customer query into Technical, Billing, or General."""
    prompt = ChatPromptTemplate.from_template(
        "Categorize the following customer query into one of these categories: "
        "Technical, Billing, General. Query: {query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    category = chain.invoke({"query": state["query"]}).content
    return {"category": category}

def analyze_sentiment(state: State) -> State:
    """Analyze the sentiment of the customer query as Positive, Neutral, or Negative."""
    prompt = ChatPromptTemplate.from_template(
        "Analyze the sentiment of the following customer query. "
        "Respond with either 'Positive', 'Neutral', or 'Negative'. Query: {query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    sentiment = chain.invoke({"query": state["query"]}).content
    return {"sentiment": sentiment}

def handle_technical(state: State) -> State:
    """Provide a technical support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a technical support response to the following query: {query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def handle_billing(state: State) -> State:
    """Provide a billing support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a billing support response to the following query: {query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def handle_general(state: State) -> State:
    """Provide a general support response to the query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide a general support response to the following query: {query}"
    )
    chain = prompt | ChatOpenAI(temperature=0)
    response = chain.invoke({"query": state["query"]}).content
    return {"response": response}

def escalate(state: State) -> State:
    """Escalate the query to a human agent due to negative sentiment."""
    return {"response": "This query has been escalated to a human agent due to its negative sentiment."}

def route_query(state: State) -> str:
    """Route the query based on its sentiment and category."""
    if state["sentiment"] == "Negative":
        return "escalate"
    elif state["category"] == "Technical":
        return "handle_technical"
    elif state["category"] == "Billing":
        return "handle_billing"
    else:
        return "handle_general"

# Create and Configure the Graph
workflow = StateGraph(State)
workflow.add_node("categorize", categorize)
workflow.add_node("analyze_sentiment", analyze_sentiment)
workflow.add_node("handle_technical", handle_technical)
workflow.add_node("handle_billing", handle_billing)
workflow.add_node("handle_general", handle_general)
workflow.add_node("escalate", escalate)
workflow.add_edge("categorize", "analyze_sentiment")
workflow.add_conditional_edges(
    "analyze_sentiment",
    route_query,
    {
        "handle_technical": "handle_technical",
        "handle_billing": "handle_billing",
        "handle_general": "handle_general",
        "escalate": "escalate"
    }
)
workflow.add_edge("handle_technical", END)
workflow.add_edge("handle_billing", END)
workflow.add_edge("handle_general", END)
workflow.add_edge("escalate", END)
workflow.set_entry_point("categorize")
app = workflow.compile()

# Streamlit App Configuration
st.set_page_config(
    page_title="QueryFlow: Customer Support Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App Title and Description
st.title("🤖 QueryFlow: Automated Customer Support Agent")
st.markdown("""
This app automates customer support by categorizing queries, analyzing sentiment, and generating responses. 
Test it with your own queries, explore dummy data, or analyze query trends!
""")

# Initialize Session State
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# Dummy Data Generator
def generate_dummy_queries(n=5):
    dummy_queries = [
        "My internet is too slow today!",
        "How do I update my payment method?",
        "Love your service, keep it up!",
        "Why is my app crashing constantly?",
        "What time do you close on weekends?",
        "I was overcharged this month!",
        "Can you help me set up my account?",
        "Great support team, thanks!",
        "Where’s my refund?",
        "How do I reset my password?"
    ]
    return random.sample(dummy_queries, min(n, len(dummy_queries)))

# Sidebar for Input and Controls
with st.sidebar:
    st.header("Customer Query")
    user_query = st.text_input("Enter your query here:", "")
    
    st.subheader("Example Queries")
    example_queries = [
        "My internet connection keeps dropping. Can you help?",
        "I need help talking to chatGPT",
        "Where can I find my receipt?",
        "What are your business hours?"
    ]
    for query in example_queries:
        if st.button(query, key=f"example_{query}"):
            user_query = query
    
    process_button = st.button("Process Query", type="primary", disabled=not user_query)
    
    st.divider()
    st.subheader("Dummy Data")
    num_dummy = st.slider("Number of dummy queries", 1, 10, 5)
    if st.button("Generate Dummy Queries"):
        dummy_queries = generate_dummy_queries(num_dummy)
        for query in dummy_queries:
            st.session_state.query_history.append(run_customer_support(query))

# Function to Process Query
def run_customer_support(query: str) -> Dict[str, str]:
    """Process a customer query through the LangGraph workflow."""
    results = app.invoke({"query": query})
    return {
        "query": query,
        "category": results["category"],
        "sentiment": results["sentiment"],
        "response": results["response"]
    }

# Tabs for Results, Visualization, and History
tab1, tab2, tab3, tab4 = st.tabs(["Query Results", "Workflow Graph", "Query History", "Sentiment Analysis"])

# Tab 1: Query Results
with tab1:
    if process_button and user_query:
        with st.spinner("Processing your query..."):
            try:
                result = run_customer_support(user_query)
                st.session_state.query_history.append(result)
                st.subheader("Query Results")
                st.write(f"**Query**: {result['query']}")
                st.write(f"**Category**: {result['category']}")
                st.write(f"**Sentiment**: {result['sentiment']}")
                st.markdown(f"**Response**: {result['response']}")
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")

# Tab 2: Workflow Graph
with tab2:
    st.subheader("Workflow Visualization")
    st.markdown("This is the flow of how your query is processed:")
    try:
        graph_image = app.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        st.image(graph_image, caption="QueryFlow Workflow", use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render graph: {str(e)}")

# Tab 3: Query History
with tab3:
    st.subheader("Query History")
    if st.session_state.query_history:
        df = pd.DataFrame(st.session_state.query_history)
        st.dataframe(df)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download History as CSV",
            data=csv,
            file_name="query_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No queries processed yet. Enter a query or generate dummy data!")

# Tab 4: Sentiment Analysis
with tab4:
    st.subheader("Sentiment Distribution")
    if st.session_state.query_history:
        df = pd.DataFrame(st.session_state.query_history)
        sentiment_counts = df["sentiment"].value_counts()
        
        # Pie Chart
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures pie is circular
        st.pyplot(fig)
        
        # Bar Chart
        fig, ax = plt.subplots()
        sentiment_counts.plot(kind='bar', ax=ax, color=['#4CAF50', '#FFC107', '#F44336'])
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Number of Queries")
        st.pyplot(fig)
    else:
        st.info("No data available for analysis. Process some queries first!")
