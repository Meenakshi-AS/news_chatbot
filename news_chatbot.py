import os
import requests
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USER_AGENT = os.getenv("USER_AGENT")


def list_databases():
    return [d for d in os.listdir(".") if d.startswith("db_news_")]


def create_database(query_input, query):
    print(f"\nSearching news for topic: {query_input}\n")

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=10)

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query_input,
        "from": start_date.date(),
        "to": end_date.date(),
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "ok":
        print("Error from News API:", data)
        return

    articles = data.get("articles", [])[:10]
    if not articles:
        print(f"No articles found for topic '{query_input}'. Try another.")
        return

    documents = []
    headers = {"User-Agent": USER_AGENT}
    for article in articles:
        url_to_scrape = article.get("url", "")
        scraped_text = ""

        if url_to_scrape:
            try:
                article_html = requests.get(url_to_scrape, headers=headers, timeout=3)
                soup = BeautifulSoup(article_html.content, "html.parser")
                paragraphs = soup.find_all("p")
                scraped_text = "\n".join(p.get_text() for p in paragraphs if len(p.get_text().strip()) > 10)
            except Exception:
                scraped_text = ""

        fallback = article.get("content") or article.get("description") or article.get("title", "")
        if not scraped_text and not fallback:
            continue

        full_text = f"Title: {article['title']}\nAuthor: {article.get('author', 'Unknown')}\nPublished: {article['publishedAt']}\nURL: {url_to_scrape}\n\n{scraped_text or fallback}"
        documents.append(Document(page_content=full_text))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)

    if not split_docs:
        print("No usable content extracted from articles.")
        return

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=f"db_news_{query}", embedding_function=embeddings)
    db.add_documents(split_docs)
    print(f"Successfully indexed {len(split_docs)} chunks. Returning to main menu.\n")


def run_chatbot(query):
    persistent_directory = f"db_news_{query}"
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    custom_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Use only the context below to answer the question.
If you're not sure, provide related information if available and mention it's approximate.
Avoid saying 'no relevant information found' unless the topic is truly missing.

Context:
{context}

Question:
{question}

Answer:
""")

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.3, model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt}
    )

    while True:
        user_query = input("\nAsk anything about the topic (type 'exit' to return to main menu):\n>> ")
        if user_query.lower() == "exit":
            break

        result = qa.invoke({"query": user_query})
        print("\nAnswer:\n")
        print(result['result'])


def main_menu():
    while True:
        print("\nMAIN MENU")
        print("1. Enter Query")
        print("2. Enter Chatbot")
        print("Type 'exit' to quit")

        choice = input("Enter your choice (1/2/exit): ").strip().lower()

        if choice == "1":
            query_input = input("Enter a news topic to search: ").strip().lower()
            query = query_input.replace(" ", "_")
            all_dbs = list_databases()

            if f"db_news_{query}" in all_dbs:
                use_existing = input("Database already exists. Use it? (y/n): ").strip().lower()
                if use_existing != "y":
                    create_database(query_input, query)
            else:
                create_database(query_input, query)

        elif choice == "2":
            all_dbs = list_databases()
            if not all_dbs:
                print("No databases available. Please enter a query first and come back.")
                continue

            print("Available topics:")
            for db in all_dbs:
                print("-", db.replace("db_news_", "").replace("_", " "))

            selected_input = input("Enter a topic to start chatbot: ").strip().lower().replace(" ", "_")
            if f"db_news_{selected_input}" in all_dbs:
                run_chatbot(selected_input)
            else:
                print("Topic not found. Please check spelling or enter the query first and come back.")
                print("Available topics:")
                for db in all_dbs:
                    print("-", db.replace("db_news_", "").replace("_", " "))

        elif choice == "exit":
            print("Exiting... Bye!!")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main_menu()
