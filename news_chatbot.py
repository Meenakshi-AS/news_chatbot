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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USER_AGENT = os.getenv("USER_AGENT")

all_dbs = [d for d in os.listdir(".") if d.startswith("db_news_")]
if all_dbs:
    print("Available topics:")
    for db in all_dbs:
        print("-", db.replace("db_news_", "").replace("_", " "))
else:
    print("No topics found. Start by searching a new topic.")

query_input = input("Enter a news topic to search: ").strip().lower()
query = query_input.replace(" ", "_")
persistent_directory = f"db_news_{query}"

if persistent_directory in all_dbs:
    use_existing = input(f"Previous database for topic '{query_input}' exists. Do you want to use it? (y/n): ").strip().lower()
else:
    print(f"No previous database found for topic '{query_input}'. Creating new database...")
    use_existing = 'n'

if use_existing != 'y':
    print(f"\nSearching news for topic: {query_input}\n")

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=3)

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
        exit()

    articles = data.get("articles", [])[:10]
    if not articles:
        print(f"No articles found for topic '{query_input}'. Please try another topic.")
        exit()

    print(f"Found {len(articles)} articles.\n")

    documents = []
    headers = {"User-Agent": USER_AGENT}
    for article in articles:
        url_to_scrape = article.get("url", "")
        scraped_text = ""

        if url_to_scrape:
            try:
                article_html = requests.get(url_to_scrape, headers=headers, timeout=5)
                soup = BeautifulSoup(article_html.content, "html.parser")
                paragraphs = soup.find_all("p")
                scraped_text = "\n".join(p.get_text() for p in paragraphs if len(p.get_text().strip()) > 50)
                if scraped_text:
                    print(f"\n--- Scraped text preview from {url_to_scrape} ---\n\n{scraped_text[:500]}\n")
            except Exception:
                scraped_text = ""

        fallback = article.get("content") or article.get("description") or ""
        if not scraped_text and not fallback:
            continue

        full_text = f"Title: {article['title']}\nAuthor: {article.get('author', 'Unknown')}\nPublished: {article['publishedAt']}\nURL: {url_to_scrape}\n\n{scraped_text or fallback}"
        documents.append(Document(page_content=full_text))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(documents)

    if not split_docs:
        print("No usable content was extracted from the articles. Try a different topic.")
        exit()

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
    db.add_documents(split_docs)
else:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": 3})

custom_prompt = PromptTemplate.from_template("""
You are an assistant answering questions only using the information below.
If the answer is not in the text, reply "Sorry, no relevant information found in the selected topic database."

Context:
{context}

Question:
{question}

Answer:
""")

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.2),
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt}
)

while True:
    user_query = input("\nAsk anything about the topic (type 'exit' to quit):\n>> ")
    if user_query.lower() == "exit":
        print("Exiting. Thank you!")
        break

    docs = retriever.get_relevant_documents(user_query)
    if not docs:
        print("\nSorry, no relevant information found in the selected topic database.")
        continue

    result = qa.invoke({"query": user_query})
    print("\nAnswer:\n")
    print(result['result'])
