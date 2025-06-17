import os
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USER_AGENT = os.getenv("USER_AGENT")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

persistent_directory = "db/news_vector_store"

def fetch_news_urls(query):
    from_date = (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    headers = {"User-Agent": USER_AGENT}
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "popularity",
        "pageSize": 10,
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    print("\n NewsAPI full response:")
    print(data)

    if data.get("status") != "ok":
        print(" Error from NewsAPI:", data.get("code"), "-", data.get("message"))
        return []

    articles = data.get("articles", [])
    print(f"Found {len(articles)} articles.")
    urls = [a["url"] for a in articles if a.get("url")]
    return urls

def extract_text_from_urls(urls):
    documents = []
    headers = {"User-Agent": USER_AGENT}
    for url in urls:
        try:
            html = requests.get(url, headers=headers, timeout=10).text
            soup = BeautifulSoup(html, "html.parser")
            for script in soup(["script", "style", "noscript"]):
                script.decompose()
            text = soup.get_text(separator="\n").strip()
            if len(text) > 100:
                documents.append(Document(page_content=text, metadata={"source": url}))
        except Exception as e:
            print(f"Error scraping {url}: {e}")
    return documents

def create_vector_store(docs):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if not os.path.exists(persistent_directory):
        db = Chroma.from_documents(split_docs, embeddings, persist_directory=persistent_directory)
        db.persist()
        print(" Vector store created and persisted.")
    else:
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)
        print(" Using existing vector store.")

    return db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def chat_loop(retriever):
    print("\nAsk anything about the topic (type 'exit' to quit):")
    while True:
        user_question = input(">> ")
        if user_question.lower() in ("exit", "quit"):
            print("Exiting. Thank you!")
            break

        results = retriever.invoke(user_question)
        if results:
            print("\nAnswer from articles:\n")
            for i, doc in enumerate(results, 1):
                print(f"[{i}] {doc.page_content[:500]}...\n")
        else:
            print("No relevant information found.")

if __name__ == "__main__":
    user_topic = input("Enter a news topic to search: ")
    urls = fetch_news_urls(user_topic)
    if urls:
        docs = extract_text_from_urls(urls)
        if docs:
            retriever = create_vector_store(docs)
            chat_loop(retriever)
        else:
            print("No text could be extracted from articles.")
    else:
        print("No articles found.")
