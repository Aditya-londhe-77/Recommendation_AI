from gui_template import ChatGUI
from datetime import datetime
import re
import pandas as pd
from dotenv import dotenv_values
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Load API key
env_vars = dotenv_values(".env")
GroqAPIKey = env_vars.get("GroqAPIKey")
if not GroqAPIKey:
    raise ValueError("❌ Missing GroqAPIKey in .env file")

model = ChatGroq(api_key=GroqAPIKey, model_name="llama-3.1-8b-instant")

# Load product data
df = pd.read_csv("new_export.csv", encoding="ISO-8859-1").fillna("")
df.columns = df.columns.str.strip()
df["Regular_price"] = pd.to_numeric(df.get("Regular_price", 0), errors="coerce").fillna(0).astype(int)

embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    collection_name="water_product_recommendation",
    persist_directory="./chroma_water_products",
    embedding_function=embedding_fn
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 15})

prompt = ChatPromptTemplate.from_template("""
You are a friendly and knowledgeable sales assistant for a water treatment company.
You help customers choose the right water purifier or system based on their needs.
Be persuasive but honest. Explain things simply. Ask questions to understand needs.

📝 Product List:
{info}

🗣️ Customer Question:
{question}

💬 Conversation History:
{history}

Instructions:
- Only use information from the product list above.
- Do not invent any product names or specs.
- If the product is listed, give accurate and helpful details about it.
- Do not hallucinate anything.
- Do not suggest any products unless the customer explicitly asks.
- If the user says "bye", "goodbye", or anything similar, reply politely and briefly.
""")

chain = prompt | model
conversation_history = []

def extract_keywords(user_input):
    stopwords = {
        "do","you","have","has","is","the","a","an","i","we","are","with",
        "any","of","in","for","to","on","and","me","can","could","please",
        "would","like","need","want","tell","know","if","it","this","that",
        "there","be","at","ave","products","what"
    }
    words = re.findall(r"\b\w{2,}\b", user_input.lower())
    return [w for w in words if w not in stopwords]

def normalize_keywords(keywords):
    synonyms = {
        "ros": ["ro"],
        "atm": ["atm","vending","dispenser","coin"],
        "softener": ["softener","softner"],
        "machine": ["machine","unit","system"],
        "plant": ["plant","system","unit"]
    }
    normalized = set()
    for w in keywords:
        matched = False
        for group in synonyms.values():
            if w in group:
                normalized.update(group)
                matched = True
                break
        if not matched:
            normalized.add(w)
    return list(normalized)

def matches_keywords(row, keywords):
    text = " ".join([row.get("Name", ""), row.get("Short description", ""), row.get("Category", "")]).lower()
    hits = sum(1 for k in keywords if k in text)
    name_hits = sum(1 for k in keywords if k in row.get("Name","" ).lower())
    min_hits = 1 if len(keywords) <= 1 else 2
    return hits >= min_hits and name_hits >= 1

def apply_filters(user_input):
    price_match = re.search(r"under\s*\u20b9?\s*(\d{2,6})", user_input.lower())
    max_price = int(price_match.group(1)) if price_match else None

    raw = extract_keywords(user_input)
    keywords = normalize_keywords(raw)

    filtered = df.copy()

    variant_kw = ["premium","classic","nx","advance","basic"]
    if any(v in keywords for v in variant_kw):
        pat = "|".join(v for v in variant_kw if v in keywords)
        filtered = filtered[
            filtered["Name"].str.contains(pat, case=False, na=False) |
            filtered["Short description"].str.contains(pat, case=False, na=False)
        ]

    known_cats = ["industrial","commercial","domestic","softener","vending","atm","cooler","plant","machine"]
    cats = [c for c in known_cats if c in keywords]
    if cats:
        pat = "|".join(cats)
        filtered = filtered[filtered["Category"].str.contains(pat, case=False, na=False)]

    if max_price is not None:
        filtered = filtered[filtered["Regular_price"] <= max_price]

    if keywords:
        pat = "|".join(re.escape(k) for k in keywords)
        filtered = filtered[
            filtered["Name"].str.contains(pat, case=False, na=False) |
            filtered["Short description"].str.contains(pat, case=False, na=False) |
            filtered["Category"].str.contains(pat, case=False, na=False)
        ]

    if any(k in ["atm","vending"] for k in keywords) and filtered.empty:
        atm_df = df[
            df["Name"].str.contains("atm|vending|coin", case=False, na=False) |
            df["Category"].str.contains("atm|vending|coin", case=False, na=False) |
            df["Short description"].str.contains("atm|vending|coin", case=False, na=False)
        ]
        filtered = atm_df

    return filtered

def handle_input(user_input):
    try:
        raw_kw = extract_keywords(user_input)
        keywords = normalize_keywords(raw_kw)

        docs = []
        exact_df = df[df.apply(lambda r: matches_keywords(r, keywords), axis=1)]
        for _, row in exact_df.iterrows():
            docs.append(Document(page_content= (
                f"Name: {row['Name']}\nPrice: ₹{row['Regular_price']}\nCategory: {row['Category']}\nDescription: {row['Short description']}"
            )))

        if not docs:
            filtered_df = apply_filters(user_input)
            for _, row in filtered_df.head(5).iterrows():
                docs.append(Document(page_content=(
                    f"Name: {row['Name']}\nPrice: ₹{row['Regular_price']}\nCategory: {row['Category']}\nDescription: {row['Short description']}"
                )))

        if not docs:
            vector_docs = retriever.get_relevant_documents(user_input)
            docs.extend(vector_docs)

        if not docs:
            gui.display_reply("❌ Sorry, no matching products were found.")
            return

        product_info = "\n\n".join(d.page_content for d in docs)
        recent_hist = "\n".join(conversation_history[-4:])

        payload = {
            "history": recent_hist,
            "question": user_input,
            "info": product_info
        }

        response = chain.invoke(payload)
        final = response.content.strip()

        gui.display_reply(final)

        #if single product matched shows image 
        if len(docs) == 1:
            product_name = docs[0].page_content.split("\n")[0].replace("Name: ", "").strip()
            product_row = df[df["Name"].str.lower() == product_name.lower()]
            if not product_row.empty:
                image_url = product_row.iloc[0].get("Images", "")
                if image_url.startswith("http"):
                    gui.display_image(image_url)
                    if len(docs) == 1:
                     product_name = docs[0].page_content.split("\n")[0].replace("Name: ", "").strip()
                     print(f"[DEBUG] Matched product name: {product_name}")
                     product_row = df[df["Name"].str.lower() == product_name.lower()]
                     if not product_row.empty:
                         image_url = product_row.iloc[0].get("images", "")
                         print(f"[DEBUG] Image URL: {image_url}")
                         if image_url.startswith("http"):
                            gui.display_image(image_url)
                             
                         else:
                             print("[DEBUG] No valid image URL found.")
                     else:
                         print("[DEBUG] No matching product row found.")

        

        conversation_history.append(f"User: {user_input}")
        conversation_history.append(f"Bot: {final}")

    except Exception as e:
       gui.display_reply(f"❌ Error occurred: {str(e)}")

gui = ChatGUI(on_submit=handle_input)
gui.run()
