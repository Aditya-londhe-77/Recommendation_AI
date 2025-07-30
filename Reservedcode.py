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
You are an experienced sales representative for a premium water treatment company. You talk like a knowledgeable human salesperson who genuinely cares about finding the right water solution for each customer.

📝 Available Products:
{info}

🗣️ Customer Query:
{question}

💬 Previous Conversation:
{history}

Your Sales Approach:
- NEVER start with greetings like "Hello", "Hi", "Good morning" etc. - go straight to addressing their question
- Be conversational and natural, like talking to a friend or neighbor
- Focus on understanding their specific water needs and usage
- Highlight key benefits and value propositions of recommended products
- Use persuasive but honest sales language
- Ask follow-up questions to better understand their requirements
- Compare products when appropriate to help them decide
- Mention pricing confidently and explain value for money
- Only recommend products from the list above - never invent products
- If they want to end the conversation, be brief and professional
- Be enthusiastic about the products but not pushy

Remember: You're here to help them solve their water problems, not just list products. Think like a human sales expert who builds trust through knowledge and genuine care.
""")

chain = prompt | model
conversation_history = []

def extract_keywords(user_input):
    stopwords = {
        "do","you","have","has","is","the","a","an","i","we","are","with",
        "any","of","in","for","to","on","and","me","can","could","please",
        "would","like","need","want","tell","know","if","it","this","that",
        "there","be","at","ave","products","what","looking","help","show",
        "hello","hi","hey","good","morning","afternoon","evening"
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
        # Check for goodbye/exit messages
        goodbye_words = ["bye", "goodbye", "thanks", "thank you", "exit", "quit"]
        if any(word in user_input.lower() for word in goodbye_words):
            gui.display_reply("Thank you for your interest! Feel free to reach out anytime for your water treatment needs. Have a great day!")
            return
            
        # Handle general greetings and convert to sales opportunity
        greeting_words = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if any(greeting in user_input.lower() for greeting in greeting_words) and len(user_input.split()) <= 3:
            gui.display_reply("What kind of water treatment solution are you looking for today? We have everything from home RO systems to large industrial plants.")
            return
            
        raw_kw = extract_keywords(user_input)
        keywords = normalize_keywords(raw_kw)

        docs = []
        # First try exact keyword matching
        exact_df = df[df.apply(lambda r: matches_keywords(r, keywords), axis=1)]
        for _, row in exact_df.head(8).iterrows():  # Increased to 8 for better selection
            docs.append(Document(page_content= (
                f"Product Name: {row['Name']}\n"
                f"Price: ₹{row['Regular_price']:,}\n"
                f"Category: {row['Category']}\n"
                f"Key Features: {row['Short description']}\n"
                f"Variants Available: {row.get('Attribute 1 value(s)', 'Standard')}"
            )))

        # If no exact matches, try filtered search
        if not docs:
            filtered_df = apply_filters(user_input)
            for _, row in filtered_df.head(6).iterrows():
                docs.append(Document(page_content=(
                    f"Product Name: {row['Name']}\n"
                    f"Price: ₹{row['Regular_price']:,}\n"
                    f"Category: {row['Category']}\n"
                    f"Key Features: {row['Short description']}\n"
                    f"Variants Available: {row.get('Attribute 1 value(s)', 'Standard')}"
                )))

        # Last resort: vector similarity search
        if not docs:
            vector_docs = retriever.get_relevant_documents(user_input)
            docs.extend(vector_docs[:5])

        if not docs:
            gui.display_reply("I don't see any products matching your specific requirements right now. Could you tell me more about what type of water treatment you're looking for? For example, are you looking for home purifiers, industrial systems, or something specific for your business?")
            return

        product_info = "\n" + "="*50 + "\n".join(d.page_content for d in docs)
        recent_hist = "\n".join(conversation_history[-6:])  # More context

        payload = {
            "history": recent_hist,
            "question": user_input,
            "info": product_info
        }

        response = chain.invoke(payload)
        final = response.content.strip()

        gui.display_reply(final)

        # Display image if single product matched
        if len(docs) == 1:
            product_name = docs[0].page_content.split("\n")[0].replace("Name: ", "").strip()
            print(f"[DEBUG] Matched product name: {product_name}")
            product_row = df[df["Name"].str.lower() == product_name.lower()]
            if not product_row.empty:
                image_url = product_row.iloc[0].get("Images", "")
                print(f"[DEBUG] Image URL: {image_url}")
                if image_url and str(image_url).startswith("http"):
                    # Get first image URL if multiple URLs are comma-separated
                    first_image_url = str(image_url).split(",")[0].strip()
                    gui.display_image(first_image_url)
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
