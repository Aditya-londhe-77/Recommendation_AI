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
Be persuasive but honest. Explain things simply.

📝 Product List:
{info}

🗣️ Customer Question:
{question}

💬 Previous Context:
{history}

🧠 User Preferences (DO NOT ask again about these):
{user_context}

Instructions:
- Only use information from the product list above.
- Do not invent any product names or specs.
- If the product is listed, give accurate and helpful details about it.
- Do not hallucinate anything.
- Pay attention to user preferences already established - don't ask about them again.
- If user has specified price range, category, or other preferences previously, remember them.
- If the user says "bye", "goodbye", or anything similar, reply politely and briefly.
- When showing multiple products, always sort them by price (lowest to highest) unless user asks otherwise.
""")

chain = prompt | model
conversation_history = []

# Enhanced user context tracking
class UserContext:
    def __init__(self):
        self.price_range = None
        self.preferred_category = None
        self.preferred_type = None
        self.capacity_requirement = None
        self.installation_type = None
        self.asked_questions = set()
        self.mentioned_keywords = set()
    
    def update_from_input(self, user_input):
        # Extract and remember price preferences
        price_patterns = [
            r"under\s*\u20b9?\s*(\d{2,6})",
            r"below\s*\u20b9?\s*(\d{2,6})",
            r"less\s+than\s*\u20b9?\s*(\d{2,6})",
            r"budget\s*\u20b9?\s*(\d{2,6})",
            r"around\s*\u20b9?\s*(\d{2,6})",
            r"between\s*\u20b9?\s*(\d{2,6})\s*and\s*\u20b9?\s*(\d{2,6})"
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                if "between" in pattern:
                    self.price_range = {"min": int(match.group(1)), "max": int(match.group(2))}
                else:
                    self.price_range = {"max": int(match.group(1))}
                break
        
        # Extract category preferences
        categories = ["industrial", "commercial", "domestic", "home", "office", "restaurant"]
        for cat in categories:
            if cat in user_input.lower():
                self.preferred_category = cat
                break
        
        # Extract type preferences
        types = ["ro", "uv", "softener", "atm", "vending", "plant", "purifier"]
        for type_name in types:
            if type_name in user_input.lower():
                self.preferred_type = type_name
                self.mentioned_keywords.add(type_name)
        
        # Extract capacity requirements
        capacity_match = re.search(r"(\d+)\s*(lph|liter|litre)", user_input.lower())
        if capacity_match:
            self.capacity_requirement = int(capacity_match.group(1))
    
    def get_context_string(self):
        context_parts = []
        if self.price_range:
            if "min" in self.price_range:
                context_parts.append(f"Budget: ₹{self.price_range['min']}-₹{self.price_range['max']}")
            else:
                context_parts.append(f"Budget: Under ₹{self.price_range['max']}")
        if self.preferred_category:
            context_parts.append(f"Category: {self.preferred_category}")
        if self.preferred_type:
            context_parts.append(f"Type: {self.preferred_type}")
        if self.capacity_requirement:
            context_parts.append(f"Capacity: {self.capacity_requirement} LPH")
        if self.mentioned_keywords:
            context_parts.append(f"Interested in: {', '.join(self.mentioned_keywords)}")
        
        return " | ".join(context_parts) if context_parts else "No specific preferences established yet"

user_context = UserContext()

def extract_keywords(user_input):
    stopwords = {
        "do","you","have","has","is","the","a","an","i","we","are","with",
        "any","of","in","for","to","on","and","me","can","could","please",
        "would","like","need","want","tell","know","if","it","this","that",
        "there","be","at","ave","products","what","show","find","looking"
    }
    words = re.findall(r"\b\w{2,}\b", user_input.lower())
    return [w for w in words if w not in stopwords]

def normalize_keywords(keywords):
    synonyms = {
        "ro": ["ro", "reverse", "osmosis"],
        "uv": ["uv", "ultraviolet"],
        "atm": ["atm","vending","dispenser","coin"],
        "softener": ["softener","softner","water","softening"],
        "machine": ["machine","unit","system","equipment"],
        "plant": ["plant","system","unit","treatment"],
        "purifier": ["purifier", "filter", "filtration", "clean"],
        "industrial": ["industrial", "commercial", "business"],
        "domestic": ["domestic", "home", "household", "residential"]
    }
    normalized = set()
    for w in keywords:
        matched = False
        for canonical, group in synonyms.items():
            if w in group:
                normalized.add(canonical)
                matched = True
                break
        if not matched:
            normalized.add(w)
    return list(normalized)

def enhanced_price_filter(df_filtered, user_input, user_context):
    """Enhanced price filtering with better parsing and sorting"""
    
    # Check user context first
    if user_context.price_range:
        if "min" in user_context.price_range:
            df_filtered = df_filtered[
                (df_filtered["Regular_price"] >= user_context.price_range["min"]) &
                (df_filtered["Regular_price"] <= user_context.price_range["max"])
            ]
        else:
            df_filtered = df_filtered[df_filtered["Regular_price"] <= user_context.price_range["max"]]
    
    # Also check current input for new price mentions
    price_patterns = [
        (r"under\s*\u20b9?\s*(\d{2,6})", "max"),
        (r"below\s*\u20b9?\s*(\d{2,6})", "max"),
        (r"less\s+than\s*\u20b9?\s*(\d{2,6})", "max"),
        (r"budget\s*\u20b9?\s*(\d{2,6})", "max"),
        (r"around\s*\u20b9?\s*(\d{2,6})", "around"),
        (r"between\s*\u20b9?\s*(\d{2,6})\s*and\s*\u20b9?\s*(\d{2,6})", "range"),
        (r"above\s*\u20b9?\s*(\d{2,6})", "min"),
        (r"more\s+than\s*\u20b9?\s*(\d{2,6})", "min")
    ]
    
    for pattern, filter_type in price_patterns:
        match = re.search(pattern, user_input.lower())
        if match:
            if filter_type == "max":
                max_price = int(match.group(1))
                df_filtered = df_filtered[df_filtered["Regular_price"] <= max_price]
            elif filter_type == "min":
                min_price = int(match.group(1))
                df_filtered = df_filtered[df_filtered["Regular_price"] >= min_price]
            elif filter_type == "around":
                target_price = int(match.group(1))
                # ±20% range for "around"
                min_price = int(target_price * 0.8)
                max_price = int(target_price * 1.2)
                df_filtered = df_filtered[
                    (df_filtered["Regular_price"] >= min_price) &
                    (df_filtered["Regular_price"] <= max_price)
                ]
            elif filter_type == "range":
                min_price = int(match.group(1))
                max_price = int(match.group(2))
                df_filtered = df_filtered[
                    (df_filtered["Regular_price"] >= min_price) &
                    (df_filtered["Regular_price"] <= max_price)
                ]
            break
    
    return df_filtered

def smart_sort_products(df_filtered, user_input):
    """Smart sorting based on user intent"""
    
    # Default sort by price (ascending)
    sort_column = "Regular_price"
    ascending = True
    
    # Check for sorting preferences in user input
    if any(word in user_input.lower() for word in ["expensive", "premium", "high price", "costly"]):
        ascending = False  # Most expensive first
    elif any(word in user_input.lower() for word in ["cheap", "affordable", "budget", "low price"]):
        ascending = True   # Cheapest first
    elif any(word in user_input.lower() for word in ["popular", "best", "recommended"]):
        # Sort by name for "popular" items (assuming alphabetical represents some popularity)
        sort_column = "Name"
        ascending = True
    
    return df_filtered.sort_values(by=sort_column, ascending=ascending)

def apply_filters(user_input):
    """Enhanced filtering with better logic"""
    
    # Update user context
    user_context.update_from_input(user_input)
    
    raw = extract_keywords(user_input)
    keywords = normalize_keywords(raw)
    
    filtered = df.copy()
    
    # Apply price filtering first
    filtered = enhanced_price_filter(filtered, user_input, user_context)
    
    # Category filtering with context awareness
    if user_context.preferred_category:
        filtered = filtered[filtered["Category"].str.contains(user_context.preferred_category, case=False, na=False)]
    
    # Current input category filtering
    known_cats = ["industrial","commercial","domestic","softener","vending","atm","cooler","plant","machine"]
    cats = [c for c in known_cats if c in keywords]
    if cats:
        pat = "|".join(cats)
        filtered = filtered[filtered["Category"].str.contains(pat, case=False, na=False)]
    
    # Variant filtering (more precise)
    variant_kw = ["premium","classic","nx","advance","basic"]
    matching_variants = [v for v in variant_kw if v in keywords]
    if matching_variants:
        pat = "|".join(matching_variants)
        filtered = filtered[
            filtered["Name"].str.contains(pat, case=False, na=False) |
            filtered["Short description"].str.contains(pat, case=False, na=False) |
            filtered["Attribute 1 value(s)"].str.contains(pat, case=False, na=False)
        ]
    
    # Main keyword filtering (improved)
    if keywords:
        # Remove already processed keywords
        search_keywords = [k for k in keywords if k not in variant_kw and k not in known_cats]
        
        if search_keywords:
            pat = "|".join(re.escape(k) for k in search_keywords)
            keyword_filter = (
                filtered["Name"].str.contains(pat, case=False, na=False) |
                filtered["Short description"].str.contains(pat, case=False, na=False) |
                filtered["Category"].str.contains(pat, case=False, na=False)
            )
            filtered = filtered[keyword_filter]
    
    # Special handling for ATM/Vending machines
    if any(k in ["atm","vending"] for k in keywords) and filtered.empty:
        atm_df = df[
            df["Name"].str.contains("atm|vending|coin", case=False, na=False) |
            df["Category"].str.contains("atm|vending|coin", case=False, na=False) |
            df["Short description"].str.contains("atm|vending|coin", case=False, na=False)
        ]
        filtered = atm_df
    
    # Apply smart sorting
    filtered = smart_sort_products(filtered, user_input)
    
    return filtered

def matches_keywords(row, keywords):
    """Improved keyword matching"""
    text_fields = [
        row.get("Name", ""), 
        row.get("Short description", ""), 
        row.get("Category", ""),
        row.get("Attribute 1 value(s)", "")
    ]
    text = " ".join(text_fields).lower()
    
    hits = sum(1 for k in keywords if k in text)
    name_hits = sum(1 for k in keywords if k in row.get("Name", "").lower())
    
    # More flexible matching criteria
    if len(keywords) == 1:
        return hits >= 1
    elif len(keywords) == 2:
        return hits >= 2 or name_hits >= 1
    else:
        return hits >= 2 and name_hits >= 1

def handle_input(user_input):
    try:
        # Update user context
        user_context.update_from_input(user_input)
        
        raw_kw = extract_keywords(user_input)
        keywords = normalize_keywords(raw_kw)

        docs = []
        
        # Try exact matching first
        exact_df = df[df.apply(lambda r: matches_keywords(r, keywords), axis=1)]
        
        if not exact_df.empty:
            # Sort the exact matches
            exact_df = smart_sort_products(exact_df, user_input)
            
            # Limit to top 8 results for better performance
            for _, row in exact_df.head(8).iterrows():
                docs.append(Document(page_content= (
                    f"Name: {row['Name']}\nPrice: ₹{row['Regular_price']}\nCategory: {row['Category']}\nDescription: {row['Short description']}"
                )))

        # Fallback to filtered search
        if not docs:
            filtered_df = apply_filters(user_input)
            for _, row in filtered_df.head(6).iterrows():
                docs.append(Document(page_content=(
                    f"Name: {row['Name']}\nPrice: ₹{row['Regular_price']}\nCategory: {row['Category']}\nDescription: {row['Short description']}"
                )))

        # Final fallback to vector search
        if not docs:
            vector_docs = retriever.get_relevant_documents(user_input)
            docs.extend(vector_docs[:5])

        if not docs:
            gui.display_reply("❌ Sorry, no matching products were found. Could you try different keywords or check your price range?")
            return

        product_info = "\n\n".join(d.page_content for d in docs)
        recent_hist = "\n".join(conversation_history[-6:])  # Increased history context
        context_info = user_context.get_context_string()

        payload = {
            "history": recent_hist,
            "question": user_input,
            "info": product_info,
            "user_context": context_info
        }

        response = chain.invoke(payload)
        final = response.content.strip()

        gui.display_reply(final)

        # Image display logic (cleaned up)
        if len(docs) == 1:
            product_name = docs[0].page_content.split("\n")[0].replace("Name: ", "").strip()
            product_row = df[df["Name"].str.lower() == product_name.lower()]
            if not product_row.empty:
                image_url = product_row.iloc[0].get("Images", "")
                if image_url and image_url.startswith("http"):
                    # Take first image if multiple URLs
                    first_image = image_url.split(",")[0].strip()
                    gui.display_image(first_image)

        # Update conversation history (limit to last 10 exchanges)
        conversation_history.append(f"User: {user_input}")
        conversation_history.append(f"Bot: {final}")
        
        # Keep only recent history to prevent context overflow
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

    except Exception as e:
       gui.display_reply(f"❌ Error occurred: {str(e)}")

gui = ChatGUI(on_submit=handle_input)
gui.run()
