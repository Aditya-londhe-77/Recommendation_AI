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
You are a water treatment sales expert. You ONLY help customers with water purification, RO systems, UV purifiers, water softeners, and related water treatment equipment.

STRICT RULES:
1. ONLY discuss water treatment topics - if asked about anything else, politely redirect to water solutions
2. ONLY recommend products from the list below - NEVER mention products not in this list
3. If no suitable products are found, say so honestly and ask for more specific requirements
4. Talk like a human expert, not a robot - be conversational and natural
5. NO greetings, get straight to helping with their water needs

AVAILABLE PRODUCTS ONLY:
{info}

CUSTOMER QUESTION: {question}
CONVERSATION HISTORY: {history}

RESPONSE GUIDELINES:
- If question is NOT about water treatment, say: "I specialize in water treatment solutions. What water challenges can I help you solve?"
- IMMEDIATELY recommend specific products when asked - don't ask questions first
- Check conversation history - NEVER repeat questions already asked
- If you already know details from previous conversation, use that information
- Talk naturally like a human expert would - use phrases like "I'd recommend", "That sounds like", "Based on what you're telling me"
- Be specific about product names, prices, and features from the list only
- Only ask NEW follow-up questions if genuinely needed for better recommendation
- Explain benefits simply: "This removes 99% of bacteria", "Saves ₹200 monthly on bottled water"
- Use conversational transitions: "Actually,", "You know what,", "Here's the thing,"
- Show genuine interest: "That's a common issue", "I see this a lot", "Good question"
- Never hallucinate or invent product specifications
- Reference only the exact product names from the catalog

IMMEDIATE RECOMMENDATION TRIGGERS:
- "recommend", "suggest", "need", "want", "looking for", "which one", "best option"
- When these words appear, give direct product recommendations with reasoning

CONVERSATION MEMORY RULES:
- If user mentioned budget before, remember it
- If user mentioned family size before, don't ask again
- If user mentioned usage type before, build on that
- Check history for: budget, people count, usage type, location type, water issues

HUMAN CONVERSATION EXAMPLES:
"I'd recommend the [exact product name] for ₹[price]. It handles exactly what you mentioned and works great for [their situation]."
"Perfect! The [exact product name] would be ideal. It's ₹[price] and specifically designed for [their need]."
"Based on what you told me earlier about [reference previous info], the [exact product name] is your best bet."

Remember: Be decisive, give specific recommendations immediately, and never repeat questions from the conversation history.
""")

chain = prompt | model
conversation_history = []

def extract_keywords(user_input):
    stopwords = {
        "do","you","have","has","is","the","a","an","i","we","are","with",
        "any","of","in","for","to","on","and","me","can","could","please",
        "would","like","need","want","tell","know","if","it","this","that",
        "there","be","at","ave","products","what","looking","help","show",
        "hello","hi","hey","good","morning","afternoon","evening","about",
        "some","get","find","see","recommend","suggest","give","best"
    }
    words = re.findall(r"\b\w{2,}\b", user_input.lower())
    keywords = [w for w in words if w not in stopwords]
    
    # Add common water treatment terms if they appear in compound words or phrases
    if "water" in user_input.lower():
        keywords.append("water")
    if "ro" in user_input.lower() or "reverse osmosis" in user_input.lower():
        keywords.append("ro")
    if "uv" in user_input.lower() or "ultraviolet" in user_input.lower():
        keywords.append("uv")
        
    return keywords

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

def is_water_treatment_related(query):
    """Check if the query is related to water treatment"""
    water_keywords = {
        'water', 'ro', 'reverse', 'osmosis', 'purifier', 'filter', 'filtration', 
        'purification', 'treatment', 'softener', 'softening', 'uv', 'ultraviolet',
        'plant', 'system', 'machine', 'unit', 'industrial', 'commercial', 'domestic',
        'drinking', 'clean', 'pure', 'mineral', 'tds', 'alkaline', 'atm', 'vending',
        'cooler', 'dispenser', 'membrane', 'cartridge', 'sediment', 'carbon',
        'bacteria', 'virus', 'contamination', 'hardness', 'chlorine', 'iron',
        'arsenic', 'fluoride', 'heavy', 'metals', 'bore', 'well', 'tap', 'supply'
    }
    
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    
    # Check if any water-related keywords are present
    return bool(water_keywords.intersection(query_words))

def handle_input(user_input):
    try:
        # Check for goodbye/exit messages
        goodbye_words = ["bye", "goodbye", "thanks", "thank you", "exit", "quit"]
        if any(word in user_input.lower() for word in goodbye_words):
            gui.display_reply("Thank you for your interest! Feel free to reach out anytime for your water treatment needs. Have a great day!")
            return
            
        # Handle general greetings and convert to sales opportunity with immediate options
        greeting_words = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]
        if any(greeting in user_input.lower() for greeting in greeting_words) and len(user_input.split()) <= 3:
            # Get top 3 popular products for immediate showcase
            top_products = df.head(3)
            product_showcase = "Here are some of our most popular water treatment solutions:\n\n"
            for _, row in top_products.iterrows():
                product_showcase += f"• {row['Name']} - ₹{row['Regular_price']:,} ({row['Category']})\n"
            product_showcase += "\nWhat type of water treatment are you interested in?"
            gui.display_reply(product_showcase)
            return
            
        # Check if query is water treatment related
        if not is_water_treatment_related(user_input):
            gui.display_reply("I specialize in water treatment solutions. What water challenges can I help you solve? We have RO systems, UV purifiers, water softeners, and industrial treatment plants.")
            return
            
        # Detect recommendation requests for immediate product suggestions
        recommendation_triggers = ["recommend", "suggest", "need", "want", "looking for", "which one", "best option", "show me", "what do you have"]
        is_recommendation_request = any(trigger in user_input.lower() for trigger in recommendation_triggers)
            
        raw_kw = extract_keywords(user_input)
        keywords = normalize_keywords(raw_kw)

        docs = []
        matched_products = set()  # Track products to avoid duplicates
        
        # For recommendation requests, be more aggressive in finding products
        max_products = 8 if is_recommendation_request else 6
        
        # First try exact keyword matching
        exact_df = df[df.apply(lambda r: matches_keywords(r, keywords), axis=1)]
        for _, row in exact_df.head(max_products).iterrows():
            product_name = row['Name']
            if product_name not in matched_products:
                matched_products.add(product_name)
                docs.append(Document(page_content= (
                    f"PRODUCT: {row['Name']}\n"
                    f"PRICE: ₹{row['Regular_price']:,}\n"
                    f"TYPE: {row['Category']}\n"
                    f"DETAILS: {row['Short description']}\n"
                    f"VARIANTS: {row.get('Attribute 1 value(s)', 'Standard')}\n"
                    f"[VERIFIED PRODUCT FROM CATALOG]"
                )))

        # If no exact matches, try filtered search
        if not docs:
            filtered_df = apply_filters(user_input)
            for _, row in filtered_df.head(max_products).iterrows():
                product_name = row['Name']
                if product_name not in matched_products:
                    matched_products.add(product_name)
                    docs.append(Document(page_content=(
                        f"PRODUCT: {row['Name']}\n"
                        f"PRICE: ₹{row['Regular_price']:,}\n"
                        f"TYPE: {row['Category']}\n"
                        f"DETAILS: {row['Short description']}\n"
                        f"VARIANTS: {row.get('Attribute 1 value(s)', 'Standard')}\n"
                        f"[VERIFIED PRODUCT FROM CATALOG]"
                    )))

        # If still no matches and it's a recommendation request, get popular products
        if not docs and is_recommendation_request:
            # Get some general products for recommendation
            general_products = df.head(6)  # Get first 6 products as general recommendations
            for _, row in general_products.iterrows():
                product_name = row['Name']
                if product_name not in matched_products:
                    matched_products.add(product_name)
                    docs.append(Document(page_content=(
                        f"PRODUCT: {row['Name']}\n"
                        f"PRICE: ₹{row['Regular_price']:,}\n"
                        f"TYPE: {row['Category']}\n"
                        f"DETAILS: {row['Short description']}\n"
                        f"VARIANTS: {row.get('Attribute 1 value(s)', 'Standard')}\n"
                        f"[VERIFIED PRODUCT FROM CATALOG - GENERAL RECOMMENDATION]"
                    )))

        # Last resort: vector similarity search from CSV only
        if not docs:
            vector_docs = retriever.get_relevant_documents(user_input)
            for doc in vector_docs[:4]:
                # Extract product name and verify it exists in CSV
                content_lines = doc.page_content.split('\n')
                if content_lines:
                    # Try to find the product in our CSV
                    for _, row in df.iterrows():
                        if row['Name'].lower() in doc.page_content.lower():
                            product_name = row['Name']
                            if product_name not in matched_products:
                                matched_products.add(product_name)
                                docs.append(Document(page_content=(
                                    f"PRODUCT: {row['Name']}\n"
                                    f"PRICE: ₹{row['Regular_price']:,}\n"
                                    f"TYPE: {row['Category']}\n"
                                    f"DETAILS: {row['Short description']}\n"
                                    f"VARIANTS: {row.get('Attribute 1 value(s)', 'Standard')}\n"
                                    f"[VERIFIED PRODUCT FROM CATALOG]"
                                )))
                                break

        if not docs:
            if is_recommendation_request:
                gui.display_reply("Let me show you some of our popular water treatment solutions. What specific type are you interested in - home purifiers, commercial systems, or industrial plants? I can recommend the best options for your needs.")
            else:
                gui.display_reply("I don't have any products in our current catalog that match what you're looking for. Could you be more specific about your water needs? Are you looking for home water purifiers, commercial systems, or industrial treatment plants? I'd be happy to suggest something from our available range.")
            return

        # Create product list for AI with strict verification
        product_info = "\n\n".join(d.page_content for d in docs)
        product_names = list(matched_products)
        product_info += f"\n\nAVAILABLE PRODUCT NAMES ONLY: {', '.join(product_names)}"
        product_info += "\n\nIMPORTANT: Only mention products from the list above. Never suggest products not listed."
        
        # Add special instruction for recommendation requests
        if is_recommendation_request:
            product_info += "\n\nUSER IS ASKING FOR RECOMMENDATIONS: Provide specific product suggestions immediately with reasons why each product is suitable. Don't ask questions they already answered."
        
        recent_hist = "\n".join(conversation_history[-8:])  # More context to avoid repetition
        
        # Extract previous context to avoid repetitive questions
        context_info = ""
        if recent_hist:
            context_info = "\n\nPREVIOUS CONTEXT: The user has already provided some information in the conversation above. Use this information and don't ask the same questions again."

        payload = {
            "history": recent_hist + context_info,
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
