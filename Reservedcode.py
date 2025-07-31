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
You are a knowledgeable water treatment systems sales assistant. Help customers find the right products.

IMPORTANT RULES:
1. NEVER ask repetitive questions if the customer has already provided information
2. If you have already shown products, don't ask the same questions again
3. Provide detailed specifications and technical details for recommended products
4. Focus on product benefits and features rather than asking more questions
5. If the customer has asked about specific products, provide comprehensive information

📝 Available Products:
{info}

🗣️ Customer Question:
{question}

💬 Previous Conversation:
{history}

🎯 Context Analysis:
{context_analysis}

RESPONSE GUIDELINES:
- If products are listed above, provide detailed information about them
- Include technical specifications, features, capacity, price, and benefits
- Mention installation requirements, maintenance, and warranty if relevant
- Compare products if multiple options are available
- Only ask clarifying questions if absolutely necessary and not asked before
- Avoid repetitive greetings or the same questions
- If customer says goodbye, respond briefly and politely
- Focus on being helpful with detailed product information
""")

chain = prompt | model
conversation_history = []
user_context = {
    "asked_questions": set(),
    "shown_products": set(),
    "user_preferences": {},
    "current_filter": None
}

def extract_keywords(user_input):
    """Enhanced keyword extraction with better water treatment terminology"""
    stopwords = {
        "do","you","have","has","is","the","a","an","i","we","are","with",
        "any","of","in","for","to","on","and","me","can","could","please",
        "would","like","need","want","tell","know","if","it","this","that",
        "there","be","at","ave","products","what","show","looking"
    }
    words = re.findall(r"\b\w{2,}\b", user_input.lower())
    keywords = [w for w in words if w not in stopwords]
    
    # Add price extraction
    price_match = re.search(r"under\s*\₹?\s*(\d{2,6})|below\s*\₹?\s*(\d{2,6})|less\s*than\s*\₹?\s*(\d{2,6})", user_input.lower())
    if price_match:
        price_value = next(filter(None, price_match.groups()))
        user_context["user_preferences"]["max_price"] = int(price_value)
    
    return keywords

def normalize_keywords(keywords):
    """Enhanced keyword normalization with comprehensive synonyms"""
    synonyms = {
        "ro": ["ro", "reverse", "osmosis"],
        "uv": ["uv", "ultraviolet", "ultra", "violet"],
        "uf": ["uf", "ultrafiltration", "ultra", "filtration"],
        "atm": ["atm", "vending", "dispenser", "coin", "operated"],
        "softener": ["softener", "softner", "soft", "water"],
        "machine": ["machine", "unit", "system", "device"],
        "plant": ["plant", "system", "unit", "treatment"],
        "industrial": ["industrial", "commercial", "business"],
        "domestic": ["domestic", "home", "household", "residential"],
        "purifier": ["purifier", "filter", "filtration", "purification"],
        "lph": ["lph", "liters", "per", "hour", "capacity"],
        "gpd": ["gpd", "gallons", "per", "day"]
    }
    
    normalized = set()
    for word in keywords:
        matched = False
        for key, group in synonyms.items():
            if word in group:
                normalized.add(key)
                matched = True
                break
        if not matched:
            normalized.add(word)
    return list(normalized)

def enhanced_product_filtering(user_input):
    """Improved product filtering with better matching logic"""
    keywords = normalize_keywords(extract_keywords(user_input))
    filtered_df = df.copy()
    
    # Apply price filter if specified
    max_price = user_context["user_preferences"].get("max_price")
    if max_price:
        filtered_df = filtered_df[filtered_df["Regular_price"] <= max_price]
    
    # Category-based filtering
    category_keywords = {
        "industrial": ["industrial", "commercial"],
        "domestic": ["domestic", "home"],
        "atm": ["atm", "vending", "coin"],
        "softener": ["softener"]
    }
    
    for category, cat_keywords in category_keywords.items():
        if any(kw in keywords for kw in cat_keywords):
            category_filter = "|".join(cat_keywords)
            filtered_df = filtered_df[
                filtered_df["Category"].str.contains(category_filter, case=False, na=False) |
                filtered_df["Name"].str.contains(category_filter, case=False, na=False)
            ]
    
    # Technology-based filtering (RO, UV, UF)
    tech_keywords = ["ro", "uv", "uf"]
    tech_present = [kw for kw in tech_keywords if kw in keywords]
    if tech_present:
        tech_pattern = "|".join(tech_present)
        filtered_df = filtered_df[
            filtered_df["Name"].str.contains(tech_pattern, case=False, na=False) |
            filtered_df["Short description"].str.contains(tech_pattern, case=False, na=False)
        ]
    
    # General keyword matching
    if keywords:
        keyword_pattern = "|".join(re.escape(kw) for kw in keywords)
        filtered_df = filtered_df[
            filtered_df["Name"].str.contains(keyword_pattern, case=False, na=False) |
            filtered_df["Short description"].str.contains(keyword_pattern, case=False, na=False) |
            filtered_df["Category"].str.contains(keyword_pattern, case=False, na=False) |
            filtered_df["Description"].str.contains(keyword_pattern, case=False, na=False)
        ]
    
    return filtered_df

def create_detailed_product_info(row):
    """Create comprehensive product information with all available details"""
    name = row.get("Name", "")
    price = row.get("Regular_price", 0)
    category = row.get("Category", "")
    short_desc = row.get("Short description", "")
    full_desc = row.get("Description", "")
    attributes = row.get("Attribute 1 value(s)", "")
    
    # Extract technical specifications from description
    specs = []
    if "lph" in full_desc.lower() or "liters per hour" in full_desc.lower():
        lph_match = re.search(r"(\d+)\s*lph|\b(\d+)\s*liters?\s*per\s*hour", full_desc.lower())
        if lph_match:
            lph_value = lph_match.group(1) or lph_match.group(2)
            specs.append(f"Flow Rate: {lph_value} LPH")
    
    if "gpd" in full_desc.lower():
        gpd_match = re.search(r"(\d+)\s*gpd", full_desc.lower())
        if gpd_match:
            specs.append(f"Capacity: {gpd_match.group(1)} GPD")
    
    product_info = f"""
🏷️ PRODUCT: {name}
💰 PRICE: ₹{price:,}
📂 CATEGORY: {category}
⚡ KEY FEATURES: {short_desc}

🔧 TECHNICAL SPECIFICATIONS:
{chr(10).join(f"• {spec}" for spec in specs) if specs else "• Detailed specs available in full description"}

📋 DETAILED DESCRIPTION:
{full_desc[:500]}{'...' if len(full_desc) > 500 else ''}

🎛️ AVAILABLE VARIANTS: {attributes if attributes else 'Standard model'}

✅ SUITABLE FOR: {category.split('>')[0].strip() if '>' in category else category}
    """.strip()
    
    return product_info

def analyze_conversation_context():
    """Analyze conversation context to avoid repetitive questions"""
    context = []
    
    if user_context["shown_products"]:
        context.append(f"Already shown {len(user_context['shown_products'])} products")
    
    if user_context["user_preferences"]:
        prefs = ", ".join(f"{k}: {v}" for k, v in user_context["user_preferences"].items())
        context.append(f"User preferences: {prefs}")
    
    if user_context["asked_questions"]:
        context.append(f"Previously asked about: {', '.join(list(user_context['asked_questions'])[:3])}")
    
    return " | ".join(context) if context else "Fresh conversation"

def handle_input(user_input):
    global conversation_history, user_context
    
    try:
        # Update conversation context
        user_context["asked_questions"].add(user_input.lower()[:50])  # Track question patterns
        
        # Enhanced product filtering
        filtered_df = enhanced_product_filtering(user_input)
        
        # Get top relevant products (limit to avoid overwhelming)
        top_products = filtered_df.head(8) if not filtered_df.empty else df.head(5)
        
        # Create detailed product documents
        docs = []
        for _, row in top_products.iterrows():
            product_name = row.get("Name", "")
            user_context["shown_products"].add(product_name)
            
            detailed_info = create_detailed_product_info(row)
            docs.append(Document(page_content=detailed_info))
        
        # Fallback to vector search if no direct matches
        if filtered_df.empty:
            vector_docs = retriever.get_relevant_documents(user_input)
            docs.extend(vector_docs[:3])  # Limit vector results
        
        if not docs:
            gui.display_reply("❌ Sorry, I couldn't find any matching water treatment products. Could you try describing what type of system you need?")
            return
        
        # Prepare context for LLM
        product_info = "\n\n".join(d.page_content for d in docs[:5])  # Limit to top 5 products
        recent_history = "\n".join(conversation_history[-6:])  # Last 3 exchanges
        context_analysis = analyze_conversation_context()
        
        payload = {
            "history": recent_history,
            "question": user_input,
            "info": product_info,
            "context_analysis": context_analysis
        }
        
        response = chain.invoke(payload)
        final_response = response.content.strip()
        
        # Display response
        gui.display_reply(final_response)
        
        # Handle image display for single product matches
        if len(docs) == 1 and hasattr(docs[0], 'page_content'):
            try:
                # Extract product name from detailed info
                product_name_match = re.search(r"🏷️ PRODUCT: (.+)", docs[0].page_content)
                if product_name_match:
                    product_name = product_name_match.group(1).strip()
                    product_row = df[df["Name"].str.lower() == product_name.lower()]
                    if not product_row.empty:
                        image_url = product_row.iloc[0].get("Images", "")
                        if image_url and image_url.startswith("http"):
                            # Take first image if multiple URLs
                            first_image = image_url.split(",")[0].strip()
                            gui.display_image(first_image)
            except Exception as img_error:
                print(f"[DEBUG] Image display error: {img_error}")
        
        # Update conversation history
        conversation_history.append(f"User: {user_input}")
        conversation_history.append(f"Bot: {final_response[:200]}...")  # Truncate for memory
        
        # Keep conversation history manageable
        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]
    
    except Exception as e:
        gui.display_reply(f"❌ Error occurred: {str(e)}")
        print(f"[DEBUG] Error in handle_input: {e}")

gui = ChatGUI(on_submit=handle_input)
gui.run()
