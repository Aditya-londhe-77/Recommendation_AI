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

# Water Education Knowledge Base
WATER_KNOWLEDGE_BASE = {
    "alkaline_water": {
        "title": "💧 Alkaline Water Benefits",
        "content": """
🌟 ALKALINE WATER BENEFITS:

📊 pH Level: 8.5-9.5 (vs regular water 6.5-7.5)

🔬 HEALTH BENEFITS:
• Better Hydration: Smaller molecular clusters for easier absorption
• Antioxidant Properties: Helps neutralize free radicals in the body
• pH Balance: May help balance body's acidic levels
• Improved Metabolism: Enhanced nutrient absorption
• Detoxification: Assists in flushing out toxins
• Bone Health: May help reduce bone loss

⚡ HOW IT WORKS:
Alkaline water is created through ionization process that increases pH and adds beneficial minerals like calcium, magnesium, and potassium.

🥤 RECOMMENDED CONSUMPTION:
• Start with 1-2 glasses daily
• Gradually increase to 6-8 glasses
• Best consumed 30 minutes before meals

⚠️ CONSIDERATIONS:
• Not recommended for people with kidney disease
• Consult doctor if on medication
• Natural alkaline sources preferred over artificial

💡 SOURCES:
• Natural spring water from alkaline rocks
• Alkaline water ionizers (our RO+Alkaline systems)
• Adding alkaline minerals to filtered water
        """
    },
    "tds_information": {
        "title": "🧪 TDS (Total Dissolved Solids) Guide",
        "content": """
📋 TDS EXPLANATION:

🔍 WHAT IS TDS?
Total Dissolved Solids - measure of dissolved minerals, salts, and metals in water (measured in ppm - parts per million)

📊 TDS LEVELS GUIDE:
• 0-50 ppm: Excellent (may lack essential minerals)
• 50-150 ppm: Good (ideal for drinking)
• 150-300 ppm: Fair (acceptable)
• 300-500 ppm: Poor (needs treatment)
• 500+ ppm: Unacceptable (requires purification)

⚖️ BENEFITS OF OPTIMAL TDS:
• 50-150 ppm provides essential minerals
• Calcium for bone health
• Magnesium for heart function
• Potassium for muscle function
• Trace minerals for overall health

🚰 TDS ADJUSTMENT:
• RO reduces TDS significantly (may go too low)
• TDS Controller maintains essential minerals
• Mineralizer adds back beneficial minerals
• UV/UF preserves natural TDS levels

💡 OUR SOLUTIONS:
• RO + TDS Controller systems
• Alkaline + Mineral cartridges
• Smart TDS monitoring systems
        """
    },
    "ro_vs_uv_uf": {
        "title": "🔬 RO vs UV vs UF Technology Comparison",
        "content": """
⚡ WATER PURIFICATION TECHNOLOGIES:

🌊 REVERSE OSMOSIS (RO):
✅ Removes: Heavy metals, chemicals, salts, bacteria, viruses
✅ TDS Reduction: 80-95%
✅ Best For: High TDS water, chemical contamination
❌ Cons: Removes beneficial minerals, water wastage

🔆 ULTRAVIOLET (UV):
✅ Removes: Bacteria, viruses, microorganisms
✅ No Chemical Addition: Chemical-free purification
✅ Retains: All minerals and TDS
❌ Cons: Doesn't remove chemicals or heavy metals

🧽 ULTRAFILTRATION (UF):
✅ Removes: Bacteria, cysts, suspended particles
✅ Retains: Essential minerals and salts
✅ No Electricity: Gravity-based operation possible
❌ Cons: Doesn't remove dissolved salts or chemicals

🏆 BEST COMBINATIONS:
• RO + UV + UF: Complete protection (our premium systems)
• UV + UF: For low TDS water sources
• RO + Mineralizer: RO benefits with mineral retention
• Pre-filter + UV: Basic protection for clean sources

💧 CHOOSING RIGHT TECHNOLOGY:
• High TDS (>300): RO-based systems
• Low TDS + bacterial risk: UV + UF
• Chemical contamination: RO mandatory
• Natural spring water: UV sufficient
        """
    },
    "water_ph_levels": {
        "title": "⚗️ Water pH Levels & Health Impact",
        "content": """
🔬 pH SCALE UNDERSTANDING:

📊 pH RANGE: 0-14
• 0-6.9: Acidic
• 7.0: Neutral
• 7.1-14: Alkaline/Basic

🥤 DRINKING WATER pH STANDARDS:
• WHO Standard: 6.5-8.5
• Ideal Drinking: 7.0-8.5
• Alkaline Water: 8.5-9.5
• Pure RO Water: 6.0-7.0

💊 HEALTH EFFECTS:

🔸 ACIDIC WATER (Below 6.5):
❌ May cause metallic taste
❌ Can leach metals from pipes
❌ Potential tooth enamel erosion
❌ May contribute to acid reflux

🔹 ALKALINE WATER (8.5-9.5):
✅ May help neutralize body acid
✅ Better hydration properties
✅ Antioxidant benefits
✅ Enhanced mineral absorption

⚖️ BALANCED pH BENEFITS:
• Optimal nutrient absorption
• Better taste and odor
• Safe for daily consumption
• Supports body's natural pH

🔧 pH ADJUSTMENT METHODS:
• Alkaline cartridges (increases pH)
• Mineral stones (natural alkalinization)
• Carbon filters (removes chlorine affecting pH)
• Remineralization (balances pH post-RO)

💡 OUR pH SOLUTIONS:
• Alkaline + RO systems
• pH balancing cartridges
• Mineral enhancement filters
• Smart pH monitoring systems
        """
    },
    "water_hardness": {
        "title": "💎 Water Hardness & Softening Solutions",
        "content": """
🧪 WATER HARDNESS EXPLAINED:

📏 HARDNESS LEVELS (ppm CaCO3):
• Soft: 0-75 ppm
• Moderately Hard: 75-150 ppm
• Hard: 150-300 ppm
• Very Hard: 300+ ppm

🔍 CAUSES OF HARDNESS:
• Calcium ions (Ca²⁺)
• Magnesium ions (Mg²⁺)
• Dissolved from limestone, chalk, gypsum

⚠️ HARD WATER PROBLEMS:
• Scale buildup in pipes and appliances
• Soap scum and reduced lathering
• Dry skin and hair issues
• Increased detergent consumption
• Reduced appliance lifespan

💧 WATER SOFTENING METHODS:

🧂 ION EXCHANGE SOFTENERS:
✅ Removes calcium and magnesium
✅ Replaces with sodium ions
✅ Complete hardness removal
❌ Adds sodium to water

🔄 SALT-FREE CONDITIONERS:
✅ Changes mineral structure
✅ Reduces scale formation
✅ No sodium addition
❌ Doesn't remove hardness completely

🌊 REVERSE OSMOSIS:
✅ Removes hardness minerals
✅ Also removes other contaminants
✅ Produces soft, pure water
❌ Removes beneficial minerals too

💡 OUR SOFTENING SOLUTIONS:
• Automatic water softeners
• Salt-free water conditioners
• RO with mineral retention
• Combo softener + purifier systems

🏠 APPLICATIONS:
• Whole house softening systems
• Point-of-use softeners
• Commercial softening plants
• Industrial water treatment
        """
    },
    "chlorine_removal": {
        "title": "🧪 Chlorine in Water & Removal Methods",
        "content": """
☢️ CHLORINE IN DRINKING WATER:

🔬 WHY CHLORINE IS ADDED:
• Disinfects water supply
• Kills bacteria and viruses
• Prevents waterborne diseases
• Maintains water safety in distribution

⚠️ CHLORINE SIDE EFFECTS:
• Taste and odor issues
• Skin and eye irritation
• May form harmful byproducts (THMs)
• Can affect beneficial gut bacteria
• Potential respiratory irritation

📊 SAFE CHLORINE LEVELS:
• WHO Standard: Up to 5 ppm
• Typical Municipal: 0.2-1.0 ppm
• Taste/Odor Threshold: 0.2-0.6 ppm

🔧 CHLORINE REMOVAL METHODS:

🥥 ACTIVATED CARBON:
✅ Highly effective chlorine removal
✅ Improves taste and odor
✅ Cost-effective solution
✅ Easy maintenance

💨 BOILING:
✅ Simple home method
✅ 100% effective
❌ Time-consuming
❌ Energy consumption

🌊 REVERSE OSMOSIS:
✅ Removes chlorine + other contaminants
✅ Comprehensive purification
❌ Higher cost
❌ Water wastage

🔆 UV TREATMENT:
❌ Does NOT remove chlorine
✅ Kills chlorine-resistant organisms
✅ Works well with carbon pre-filter

💡 OUR CHLORINE SOLUTIONS:
• Multi-stage carbon filters
• RO systems with carbon stages
• Whole house carbon filters
• Shower and bath filters

🏠 BEST APPLICATIONS:
• Kitchen: Under-sink carbon filters
• Whole House: POE carbon systems
• Shower: Carbon shower filters
• Drinking: Multi-stage purifiers
        """
    }
}

def get_water_education_info(query):
    """Get relevant water education information based on user query"""
    query_lower = query.lower()
    
    # Keyword mapping for different topics
    topic_keywords = {
        "alkaline_water": ["alkaline", "ph", "alkaline water", "ionized", "antioxidant"],
        "tds_information": ["tds", "total dissolved solids", "minerals", "ppm", "dissolved"],
        "ro_vs_uv_uf": ["ro vs uv", "technology", "reverse osmosis", "ultraviolet", "ultrafiltration", "difference", "comparison"],
        "water_ph_levels": ["ph level", "acidic", "basic", "ph scale", "acidity"],
        "water_hardness": ["hard water", "softener", "hardness", "scale", "calcium", "magnesium"],
        "chlorine_removal": ["chlorine", "taste", "odor", "chemical", "disinfection"]
    }
    
    # Find matching topics
    matching_topics = []
    for topic, keywords in topic_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            matching_topics.append(topic)
    
    # Return relevant information
    if matching_topics:
        info_parts = []
        for topic in matching_topics[:2]:  # Limit to 2 topics to avoid overwhelming
            topic_info = WATER_KNOWLEDGE_BASE.get(topic, {})
            if topic_info:
                info_parts.append(f"{topic_info['title']}\n{topic_info['content']}")
        return "\n\n" + "="*50 + "\n\n".join(info_parts)
    
    return None

def is_educational_query(user_input):
    """Check if the query is asking for educational information about water"""
    educational_keywords = [
        "what is", "benefits of", "advantage", "disadvantage", "how does", "why",
        "explain", "difference", "comparison", "help", "information", "tell me about",
        "alkaline", "ph", "tds", "hardness", "chlorine", "purification", "filtration"
    ]
    
    query_lower = user_input.lower()
    return any(keyword in query_lower for keyword in educational_keywords)

prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable water treatment systems expert and educator. Help customers with both product recommendations and water education.

IMPORTANT RULES:
1. NEVER ask repetitive questions if the customer has already provided information
2. If you have already shown products, don't ask the same questions again
3. Provide detailed specifications and technical details for recommended products
4. Answer educational questions about water treatment, health benefits, and water science
5. Focus on being informative and educational while remaining conversational
6. If asked about water benefits, alkaline water, TDS, pH, etc., provide comprehensive information

📝 Available Products:
{info}

🎓 Water Education Content:
{education_info}

🗣️ Customer Question:
{question}

💬 Previous Conversation:
{history}

🎯 Context Analysis:
{context_analysis}

RESPONSE GUIDELINES:
- If products are listed above, provide detailed information about them
- If educational content is provided above, use it to answer water-related questions
- Include technical specifications, features, capacity, price, and benefits for products
- For educational queries, provide comprehensive, accurate information about water science
- Mention installation requirements, maintenance, and warranty if relevant for products
- Compare products if multiple options are available
- Only ask clarifying questions if absolutely necessary and not asked before
- Avoid repetitive greetings or the same questions
- If customer says goodbye, respond briefly and politely
- Balance product recommendations with educational information as appropriate
- Use the educational content provided to give detailed explanations about water benefits
""")

chain = prompt | model
conversation_history = []
user_context = {
    "asked_questions": set(),
    "shown_products": set(),
    "user_preferences": {},
    "current_filter": None,
    "educational_topics_covered": set()
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
    
    if user_context["educational_topics_covered"]:
        context.append(f"Educational topics covered: {', '.join(list(user_context['educational_topics_covered'])[:3])}")
    
    return " | ".join(context) if context else "Fresh conversation"

def handle_input(user_input):
    global conversation_history, user_context
    
    try:
        # Update conversation context
        user_context["asked_questions"].add(user_input.lower()[:50])
        
        # Check if this is an educational query
        is_educational = is_educational_query(user_input)
        education_info = ""
        
        if is_educational:
            education_content = get_water_education_info(user_input)
            if education_content:
                education_info = education_content
                # Track educational topics covered
                for topic in WATER_KNOWLEDGE_BASE.keys():
                    if any(keyword in user_input.lower() for keyword in topic.split('_')):
                        user_context["educational_topics_covered"].add(topic)
        
        # Enhanced product filtering (only if not purely educational)
        product_info = ""
        docs = []
        
        if not is_educational or any(keyword in user_input.lower() for keyword in ["system", "purifier", "recommend", "buy", "price"]):
            filtered_df = enhanced_product_filtering(user_input)
            
            # Get top relevant products (limit to avoid overwhelming)
            top_products = filtered_df.head(5) if not filtered_df.empty else df.head(3)
            
            # Create detailed product documents
            for _, row in top_products.iterrows():
                product_name = row.get("Name", "")
                user_context["shown_products"].add(product_name)
                
                detailed_info = create_detailed_product_info(row)
                docs.append(Document(page_content=detailed_info))
            
            # Fallback to vector search if no direct matches and not purely educational
            if filtered_df.empty and not is_educational:
                vector_docs = retriever.get_relevant_documents(user_input)
                docs.extend(vector_docs[:2])
            
            if docs:
                product_info = "\n\n".join(d.page_content for d in docs[:3])
        
        # If no products found and no educational content, provide helpful message
        if not docs and not education_info:
            gui.display_reply("❌ I couldn't find specific products for your query. Could you try asking about specific water treatment systems or water-related topics? I can help with product recommendations and explain water treatment benefits.")
            return
        
        # Prepare context for LLM
        recent_history = "\n".join(conversation_history[-6:])
        context_analysis = analyze_conversation_context()
        
        payload = {
            "history": recent_history,
            "question": user_input,
            "info": product_info,
            "education_info": education_info,
            "context_analysis": context_analysis
        }
        
        response = chain.invoke(payload)
        final_response = response.content.strip()
        
        # Display response
        gui.display_reply(final_response)
        
        # Handle image display for single product matches
        if len(docs) == 1 and hasattr(docs[0], 'page_content'):
            try:
                product_name_match = re.search(r"🏷️ PRODUCT: (.+)", docs[0].page_content)
                if product_name_match:
                    product_name = product_name_match.group(1).strip()
                    product_row = df[df["Name"].str.lower() == product_name.lower()]
                    if not product_row.empty:
                        image_url = product_row.iloc[0].get("Images", "")
                        if image_url and image_url.startswith("http"):
                            first_image = image_url.split(",")[0].strip()
                            gui.display_image(first_image)
            except Exception as img_error:
                print(f"[DEBUG] Image display error: {img_error}")
        
        # Update conversation history
        conversation_history.append(f"User: {user_input}")
        conversation_history.append(f"Bot: {final_response[:200]}...")
        
        # Keep conversation history manageable
        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]
    
    except Exception as e:
        gui.display_reply(f"❌ Error occurred: {str(e)}")
        print(f"[DEBUG] Error in handle_input: {e}")

gui = ChatGUI(on_submit=handle_input)
gui.run()
