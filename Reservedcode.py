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

def is_product_inquiry(user_input):
    """Check if the user is asking about products or wants to buy something"""
    product_keywords = [
        "show me", "i need", "i want", "looking for", "recommend", "suggest",
        "buy", "purchase", "price", "cost", "system", "purifier", "filter",
        "ro", "uv", "uf", "plant", "machine", "softener", "products"
    ]
    
    query_lower = user_input.lower()
    return any(keyword in query_lower for keyword in product_keywords)

def extract_user_requirements(user_input):
    """Extract user requirements from their input"""
    query_lower = user_input.lower()
    needs = user_context["needs_assessment"]
    
    # Extract usage type
    if any(word in query_lower for word in ["home", "house", "family", "domestic", "residential"]):
        needs["usage_type"] = "domestic"
    elif any(word in query_lower for word in ["office", "commercial", "business", "company"]):
        needs["usage_type"] = "commercial"
    elif any(word in query_lower for word in ["factory", "industrial", "plant", "manufacturing"]):
        needs["usage_type"] = "industrial"
    
    # Extract capacity hints
    if any(word in query_lower for word in ["small family", "2-3 people", "few people"]):
        needs["capacity_needed"] = "small"
    elif any(word in query_lower for word in ["large family", "5-6 people", "big family"]):
        needs["capacity_needed"] = "large"
    elif any(word in query_lower for word in ["office", "50 people", "100 people"]):
        needs["capacity_needed"] = "office"
    
    # Extract budget information
    budget_match = re.search(r"budget.*?(\d{4,6})|under.*?(\d{4,6})|below.*?(\d{4,6})", query_lower)
    if budget_match:
        budget_value = next(filter(None, budget_match.groups()))
        needs["budget_range"] = int(budget_value)
    
    # Extract water source
    if any(word in query_lower for word in ["borewell", "bore well", "groundwater", "well water"]):
        needs["water_source"] = "borewell"
    elif any(word in query_lower for word in ["municipal", "corporation", "tap water"]):
        needs["water_source"] = "municipal"
    elif any(word in query_lower for word in ["tanker", "tank water", "delivered water"]):
        needs["water_source"] = "tanker"
    
    # Extract specific concerns
    concerns = []
    if any(word in query_lower for word in ["taste", "bad taste", "bitter"]):
        concerns.append("taste issues")
    if any(word in query_lower for word in ["hard water", "scale", "soap"]):
        concerns.append("water hardness")
    if any(word in query_lower for word in ["high tds", "tds", "dissolved solids"]):
        concerns.append("high TDS")
    if any(word in query_lower for word in ["bacteria", "contamination", "infection"]):
        concerns.append("bacterial contamination")
    if any(word in query_lower for word in ["chlorine", "chemical smell"]):
        concerns.append("chlorine/chemicals")
    
    if concerns:
        needs["specific_concerns"].extend(concerns)
        needs["specific_concerns"] = list(set(needs["specific_concerns"]))  # Remove duplicates

def get_needs_assessment_questions():
    """Generate questions to understand customer needs"""
    needs = user_context["needs_assessment"]
    questions = []
    
    if not needs["usage_type"]:
        questions.append("🏠 **Where will you be using this water treatment system?** (Home, Office, or Industrial facility)")
    
    if not needs["capacity_needed"] and needs["usage_type"]:
        if needs["usage_type"] == "domestic":
            questions.append("👨‍👩‍👧‍👦 **How many people will be using the system?** (Family size helps determine capacity)")
        elif needs["usage_type"] == "commercial":
            questions.append("🏢 **How many people work in your office?** (This helps determine daily water requirement)")
        elif needs["usage_type"] == "industrial":
            questions.append("🏭 **What's your daily water requirement?** (In liters per hour or per day)")
    
    if not needs["water_source"]:
        questions.append("🚰 **What's your water source?** (Municipal supply, Borewell, or Tanker water)")
    
    if not needs["budget_range"]:
        questions.append("💰 **What's your budget range?** (This helps me recommend the best system for your needs)")
    
    if not needs["specific_concerns"]:
        questions.append("⚠️ **Any specific water quality issues?** (Bad taste, hardness, high TDS, contamination concerns)")
    
    return questions[:2]  # Limit to 2 questions at a time to avoid overwhelming

def check_if_requirements_sufficient():
    """Check if we have enough information to make recommendations"""
    needs = user_context["needs_assessment"]
    essential_info = [
        needs["usage_type"] is not None,
        needs["capacity_needed"] is not None or needs["usage_type"] == "industrial",
        needs["water_source"] is not None or len(needs["specific_concerns"]) > 0
    ]
    
    return sum(essential_info) >= 2  # Need at least 2 essential pieces of information

def generate_needs_assessment_response():
    """Generate response for needs assessment"""
    questions = get_needs_assessment_questions()
    
    if not questions:
        user_context["needs_assessment"]["requirements_gathered"] = True
        return None
    
    response = """🔍 **To recommend the perfect water treatment system for you, I'd like to understand your needs better:**

"""
    
    for i, question in enumerate(questions, 1):
        response += f"{question}\n\n"
    
    response += "💡 This information helps me suggest the most suitable and cost-effective solution for your specific requirements!"
    
    return response

def is_greeting(user_input):
    """Check if the user input is a greeting"""
    greeting_keywords = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "greetings", "howdy", "what's up", "whats up", "sup", "hiya", "hola"
    ]
    
    query_lower = user_input.lower().strip()
    
    # Check for exact greetings or greetings with punctuation
    if query_lower in greeting_keywords:
        return True
    
    # Check for greetings with punctuation
    clean_query = ''.join(char for char in query_lower if char.isalpha() or char.isspace()).strip()
    if clean_query in greeting_keywords:
        return True
    
    # Check if query starts with a greeting
    for greeting in greeting_keywords:
        if query_lower.startswith(greeting):
            return True
    
    return False

def get_greeting_response():
    """Generate a friendly greeting response"""
    import random
    
    responses = [
        """👋 Hello there! Great to meet you!

I'm your Water Treatment Expert, and I'm here to help you with:

🔹 **Product Recommendations**: Find the perfect RO, UV, or UF system
🔹 **Water Education**: Learn about alkaline water, TDS, pH levels
🔹 **Technology Guidance**: Understand different purification methods
🔹 **Health Benefits**: Discover how water quality affects your health

What would you like to know about water treatment today? 😊""",

        """👋 Hi! Welcome to your personal water treatment consultation!

I'm excited to help you with:

💧 **Smart Product Matching**: Get systems tailored to your needs
🧪 **Water Science Made Simple**: Easy explanations of complex topics  
⚡ **Technology Comparisons**: RO vs UV vs UF - what's best for you?
🏠 **Custom Solutions**: For home, office, or industrial use

How can I assist you with your water treatment needs? 🌟""",

        """👋 Hello! I'm delighted you're here!

As your Water Treatment Specialist, I can help with:

🎯 **Perfect System Selection**: Find exactly what you need
📚 **Educational Insights**: Learn about water quality and health
🔧 **Technical Support**: Understand specifications and features
💡 **Expert Advice**: Get professional recommendations

What brings you here today? I'm ready to help! ✨"""
    ]
    
    return random.choice(responses)

prompt = ChatPromptTemplate.from_template("""
You are a knowledgeable water treatment systems consultant and educator. You follow a consultative approach - understanding customer needs before recommending products.

IMPORTANT RULES:
1. NEVER ask repetitive questions if the customer has already provided information
2. If products are shown, they are already filtered based on customer requirements
3. Provide detailed specifications and technical details for recommended products
4. Answer educational questions about water treatment, health benefits, and water science
5. Focus on being informative and educational while remaining conversational
6. If asked about water benefits, alkaline water, TDS, pH, etc., provide comprehensive information
7. Do NOT handle greetings or goodbyes - they are handled separately
8. When recommending products, explain WHY they suit the customer's specific needs

📝 Recommended Products (Pre-filtered based on customer needs):
{info}

🎓 Water Education Content:
{education_info}

🗣️ Customer Question:
{question}

💬 Previous Conversation:
{history}

🎯 Context Analysis:
{context_analysis}

RESPONSE GUIDELINES FOR PRODUCT RECOMMENDATIONS:
- Products listed above are already filtered for the customer's specific requirements
- Explain HOW each product addresses their particular needs (usage type, capacity, budget, concerns)
- Include technical specifications, features, capacity, price, and benefits
- Mention why this product is suitable for their specific situation
- Compare products if multiple options are available, highlighting differences
- Include installation requirements, maintenance, and warranty information
- Prioritize products that best match their stated requirements

RESPONSE GUIDELINES FOR EDUCATION:
- If educational content is provided above, use it to answer water-related questions
- Provide comprehensive, accurate information about water science
- Connect educational content to practical applications when relevant

GENERAL GUIDELINES:
- Be helpful, informative, and professional
- Avoid repetitive content or questions
- Balance product recommendations with educational information as appropriate
- Focus on customer value and solving their specific problems
""")

chain = prompt | model
conversation_history = []
user_context = {
    "asked_questions": set(),
    "shown_products": set(),
    "user_preferences": {},
    "current_filter": None,
    "educational_topics_covered": set(),
    "has_greeted": False,
    "needs_assessment": {
        "usage_type": None,  # domestic, commercial, industrial
        "capacity_needed": None,  # family size, office size, etc.
        "budget_range": None,
        "water_source": None,  # municipal, borewell, tanker
        "specific_concerns": [],  # taste, hardness, tds, bacteria
        "location": None,  # home, office, factory
        "current_system": None,  # existing system if any
        "requirements_gathered": False
    }
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
    """Improved product filtering based on user requirements and query"""
    keywords = normalize_keywords(extract_keywords(user_input))
    filtered_df = df.copy()
    needs = user_context["needs_assessment"]
    
    # Apply budget filter from needs assessment or query
    budget_limit = needs.get("budget_range") or user_context["user_preferences"].get("max_price")
    if budget_limit:
        filtered_df = filtered_df[filtered_df["Regular_price"] <= budget_limit]
    
    # Filter by usage type from needs assessment
    if needs["usage_type"]:
        if needs["usage_type"] == "domestic":
            domestic_filter = "domestic|home|residential"
            filtered_df = filtered_df[
                filtered_df["Category"].str.contains(domestic_filter, case=False, na=False)
            ]
        elif needs["usage_type"] == "commercial":
            commercial_filter = "commercial|office|business"
            filtered_df = filtered_df[
                filtered_df["Category"].str.contains(commercial_filter, case=False, na=False) |
                filtered_df["Name"].str.contains(commercial_filter, case=False, na=False)
            ]
        elif needs["usage_type"] == "industrial":
            industrial_filter = "industrial|plant"
            filtered_df = filtered_df[
                filtered_df["Category"].str.contains(industrial_filter, case=False, na=False)
            ]
    
    # Filter by specific concerns
    if needs["specific_concerns"]:
        concern_filters = []
        for concern in needs["specific_concerns"]:
            if "hardness" in concern:
                concern_filters.append("softener|softner")
            elif "high TDS" in concern:
                concern_filters.append("ro|reverse osmosis")
            elif "bacterial" in concern:
                concern_filters.append("uv|uv")
            elif "chlorine" in concern:
                concern_filters.append("carbon|activated")
        
        if concern_filters:
            concern_pattern = "|".join(concern_filters)
            filtered_df = filtered_df[
                filtered_df["Name"].str.contains(concern_pattern, case=False, na=False) |
                filtered_df["Short description"].str.contains(concern_pattern, case=False, na=False)
            ]
    
    # Filter by water source considerations
    if needs["water_source"]:
        if needs["water_source"] == "borewell":
            # Borewell water typically needs RO due to high TDS
            filtered_df = filtered_df[
                filtered_df["Name"].str.contains("ro", case=False, na=False) |
                filtered_df["Short description"].str.contains("ro", case=False, na=False)
            ]
        elif needs["water_source"] == "municipal":
            # Municipal water typically needs UV/UF for bacterial protection
            filtered_df = filtered_df[
                filtered_df["Name"].str.contains("uv|uf", case=False, na=False) |
                filtered_df["Short description"].str.contains("uv|uf", case=False, na=False)
            ]
    
    # Apply capacity-based filtering
    if needs["capacity_needed"]:
        if needs["capacity_needed"] == "small":
            # Small capacity systems (typically under 15 LPH)
            small_systems = filtered_df[
                filtered_df["Description"].str.contains("12|15|10", case=False, na=False) |
                filtered_df["Name"].str.contains("domestic|home", case=False, na=False)
            ]
            if not small_systems.empty:
                filtered_df = small_systems
        elif needs["capacity_needed"] == "large":
            # Large capacity systems
            large_systems = filtered_df[
                filtered_df["Description"].str.contains("20|25|30", case=False, na=False) |
                filtered_df["Name"].str.contains("premium|advance", case=False, na=False)
            ]
            if not large_systems.empty:
                filtered_df = large_systems
        elif needs["capacity_needed"] == "office":
            # Office/commercial systems
            office_systems = filtered_df[
                filtered_df["Category"].str.contains("commercial", case=False, na=False) |
                filtered_df["Description"].str.contains("office|commercial", case=False, na=False)
            ]
            if not office_systems.empty:
                filtered_df = office_systems
    
    # Apply keyword-based filtering from current query
    if keywords:
        tech_keywords = ["ro", "uv", "uf"]
        tech_present = [kw for kw in tech_keywords if kw in keywords]
        if tech_present:
            tech_pattern = "|".join(tech_present)
            tech_filtered = filtered_df[
                filtered_df["Name"].str.contains(tech_pattern, case=False, na=False) |
                filtered_df["Short description"].str.contains(tech_pattern, case=False, na=False)
            ]
            if not tech_filtered.empty:
                filtered_df = tech_filtered
        
        # General keyword matching
        keyword_pattern = "|".join(re.escape(kw) for kw in keywords)
        keyword_filtered = filtered_df[
            filtered_df["Name"].str.contains(keyword_pattern, case=False, na=False) |
            filtered_df["Short description"].str.contains(keyword_pattern, case=False, na=False) |
            filtered_df["Category"].str.contains(keyword_pattern, case=False, na=False) |
            filtered_df["Description"].str.contains(keyword_pattern, case=False, na=False)
        ]
        if not keyword_filtered.empty:
            filtered_df = keyword_filtered
    
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
    needs = user_context["needs_assessment"]
    
    # Include needs assessment status
    if needs["requirements_gathered"]:
        requirements = []
        if needs["usage_type"]:
            requirements.append(f"Usage: {needs['usage_type']}")
        if needs["capacity_needed"]:
            requirements.append(f"Capacity: {needs['capacity_needed']}")
        if needs["budget_range"]:
            requirements.append(f"Budget: ₹{needs['budget_range']:,}")
        if needs["water_source"]:
            requirements.append(f"Source: {needs['water_source']}")
        if needs["specific_concerns"]:
            requirements.append(f"Concerns: {', '.join(needs['specific_concerns'][:2])}")
        
        if requirements:
            context.append(f"Customer requirements: {' | '.join(requirements[:3])}")
    else:
        context.append("Requirements being assessed")
    
    if user_context["shown_products"]:
        context.append(f"Products shown: {len(user_context['shown_products'])}")
    
    if user_context["educational_topics_covered"]:
        context.append(f"Education provided: {', '.join(list(user_context['educational_topics_covered'])[:2])}")
    
    return " | ".join(context) if context else "Fresh conversation"

def handle_input(user_input):
    global conversation_history, user_context
    
    try:
        # Update conversation context
        user_context["asked_questions"].add(user_input.lower()[:50])
        
        # Handle greetings first - no product suggestions
        if is_greeting(user_input):
            greeting_response = get_greeting_response()
            gui.display_reply(greeting_response)
            
            # Mark that greeting has occurred
            user_context["has_greeted"] = True
            
            # Update conversation history
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Bot: [Greeting response]")
            return
        
        # Check for goodbye/farewell
        goodbye_keywords = ["bye", "goodbye", "see you", "farewell", "thanks", "thank you", "that's all", "thats all"]
        if any(keyword in user_input.lower() for keyword in goodbye_keywords):
            farewell_response = """👋 Thank you for using our Water Treatment Assistant! 

I hope I was able to help you with your water treatment needs. If you have any more questions about products, water science, or anything else related to water treatment, feel free to ask anytime.

Have a great day and stay hydrated! 💧✨"""
            gui.display_reply(farewell_response)
            
            # Update conversation history
            conversation_history.append(f"User: {user_input}")
            conversation_history.append(f"Bot: [Farewell response]")
            return
        
        # Always extract user requirements from their input
        extract_user_requirements(user_input)
        
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
        
        # Check if this is a product inquiry
        is_product_request = is_product_inquiry(user_input)
        
        # If it's a product inquiry but we don't have enough requirements, ask for needs assessment
        if is_product_request and not user_context["needs_assessment"]["requirements_gathered"]:
            if not check_if_requirements_sufficient():
                needs_response = generate_needs_assessment_response()
                if needs_response:
                    gui.display_reply(needs_response)
                    
                    # Update conversation history
                    conversation_history.append(f"User: {user_input}")
                    conversation_history.append(f"Bot: [Needs assessment questions]")
                    return
            else:
                # Mark requirements as gathered if we have sufficient information
                user_context["needs_assessment"]["requirements_gathered"] = True
        
        # Enhanced product filtering (only if requirements are gathered or it's educational)
        product_info = ""
        docs = []
        
        if (is_product_request and user_context["needs_assessment"]["requirements_gathered"]) or \
           (not is_educational and not is_product_request and user_context["needs_assessment"]["requirements_gathered"]):
            filtered_df = enhanced_product_filtering(user_input)
            
            # Get top relevant products (limit to avoid overwhelming)
            top_products = filtered_df.head(5) if not filtered_df.empty else df.head(3)
            
            # Create detailed product documents
            for _, row in top_products.iterrows():
                product_name = row.get("Name", "")
                user_context["shown_products"].add(product_name)
                
                detailed_info = create_detailed_product_info(row)
                docs.append(Document(page_content=detailed_info))
            
            # Fallback to vector search if no direct matches
            if filtered_df.empty:
                vector_docs = retriever.get_relevant_documents(user_input)
                docs.extend(vector_docs[:2])
            
            if docs:
                product_info = "\n\n".join(d.page_content for d in docs[:3])
        
        # If it's a product request but no products found and no educational content
        if is_product_request and not docs and not education_info and user_context["needs_assessment"]["requirements_gathered"]:
            gui.display_reply("❌ I couldn't find specific products matching your requirements. Could you provide more details about what you're looking for? I can help you find the right water treatment solution.")
            return
        
        # If no relevant response can be generated
        if not docs and not education_info and not is_product_request:
            gui.display_reply("💬 I'm here to help with water treatment systems and water quality questions. You can ask me about:\n\n• Product recommendations (just tell me your needs first)\n• Water science (alkaline water, TDS, pH levels)\n• Technology comparisons (RO vs UV vs UF)\n• Water quality issues and solutions\n\nWhat would you like to know?")
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
