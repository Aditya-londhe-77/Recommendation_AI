# 💧 Water Treatment Systems Chatbot

An intelligent chatbot for water treatment product recommendations using advanced filtering and conversation management.

## 🚀 Key Improvements Made

### 1. **Enhanced Product Filtering**
- **Smart Keyword Extraction**: Improved recognition of water treatment terminology (RO, UV, UF, LPH, GPD)
- **Multi-criteria Filtering**: Filters by technology type, category, price range, and capacity
- **Synonym Handling**: Understands variations like "reverse osmosis" = "RO", "softener" = "water softener"
- **Price Range Support**: Recognizes queries like "under ₹50000" or "below ₹30000"

### 2. **Conversation State Management**
- **Anti-repetition System**: Tracks previously asked questions to avoid repetitive interactions
- **Context Awareness**: Remembers what products were already shown to the user
- **User Preference Tracking**: Stores user requirements (price limits, technology preferences)
- **Smart Follow-up**: Provides relevant follow-up information instead of asking same questions

### 3. **Detailed Product Information**
- **Comprehensive Specs**: Displays technical specifications, capacity, flow rates
- **Structured Format**: Uses emojis and clear formatting for better readability
- **Feature Extraction**: Automatically extracts LPH, GPD, storage capacity from descriptions
- **Variant Information**: Shows available product variants (Premium, Classic, NX, etc.)
- **Application Details**: Clearly indicates suitable applications (Industrial, Domestic, etc.)

### 4. **Improved AI Responses**
- **No Repetitive Questions**: AI focuses on providing information rather than asking same questions
- **Detailed Explanations**: Includes technical features, benefits, and specifications
- **Contextual Responses**: Considers conversation history to provide relevant information
- **Professional Tone**: Maintains expertise while being conversational

### 5. **Water Education & Knowledge Base**
- **Comprehensive Water Science**: Explains alkaline water benefits, pH levels, TDS importance
- **Technology Comparisons**: Detailed RO vs UV vs UF technology explanations
- **Health Benefits**: Information about water quality impact on health
- **Water Treatment Education**: Explains chlorine removal, water hardness, softening methods
- **Smart Topic Detection**: Automatically provides relevant educational content
- **FAQ Coverage**: Answers common water treatment questions

## 📋 Features

- **GUI Interface**: Clean, modern chat interface with dark theme
- **Image Display**: Shows product images for single product matches
- **Vector Search**: Uses advanced semantic search for product recommendations
- **Price Filtering**: Support for budget-based filtering
- **Category Intelligence**: Smart categorization (Industrial, Domestic, Commercial)
- **Technology Detection**: Recognizes RO, UV, UF technologies automatically
- **Water Education**: Comprehensive knowledge base about water science and health
- **Educational Q&A**: Answers questions about alkaline water, TDS, pH levels, water hardness
- **Smart Content Delivery**: Combines product recommendations with educational information

## 🛠️ Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install pandas langchain-huggingface langchain-chroma langchain-groq tkinter pillow requests python-dotenv
   ```

2. **Configure API Key**:
   - Get a Groq API key from [console.groq.com](https://console.groq.com/)
   - Add it to `.env` file:
     ```
     GroqAPIKey=your_actual_api_key_here
     ```

3. **Prepare Data**:
   - Ensure `new_export.csv` contains product data
   - Run vector database setup (first run will create the database)

4. **Run the Chatbot**:
   ```bash
   python Reservedcode.py
   ```

## 💡 Usage Examples

### Before Improvements:
```
User: "Show me RO systems"
Bot: "What's your budget? What capacity do you need? For home or office?"
User: "Under 50000"
Bot: "What's your budget? What capacity do you need?" [Repetitive]
```

### After Improvements:
```
User: "Show me RO systems under 50000"
Bot: "Here are RO systems under ₹50,000:

🏷️ PRODUCT: ATROS Swift
💰 PRICE: ₹25,000
📂 CATEGORY: Domestic RO Systems
⚡ KEY FEATURES: RO+UV+UF with TDS Adjuster

🔧 TECHNICAL SPECIFICATIONS:
• Flow Rate: 15 LPH
• Storage: 12-15 liters
• UV Purification
• RO Membrane included

✅ SUITABLE FOR: Home, Office, Small families"

User: "What are the benefits of alkaline water?"
Bot: "💧 Alkaline Water Benefits:

📊 pH Level: 8.5-9.5 (vs regular water 6.5-7.5)

🔬 HEALTH BENEFITS:
• Better Hydration: Smaller molecular clusters for easier absorption
• Antioxidant Properties: Helps neutralize free radicals
• pH Balance: May help balance body's acidic levels
• Improved Metabolism: Enhanced nutrient absorption
• Detoxification: Assists in flushing out toxins

⚡ HOW IT WORKS:
Alkaline water is created through ionization process that increases pH and adds beneficial minerals like calcium, magnesium, and potassium.

💡 OUR SOLUTIONS:
• RO+Alkaline systems that provide alkaline water
• Alkaline + Mineral cartridges for existing systems"
```

## 🔧 Technical Architecture

- **Frontend**: Tkinter-based GUI with modern styling
- **Backend**: LangChain + Groq LLM for intelligent responses
- **Database**: ChromaDB for vector-based product search
- **Data Processing**: Pandas for CSV handling and filtering
- **Embeddings**: Sentence Transformers for semantic search

## 📁 File Structure

- `Reservedcode.py` - Main chatbot application with improved logic
- `gui_template.py` - GUI interface components
- `vector.py` - Enhanced vector database setup
- `new_export.csv` - Product data with specifications
- `.env` - Environment variables (API keys)

## 🎯 Key Improvements Summary

1. ✅ **Better Product Filtering**: More accurate product matching
2. ✅ **No Repetitive Questions**: Smart conversation flow
3. ✅ **Detailed Specifications**: Comprehensive product information
4. ✅ **Context Awareness**: Remembers conversation history
5. ✅ **Enhanced User Experience**: Professional, informative responses
6. ✅ **Water Education**: Comprehensive knowledge base about water science
7. ✅ **Educational Q&A**: Answers water treatment and health questions

## 🧪 Educational Topics Covered

- **Alkaline Water**: Benefits, pH levels, health impact, consumption guidelines
- **TDS (Total Dissolved Solids)**: Explanation, optimal levels, health significance
- **Water Technologies**: RO vs UV vs UF comparison, pros and cons
- **pH Levels**: Understanding water acidity/alkalinity and health effects
- **Water Hardness**: Causes, problems, softening solutions
- **Chlorine Removal**: Why it's added, side effects, removal methods

The chatbot now serves as both a product recommendation system and a comprehensive water education platform, helping customers make informed decisions about water treatment solutions.