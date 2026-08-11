# 💧 AI Water Treatment Sales Chatbot

A professional AI-powered sales chatbot designed to help customers find the right water treatment solutions. The bot acts like an experienced human sales representative, providing personalized product recommendations without excessive greetings.

## 🚀 Features

- **Human-like Sales Interaction**: Talks like a knowledgeable sales person, not a robotic assistant
- **No Repetitive Greetings**: Jumps straight to addressing customer queries
- **Smart Product Matching**: Uses vector search and keyword matching to find relevant products
- **Visual Product Display**: Shows product images for single product matches
- **Conversation Memory**: Maintains context throughout the conversation
- **Professional GUI**: Clean, modern chat interface

## 📋 Setup Instructions

1. **Install Dependencies**:
```bash
pip install tkinter pandas python-dotenv langchain-huggingface langchain-chroma langchain-groq langchain-core pillow requests
```

2. **Configure API Key**:
   - Copy `.env.template` to `.env`
   - Get your Groq API key from https://console.groq.com/
   - Add your API key to the `.env` file:
   ```
   GroqAPIKey=your_actual_api_key_here
   ```

3. **Run the Application**:
```bash
python Reservedcode.py
```

## 🎯 How It Works

- **Product Database**: Loads from `new_export.csv` containing water treatment products
- **Vector Search**: Uses sentence transformers for semantic product matching  
- **Sales Persona**: Configured to act as an experienced sales representative
- **Image Display**: Shows product images when a single product is recommended

## 💬 Sample Interactions

- "I need an RO system for my home" → Recommends domestic RO systems
- "Industrial water treatment plant" → Shows commercial/industrial options
- "Under ₹50000 budget" → Filters products by price range
- "UV purifier" → Finds UV-based water purification systems

## 📁 File Structure

- `Reservedcode.py` - Main application with AI logic
- `gui_template.py` - Chat GUI interface
- `vector.py` - Vector database setup
- `new_export.csv` - Product catalog
- `.env` - API configuration (create from template)

## 🔧 Key Improvements Made

- Removed automatic welcome messages
- Enhanced sales-oriented conversation flow
- Fixed duplicate image display code
- Improved product recommendation accuracy
- Added proper error handling and fallbacks