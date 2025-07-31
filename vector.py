from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
import os
import re

# -------------------------
# 📦 Load CSV product data
# -------------------------
df = pd.read_csv("new_export.csv", encoding="ISO-8859-1").fillna("")

# 💰 Ensure numeric price
df["Regular_price"] = pd.to_numeric(df.get("Regular_price", 0), errors="coerce").fillna(0)

# -------------------------
# 🤖 Embeddings
# -------------------------
embedding_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# -------------------------
# 📦 Vector DB Setup
# -------------------------
db_location = "./chroma_water_products"
add_documents = not os.path.exists(db_location)

vector_store = Chroma(
    collection_name="water_product_recommendation",
    persist_directory=db_location,
    embedding_function=embedding_fn
)

def extract_technical_specs(description):
    """Extract technical specifications from product description"""
    specs = []
    
    # Extract flow rate (LPH)
    lph_match = re.search(r"(\d+)\s*lph|\b(\d+)\s*liters?\s*per\s*hour", description.lower())
    if lph_match:
        lph_value = lph_match.group(1) or lph_match.group(2)
        specs.append(f"Flow Rate: {lph_value} LPH")
    
    # Extract GPD capacity
    gpd_match = re.search(r"(\d+)\s*gpd", description.lower())
    if gpd_match:
        specs.append(f"Capacity: {gpd_match.group(1)} GPD")
    
    # Extract storage capacity
    storage_match = re.search(r"(\d+)\s*liters?\s*storage|storage\s*capacity\s*of\s*(\d+)", description.lower())
    if storage_match:
        storage_value = storage_match.group(1) or storage_match.group(2)
        specs.append(f"Storage: {storage_value} liters")
    
    # Extract membrane information
    if "membrane" in description.lower():
        membrane_match = re.search(r"(\d+)\s*membrane|membrane.*?(\d+)", description.lower())
        if membrane_match:
            specs.append("RO Membrane included")
    
    # Extract UV/UF information
    if "uv" in description.lower():
        specs.append("UV Purification")
    if "uf" in description.lower():
        specs.append("UF Filtration")
    
    return specs

if add_documents:
    documents = []
    ids = []

    for i, row in df.iterrows():
        name = row.get("Name", "")
        category = row.get("Category", "")
        price = int(float(row.get("Regular_price", 0)))
        short_desc = row.get("Short description", "")
        desc = row.get("Description", "")
        img = row.get("Images", "")
        attributes = row.get("Attribute 1 value(s)", "")
        
        # Extract technical specifications
        tech_specs = extract_technical_specs(desc)
        specs_text = " | ".join(tech_specs) if tech_specs else "Standard specifications"
        
        # Create comprehensive product content for better search
        page_content = f"""
🏷️ Product Name: {name}
📂 Category: {category}
💰 Price: ₹{price:,}
⚡ Key Features: {short_desc}

🔧 Technical Specifications: {specs_text}

📋 Full Description: {desc}

🎛️ Available Variants: {attributes if attributes else 'Standard model'}

🏢 Application: {category.split('>')[0].strip() if '>' in category else category}

💧 Water Treatment Technology: {"RO" if "ro" in name.lower() or "ro" in desc.lower() else ""} {"UV" if "uv" in name.lower() or "uv" in desc.lower() else ""} {"UF" if "uf" in name.lower() or "uf" in desc.lower() else ""}

🏭 Suitable For: Industrial Commercial Domestic Home Office School Hospital
        """.strip()

        doc = Document(
            page_content=page_content,
            metadata={
                "index": i,
                "price": price,
                "name": name,
                "category": category,
                "image_url": img.split(",")[0].strip() if img else "",
                "has_ro": "ro" in name.lower() or "ro" in desc.lower(),
                "has_uv": "uv" in name.lower() or "uv" in desc.lower(),
                "has_uf": "uf" in name.lower() or "uf" in desc.lower(),
                "is_industrial": "industrial" in category.lower(),
                "is_domestic": "domestic" in category.lower(),
                "attributes": attributes
            }
        )
        documents.append(doc)
        ids.append(str(i))

    # Add and persist documents
    vector_store.add_documents(documents=documents, ids=ids)
    print(f"✅ {len(documents)} products added to vector store with enhanced metadata.")

else:
    print("✅ Vector store already exists. Loaded from disk.")
