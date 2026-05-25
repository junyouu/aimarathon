# import os
# import json
# import re
# from dotenv import load_dotenv
# import firebase_admin
# from firebase_admin import credentials, firestore
# from google.cloud.firestore_v1.vector import Vector
# from sentence_transformers import SentenceTransformer
# from openai import OpenAI

# try:
#     load_dotenv(override=True, encoding='utf-8')
# except Exception as e:
#     print(f"Warning: Could not load .env file: {e}")

# api_key = os.getenv("CHUTES_API_KEY")
# if not api_key:
#     raise ValueError("CHUTES_API_KEY not set in .env")

# class LLM:
#   def __init__(self):
#       self.client = OpenAI(
#         api_key=api_key,
#         base_url="https://llm.chutes.ai/v1"
#       )

#       self.embedding_model = SentenceTransformer(
#           "BAAI/bge-base-en-v1.5"
#       )

#   def generate(self, prompt, max_tokens=512, temperature=0.1):
#       response = self.client.chat.completions.create(
#           model="Qwen/Qwen3-32B-TEE",
#           messages=[
#               {"role": "user", "content": prompt}
#           ]
#       )

#       cleaned = re.sub(r"<think>.*?</think>", "", response.choices[0].message.content, flags=re.DOTALL)
#       return cleaned.strip()

#   def embedding(self, content_to_embed):
#       return self.embedding_model.encode(content_to_embed).tolist()

# llm = LLM()

# # =====================================================================
# # 1. Initialize Firebase Safely (Notebook Re-run Safe)
# # =====================================================================
# try:
#     firebase_admin.get_app()
#     print("[Firebase] Existing default app detected. Reusing active connection safely.")
# except ValueError:
#     cred_file = "./firebase-key.json"
#     cred = credentials.Certificate(cred_file)
#     firebase_admin.initialize_app(cred)
#     print("[Firebase] Cloud database connection initialized successfully.")

# db = firestore.client()
# collection_ref = db.collection("catalog")

# # =====================================================================
# # 2. Define Updated Catalog Mapping (All 7 Asset Files Included)
# # =====================================================================
# files_map = {
#     "CCTV Camera": ["01_cameras_hdcvi.json", "03_cameras_ip.json"],
#     "Recorder": ["02_recorders.json"],
#     "Hard Drive": ["04_consumer_vms_storage.json"],
#     "Mounts & Enclosures": ["05_tools_mounts.json"],
#     "Switches": ["06_networking.json"],
#     "Cable": ["07_cables_power_converters.json"]
# }

# print("\n🚀 Starting catalog vector embedding generation and cloud upload...")

# # =====================================================================
# # 3. Process and Store Data in Firestore (With Path Sanitation)
# # =====================================================================
# for category, filenames in files_map.items():
#     for filename in filenames:
#         if os.path.exists(filename):
#             print(f"\nProcessing configuration file: {filename} [{category}]...")

#             with open(filename, 'r', encoding='utf-8') as f:
#                 hardware_list = json.load(f)

#                 if isinstance(hardware_list, list):
#                     for item in hardware_list:
#                         product_name = str(item.get("name", item.get("Product Name", "")))
#                         raw_prod_id = str(item.get("id", item.get("ID", ""))).strip()

#                         # Filter to isolate relevant surveillance storage components
#                         if category == "Hard Drive" and "Hard Drive" not in product_name:
#                             continue

#                         # Skip empty rows if any exist
#                         if not product_name or product_name == "nan" or not raw_prod_id:
#                             continue

#                         # 🌟 FIXED: Sanitize forward slashes to prevent Firestore path path splitting errors
#                         prod_id = raw_prod_id.replace("/", "-")

#                         # Extract features list (handles arrays or comma strings safely)
#                         features = item.get("features", item.get("Features", []))
#                         if isinstance(features, str):
#                             features = [f.strip() for f in features.split(", ")]

#                         # Map specification dictionary or description string into a clean string parameter
#                         specification = item.get("specification", item.get("Description", ""))
#                         if isinstance(specification, dict):
#                             description_str = ", ".join(f"{k}: {v}" for k, v in specification.items())
#                         else:
#                             description_str = str(specification)

#                         use_case = str(item.get("use_case", item.get("Potential Use Case", "")))

#                         # --- PRICE CONVERSION LOGIC (Preserving Your Exact Rules) ---
#                         price_raw = str(item.get("price", item.get("Price", "")))
#                         try:
#                             # 1. Remove '$' and extra spaces
#                             clean_price = price_raw.replace("$", "").strip()
#                             # 2. Remove the thousands separator (e.g., "2.730,00" -> "2730,00")
#                             clean_price = clean_price.replace(".", "")
#                             # 3. Replace the decimal comma with a decimal point (e.g., "2730,00" -> "2730.00")
#                             clean_price = clean_price.replace(",", ".")

#                             # 4. Convert to float
#                             price_float = float(clean_price)
#                         except ValueError:
#                             price_float = 0.0

#                         # Combine data into a rich text chunk for the AI embedding model to process
#                         content_to_embed = (
#                             f"Product Name: {product_name}. "
#                             f"Category: {category}. "
#                             f"Description: {description_str}. "
#                             f"Use Case: {use_case}"
#                         )

#                         # Generate the dense token vector embedding via your model
#                         embedding = llm.embedding(content_to_embed)

#                         # Create the unified document payload structure
#                         doc_data = {
#                             "product_name": product_name,
#                             "features": features,
#                             "price": price_float,
#                             "use_case": use_case,
#                             "description": description_str,
#                             "category": category,
#                             "embedding_vector": Vector(embedding)
#                         }

#                         # Push record to Firebase using the sanitized alphanumeric code as the unique Document ID
#                         collection_ref.document(prod_id).set(doc_data)
#                         print(f"Uploaded [{category}] -> {prod_id} with verified price: ${price_float}")
#         else:
#             print(f"⚠️ File not found in folder path: {filename}")

# print("\n🎉 Cloud vectorization database upload complete with clean path segments!")