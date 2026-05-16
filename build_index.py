import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

def build_index():
    with open("catalog_clean.json") as f:
        catalog = json.load(f)
    
    docs = []
    for item in catalog:
        content = f"""Name: {item['name']}
Description: {item['description']}
Test Type: {item['test_type']}
Job Levels: {', '.join(item['job_levels'])}
Duration: {item['duration'] or 'Not specified'}
Remote: {item['remote']}
Categories: {', '.join(item['keys'])}"""
        
        docs.append(Document(
            page_content=content,
            metadata={
                "name": item["name"],
                "url": item["url"],
                "test_type": item["test_type"]
            }
        ))
    
    print("Building embeddings... (1-2 min first time)")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")
    print(f"Done! Index built with {len(docs)} documents")

if __name__ == "__main__":
    build_index()