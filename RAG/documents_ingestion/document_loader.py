#Step of loading documents 

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader,JSONLoader
from pathlib import Path

def load_documents(dir_path):
    """This function take the path of specific directory and load the document inside it"""
    
    path = Path(dir_path)
    path_check = path.exists()
    
    if not path_check:
         print(f"{dir_path} does not exist. Please check")
         return {
            "documents": [],
            "files_loaded": 0,
            "total_chunks": 0,
            "failed_files": []
        }

    all_documents = []
    files_loaded = 0
    failed_files = []

    for item in path.rglob("*"):
        if item.is_file():
            suffix = item.suffix.lower()
            if suffix == ".pdf":
                print(f"{item.name} loading...")
                loader = PyPDFLoader(str(item))

            elif suffix == ".csv":
                print(f"{item.name} loading...")
                loader = CSVLoader(str(item))

            elif suffix == ".txt":
                print(f"{item.name} loading...")
                loader = TextLoader(str(item))
            elif suffix == ".json":
                print(f"{item.name} loading...")
                loader = JSONLoader(
                str(item),
                jq_schema='.',
                text_content=False
                )
            
            else:
                print(f"Skipping {item.name} – unsupported type")
                continue

            # Attempt to load the file
            try:
                docs = loader.load()
                all_documents.extend(docs)
                files_loaded += 1
                print(f"{item.name} Done ✅")
            except Exception as e:
                failed_files.append({"file": str(item), "error": str(e)})
                print(f"❌ {item.name} failed: {e}")
    
    return {
        "documents": all_documents,
        "files_loaded": files_loaded,
        "total_chunks": len(all_documents),   
        "failed_files": failed_files
    }