from inverted_index import InvertedIndex
import os, json

if __name__ == "__main__":
    InvertedIndex = InvertedIndex()

    for root, dir, files in os.walk('DEV'):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                page_id = file.split('.')[0]
                page_json = json.load(f) # each file json
                token_dict = InvertedIndex.scrape_page(page_json)
                InvertedIndex.create_postings(page_id, token_dict)
                
