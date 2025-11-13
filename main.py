from inverted_index import InvertedIndex
from generate_report import Report
import os, json

if __name__ == "__main__":
    Report = Report()
    InvertedIndex = InvertedIndex(Report)

    for root, dir, files in os.walk('DEV'):
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)
            with open(file_path, 'r') as f:
                page_json = json.load(f) # each file json
                token_dict = InvertedIndex.scrape_page(page_json)
                InvertedIndex.create_postings(Report.indexed_documents, token_dict)

                Report.indexed_documents += 1

    # generate report at the end
        InvertedIndex.write_inverted_index()
        Report.read_disk_size()
        Report.write_report()
                
