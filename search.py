class Search:
    def __init__(self):
        self.inverted_index = {}
        self.doc_id_to_url = {}

    def load_inverted_index_from_file(self, index_file_path, doc_mapping_file_path):
        with open(index_file_path, 'r') as f:
            self.inverted_index = json.load(f)
        
        with open(doc_mapping_file_path, 'r') as f:
            self.doc_id_to_url = json.load(f)
    
    def search(self, query)