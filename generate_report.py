class Report:
    def __init__(self):
        indexed_documents = 0
        unique_tokens = set()
        index_disk_size = 0 

    def write_report(self):
        # write report at the end of the scrapers run
        with open('report.txt', 'w') as f:
            f.write(f"Indexed documents: {self.indexed_documents} \n \n \n")
            f.write(f"Unique tokens: {len(self.unique_tokens)} \n \n \n")
            f.write(f"Total size of index on disk: {self.index_disk_size:.2f} KB \n \n \n")
    
    def increment_unique_tokens(self, tokens):
        for token in tokens:
            self.unique_tokens.add(token)

    def read_disk_size(self):
        size_bytes = os.path.getsize("inverted_index.json")
        self.index_disk_size = size_bytes / 1024        
    
            