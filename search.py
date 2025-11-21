class Search:
    def __init__(self):
        self.inverted_index = {}
        self.doc_id_to_url = {}

    def load_inverted_index_from_file(self, index_file_path, doc_mapping_file_path):
        with open(index_file_path, 'r') as f:
            self.inverted_index = json.load(f)
        
        with open(doc_mapping_file_path, 'r') as f:
            self.doc_id_to_url = json.load(f)
    
    def bool_search(self, query):
        
        # split up query into AND chunks of words
        query_words = query.split()
        if not query_words: return []

        combined_postings = _merge_lists(query_words)
        if not combined_postings: return [] # for AND, no documents were a match



    def _merge_lists(self, query_words):
        """ return one final list of combined postings """
        combined_postings = []
        
        # init combined_postings to the first word result
        if query_words[0] not in self.inverted_index:
            return []
        combined_postings = self.inverted_index[query_words[0]]

        # now just condense the combined results into a singular list by comparing
        for word in query_words[1:]:
            if word not in self.inverted_index:
                return []
            pointer1, pointer2 = 0, 0
            new_combined_postings = []
            other_postings = self.inverted_index[word]
            while pointer1 < len(combined_postings) and pointer2 < len(other_postings):
                if combined_postings[pointer1]["document_id"] == other_postings[pointer2]["document_id"]:
                    ## append the whole posting with its scores
                    new_combined_postings.append(combined_postings[pointer1])
                    pointer1 += 1
                    pointer2 += 1
                elif combined_postings[pointer1]["document_id"] > other_postings[pointer2]["document_id"]:
                    pointer2 += 1
                else:
                    pointer1 += 1
            combined_postings = new_combined_postings
        
        return combined_postings
        

    def _k_search_results(self, k):
        """ keep a max heap of k size """
        # doc_tuple = (score, doc_id)

        
