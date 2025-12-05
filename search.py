import json
import re
import heapq
import math
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from inverted_index import InvertedIndex

class Search:
    def __init__(self):
        self.inverted_index = {}
        self.doc_id_to_url = {}
        self.total_documents = 0


    def load_inverted_index_from_file(self, index_file_path, doc_mapping_file_path):
        with open(index_file_path, 'r') as f:
            self.inverted_index = json.load(f)
        
        with open(doc_mapping_file_path, 'r') as f:
            self.doc_id_to_url = json.load(f)
        
        # Track total number of documents for IDF calculation
        self.total_documents = len(self.doc_id_to_url)
    

    def bool_search(self, query, use_or=False):
        """
        Search with boolean logic.
        use_or=False: AND search (all terms must appear)
        use_or=True: OR search (any term can appear)
        """
        query_words = self._tokenize_query(query)
        if not query_words: return []
        
        if use_or:
            combined_postings = self._merge_lists_or(query_words)
        else:
            combined_postings = self._merge_lists(query_words)
            
        if not combined_postings: return []

        results = self.TF_IDF_Search(combined_postings, 5, query_words)

        return results


    def _tokenize_query(self, query):
        InvertedIndex._check_punkt()
        query_tokens = word_tokenize(query.lower())
        query_stems = []
        ps = PorterStemmer()
        
        for tok in query_tokens:
            if re.fullmatch(r"[a-z0-9]+", tok):
                stem = ps.stem(tok)
                query_stems.append(stem)
        return query_stems

    def _merge_lists(self, query_words):
        """ return one final list of combined postings (AND operation) """
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
                doc1 = combined_postings[pointer1]["document_id"]
                doc2 = other_postings[pointer2]["document_id"]
                
                if doc1 == doc2:
                    new_combined_postings.append(combined_postings[pointer1])
                    pointer1 += 1
                    pointer2 += 1
                elif doc1 > doc2:
                    pointer2 += 1
                else:
                    pointer1 += 1
            
            combined_postings = new_combined_postings
        
        return combined_postings
    

    def TF_IDF_Search(self, postings, k, query_words):
        """ 
        Rank documents using tf-idf with cosine similarity.
        Also incorporates important words weighting (title, headers, bold).
        """
        # Calculate IDF for each query term
        idf_scores = {}
      
        for word in query_words:
            if word in self.inverted_index:
                df = len(self.inverted_index[word])
                idf_scores[word] = math.log(self.total_documents / df)
            else:
                idf_scores[word] = 0

        #Creates the query Vector: keys = values & terms = frequency
        query_vector = defaultdict(int)
        for word in query_words:
            query_vector[word] += 1
        
        counter = 0
        for word in query_vector:
            count = query_vector[word]
            square = count ** 2
            counter += square

        query_magnitude = math.sqrt(counter)
        
        #Calculate TFIDF scores for each document by looking up term frequencies for all query words in doc
        heap_lists = []
        for posting in postings:
            document_id = posting["document_id"]
            document_vector = {}
            document_mag_sq = 0
            
            for word in query_words:
                if word in self.inverted_index:
                    word_postings = self.inverted_index[word]
    
                    matched_posting = None
                    for proper_posting in word_postings:
                        if proper_posting["document_id"] == document_id:
                            matched_posting = proper_posting
                            break
                            
                    #Weigh boosts terms if Title, Bold, Heading
                    if matched_posting is not None:
                        tf = matched_posting["frequency"]
                        weight = 1.0
                        if matched_posting["title_count"] > 0:
                            weight += 2.0
                        if matched_posting["header_bold_count"] > 0:
                            weight += 1.0
                            
                        #Calc for TFIDF
                        weighted_tf = tf * weight
                        tf_idf = weighted_tf * idf_scores[word]
                        document_vector[word] = tf_idf
                        document_mag_sq += tf_idf * tf_idf
    
            #Dot Product for Cosine Similarity 
            dot_product = 0
        
            for word in query_words:
                if word in document_vector:
                    dot_product += query_vector[word] * document_vector[word]
    
            document_magnitude = math.sqrt(document_mag_sq)
            
            if document_magnitude > 0 and query_magnitude > 0:
                cosine_score = dot_product / (query_magnitude * document_magnitude)
            else:
                cosine_score = 0

            #Sort by TFIDF w/ Heap
            if len(heap_lists) < k:
                heapq.heappush(heap_lists, (cosine_score, document_id))
            elif cosine_score > heap_lists[0][0]: 
                heapq.heapreplace(heap_lists, (cosine_score, document_id))

        # Extract from heap in descending order and reverse
        final_results = []
        while heap_lists:
            score, document_id = heapq.heappop(heap_lists)
            final_results.append((self.doc_id_to_url[str(document_id)], document_id, score))
        
        final_results.reverse()
         
        return final_results
                    
                   