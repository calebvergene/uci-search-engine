import heapq
import json
import re
import math
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from inverted_index import InvertedIndex

class Search:
    def __init__(self):
        self.lookup_table = {}
        self.doc_id_to_url = {}
        self.index_file = None  # Keep file open for performance
        self.ps = PorterStemmer()  # Reuse stemmer
        self.total_documents = 0


    def load_inverted_index_from_file(self, lookup_file_path, doc_mapping_file_path):
        """Load ONLY the lookup table and URL mappings (not the full index)"""
        print("Loading lookup table...")
        with open(lookup_file_path, 'r', encoding='utf-8') as f:
            self.lookup_table = json.load(f)
        
        print("Loading document mappings...")
        with open(doc_mapping_file_path, 'r', encoding='utf-8') as f:
            self.doc_id_to_url = json.load(f)
        
        # Open index file once and keep it open
        self.index_file = open("final_index.txt", 'r', encoding='utf-8')
        
        # Store total document count
        self.total_documents = len(self.doc_id_to_url)
        
        print(f"✓ Loaded lookup table with {len(self.lookup_table)} terms")
        print(f"✓ Loaded {len(self.doc_id_to_url)} document mappings")
    

    def get_postings(self, term):
        """Get postings for a term WITHOUT loading entire index"""
        if term not in self.lookup_table:
            return []
        
        offset = self.lookup_table[term]["offset"]
        length = self.lookup_table[term]["length"]
        
        # Seek to exact location (file already open)
        self.index_file.seek(offset)
        line = self.index_file.read(length).strip()
        
        parts = line.split(' ', 1)
        if len(parts) != 2:
            return []
        
        term_read, postings_str = parts
        postings = []
        
        # Parse postings
        for posting_str in postings_str.split('|'):
            posting_parts = posting_str.split(':', 4)
            if len(posting_parts) == 5:
                docid = int(posting_parts[0])
                freq = int(posting_parts[1])
                hbc = int(posting_parts[2])
                tc = int(posting_parts[3])
                
                posting = {
                    "document_id": docid,
                    "frequency": freq,
                    "header_bold_count": hbc,
                    "title_count": tc
                }
                postings.append(posting)
        
        return postings


    def bool_search(self, query):
        """OR search - documents contain ANY query term"""
        query_words = self._tokenize_query(query)
        if not query_words: 
            return []
        
        # Get all documents that contain ANY of the query terms
        all_doc_ids = set()
        postings_by_doc = defaultdict(lambda: {"frequency": 0, "header_bold_count": 0, "title_count": 0})
        
        for word in query_words:
            if word not in self.lookup_table:
                continue
            
            word_postings = self.get_postings(word)
            for posting in word_postings:
                doc_id = posting["document_id"]
                all_doc_ids.add(doc_id)
                
                # Accumulate scores for documents
                postings_by_doc[doc_id]["frequency"] += posting["frequency"]
                postings_by_doc[doc_id]["header_bold_count"] += posting["header_bold_count"]
                postings_by_doc[doc_id]["title_count"] += posting["title_count"]
        
        if not all_doc_ids:
            return []
        
        # Convert to posting list format
        combined_postings = []
        for doc_id in all_doc_ids:
            combined_postings.append({
                "document_id": doc_id,
                "frequency": postings_by_doc[doc_id]["frequency"],
                "header_bold_count": postings_by_doc[doc_id]["header_bold_count"],
                "title_count": postings_by_doc[doc_id]["title_count"]
            })
        
        results = self._tf_idf_search(combined_postings, 20, query_words)
        return results


    def _tokenize_query(self, query):
        """Tokenize and stem query"""
        InvertedIndex._check_punkt()
        query_tokens = word_tokenize(query.lower())
        query_stems = []
        
        for tok in query_tokens:
            if re.fullmatch(r"[a-z0-9]+", tok):
                stem = self.ps.stem(tok)
                query_stems.append(stem)
        return query_stems
    

    def _tf_idf_search(self, postings, k, query_words):
        """ 
        Rank documents using TF-IDF with cosine similarity.
        Uses disk-based lookup to avoid loading entire index.
        """
        # Calculate IDF for each query term
        idf_scores = {}
        for word in query_words:
            if word in self.lookup_table:
                # Get document frequency from postings
                word_postings = self.get_postings(word)
                df = len(word_postings)
                idf_scores[word] = math.log(self.total_documents / df) if df > 0 else 0
            else:
                idf_scores[word] = 0

        # Create query vector: term -> frequency in query
        query_vector = defaultdict(int)
        for word in query_words:
            query_vector[word] += 1
        
        # Calculate query magnitude
        query_mag_sq = sum(count ** 2 for count in query_vector.values())
        query_magnitude = math.sqrt(query_mag_sq)
        
        # Calculate TF-IDF scores for each document
        document_scores = {}
        
        for posting in postings:
            document_id = posting["document_id"]
            document_vector = {}
            document_mag_sq = 0
            
            # For each query word, get its TF-IDF in this document
            for word in query_words:
                if word not in self.lookup_table:
                    continue
                
                # Get postings for this word
                word_postings = self.get_postings(word)
                
                # Find the posting for this specific document
                matched_posting = None
                for p in word_postings:
                    if p["document_id"] == document_id:
                        matched_posting = p
                        break
                
                if matched_posting is not None:
                    tf = matched_posting["frequency"]
                    
                    # Weight boost for title, headers, bold
                    weight = 1.0
                    if matched_posting["title_count"] > 0:
                        weight += 2.0
                    if matched_posting["header_bold_count"] > 0:
                        weight += 1.0
                    
                    # Calculate TF-IDF
                    weighted_tf = tf * weight
                    tf_idf = weighted_tf * idf_scores[word]
                    document_vector[word] = tf_idf
                    document_mag_sq += tf_idf * tf_idf
            
            # Calculate cosine similarity (dot product / magnitudes)
            dot_product = 0
            for word in query_words:
                if word in document_vector:
                    dot_product += query_vector[word] * document_vector[word]
            
            document_magnitude = math.sqrt(document_mag_sq)
            
            if document_magnitude > 0 and query_magnitude > 0:
                cosine_score = dot_product / query_magnitude
            else:
                cosine_score = 0
            
            document_scores[document_id] = cosine_score
        
        # Sort and return top k results
        sorted_scores = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = []
        for document_id, score in sorted_scores[:k]:
            final_results.append((self.doc_id_to_url[str(document_id)], document_id, score))
        
        return final_results
    
    
    def close(self):
        """Close the index file when done"""
        if self.index_file:
            self.index_file.close()
    
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close()