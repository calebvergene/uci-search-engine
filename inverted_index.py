import token
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.data import find
from collections import defaultdict
import warnings
from generate_report import Report
import os, json
from urllib.parse import urldefrag

# suppress BeautifulSoup warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

# Inverted index class, handles scraping and processing each posting
# ==========================================================

class InvertedIndex:
    def __init__(self, Report) -> None:
        self.Report = Report
        self.inverted_index = {}
        self.docid_to_url = {}
        self.url_to_docid = {}
        self.doc_lengths = {}
        self._check_punkt()
        self.ps = PorterStemmer()
        self.seen_fingerprints = set()


    def scrape_page(self, page_json) -> dict:
        """
        The main function of this class, it will tokenize raw HTML data and return relevant data
        @parameters: Take a page_json as the information to scrape
        @return: Return a hash_map with key=token:str, body=dict{freq:int, positions:list[int], header_bold_count:int, title_count:int}
        """
        # first: Use beautiful soup to scrape the page for the relevent text
        html = page_json.get("content", "")

        if not html:
            return {}

        # parse HTML
        soup = BeautifulSoup(html, "html.parser")

        # remove noise globally
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # get the title first (weighed heavily)
        title_counts = self._get_title_counts(soup)
        
        # get the header counts next
        header_counts = self._get_head_bold_counts(soup)

        # call the helper function to build final output combining the title, header, and body counts
        output = self._build_final_output(soup, title_counts, header_counts)
        return output

    def _get_title_counts(self, soup: BeautifulSoup) -> dict:
        """
        Helper function that creats a hash_map that counts how many times each token was in the title (porter stemmed)
        @parameters: soup is the beautiful soup to scrap the html for text words in the title
        @return: a hash_map of key=token:str : value=freq:int
        """
        # first get the title
        if not soup or not soup.title or not soup.title.string:
            return {}
            
        title = soup.title.string.strip()
        title_counts = defaultdict(int)

        for tok in word_tokenize(title):
            tok = tok.lower()
            if re.fullmatch(r"[a-z0-9]+", tok):
                stem = self.ps.stem(tok)
                title_counts[stem] += 1
        
        return title_counts
    
    def _get_head_bold_counts(self, soup: BeautifulSoup) -> dict:
        """
        Helper function that creats a hash_map that counts how many times each token was in the header, bold, or strong (porter stemmed)
        @parameters: soup is the beautiful soup to scrap the html for text words in the title
        @return: a hash_map of key=token:str : value=freq:int
        """
        # first get all relevant text split by each "node" (section of text)
        header_nodes = soup.find_all(["h1","h2","h3","h4","h5","h6","b","strong"])

        if not header_nodes:
            return {}

        # because each header node is a body of text this will run in O(n^2) as it will iterate through each node and every word inside each node
        header_counts = defaultdict(int)
        for node in header_nodes:
            htext = node.get_text(separator=" ", strip=True)
            for tok in word_tokenize(htext):
                tok = tok.lower()
                if not re.fullmatch(r"[a-z0-9]+", tok):
                    continue
                stem = self.ps.stem(tok)
                header_counts[stem] += 1

        return header_counts


    def _build_final_output(self, soup: BeautifulSoup, titles: defaultdict, headers: defaultdict) -> dict:
        """
        Helper function that creates the final output for the scrape page function
        @parameters: soup is the beautiful soup to scrap the html for text words in the title
        @return: look at the return typing for scrape_page()
        """
        # grab all the text
        full_text = soup.get_text(separator=" ", strip=True)
        full_text = re.sub(r"\s+", " ", full_text)

        tokens = word_tokenize(full_text)
        # keep track of the frequency and position of each token
        freq_map = defaultdict(int)
        pos_map = defaultdict(list)

        # Keep track of stems in order for 2-gram and 3-gram
        # [(stem, pos), ...]
        ordered_stems = []

        for pos, token in enumerate(tokens):
            token = token.lower()
            if not re.fullmatch(r"[a-z0-9]+", token):
                continue

            stem = self.ps.stem(token)
            # add stem to report token set to track unique tokens
            self.Report.unique_tokens.add(stem)

            freq_map[stem] += 1
            pos_map[stem].append(pos)

            ordered_stems.append((stem, pos))

        # Build 2-grams and 3-grams
        for n in [2, 3]:
            # when we have less words then grams move to next iteration
            if len(ordered_stems) < n:
                continue
                
            for i in range(len(ordered_stems) - n + 1):
                # For each stem get n stems together
                stem_slice = []
                for j in range(i, i + n):
                    stem_slice.append(ordered_stems[j][0])
                
                # get the position for the whole slice
                start_position = ordered_stems[i][1]

                # join with symbol so we know its a n-gram
                ngram_token = "_".join(stem_slice)

                # add n-gram to frequency map and position map 
                freq_map[ngram_token] += 1
                pos_map[ngram_token].append(start_position)

        output = {}
        all_stems = set(freq_map.keys()) | set(titles.keys()) | set(headers.keys())
        for stem in all_stems:
            output[stem] = {
                "freq": freq_map.get(stem, 0),
                "positions": pos_map.get(stem, []),
                "header_bold_count": headers.get(stem, 0),
                "title_count": titles.get(stem, 0)
            }
        
        return output, len(tokens)
    
    
    def create_postings(self, document_id, token_dict, doc_length):
        """
        Updates the inverted index with each new token from the document
        token_dict format:
        key=token:str, 
        value={"freq":int, "positions":list[int], "header_bold_count":int, "title_count":int}
        """
        self.doc_lengths[document_id] = doc_length
        
        for token in token_dict:
            if token not in self.inverted_index:
                self.inverted_index[token] = []
            
            posting = {
                "document_id": document_id,
                "frequency": token_dict[token]["freq"],
                "positions": token_dict[token]["positions"],
                "header_bold_count": token_dict[token]["header_bold_count"],
                "title_count": token_dict[token]["title_count"]
            }
            
            self.inverted_index[token].append(posting)


    @staticmethod
    def _check_punkt() -> None:
        """
        Checks if punkt_tab is installed and if it isnt then it will download it for nltk
        """
        try:
            find("tokenizers/punkt_tab")
        except LookupError:
            print("Downloading NLTK punkt_tab tokenizer...")
            nltk.download("punkt_tab")

    




    # for detecting duplicate pages
    # ==========================================================
    def hash(self, text):
        result = 0
        for i, char in enumerate(text):
            result = result * 37 + ord(char)
            result = result % 1000000007
        return str(result).zfill(16)

    def hash2(self, text):
        h = 5381 # prime long number
        for char in text:
            h = ((h * 33) + ord(char)) % (2**32)
        
        # make it look more random
        h = h ^ (h >> 16)
        h = h * 2654435761
        h = h % (2**32)
        
        return hex(h)[2:].zfill(16)

    def make_ngrams(self, words):
        # 5 grams
        chunks = []
        for i in range(len(words) - 4):
            chunk = " ".join(words[i:i+5])
            chunks.append(chunk)
        return chunks

    def randomize_ngrams(self, items):
        if len(items) <= 100:
            return items
        items = sorted(items, key=lambda x: self.hash(x))
        return items[:100]

    def hash_chunks(self, chunks):
        chosen = self.randomize_ngrams(chunks)
        result = []
        for c in chosen:
            hashed = self.hash2(c)
            result.append(hashed)
        return result

    def check_similar(self, hashes):
        # THRESHOLD = 0.8, pretty leniant
        # true if dup
        if not hashes:
            return False
        similar = 0
        for h in hashes:
            if h in self.seen_fingerprints:
                similar += 1
        score = similar / len(hashes)
        return score > 0.8

    def add_all_hashes(self, hashes):
        for h in hashes:
            self.seen_fingerprints.add(h)

    def page_too_similar(self, words):
        ngrams = self.make_ngrams(words)
        hashed = self.hash_chunks(ngrams)

        if not self.check_similar(hashed):
            self.add_all_hashes(hashed)
            # good
            return False

        else:
            return True








    # handles writing the indexes to split files, then merge
    # =============================================================================
    
    def write_partial_index(self, partial_num):
        filename = f"partial_index_{partial_num}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            for term in sorted(self.inverted_index.keys()):
                postings = self.inverted_index[term]
                # term docid:freq:hbc:tc:pos1,pos2,...
                # basically, seperated by : so that it can be easily parsed
                postings_list = []
                for posting in postings:
                    positions_str = ','.join(map(str, posting['positions']))
                    posting_str = f"{posting['document_id']}:{posting['frequency']}:{posting['header_bold_count']}:{posting['title_count']}:{positions_str}"
                    postings_list.append(posting_str)
                
                postings_str = '|'.join(postings_list)
                f.write(f"{term} {postings_str}\n")
        
        print(f"✓ Wrote {filename} with {len(self.inverted_index)} terms")
    
    
    # merge indexes into one index
    def merge_partial_indices(self, num_partials):
        partial_files = [f"partial_index_{i}.txt" for i in range(num_partials)]
        file_handles = [open(f, 'r', encoding='utf-8') for f in partial_files]
        current_lines = []

        for fh in file_handles:
            line = fh.readline().strip()
            if line:
                term = line.split(' ', 1)[0]
                current_lines.append((term, line, fh))
        
        lookup_table = {}
        terms_merged = 0
        
        with open("final_index.txt", 'w', encoding='utf-8') as out_file:
            while current_lines:
                current_lines.sort(key=lambda x: x[0])
                min_term = current_lines[0][0]
                merged_postings = {}  # docid -> posting data
                i = 0
                while i < len(current_lines):
                    term, line, fh = current_lines[i]
                    
                    if term == min_term:
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            postings_str = parts[1]
                            # docid:freq:hbc:tc:pos1,pos2,...|docid:freq:hbc:tc:pos1,pos2,...
                            for posting_str in postings_str.split('|'):
                                posting_parts = posting_str.split(':', 4)
                                if len(posting_parts) == 5:
                                    docid = int(posting_parts[0])
                                    freq = int(posting_parts[1])
                                    hbc = int(posting_parts[2])
                                    tc = int(posting_parts[3])
                                    positions = posting_parts[4]
                                    
                                    # merge postings for same docid (shouldn't happen but just in case)
                                    if docid in merged_postings:
                                        merged_postings[docid]['freq'] += freq
                                        merged_postings[docid]['hbc'] += hbc
                                        merged_postings[docid]['tc'] += tc
                                    else:
                                        merged_postings[docid] = {
                                            'freq': freq,
                                            'hbc': hbc,
                                            'tc': tc,
                                            'positions': positions
                                        }
                        
                        next_line = fh.readline().strip()
                        if next_line:
                            next_term = next_line.split(' ', 1)[0]
                            current_lines[i] = (next_term, next_line, fh)
                            i += 1
                        else:
                            current_lines.pop(i)
                    else:
                        i += 1
                
                # write merged postings for this term
                offset = out_file.tell()
                postings_list = []
                for docid in sorted(merged_postings.keys()):
                    p = merged_postings[docid]
                    posting_str = f"{docid}:{p['freq']}:{p['hbc']}:{p['tc']}:{p['positions']}"
                    postings_list.append(posting_str)
                
                postings_str = '|'.join(postings_list)
                line = f"{min_term} {postings_str}\n"
                out_file.write(line)
                
                # THIS CREATES LOOKUP TABLE
                lookup_table[min_term] = {
                    "offset": offset,
                    "length": len(line.encode('utf-8'))
                }
                
                terms_merged += 1
                if terms_merged % 10000 == 0:
                    print(f"  Merged {terms_merged} terms...")
        
        for fh in file_handles:
            fh.close()
        
        with open("index_lookup.json", 'w', encoding='utf-8') as f:
            json.dump(lookup_table, f)

        
        for partial_file in partial_files:
            if os.path.exists(partial_file):
                os.remove(partial_file)
                print(f"  Deleted {partial_file}")
    
    
    def save_url_mappings(self):
        """Save docID <-> URL mappings AND document lengths"""
        with open("docid_to_url.json", 'w', encoding='utf-8') as f:
            json.dump(self.docid_to_url, f)
        with open("url_to_docid.json", 'w', encoding='utf-8') as f:
            json.dump(self.url_to_docid, f)
        
        with open("doc_lengths.json", 'w', encoding='utf-8') as f:
            json.dump(self.doc_lengths, f)
        

    def write_inverted_index(self):
        """Legacy method - keeping for backwards compatibility"""
        with open("inverted_index.json", "w") as f:
            json.dump(self.inverted_index, f)
        
        with open("doc_id_to_url.json", "w") as f:
            json.dump(self.docid_to_url, f)


def create_inverted_index():
    report = Report()
    inverted_index = InvertedIndex(report)
    
    # partial indices
    partial_count = 0
    OFFLOAD_THRESHOLD = 10000  # basically, offload every 10k documents

    for root, dir, files in os.walk('DEV'):
        for file in files:
            if not file.endswith('.json'):
                continue
            
            file_path = os.path.join(root, file)
            print(file_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    page_json = json.load(f)
                    
                    # remove fragment from URL
                    original_url = page_json['url']
                    url_without_fragment, _ = urldefrag(original_url)
                    if url_without_fragment in inverted_index.url_to_docid:
                        continue
                    
                    token_dict, doc_length = inverted_index.scrape_page(page_json)

                    # Change duplicate check to only look at single word tokens not n-grams
                    words = [t for t in token_dict.keys() if "_" not in t]
                    
                    # duplicate check
                    if inverted_index.page_too_similar(words):
                        continue

                    inverted_index.create_postings(report.indexed_documents, token_dict, doc_length)
                    inverted_index.docid_to_url[report.indexed_documents] = url_without_fragment
                    inverted_index.url_to_docid[url_without_fragment] = report.indexed_documents
                    report.indexed_documents += 1
                    print('Document #', report.indexed_documents)
                    
                    # OFFLOAD TO DISK when threshold reached
                    if report.indexed_documents % OFFLOAD_THRESHOLD == 0:
                        print(f"OFFLOADING PARTIAL INDEX #{partial_count}")
                        inverted_index.write_partial_index(partial_count)
                        partial_count += 1
                        inverted_index.inverted_index.clear()
                        report.read_disk_size()
                        report.write_report()
            
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                print(f"Error reading {file_path}: {e}")
                print("Skipping this file...")
                continue
            except Exception as e:
                print(f"Unexpected error with {file_path}: {e}")
                print("Skipping this file...")
                continue

    # final partial index (leftover documents)
    if inverted_index.inverted_index:
        inverted_index.write_partial_index(partial_count)
        partial_count += 1
    
    # MERGE all partial indices into final index
    if partial_count > 0:
        inverted_index.merge_partial_indices(partial_count)
    
    inverted_index.save_url_mappings()
    
    # final report
    report.read_disk_size()
    report.write_report()