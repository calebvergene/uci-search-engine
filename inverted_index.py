from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.data import find
from collections import defaultdict
import json


class InvertedIndex:
    def __init__(self) -> None:
        inverted_index = {}
        self._check_punkt()
        self.ps = PorterStemmer()
    

    def scrape_page(self, page_json) -> dict:
        """
        The main function of this class, it will tokenize raw HTML data and return relevant data
        @parameters: Take a page_json as the information to scrape
        @return: Return a hash_map with key=token:str, body=dict{freq:int, positions:list[int], header_bold_count:int, title_count:int}
        """
        # First: Use beautiful soup to scrape the page for the relevent text
        html = page_json.get("content", "")

        if not html:
            return {}

        # Parse HTML
        soup = BeautifulSoup(html, "html.parser")

        # Remove noise globally
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Get the title first (weighed heavily)
        title_counts = self._get_title_counts(soup)
        
        # Get the header counts next
        header_counts = self._get_head_bold_counts(soup)

        # Call the helper function to build final output combining the title, header, and body counts
        output = self._build_final_output(soup, title_counts, header_counts)
        return output

    def _get_title_counts(self, soup: BeautifulSoup) -> dict:
        """
        Helper function that creats a hash_map that counts how many times each token was in the title (porter stemmed)
        @parameters: soup is the beautiful soup to scrap the html for text words in the title
        @return: a hash_map of key=token:str : value=freq:int
        """
        # First get the title
        title = soup.title.string.strip()

        if not title:
            return {}

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
        # First get all relevant text split by each "node" (section of text)
        header_nodes = soup.find_all(["h1","h2","h3","h4","h5","h6","b","strong"])

        if not header_nodes:
            return {}

        # Because each header node is a body of text this will run in O(n^2) as it will iterate through each node and every word inside each node
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
        # Grab the soup
        full_text = soup.get_text(separator=" ", strip=True)
        full_text = re.sub(r"\s+", " ", full_text)

        tokens = word_tokenize(full_text)

        freq_map = defaultdict(int)
        pos_map = defaultdict(list)

        for pos, tok in enumerate(tokens):
            tok = tok.lower()
            if not re.fullmatch(r"[a-z0-9]+", tok):
                continue

            stem = self.ps.stem(tok)

            freq_map[stem] += 1
            pos_map[stem].append(pos)

        output = {}

        for stem in freq_map:
            output[stem] = {
                "freq": freq_map[stem],
                "positions": pos_map[stem],
                "header_bold_count": headers[stem],
                "title_count": titles[stem]
            }

        return output

    @staticmethod
    def _check_punkt() -> None:
        """
        Checks if punkt is installed and if it isnt then it will download it for nltk
        """
        try:
            find("tokenizers/punkt")
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download("punkt")


    def create_postings(self, document_id, token_dict):
        # updates the inverted index with each new token from the document
        """
        token_dict:
        key=token:str, 
        value=[freq:int, positions:list[int], header/bold_count:int, title_count:int]
        """
        for token in token_dict:
            if token not in self.inverted_index:
                self.inverted_index[token] = []
            posting = {} # document_id, frequency, proximity, header/bolded
            posting[document_id] = document_id
            posting[frequency] = token_dict[token][freq]
            posting[positions] = token_dict[token][positions]
            posting[header_bold_count] = token_dict[token][header_bold_count]
            posting[title_count] = token_dict[token][title_count]

            posting_list = self.inverted_index[token]
            posting_list.append(posting)
    
    
    def write_inverted_index(self):
        with open("inverted_index.json", "w") as f:
            json.dump(self.inverted_index, f)