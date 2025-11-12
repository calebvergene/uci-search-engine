import sys

class Tokenizer:
    # Time complexity: O(n), where n = number of chars in text file
    # It has to process each character in the file to determine tokens, and the runtime is linear because we do this in one pass, which is efficient!
    @staticmethod
    def tokenize(file_path):
        # my approach to special chars: remove at the beginning and end to prioritize middle important chars that have more meaning.
        # nevermind! instructions say "sequences of alphanumeric characters" did so much work for nothing oopsies
        tokens = []
        curr_token = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                while True:
                    char = file.read(1)

                    if not char: 
                        if curr_token: # last token, need to add
                            tokens.append(''.join(curr_token))
                        break

                    try:
                        if char.isalnum():
                            curr_token.append(char.lower())
                        else:
                            if curr_token:
                                tokens.append(''.join(curr_token))
                                curr_token = []
                                
                    except Exception as e:
                        print(f"Error processing char: {e}")
                        if curr_token:
                                tokens.append(''.join(curr_token))
                                curr_token = []
                        continue

        except Exception as e:
            print(f"Could not read file! {e}")
        
        return tokens


    # Time complexity: O(n), where n = len(tokens).
    # Iterates through tokens. Then, for each token, increment it in the frequency hash, which is O(1). So, O(1) * n = O(n)
    @staticmethod
    def computeWordFrequencies(tokens):
        """
        Write another method/function that counts the number of occurrences of each token in the token list. 
        """
        frequency = {}
        for token in tokens:
            frequency[token] = frequency.get(token, 0) + 1
        return frequency


    # Time complexity: O(nlogn), where n = number of unique tokens.
    # Converting to list is O(n) time
    # Python's sorting function takes an average time of O(nlogn), which is the domininant runtime
    # Then, has to go through each key of the frequency hash and prints the key & its value (print = contstant time). This is O(n)
    # O(nlogn) overpowers O(n)!
    @staticmethod
    def print(frequencies):
        """
        prints out the word frequency count onto the screen. The print out should be ordered by decreasing frequency 
        (so, the highest frequency words first). 
        """
        sorted_freqs = list(frequencies.items())
        sorted_freqs.sort(key=lambda x : x[1], reverse=True)
        for token in sorted_freqs:
            print(f'{token[0]} - {token[1]}')
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("No file given.")
        sys.exit(1)

    file_path = sys.argv[1]
    tokens = Tokenizer.tokenize(file_path)
    freqs = Tokenizer.computeWordFrequencies(tokens)
    Tokenizer.print(freqs)