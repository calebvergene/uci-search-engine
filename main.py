from search import Search
from inverted_index import create_inverted_index


if __name__ == "__main__":
    # Uncomment below to create the inverted index
    create_inverted_index()


    # For searching
    search_engine = Search()
    print('SYSTEM: Loading index...')
    search_engine.load_inverted_index_from_file("index_lookup.json", "docid_to_url.json")
    print('SYSTEM: Finished loading\n')
    
    while True:
        print("(Enter 's' to stop)")
        query = input("Search for something in the UCI database: ")
        if query == 's': 
            break

        results = search_engine.bool_search(query)
        
        if not results:
            print("No results found.\n")
            continue
            
        for i, result in enumerate(results):
            print(f"\n{i+1}. {result[0]}")
            print(f"   Score: {result[2]}")
        print("\n")