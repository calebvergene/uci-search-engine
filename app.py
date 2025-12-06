from flask import Flask, render_template, request
from search import Search

#Create Flask Web Broswer
app = Flask(__name__)
search_engine = Search()

#Creates Search Engine In Flask Browser
search_engine.load_inverted_index_from_file("index_lookup.json", "docid_to_url.json")

#Loads "index template"
@app.route('/')
def index():
    return render_template('index.html')

#Search function 
@app.route('/search')
def search():
    query = request.args.get('query', '')

    #reloads search page if query left blank
    if not query:
        return render_template('index.html')
    #Tries looking for query in index. If found returns results if not returns "error"
    try:
        results = search_engine.bool_search(query)
        return  render_template('index.html', results=results, query=query)
    except:
        return render_template('index.html', error="Search error", query=query)

if __name__ == '__main__':
    app.run(port = 1235)
