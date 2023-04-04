import requests

# Wikidata example
url = 'https://query.wikidata.org/sparql'
query = ''' ASK { wd:Q615 wdt:P22 wd:Q615 } '''
r = requests.get(url, params = {'format': 'json', 'query': query})
data = r.json()
print(data['boolean'])

#DBPedia example
url = 'https://dbpedia.org/sparql'
query = ''' ASK { <http://dbpedia.org/resource/Claude_Monet> <http://dbpedia.org/property/birthPlace> ?birthPlace } '''
r = requests.get(url, params = {'format': 'json', 'query': query})
data = r.json()
print(data['boolean'])