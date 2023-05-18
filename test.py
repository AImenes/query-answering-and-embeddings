import requests
import json

url = 'https://dbpedia.org/sparql'
query = "SELECT ?w WHERE {  VALUES ?w { <http://dbpedia.org/resource/64th_Academy_Awards> <http://dbpedia.org/resource/Angela_Lansbury> <http://dbpedia.org/resource/August_Strindberg> <http://dbpedia.org/resource/Allen_Ginsberg> <http://dbpedia.org/resource/Arthur_Conan_Doyle> <http://dbpedia.org/resource/Alameda_County,_California> <http://dbpedia.org/resource/Arcadia,_California> <http://dbpedia.org/resource/Anchorage,_Alaska> <http://dbpedia.org/resource/American_Academy_of_Dramatic_Arts> <http://dbpedia.org/resource/Algiers> <http://dbpedia.org/resource/Atlantic_City,_New_Jersey> <http://dbpedia.org/resource/Atlante_F.C.> <http://dbpedia.org/resource/20th_Century_Fox> <http://dbpedia.org/resource/Adobe_Systems> <http://dbpedia.org/resource/Anaheim,_California> <http://dbpedia.org/resource/Auburn,_New_York> <http://dbpedia.org/resource/Arlington,_Texas> <http://dbpedia.org/resource/Auckland> <http://dbpedia.org/resource/Amsterdam> <http://dbpedia.org/resource/Amarillo,_Texas> <http://dbpedia.org/resource/Annapolis,_Maryland> <http://dbpedia.org/resource/Aegean_Sea> <http://dbpedia.org/resource/AT&T> <http://dbpedia.org/resource/American_University_of_Beirut> <http://dbpedia.org/resource/Appalachian_State_University> <http://dbpedia.org/resource/Adriatic_Sea> <http://dbpedia.org/resource/Barnard_College> <http://dbpedia.org/resource/Aberdeen> <http://dbpedia.org/resource/Baltimore> <http://dbpedia.org/resource/Atlanta> <http://dbpedia.org/resource/Arizona_State_University> <http://dbpedia.org/resource/20th_Century_Fox> <http://dbpedia.org/resource/Augsburg> <http://dbpedia.org/resource/Aisne> <http://dbpedia.org/resource/Asheville,_North_Carolina> <http://dbpedia.org/resource/Athens,_Georgia> <http://dbpedia.org/resource/Agra> <http://dbpedia.org/resource/Bagrationi_dynasty> <http://dbpedia.org/resource/American_Idol> <http://dbpedia.org/resource/Babylon_5:_The_Gathering> <http://dbpedia.org/resource/A_Serious_Man> <http://dbpedia.org/resource/1941_(film)> <http://dbpedia.org/resource/28_Days_Later> <http://dbpedia.org/resource/A_History_of_Violence> <http://dbpedia.org/resource/A_Simple_Life> <http://dbpedia.org/resource/Academy_Award_for_Best_Animated_Feature> <http://dbpedia.org/resource/Academy_Award_for_Best_Actress> <http://dbpedia.org/resource/Academy_Award_for_Best_Picture> <http://dbpedia.org/resource/Academy_Award_for_Best_Original_Song> <http://dbpedia.org/resource/Anthony_Anderson> <http://dbpedia.org/resource/Balochistan,_Pakistan> <http://dbpedia.org/resource/Arunachal_Pradesh> <http://dbpedia.org/resource/Arista_Records> <http://dbpedia.org/resource/Acoustic_music> <http://dbpedia.org/resource/Amyotrophic_lateral_sclerosis> <http://dbpedia.org/resource/Backstairs_at_the_White_House> <http://dbpedia.org/resource/Arlington_National_Cemetery> <http://dbpedia.org/resource/Alec_Guinness> <http://dbpedia.org/resource/Atlantic_Time_Zone> <http://dbpedia.org/resource/Art_Deco> <http://dbpedia.org/resource/Amy_Winehouse> <http://dbpedia.org/resource/Antioch_College> <http://dbpedia.org/resource/Albert_Brenner> <http://dbpedia.org/resource/Asian_people> <http://dbpedia.org/resource/BBC_Films> <http://dbpedia.org/resource/Bahrain> <http://dbpedia.org/resource/2009_Toronto_International_Film_Festival> }  ?x <http://dbpedia.org/ontology/type> ?w . ?w rdf:type <http://dbpedia.org/ontology/Agent> . } "
r = requests.get(url, params={'format': 'json', 'query': query})

data = json.dumps({})

if r.status_code == 206:
    data = r.json()