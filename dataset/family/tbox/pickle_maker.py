import pickle

classes = {
    0: "Q1196129", 
    1: "Q31184",
    2: "Q10861465",
    3: "Q595094",
    4: "Q7566",
    5: "Q7560",
    6: "Q7565",
    7: "Q7569",
    8: "Q177232",
    9: "Q308194"
    }

properties = {
    0: "P1038", 
    1: "P26",
    2: "P3373",
    3: "P40",
    4: "P8810",
    5: "P22",
    6: "P25",
    }

with open('id2ent.pkl', 'wb') as handle:
    pickle.dump(classes, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('id2rel.pkl', 'wb') as handle:
    pickle.dump(properties, handle, protocol=pickle.HIGHEST_PROTOCOL)