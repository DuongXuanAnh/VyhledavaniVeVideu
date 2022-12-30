import annoy
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Načtení CSV souboru s deskriptory obrázků do Pandas dataframe
df = pd.read_csv("CLIP_VITB32.csv", names=["ID"] + list(range(512)), sep=";", dtype=str)

# Vytvoření slovníku, který mapuje ID obrázku na jeho vektor deskriptoru
descriptor_dict = df.set_index("ID").to_dict()[0]

# Vytvoření Annoy indexu pro rychlé vektorové vyhledávání
index = annoy.AnnoyIndex(512, metric="euclidean")

# Přidání vektorů deskriptorů do indexu
for idx, descriptor in enumerate(df[0].values):
     # Převod vektoru deskriptoru na seznam pomocí funkce tolist()
    descriptor_list = [float(x) for x in descriptor.split(";")]
    index.add_item(idx, descriptor_list)

def generate_descriptor_for_text(text):
    # Vytvoření bag-of-words pomocí CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform([text])
    
    # Vrátit vektor deskriptoru pro zadaný text
    return X.toarray()[0]

text = "a dog"
# Vytvoření vektoru deskriptoru pro zadaný text
query_descriptor = generate_descriptor_for_text(text)

# Vyhledání nejpodobnějších obrázků pomocí Annoy indexu
indices, distances = index.get_nns_by_vector(query_descriptor, n=5, include_distances=True)

# Vypsání názvů nejpodobnějších obrázků
for idx, distance in zip(indices, distances):
    image_id = df.iloc[idx]["ID"]
    image_name = f"Image_{image_id}.jpg"
    print(f"{image_name}: {distance}")
