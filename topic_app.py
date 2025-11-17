#######################################################################
#   TÜRKÇE HABER GÜNDEM ANALİZİ — FULL VERSION (NO ERRORS)
#   AUTO INSTALL + BERTopic + BERT NER + 33 RSS + HEADLINE LIST
#######################################################################

import subprocess
import sys
import importlib
import feedparser
import re
import nltk
from nltk.corpus import stopwords


#######################################################################
#                     AUTO INSTALLER
#######################################################################

def auto_install(package):
    try:
        importlib.import_module(package)
        print(f"[OK] {package} zaten yüklü.")
    except ImportError:
        print(f"[INSTALL] {package} yükleniyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "sentence-transformers",
    "bertopic",
    "feedparser",
    "nltk"
]

for pkg in REQUIRED_PACKAGES:
    auto_install(pkg)

print("\n[+] Tüm paketler hazır!")


#######################################################################
#                     STOPWORDS
#######################################################################

nltk.download("stopwords")
stop_words = set(stopwords.words("turkish"))


#######################################################################
#                 BERT TABANLI TÜRKÇE NER
#######################################################################

print("[+] Türkçe BERT NER modeli yükleniyor...")

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer_ner = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-ner-cased")
model_ner = AutoModelForTokenClassification.from_pretrained("savasy/bert-base-turkish-ner-cased")

ner_pipeline = pipeline("ner", model=model_ner, tokenizer=tokenizer_ner, aggregation_strategy="simple")

print("[+] NER modeli hazır!")


#######################################################################
#                     HABER KAYNAKLARI (33 ADET)
#######################################################################

RSS_FEEDS = {

    # Uluslararası
    "AA": "https://www.aa.com.tr/tr/rss/default?cat=guncel",
    "Reuters": "https://feeds.reuters.com/Reuters/worldNews",
    "BBC": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "CNN International": "http://rss.cnn.com/rss/edition.rss",
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "NY Times": "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "Guardian": "https://www.theguardian.com/world/rss",
    "Washington Post": "http://feeds.washingtonpost.com/rss/world",
    "AP News": "https://apnews.com/rss",
    "DW": "https://rss.dw.com/xml/feed/rss-en-all",
    "Euronews": "https://www.euronews.com/api/rss/most-read",

    # Ulusal
    "Hürriyet": "https://www.hurriyet.com.tr/rss/gundem",
    "Sabah": "https://www.sabah.com.tr/rss/gundem.xml",
    "Milliyet": "https://www.milliyet.com.tr/rss/rssNew/gundem.xml",
    "NTV": "https://www.ntv.com.tr/gundem.rss",
    "CNN Türk": "https://www.cnnturk.com/feed/rss/all/news",
    "Sözcü": "https://www.sozcu.com.tr/rss/gundem.xml",
    "TRT Haber": "https://www.trthaber.com/rss/gundem.rss",
    "Habertürk": "https://www.haberturk.com/rss",
    "Yeni Şafak": "https://www.yenisafak.com/rss?xml=gundem",
    "Cumhuriyet": "https://www.cumhuriyet.com.tr/rss/gundem.xml",
    "T24": "https://t24.com.tr/rss",
    "Diken": "https://www.diken.com.tr/feed/",
    "OdaTV": "https://odatv4.com/rss.php",
    "Ensonhaber": "https://www.ensonhaber.com/rss/ensonhaber.xml",
    "A Haber": "https://www.ahaber.com.tr/rss/anasayfa.xml",
    "Haber7": "https://www.haber7.com/rss/haber",
    "Karar": "https://www.karar.com/rss/haber",
    "BirGün": "https://www.birgun.net/rss",
    "Akşam": "https://www.aksam.com.tr/rss/haber",
    "Star": "https://www.star.com.tr/rss/rss.asp?cid=1",
    "Milli Gazete": "https://www.milligazete.com.tr/rss",
    "Evrensel": "https://www.evrensel.net/rss/haber.xml"
}

INTERNATIONAL = list(RSS_FEEDS.keys())[:11]
NATIONAL = list(RSS_FEEDS.keys())[11:]


#######################################################################
#                     HABER ÇEKME
#######################################################################

def fetch_rss(url):
    try:
        feed = feedparser.parse(url)
        return [
            (getattr(e, "title", "") + " " + getattr(e, "summary", "")).strip()
            for e in feed.entries
        ]
    except:
        return []


#######################################################################
#                     METİN TEMİZLEME
#######################################################################

def clean_text(t):
    t = t.lower()
    t = re.sub(r"[^a-zğüşöçı0-9 ]", " ", t)
    return " ".join([w for w in t.split() if w not in stop_words])


#######################################################################
#                     NER ÇIKARIMI
#######################################################################

def extract_entities(text):
    ent_raw = ner_pipeline(text)
    out = {"PER": [], "LOC": [], "ORG": [], "DATE": [], "OTHER": []}

    for item in ent_raw:
        tag = item["entity_group"]
        word = item["word"]

        if tag in out:
            out[tag].append(word)
        else:
            out["OTHER"].append(word)

    return out


#######################################################################
#                      BERTopic MODELİ
#######################################################################

print("[+] BERTopic için embedding modeli yükleniyor...")

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

from bertopic import BERTopic
topic_model = BERTopic(language="turkish", embedding_model=embedding_model)


#######################################################################
#                     TÜM HABERLERİ ÇEK
#######################################################################

print("[+] Haberler toplanıyor...")

all_texts = []
news_by_source = {}

for source, url in RSS_FEEDS.items():
    items = fetch_rss(url)
    if items:
        news_by_source[source] = items
        all_texts.extend(items)

print(f"[+] Toplam {len(all_texts)} haber alındı.")


#######################################################################
#                      TOPIC MODEL EĞİT
#######################################################################

cleaned_all = [clean_text(t) for t in all_texts]

print("[+] BERTopic modeli eğitiliyor (biraz sürebilir)...")
topics, _ = topic_model.fit_transform(cleaned_all)


#######################################################################
#           TOPIC → GENEL KATEGORİ EŞLEŞTİRME
#######################################################################

def categorize(words):
    joined = " ".join(w[0] for w in words)

    rules = {
        "savas_jeopolitik": ["saldırı","rusya","ukrayna","israil","iran","çatışma","gaza"],
        "ekonomi": ["dolar","faiz","enflasyon","piyasa","kriz"],
        "siyaset": ["meclis","bakan","cumhurbaşkanı","seçim","parti"],
        "asayis": ["tutuk","cinayet","mahkeme"],
        "saglik": ["hastane","kanser","salgın","tedavi"],
        "teknoloji": ["yapay","uzay","nasa","ai","robot"],
        "dogal_afet": ["deprem","sel","yangın"],
        "spor": ["maç","transfer","lig","hakem"],
        "toplumsal": ["belediye","trafik","eğitim","öğrenci"],
        "magazin": ["ünlü","dizi","film","oyuncu"]
    }

    for label, keys in rules.items():
        if any(k in joined for k in keys):
            return label

    return "diger"


#######################################################################
#         HER HABER SİTESİ İÇİN İSTATİSTİK HESAPLA
#######################################################################

results = {}

for source, articles in news_by_source.items():

    cleaned_src = [clean_text(t) for t in articles]
    src_topics, _ = topic_model.transform(cleaned_src)

    counter = {k: 0 for k in [
        "savas_jeopolitik","ekonomi","siyaset",
        "asayis","saglik","teknoloji",
        "dogal_afet","spor","toplumsal","magazin","diger"
    ]}

    for t in src_topics:
        topic_words = topic_model.get_topic(t)
        cat = categorize(topic_words)
        counter[cat] += 1

    results[source] = counter

    print(f"\n=== {source} ===")
    total = sum(counter.values())
    for k,v in counter.items():
        if total > 0:
            print(f"{k}: %{round(v/total*100,2)}")


#######################################################################
#           ULUSAL VE ULUSLARARASI DAĞILIM
#######################################################################

def sum_stats(src_list):
    keys = list(next(iter(results.values())).keys())
    out = {k:0 for k in keys}

    for src in src_list:
        if src in results:
            for k,v in results[src].items():
                out[k] += v

    return out

intl = sum_stats(INTERNATIONAL)
nat  = sum_stats(NATIONAL)

print("\n=== ULUSLARARASI TOPLAM ===")
print(intl)

print("\n=== ULUSAL TOPLAM ===")
print(nat)


#######################################################################
#               NER BÜYÜK VARLIK ANALİZİ
#######################################################################

print("\n[+] Haberlerde geçen kişi/şehir/ülke analiz ediliyor...")

entities = {"PER": [], "LOC": [], "ORG": [], "DATE": [], "OTHER": []}

for t in all_texts[:400]:
    e = extract_entities(t)
    for k,v in e.items():
        entities[k].extend(v)


#######################################################################
#               YORUM OLUŞTUR
#######################################################################

def gpt_summary():
    intl_main = max(intl, key=intl.get)
    nat_main = max(nat, key=nat.get)

    return f"""
=====================================================
               YAPAY ZEKA GÜNDEM ÖZETİ
=====================================================

🌍 ULUSLARARASI GÜNDEMİN ÖNE ÇIKANI:
→ {intl_main}

🇹🇷 ULUSAL GÜNDEMDE ÖNE ÇIKAN:
→ {nat_main}

📌 Haberlerde en çok geçen kişiler:
{entities['PER'][:10]}

📌 En çok geçen şehir ve bölgeler:
{entities['LOC'][:10]}

📌 En çok geçen kurumlar:
{entities['ORG'][:10]}

=====================================================
"""

print(gpt_summary())


#######################################################################
#            HABER BAŞLIKLARI LİSTELEME SON SORU
#######################################################################

while True:
    choice = input("\nHaber başlıklarını listelemek ister misiniz? (y/q): ").strip().lower()

    if choice == "q":
        print("\n[✔] Uygulama kapatıldı.")
        break

    elif choice == "y":
        print("\n================ TÜM HABER BAŞLIKLARI ================\n")

        for src, items in news_by_source.items():
            print(f"\n------------ {src} ------------\n")
            for i, t in enumerate(items, 1):
                print(f"{i}. {t}")

        print("\n======================================================")

    else:
        print("Geçersiz seçim, tekrar deneyin.")
