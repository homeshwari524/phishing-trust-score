import pandas as pd
import numpy as np
import re
import socket
import requests
import whois
import tldextract

from urllib.parse import urlparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample


# ===============================
#        URL Feature Extractors
# ===============================

def havingIP(url):
    try:
        return 1 if re.findall(r'[0-9]+(?:\.[0-9]+){3}', url) else 0
    except:
        return 0

def haveAtSign(url):
    return 1 if "@" in url else 0

def getLength(url):
    return 1 if len(url) >= 75 else 0

def getDepth(url):
    try:
        return len([i for i in urlparse(url).path.split('/') if i])
    except:
        return 0

def redirection(url):
    return 1 if "//" in urlparse(url).path else 0

def httpDomain(url):
    return 1 if "https" in urlparse(url).netloc else 0

def tinyURL(url):
    tiny_domains = [
        "bit.ly", "goo.gl", "shorte.st", "go2l.ink", "x.co", "ow.ly", "t.co",
        "tinyurl", "tr.im", "is.gd", "cli.gs", "yfrog.com", "migre.me", "ff.im",
        "tiny.cc", "url4.eu", "twit.ac", "su.pr", "twurl.nl", "snipurl.com", "short.to",
        "BudURL.com", "ping.fm", "post.ly", "Just.as", "bkite.com", "snipr.com", "fic.kr",
        "loopt.us", "doiop.com", "short.ie", "kl.am", "wp.me", "rubyurl.com", "om.ly",
        "to.ly", "bit.do", "t.ly"
    ]
    domain = urlparse(url).netloc
    return 1 if any(tiny in domain for tiny in tiny_domains) else 0

def prefixSuffix(url):
    return 1 if '-' in urlparse(url).netloc else 0

def getDomain(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"

def dnsRecord(domain):
    try:
        socket.gethostbyname(domain)
        return 0
    except:
        return 1

def web_traffic(url):
    try:
        domain = getDomain(url)
        response = requests.get(f"https://www.alexa.com/minisiteinfo/{domain}", timeout=5)
        return 0 if response.status_code == 200 else 1
    except:
        return 1

def domainAge(domain_name):
    try:
        whois_info = whois.whois(domain_name)
        creation_date = whois_info.creation_date
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if creation_date is None:
            return 1
        age = (datetime.now() - creation_date).days
        return 0 if age > 180 else 1
    except:
        return 1

def domainEnd(domain_name):
    try:
        whois_info = whois.whois(domain_name)
        expiration_date = whois_info.expiration_date
        if isinstance(expiration_date, list):
            expiration_date = expiration_date[0]
        if expiration_date is None:
            return 1
        end = (expiration_date - datetime.now()).days
        return 0 if end > 365 else 1
    except:
        return 1


# ===============================
#        Feature Extraction
# ===============================

FEATURE_NAMES = [
    'IpAddress',            # havingIP
    'AtSymbol',             # haveAtSign
    'UrlLength',            # getLength
    'PathLevel',            # getDepth
    'DoubleSlashInPath',    # redirection
    'HttpsInHostname',      # httpDomain
    'DomainInPaths',        # tinyURL
    'NumDashInHostname',    # prefixSuffix
    'SubdomainLevel',       # dnsRecord
    'NoHttps',              # web_traffic
    'DomainInSubdomains',   # domainAge
    'RandomString'          # domainEnd
]

def extract_features_from_url(url):
    domain = getDomain(url)
    return [
        havingIP(url),            # 1
        haveAtSign(url),          # 2
        getLength(url),           # 3
        getDepth(url),            # 4
        redirection(url),         # 5
        httpDomain(url),          # 6
        tinyURL(url),             # 7
        prefixSuffix(url),        # 8
        dnsRecord(domain),        # 9
        web_traffic(url),         # 10
        domainAge(domain),        # 11
        domainEnd(domain),        # 12
    ]

def extract_features_df(url):
    """
    Extracts features from a URL and returns a DataFrame
    with proper column names matching FEATURE_NAMES.
    Logs each feature with its corresponding value.
    """
    values = extract_features_from_url(url)
    print("[INFO] Feature Extraction Log:")
    for name, value in zip(FEATURE_NAMES, values):
        print(f"  - {name}: {value}")
    return pd.DataFrame([values], columns=FEATURE_NAMES)


# ===============================
#        Data Preprocessing
# ===============================

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    print(f"[INFO] Original shape: {df.shape}")
    
    df = df.drop_duplicates()
    print(f"[INFO] After removing duplicates: {df.shape}")

    #df['trust_score'] = df['CLASS_LABEL'].map({-1: 0, 1: 1})
    df['trust_score'] = df['CLASS_LABEL']

    df = df.drop(columns=['id', 'CLASS_LABEL'], errors='ignore')

    df = df[FEATURE_NAMES + ['trust_score']]

    phishing_df = df[df['trust_score'] == 0.0]
    legit_df = df[df['trust_score'] == 1.0]

    if phishing_df.empty or legit_df.empty:
        raise ValueError(
            f"[ERROR] Dataset imbalance: phishing samples = {len(phishing_df)}, legitimate samples = {len(legit_df)}"
        )

    min_len = min(len(phishing_df), len(legit_df))

    phishing_df_down = resample(phishing_df, replace=False, n_samples=min_len, random_state=42)
    legit_df_down = resample(legit_df, replace=False, n_samples=min_len, random_state=42)

    df_balanced = pd.concat([phishing_df_down, legit_df_down])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df_balanced.drop(columns=['trust_score'])
    y = df_balanced['trust_score'].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler


# ===============================
#        Test Entry Point
# ===============================

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess("dataset/phishing.csv")

    print("[TEST] Sample training feature row:", X_train[0])
    print("[TEST] Sample label:", y_train[0])

    # Test feature extraction on a live URL
    url = "http://tinyurl.com/fake-site"
    test_df = extract_features_df(url)
    print("[TEST] Extracted features for test URL:")
    print(test_df)
