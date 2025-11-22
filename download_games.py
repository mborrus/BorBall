import urllib.request
import os

url = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
filename = "games.csv"

print(f"Downloading {url} to {filename}...")
try:
    urllib.request.urlretrieve(url, filename)
    print("Download complete.")
    print(f"File size: {os.path.getsize(filename)} bytes")
except Exception as e:
    print(f"Error downloading file: {e}")
