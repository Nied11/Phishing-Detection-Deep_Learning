import requests

API_KEY = "6876cc5df49a61b88a9554d840c9f5a10bc830509d473a713d1da336e08c38aa"
headers = {"x-apikey": API_KEY}

response = requests.get("https://www.virustotal.com/api/v3/users/me", headers=headers)

print(response.json())
