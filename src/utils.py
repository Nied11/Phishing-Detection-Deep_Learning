import re
from urllib.parse import urlparse

def has_ip_address(url):
    return 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0

def count_special_chars(url):
    special_chars = ['.', '-', '@', '?', '&', '|', '=', '_', '~', '%', '/', '*', ':', ',', ';', '$', ' ']
    return {char: url.count(char) for char in special_chars}
