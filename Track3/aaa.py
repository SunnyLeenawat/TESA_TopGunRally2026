import requests

response = requests.get("http://161.246.199.191:8000/trade")
html_content = response.text
print(html_content)
