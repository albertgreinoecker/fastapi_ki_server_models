'''
curl -X POST http://10.10.11.11:2222/generate \
     -H "Content-Type: application/json" \
     -d '{"text": "a wooden table"}'
'''

import requests

url = "http://10.10.11.11:2222/generate_shape/text"

headers = {
    "Content-Type": "application/json"
}

data = {
    "prompt": "man with banana in the hand",
    "num_steps": 64
}

response = requests.post(url, headers=headers, json=data)

print("Status Code:", response.status_code)
print("Response:")
print(response.text)


'''
Download-URL: http://10.10.11.11:2222/download/3cfc116d938f40e2b0695ce752cbf8cb.ply
'''
