
import requests
import os
from dotenv import load_dotenv
load_dotenv()

with open(os.getenv("IMAGE_FILE"), "rb") as f:
    response = requests.post(
        "http://10.10.11.11:2222/generate_shape/image",
        params={
            "guidance_scale": 3.0,
            "num_steps": 64,
            "output_format": "ply"
        },
        files={"file": f}
    )

print(response.json())