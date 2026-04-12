# test_discord.py
import requests

# 🚨 Paste your actual webhook URL strictly inside the quotes below 🚨
WEBHOOK_URL = "https://discord.com/api/webhooks/1487199773256585239/Ci6G7sxUSvvlNPVdT2ng4CpbtTn69t0u7bVn9AupakQa5YYMoZkk7t3GuhnyM4BCWZov"


def test_connection():
    print("Initiating Discord handshake...")

    payload = {
        "content": "⚾ **SYSTEM CHECK SUCCESSFUL** ⚾\nSUCCESS! YOUR THE MAN!"
    }

    try:
        response = requests.post(WEBHOOK_URL, json=payload)

        if response.status_code in [200, 204]:
            print("\n[SUCCESS] The payload was delivered! Check your Discord server.")
        else:
            print(f"\n[FAILED] Discord rejected the payload. Error Code: {response.status_code}")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Could not connect: {e}")


if __name__ == "__main__":
    test_connection()