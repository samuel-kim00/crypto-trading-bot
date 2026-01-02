from binance.client import Client
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def test_connection():
    try:
        # Initialize Binance client
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        print("Testing connection to Binance...")
        print(f"API Key found: {'Yes' if api_key else 'No'}")
        print(f"API Secret found: {'Yes' if api_secret else 'No'}")
        print(f"API Key: {api_key}")
        if api_secret:
            print(f"API Secret: {api_secret[:4]}{'*' * (len(api_secret)-8)}{api_secret[-4:]}")
        else:
            print("API Secret: None")
        
        client = Client(api_key, api_secret)
        
        # Test API connection
        server_time = client.get_server_time()
        print("\nConnection successful!")
        print(f"Server time: {server_time}")
        
        # Get account information
        account = client.get_account()
        print("\nAccount Information:")
        for balance in account['balances']:
            if float(balance['free']) > 0 or float(balance['locked']) > 0:
                print(f"{balance['asset']}: Free={balance['free']}, Locked={balance['locked']}")
                
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    test_connection() 