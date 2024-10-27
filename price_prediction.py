import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import requests
import logging
import matplotlib.pyplot as plt

# Function to fetch historical data for a cryptocurrency pair
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Modify the fetch_historical_data function to include logging
def fetch_historical_data(pair, days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{pair}/market_chart?vs_currency=usd&days={days}&interval=daily"
    response = requests.get(url)
    
    if response.status_code != 200:
        logging.error(f"Error fetching data for {pair}: {response.status_code}")
        return pd.DataFrame()  # Return an empty DataFrame on error
    
    data = response.json()
    
    logging.info(f"Data fetched for {pair}: {data}")
    
    if 'prices' not in data:
        logging.warning(f"'prices' key not found in data for {pair}. Response: {data}")
        return pd.DataFrame()
    
    prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
    prices.set_index('timestamp', inplace=True)
    return prices


# Function to predict price
# Function to predict price
def predict_price(pair):
    data = fetch_historical_data(pair)
    
    # Check if data is empty
    if data.empty:
        return None

    data['day'] = np.arange(len(data))  # Add a day index for linear regression

    # Prepare data for training
    X = data[['day']]
    y = data['price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the price for the next day
    next_day = pd.DataFrame([[len(data)]], columns=['day'])  # Use DataFrame with column name
    predicted_price = model.predict(next_day)

    # Define target boundaries
    lower_bound = predicted_price * 0.95
    upper_bound = predicted_price * 1.05

    return {
        'predicted_price': predicted_price[0],
        'lower_bound': lower_bound[0],
        'upper_bound': upper_bound[0]
    }


# Example usage for multiple pairs
crypto_pairs = ['bitcoin', 'ethereum', 'ripple']  # Add up to 50 pairs as needed
predictions = {}

for pair in crypto_pairs:
    predictions[pair] = predict_price(pair)

# Display predictions
for pair, pred in predictions.items():
    print(f"{pair.capitalize()}: Predicted Price: {pred['predicted_price']:.2f}, "
          f"Lower Bound: {pred['lower_bound']:.2f}, Upper Bound: {pred['upper_bound']:.2f}")





def visualize_prediction(pair, prices, predicted_price, lower_bound, upper_bound):
    plt.figure(figsize=(10, 5))
    plt.plot(prices.index, prices['price'], label='Historical Prices', color='blue')
    plt.axhline(y=predicted_price, color='green', linestyle='--', label='Predicted Price')
    plt.axhline(y=lower_bound, color='red', linestyle='--', label='Lower Bound')
    plt.axhline(y=upper_bound, color='orange', linestyle='--', label='Upper Bound')
    plt.title(f"{pair.capitalize()} Price Prediction")
    plt.xlabel('Date')
    plt.ylabel('Price in USD')
    plt.legend()
    plt.show()

# Update the prediction loop to include visualization
for pair in crypto_pairs:
    prediction = predict_price(pair)
    if prediction:
        prices = fetch_historical_data(pair)  # Fetch historical data again for plotting
        visualize_prediction(pair, prices, prediction['predicted_price'], prediction['lower_bound'], prediction['upper_bound'])



       

