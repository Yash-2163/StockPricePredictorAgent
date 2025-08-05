# In stock_agent.py

import pandas as pd
import yfinance as yf
from typing import TypedDict
from langgraph.graph import StateGraph, END
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error


# Update the state to hold prediction results
class AgentState(TypedDict):
    ticker: str
    data: pd.DataFrame
    prediction: dict  # To hold the prediction for the next day
    model_confidence: float # A score for how confident we are


######################################## Agent-1  
# Add this function to stock_agent.py

def fetch_data_node(state: AgentState) -> AgentState:
    """
    Agent 1: Fetches historical stock data from yfinance and cleans it.
    This function acts as a node in our LangGraph.
    """
    print("---AGENT 1: FETCHING HISTORICAL DATA---")
    ticker = state["ticker"]

    try:
        # Fetch data for the last 2 years, which is good for training models
        stock_data = yf.download(ticker, period="2y", progress=False)

        if stock_data.empty:
            print(f"ERROR: No data found for ticker '{ticker}'.")
            return {**state, "data": pd.DataFrame()}

        # Basic data cleaning
        stock_data.reset_index(inplace=True) # Move 'Date' from index to a column
        stock_data.dropna(inplace=True)     # Remove any rows with missing values

        print(f"Successfully fetched {len(stock_data)} data points for {ticker}.")

        # Update the state with the cleaned data
        return {**state, "data": stock_data}

    except Exception as e:
        print(f"An error occurred in fetch_data_node: {e}")
        return {**state, "data": pd.DataFrame()}


########################################### Agent-2  
# Add these imports at the top of your file
# import xgboost as xgb
# from sklearn.metrics import mean_absolute_percentage_error

def model_runner_node(state: AgentState) -> AgentState:
    """
    Agent 2: Trains an XGBoost model and makes a prediction.
    """
    print("---AGENT 2: RUNNING PREDICTION MODEL---")
    data = state["data"]

    if data.empty or len(data) < 50: # Need enough data to train
        print("Not enough data to train model. Aborting.")
        return {**state, "prediction": {}, "model_confidence": 0.0}

    # 1. Feature Engineering: Create features from the date
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['lag_1'] = df['Close'].shift(1) # Add previous day's close as a feature
    df.dropna(inplace=True)

    # 2. Define Features (X) and Target (y)
    features = ['dayofweek', 'month', 'year', 'lag_1', 'Open', 'High', 'Low', 'Volume']
    target = 'Close'
    X, y = df[features], df[target]

    # 3. Train the XGBoost Model
    # We use the last 30 days for validation to prevent overfitting
    X_train, X_val = X[:-30], X[-30:]
    y_train, y_val = y[:-30], y[-30:]

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        early_stopping_rounds=10, # Stops training if validation error doesn't improve
        n_jobs=-1 # Use all available CPU cores
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # 4. Predict the Next Day
    # We use the features from the very last day of data to predict the next one
    next_day_features = X.iloc[[-1]]
    prediction_value = model.predict(next_day_features)[0]

    # 5. Calculate a Confidence Score
    # We use 1 minus the error on our validation set (1 - MAPE)
    val_preds = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, val_preds)
    confidence = max(0, 1 - mape) # Ensure confidence is not negative

    print(f"Prediction for next day's close: ${prediction_value:.2f}")
    print(f"Model Confidence (1 - MAPE): {confidence:.2%}")

    prediction_result = {
        "predicted_close": float(prediction_value),
        "last_actual_close": y.iloc[-1]
    }

    return {**state, "prediction": prediction_result, "model_confidence": confidence}
    

# Add this to the end of stock_agent.py to make it runnable for testing

# from langgraph.graph import StateGraph, END

def build_and_run_graph():
    # Define the workflow
    workflow = StateGraph(AgentState)

    # Add the nodes for each agent
    workflow.add_node("data_collector", fetch_data_node)
    workflow.add_node("model_runner", model_runner_node)

    # Define the edges that connect the agents
    workflow.set_entry_point("data_collector")
    workflow.add_edge("data_collector", "model_runner") # Connect Agent 1 to Agent 2
    workflow.add_edge("model_runner", END) # End after Agent 2

    # Compile the graph into a runnable application
    app = workflow.compile()

    # --- Test Run ---
    print("---STARTING AGENT WORKFLOW---")
    ticker = "NVDA" # Let's try a different stock like Nvidia
    # Define the full initial state
    initial_state = {
        "ticker": ticker,
        "data": pd.DataFrame(),
        "prediction": {},
        "model_confidence": 0.0
    }
    final_state = app.invoke(initial_state)

    print("\n---WORKFLOW COMPLETE: FINAL STATE---")
    print(f"Ticker: {final_state['ticker']}")
    print(f"Prediction: {final_state['prediction']}")
    print(f"Confidence: {final_state['model_confidence']:.2%}")

# Update the main execution block
if __name__ == "__main__":
    build_and_run_graph()