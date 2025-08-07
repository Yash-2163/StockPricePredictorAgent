# In stock_agent.py

import pandas as pd
import yfinance as yf
from typing import TypedDict
from langgraph.graph import StateGraph, END
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
import os
from tavily import TavilyClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
# from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
# --- Pydantic Version Compatibility Shim ---
# This block checks the installed Pydantic version at runtime and imports the
# correct BaseModel to ensure compatibility with LangChain's requirements.
import pydantic

# Use pydantic.VERSION to check the major version
if pydantic.VERSION.startswith('1.'):
    from pydantic.v1 import BaseModel, Field
    print("--- Using Pydantic V1 compatibility mode. ---")
else:
    # For modern Pydantic V2 environments
    from pydantic import BaseModel, Field
    print("--- Using Pydantic V2. ---")
# --- End of Shim ---


# Add these imports from pydantic
# from pydantic import BaseModel, Field

# This class IS our fixed schema for the final analysis
class StrategyAnalysis(BaseModel):
    timeline_strategy: str = Field(description="A clear, time-based action. Examples: 'This looks like a solid short-term (1-2 weeks) buying opportunity.' or 'High risk suggests avoiding this stock for the next month.'")
    adjusted_prediction: str = Field(description="The final adjusted prediction with a specific price and timeframe. Example: 'I believe the price could reach ~$155 in the next 7-10 days.'")
    confidence_score: str = Field(description="A confidence score for the final recommendation. Examples: 'High', 'Medium', 'Low'.")
    justification: str = Field(description="A brief paragraph explaining the reasoning that synthesizes all the data points into the final strategy.")


# Update the state to hold prediction results
class AgentState(TypedDict):
    ticker: str
    data: pd.DataFrame
    prediction: list[dict] # Changed to a list for the 30-day forecast
    model_confidence: float # A score for how confident we are
    news: list[dict] # New field to hold a list of news articles
    # analysis: dict  # New field for the final verdict and strategy
    analysis: StrategyAnalysis # Changed from dict to our new Pydantic model
    communication_status: str # New field for email status
    anomalies: list[dict] # New field for price anomalies
    analyst_ratings: dict # New field for Wall Street consensus
    upcoming_events: str # New field for calendar events
    risk_assessment: dict # New field for the formal risk report


# In stock_agent.py


######################################## Agent-1  
# Add this function to stock_agent.py

# In stock_agent.py

def fetch_data_node(state: AgentState) -> AgentState:
    """
    Agent 1: Fetches historical stock data from yfinance and cleans it. (Corrected)
    """
    print("---AGENT 1: FETCHING HISTORICAL DATA---")
    ticker = state["ticker"]

    try:
        stock_data = yf.download(ticker, period="2y", progress=False)

        # --- NEW FIX: FLATTEN COLUMN INDEX ---
        # yfinance can return a multi-level column index. We flatten it here.
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)
            print("Column index flattened.")

        if stock_data.empty:
            print(f"ERROR: No data found for ticker '{ticker}'.")
            return {**state, "data": pd.DataFrame()}

        # Basic data cleaning
        stock_data.reset_index(inplace=True) # Move 'Date' from index to a column
        stock_data.dropna(inplace=True)

        print(f"Successfully fetched {len(stock_data)} data points for {ticker}.")

        return {**state, "data": stock_data}

    except Exception as e:
        print(f"An error occurred in fetch_data_node: {e}")
        return {**state, "data": pd.DataFrame()}


########################################### Agent-2  
# Add these imports at the top of your file
# import xgboost as xgb
# from sklearn.metrics import mean_absolute_percentage_error

def model_runner_node(state: AgentState) -> AgentState:
    """Agent 2: Trains a model and creates a recursive 30-day forecast."""
    print("---AGENT 2: RUNNING 30-DAY FORECAST MODEL---")
    data = state["data"]

    if data.empty or len(data) < 50:
        print("Not enough data to train model.")
        return {**state, "prediction": [], "model_confidence": 0.0}

    # 1. Feature Engineering
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['lag_1'] = df['Close'].shift(1)
    df.dropna(inplace=True)

    features = ['dayofweek', 'month', 'year', 'lag_1', 'Open', 'High', 'Low', 'Volume']
    target = 'Close'
    X, y = df[features], df[target]

    # 2. Train the Model (on all data except the last validation set)
    X_train, X_val = X[:-30], X[-30:]
    y_train, y_val = y[:-30], y[-30:]
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, early_stopping_rounds=10, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Calculate a single confidence score based on recent performance
    val_preds = model.predict(X_val)
    mape = mean_absolute_percentage_error(y_val, val_preds)
    confidence = max(0, 1 - mape)
    print(f"Model Confidence (1 - MAPE): {confidence:.2%}")

    # 3. Perform 30-Day Recursive Forecast
    future_predictions = []
    last_row = df.iloc[-1]
    current_features = X.iloc[[-1]].copy()

    for i in range(30):
        # Predict the next day
        next_pred = model.predict(current_features)[0]
        
        # Get the date for the prediction
        next_date = last_row.name + pd.Timedelta(days=i + 1)
        
        future_predictions.append({"date": next_date.strftime('%Y-%m-%d'), "predicted_close": float(next_pred)})

        # Update features for the *next* iteration
        current_features['lag_1'] = next_pred
        # We can carry forward other features or use simple estimates
        current_features['Open'] = next_pred
        current_features['High'] = next_pred * 1.01 # Estimate a 1% high
        current_features['Low'] = next_pred * 0.99  # Estimate a 1% low

    print(f"Generated a 30-day forecast, starting with {future_predictions[0]['predicted_close']:.2f}")
    return {**state, "prediction": future_predictions, "model_confidence": confidence}


############################################## Agent-3

# In stock_agent.py

def anomaly_detector_node(state: AgentState) -> AgentState:
    """
    New Agent (v1.1): Detects price anomalies using Bollinger Bands. (Corrected)
    """
    print("---ANOMALY DETECTION AGENT---")
    data = state['data'].copy()
    if data.empty or len(data) < 20:
        print("Not enough data for anomaly detection.")
        return {**state, "anomalies": []}

    # Calculate Bollinger Bands
    window = 20
    data['MA20'] = data['Close'].rolling(window=window).mean()
    data['STD20'] = data['Close'].rolling(window=window).std()
    data['Upper_Band'] = data['MA20'] + (data['STD20'] * 2)
    data['Lower_Band'] = data['MA20'] - (data['STD20'] * 2)
    
    # Drop rows with NaN values that were created by the rolling function
    data.dropna(inplace=True)

    # --- VECTORIZED FIX START ---
    # Perform vectorized comparison on the last 90 days of data
    recent_data = data.tail(90).copy()
    
    surges = recent_data[recent_data['Close'] > recent_data['Upper_Band']]
    drops = recent_data[recent_data['Close'] < recent_data['Lower_Band']]

    anomalies_found = []
    # Loop ONLY over the found anomalies to format them
    for index, row in surges.iterrows():
        anomalies_found.append({
            "date": row['Date'].strftime('%Y-%m-%d'),
            "type": "Positive Anomaly (Price Surge)",
            "details": f"Price (${row['Close']:.2f}) surged above the upper Bollinger Band (${row['Upper_Band']:.2f})."
        })
    
    for index, row in drops.iterrows():
        anomalies_found.append({
            "date": row['Date'].strftime('%Y-%m-%d'),
            "type": "Negative Anomaly (Price Drop)",
            "details": f"Price (${row['Close']:.2f}) dropped below the lower Bollinger Band (${row['Lower_Band']:.2f})."
        })
    # --- VECTORIZED FIX END ---

    if anomalies_found:
        # Sort anomalies by date for cleaner reporting
        anomalies_found.sort(key=lambda x: x['date'], reverse=True)
        print(f"Found {len(anomalies_found)} anomalies in the last 90 days.")
    else:
        print("No significant price anomalies found in the last 90 days.")
        
    return {**state, "anomalies": anomalies_found}


######################################################################## Agent4

def analyst_ratings_node(state: AgentState) -> AgentState:
    """New Agent: Fetches consensus analyst ratings and price targets."""
    print("---ANALYST RATINGS AGENT---")
    ticker_str = state['ticker']
    try:
        # We get all the info we need from the .info dictionary
        ticker_info = yf.Ticker(ticker_str).info
        
        consensus = ticker_info.get('recommendationKey', 'N/A')
        target_price = ticker_info.get('targetMeanPrice', 'N/A')
        num_analysts = ticker_info.get('numberOfAnalystOpinions', 'N/A')

        # yfinance returns 'buy', 'hold', etc. Let's make it look nice.
        if isinstance(consensus, str):
            consensus = consensus.replace('_', ' ').title()

        ratings_summary = {
            "consensus_rating": consensus,
            "target_price": target_price,
            "number_of_analysts": num_analysts
        }
        print(f"Fetched Analyst Ratings: {ratings_summary}")
        return {**state, "analyst_ratings": ratings_summary}
    except Exception as e:
        print(f"Could not fetch analyst ratings: {e}")
        return {**state, "analyst_ratings": {"error": str(e)}}



################################################################## Agent-5
# # Add this import at the top of your file
# from datetime import datetime

def event_awareness_node(state: AgentState) -> AgentState:
    """New Agent: Checks for upcoming corporate events like earnings."""
    print("---EVENT AWARENESS AGENT---")
    ticker_str = state['ticker']
    try:
        ticker = yf.Ticker(ticker_str)
        calendar_df = ticker.calendar

        if calendar_df is None or calendar_df.empty:
            event_summary = "No upcoming corporate events calendar found."
            print(event_summary)
            return {**state, "upcoming_events": event_summary}

        # The 'Earnings Date' is the key event we care about
        earnings_row = calendar_df.T['Earnings Date'] # Transpose to easily access by column name
        dates = earnings_row.dropna().tolist()

        # Check if any of these dates are in the future
        future_dates = [d for d in dates if isinstance(d, pd.Timestamp) and d.date() >= datetime.now().date()]
        
        event_list = []
        if future_dates:
            start_date = min(future_dates).strftime('%Y-%m-%d')
            end_date = max(future_dates).strftime('%Y-%m-%d')
            date_str = f"on {start_date}" if start_date == end_date else f"between {start_date} and {end_date}"
            event_list.append(f"Upcoming Earnings Date {date_str}. Expect increased volatility.")

        event_summary = " ".join(event_list) if event_list else "No significant upcoming events found."
        
        print(f"Event Check: {event_summary}")
        return {**state, "upcoming_events": event_summary}
    except (KeyError, IndexError):
        # This can happen if 'Earnings Date' isn't in the calendar or is empty
        event_summary = "No earnings date found in the calendar."
        print(event_summary)
        return {**state, "upcoming_events": event_summary}
    except Exception as e:
        print(f"Could not fetch corporate calendar: {e}")
        return {**state, "upcoming_events": "Error fetching event data."}
    
############################################## Agent-6
# Add these imports at the top of your file
# import os
# from tavily import TavilyClient

def news_fetcher_node(state: AgentState) -> AgentState:
    """
    Agent 3: Fetches recent news and events for the stock using Tavily.
    """
    print("---AGENT 3: FETCHING NEWS & EVENTS---")
    ticker = state['ticker']
    try:
        # Use yfinance to get the company's full name for a better search query
        company_info = yf.Ticker(ticker).info
        company_name = company_info.get('longName', ticker)

        # Initialize the Tavily client with your API key
        tavily_client = TavilyClient(api_key=os.environ['TAVILY_API_KEY'])
        
        # Create a detailed search query for the last month
        query = (f"What is the most recent and relevant news for {company_name} ({ticker}) "
                 f"in the last month? Focus on earnings, product launches, "
                 f"analyst ratings, and market sentiment.")

        # Use the Tavily client to search
        response = tavily_client.search(
            query=query,
            search_depth="advanced", # Use "advanced" for more thorough results
            max_results=5            # Get the top 5 most relevant articles
        )
        
        # Format the results into a clean list of dictionaries
        news_articles = [{"url": res["url"], "content": res["content"]} for res in response.get("results", [])]
        
        print(f"Found {len(news_articles)} relevant news articles for {company_name}.")
        return {**state, "news": news_articles}

    except Exception as e:
        print(f"Error in news_fetcher_node: {e}")
        return {**state, "news": []}
    
    
########################################################### Agent7

def risk_assessor_node(state: AgentState) -> AgentState:
    """New Agent: Assesses risks based on news and price anomalies."""
    print("---RISK ASSESSOR AGENT---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    parser = JsonOutputParser()

    template = """
    You are a cautious Risk Management Officer. Your sole job is to identify and summarize potential risks based on the provided data.
    You must ignore positive or neutral information and focus exclusively on negative signals.

    **Data Point 1: Price Anomaly Data (look only for 'Negative Anomaly' types):**
    {anomalies}

    **Data Point 2: News Snippets (scan for negative keywords like 'lawsuit', 'downgrade', 'disappointing', 'concerns', 'volatile', 'risk'):**
    {news}
    ----------------
    
    **Your Task:**
    1.  Review the data for any signs of risk.
    2.  Summarize the key risks you've identified in a brief paragraph for the `risk_summary`.
    3.  Assign an overall `risk_level` (one of: Low, Medium, or High). Base this on the severity and number of negative signals.

    Provide your output in the specified JSON format. If no risks are found, state that clearly in the summary and set the risk level to "Low".
    {format_instructions}
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["anomalies", "news"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Prepare inputs: filter for only negative anomalies to help the LLM focus
    negative_anomalies = [a for a in state.get('anomalies', []) if "Negative" in a.get('type', '')]
    anomaly_report = "\n".join([f"- {a['details']}" for a in negative_anomalies]) if negative_anomalies else "No negative price anomalies detected."
    news_snippets = "\n\n".join([f"Snippet: {article['content']}" for article in state.get('news', [])])

    chain = prompt | llm | parser
    try:
        risk_result = chain.invoke({
            "anomalies": anomaly_report,
            "news": news_snippets
        })
        print(f"Generated Risk Assessment: {risk_result}")
        return {**state, "risk_assessment": risk_result}
    except Exception as e:
        print(f"Error during risk assessment: {e}")
        return {**state, "risk_assessment": {"error": "Failed to generate risk assessment.", "details": str(e)}}
    
########################################################### Agent8
# Add these imports at the top of your file
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_google_genai import ChatGoogleGenerativeAI

def synthesizer_node(state: AgentState) -> AgentState:
    """Agent 'Brain': Uses Pydantic to enforce a fixed schema for its output."""
    print("---AGENT 'BRAIN': CREATING FINAL STRATEGY (WITH FIXED SCHEMA)---")
    
    # Use our new Pydantic model to create a parser
    parser = PydanticOutputParser(pydantic_object=StrategyAnalysis)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

    # The prompt is the same, but the parser will now inject more detailed format instructions
    template = """
    You are "Alpha," a lead financial strategist... (Your existing detailed prompt here) ...Your final output must be in the specified JSON format that matches the schema.
    {format_instructions}
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["quantitative_forecast", "analyst_summary", "events_summary", "risk_level", "risk_summary", "news_snippets"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Format all inputs (this part is the same as before)
    prediction_list = state.get('prediction', [])
    quantitative_forecast = "Day 1: ${:.2f}, Day 7: ${:.2f}, Day 30: ${:.2f}".format(
        prediction_list[0]['predicted_close'],
        prediction_list[6]['predicted_close'],
        prediction_list[29]['predicted_close']
    ) if len(prediction_list) >= 30 else "Forecast not available."
    ratings = state.get('analyst_ratings', {})
    risk = state.get('risk_assessment', {})
    news_snippets = "\n".join([f"- {article['content'][:150]}..." for article in state.get('news', [])])
    analyst_summary = f"Consensus: '{ratings.get('consensus_rating', 'N/A')}', Target: ${ratings.get('target_price', 'N/A')}"
    
    chain = prompt | llm | parser
    try:
        # The result is now a Pydantic object, not a dictionary!
        analysis_result = chain.invoke({
            "quantitative_forecast": quantitative_forecast,
            "analyst_summary": analyst_summary,
            "events_summary": state.get('upcoming_events', 'N/A'),
            "risk_level": risk.get('risk_level', 'N/A'),
            "risk_summary": risk.get('risk_summary', 'N/A'),
            "news_snippets": news_snippets
        })
        print("Successfully generated final strategic analysis with fixed schema.")
        return {**state, "analysis": analysis_result}
    except Exception as e:
        print(f"Error during final synthesis: {e}")
        # Create a default error object if generation fails
        error_analysis = StrategyAnalysis(
            timeline_strategy="Error", adjusted_prediction="Error", 
            confidence_score="N/A", justification=f"Failed to generate analysis: {e}"
        )
        return {**state, "analysis": error_analysis}


############################################################### Agent9
# Add these imports for email functionality
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText

# In stock_agent.py

# ... (keep all other imports and agent functions as they are) ...
def format_crisp_analysis_as_html(analysis: StrategyAnalysis, ticker: str) -> str:
    """Formats the Pydantic StrategyAnalysis object into a crisp HTML email."""
    # We now access attributes directly (e.g., analysis.timeline_strategy)
    recommendation = analysis.timeline_strategy
    rec_lower = recommendation.lower()
    rec_class = "buy" if "buy" in rec_lower or "invest" in rec_lower else \
                "sell" if "sell" in rec_lower or "avoid" in rec_lower else "hold"

    html = f"""
    <html><head><style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-size: 16px; color: #333; }}
        .container {{ width: 90%; max-width: 600px; margin: 15px auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px; background-color: #f9f9f9; }}
        h1, h2 {{ text-align: center; color: #111; }} h2 {{ font-size: 24px; font-weight: bold; margin: 15px 0; }}
        .recommendation {{ padding: 12px; border-radius: 8px; color: white; }}
        .buy {{ background-color: #28a745; }} .sell {{ background-color: #dc3545; }} .hold {{ background-color: #ffc107; color: #333; }}
        .section {{ margin-top: 20px; border-top: 1px solid #eee; padding-top: 15px; }}
        p {{ line-height: 1.5; }} b {{ color: #000; }}
        .footer {{ font-size: 12px; text-align: center; color: #777; margin-top: 20px;}}
    </style></head><body><div class="container">
        <h1>Alpha's Strategy for {ticker.upper()}</h1>
        <h2 class="recommendation {rec_class}">{recommendation}</h2>
        
        <div class="section">
            <p><b>Final Adjusted Prediction:</b> {analysis.adjusted_prediction}</p>
            <p><b>My Confidence:</b> {analysis.confidence_score}</p>
        </div>
        
        <div class="section">
            <p><b>Justification:</b> {analysis.justification}</p>
        </div>

        <p class="footer">This is an automated analysis from your personal AI strategist, Alpha.</p>
    </div></body></html>
    """
    return html

def communicator_node(state: AgentState) -> AgentState:
    """Agent 5: Formats the final STRATEGIC analysis and sends it via email."""
    print("---AGENT 5: COMMUNICATING STRATEGIC RESULTS---")
    analysis = state.get('analysis', {})
    if "error" in analysis or not analysis:
        print("Analysis contains an error or is empty. Skipping communication.")
        return {**state, "communication_status": "SKIPPED"}

    sender_email = os.environ.get("EMAIL_SENDER")
    password = os.environ.get("EMAIL_PASSWORD")
    recipient_email = os.environ.get("EMAIL_RECIPIENT")
    
    if not all([sender_email, password, recipient_email]):
        print("Email credentials not set. Skipping email.")
        return {**state, "communication_status": "FAILED: Credentials not set."}
        
    ticker = state['ticker']
    subject = f"Alpha's Strategy for {ticker.upper()}"
    
    # This call is now simpler and correctly uses the new HTML function
    html_body = format_crisp_analysis_as_html(analysis, ticker)

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(html_body, 'html'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, password)
            server.send_message(msg)
        print(f"Successfully sent strategic email report to {recipient_email}")
        return {**state, "communication_status": "SUCCESS"}
    except Exception as e:
        print(f"Failed to send email: {e}")
        return {**state, "communication_status": f"FAILED: {e}"}












# Add this to the end of stock_agent.py to make it runnable for testing

# from langgraph.graph import StateGraph, END

# def build_and_run_graph():
#     workflow = StateGraph(AgentState)

#     # Add all nodes
#     workflow.add_node("data_collector", fetch_data_node)
#     workflow.add_node("model_runner", model_runner_node)
#     workflow.add_node("anomaly_detector", anomaly_detector_node)
#     workflow.add_node("analyst_ratings", analyst_ratings_node)
#     workflow.add_node("event_awareness", event_awareness_node)
#     workflow.add_node("news_fetcher", news_fetcher_node)
#     workflow.add_node("risk_assessor", risk_assessor_node) # New node
#     workflow.add_node("synthesizer", synthesizer_node)
#     workflow.add_node("communicator", communicator_node)

#     # Define the final sequence of edges
#     workflow.set_entry_point("data_collector")
#     workflow.add_edge("data_collector", "model_runner")
#     workflow.add_edge("model_runner", "anomaly_detector")
#     workflow.add_edge("anomaly_detector", "analyst_ratings")
#     workflow.add_edge("analyst_ratings", "event_awareness")
#     workflow.add_edge("event_awareness", "news_fetcher")
#     workflow.add_edge("news_fetcher", "risk_assessor")     # New step
#     workflow.add_edge("risk_assessor", "synthesizer")      # Then synthesize everything
#     workflow.add_edge("synthesizer", "communicator")
#     workflow.add_edge("communicator", END)

#     app = workflow.compile()

#     # --- Test Run ---
#     print("---STARTING AGENT WORKFLOW (v1.3 with Risk Assessment)---")
#     ticker = "DIS" # Let's try Disney
#     initial_state = {
#         "ticker": ticker, "data": pd.DataFrame(), "prediction": {}, "model_confidence": 0.0,
#         "news": [], "analysis": {}, "communication_status": "", "anomalies": [],
#         "analyst_ratings": {}, "upcoming_events": "", "risk_assessment": {} # Add the new field
#     }
#     final_state = app.invoke(initial_state)

#     print("\n---WORKFLOW COMPLETE---")
#     print(f"Final email status: {final_state.get('communication_status')}")



# # This block ensures the code runs only when the script is executed directly
# if __name__ == "__main__":
#     build_and_run_graph()





















def run_full_analysis(ticker: str, recipient_email: str):
    """
    This function initializes and runs the entire agent graph for a given
    ticker and sends the result to the specified email.
    """
    # This is crucial: it sets the recipient's email for the communicator agent
    os.environ['EMAIL_RECIPIENT'] = recipient_email

    # The rest of this is the logic from our old build_and_run_graph function
    workflow = StateGraph(AgentState)

    # Add all nodes...
    workflow.add_node("data_collector", fetch_data_node)
    workflow.add_node("model_runner", model_runner_node)
    workflow.add_node("anomaly_detector", anomaly_detector_node)
    workflow.add_node("analyst_ratings", analyst_ratings_node)
    workflow.add_node("event_awareness", event_awareness_node)
    workflow.add_node("news_fetcher", news_fetcher_node)
    workflow.add_node("risk_assessor", risk_assessor_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("communicator", communicator_node)

    # Define the graph's edges...
    workflow.set_entry_point("data_collector")
    workflow.add_edge("data_collector", "model_runner")
    workflow.add_edge("model_runner", "anomaly_detector")
    workflow.add_edge("anomaly_detector", "analyst_ratings")
    workflow.add_edge("analyst_ratings", "event_awareness")
    workflow.add_edge("event_awareness", "news_fetcher")
    workflow.add_edge("news_fetcher", "risk_assessor")
    workflow.add_edge("risk_assessor", "synthesizer")
    workflow.add_edge("synthesizer", "communicator")
    workflow.add_edge("communicator", END)

    app = workflow.compile()

    # Define the initial state for the run
    initial_state = {
        "ticker": ticker, "data": pd.DataFrame(), "prediction": [], "model_confidence": 0.0,
        "news": [], "analysis": None, "communication_status": "", "anomalies": [],
        "analyst_ratings": {}, "upcoming_events": "", "risk_assessment": {}
    }
    
    # Run the graph and return the final state
    print(f"---INVOKING AGENT WORKFLOW FOR {ticker.upper()}---")
    final_state = app.invoke(initial_state)
    return final_state