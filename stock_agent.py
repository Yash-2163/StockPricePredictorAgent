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


# Update the state to hold prediction results
class AgentState(TypedDict):
    ticker: str
    data: pd.DataFrame
    prediction: dict  # To hold the prediction for the next day
    model_confidence: float # A score for how confident we are
    news: list[dict] # New field to hold a list of news articles
    analysis: dict  # New field for the final verdict and strategy
    communication_status: str # New field for email status




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
        "last_actual_close": float(y.iloc[-1])
    }

    return {**state, "prediction": prediction_result, "model_confidence": confidence}
    
############################################## Agent-3
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
    
########################################################### Agent4
# Add these imports at the top of your file
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_google_genai import ChatGoogleGenerativeAI

def synthesizer_node(state: AgentState) -> AgentState:
    """
    Agent 4: The "brain". Synthesizes prediction and news into a final verdict.
    """
    print("---AGENT 4: SYNTHESIZING DATA AND FORMULATING STRATEGY---")
    
    # 1. Initialize the LLM and the JSON output parser
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    parser = JsonOutputParser()

    # 2. Create a detailed prompt template
    template = """
    You are an expert, unbiased financial analyst providing a recommendation for a stock.
    You will be given a quantitative prediction from a time-series model and qualitative data from recent news articles.
    Your task is to synthesize this information into a coherent investment thesis.

    **Quantitative Data:**
    - Stock Ticker: {ticker}
    - Time-Series Model Prediction (Next Day's Close): ${predicted_close:.2f}
    - Last Known Actual Close: ${last_actual_close:.2f}
    - Model Confidence (based on historical accuracy): {model_confidence:.2%}

    **Qualitative Data (Recent News Snippets):**
    {news_snippets}
    ----------------
    
    **Analysis Instructions:**
    1.  **Summarize Sentiment:** Briefly summarize the overall sentiment from the news (Positive, Negative, Neutral).
    2.  **Identify Key Drivers:** What are the key factors from the news driving the sentiment (e.g., earnings beat, new product, regulatory concerns)?
    3.  **Synthesize:** Compare the model's quantitative prediction with the qualitative news sentiment.
        - If they align (e.g., model predicts an increase and news is positive), the confidence in the trend is high.
        - If they conflict (e.g., model predicts an increase but news is negative), explain the discrepancy and advise caution.
    4.  **Provide a Recommendation:** Based on your synthesis, provide a clear investment recommendation.
    
    You MUST provide your response in the following JSON format.
    {format_instructions}
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["ticker", "predicted_close", "last_actual_close", "model_confidence", "news_snippets"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # 3. Format the inputs to be passed to the LLM chain
    prediction_data = state['prediction']
    news_snippets = "\n\n".join(
        [f"URL: {article['url']}\nContent Snippet: {article['content']}" for article in state.get('news', [])]
    )
    
    # 4. Create and invoke the LangChain Expression Language (LCEL) chain
    chain = prompt | llm | parser
    
    try:
        analysis_result = chain.invoke({
            "ticker": state['ticker'],
            "predicted_close": prediction_data['predicted_close'],
            "last_actual_close": prediction_data['last_actual_close'],
            "model_confidence": state['model_confidence'],
            "news_snippets": news_snippets
        })
        print("Successfully generated final analysis.")
        return {**state, "analysis": analysis_result}
    except Exception as e:
        print(f"Error during analysis generation: {e}")
        return {**state, "analysis": {"error": "Failed to generate analysis.", "details": str(e)}}




############################################################### Agent5
# Add these imports for email functionality
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText

# In stock_agent.py

# ... (keep all other imports and agent functions as they are) ...

def format_crisp_analysis_as_html(analysis: dict, ticker: str, prediction: dict, confidence: float) -> str:
    """
    Helper function to format a crisp, summary-style HTML email.
    """
    rec = analysis.get('recommendation', 'Hold')
    rec_class = "buy" if rec == "Buy" else "sell" if rec == "Sell" else "hold"
    
    # Safely get the final justification
    thesis = analysis.get('investmentThesis', {})
    final_rec = thesis.get('finalRecommendation', {})
    justification = final_rec.get('justification', 'No summary available.')

    html = f"""
    <html><head><style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; font-size: 16px; color: #333; }}
        .container {{ width: 90%; max-width: 600px; margin: 15px auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px; background-color: #f9f9f9; }}
        h1 {{ font-size: 24px; color: #111; text-align: center; }}
        .recommendation {{ font-size: 28px; font-weight: bold; text-align: center; padding: 12px; margin: 15px 0; border-radius: 8px; }}
        .buy {{ background-color: #28a745; color: white; }}
        .sell {{ background-color: #dc3545; color: white; }}
        .hold {{ background-color: #ffc107; color: #333; }}
        p {{ line-height: 1.5; }}
        .footer {{ font-size: 12px; text-align: center; color: #777; margin-top: 20px;}}
    </style></head><body><div class="container">
        <h1>Stock Verdict: {ticker.upper()}</h1>
        <div class="recommendation {rec_class}">{rec}</div>
        <p><b>Quantitative Prediction:</b> The model predicts a next-day close of <b>${prediction.get('predicted_close'):.2f}</b> with <b>{confidence:.2%}</b> confidence.</p>
        <p><b>Analyst's Take:</b> {justification}</p>
        <p class="footer">This is an automated analysis and not financial advice.</p>
    </div></body></html>
    """
    return html

def communicator_node(state: AgentState) -> AgentState:
    """Agent 5: Formats the analysis and sends it via a crisp email."""
    print("---AGENT 5: COMMUNICATING CRISP RESULTS---")
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
    subject = f"Stock Verdict for {ticker.upper()}: {analysis.get('recommendation', 'N/A')}"
    
    # *** This is the key change: Using the new crisp HTML function ***
    html_body = format_crisp_analysis_as_html(
        analysis,
        ticker,
        state['prediction'],
        state['model_confidence']
    )

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
        print(f"Successfully sent crisp email report to {recipient_email}")
        return {**state, "communication_status": "SUCCESS"}
    except Exception as e:
        print(f"Failed to send email: {e}")
        return {**state, "communication_status": f"FAILED: {e}"}












# Add this to the end of stock_agent.py to make it runnable for testing

# from langgraph.graph import StateGraph, END

def build_and_run_graph():
    # Define the workflow
    workflow = StateGraph(AgentState)

    # Add all the nodes
    workflow.add_node("data_collector", fetch_data_node)
    workflow.add_node("model_runner", model_runner_node)
    workflow.add_node("news_fetcher", news_fetcher_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("communicator", communicator_node) # Add the communicator

    # Define the graph's edges, ending with the communicator
    workflow.set_entry_point("data_collector")
    workflow.add_edge("data_collector", "model_runner")
    workflow.add_edge("model_runner", "news_fetcher")
    workflow.add_edge("news_fetcher", "synthesizer")
    workflow.add_edge("synthesizer", "communicator") # Connect Agent 4 to Agent 5
    workflow.add_edge("communicator", END)           # End the workflow

    # Compile the graph
    app = workflow.compile()

    # --- Test Run ---
    print("---STARTING AGENT WORKFLOW (V1.0 COMPLETE)---")
    ticker = "META" # Let's try Meta
    initial_state = {
        "ticker": ticker,
        "data": pd.DataFrame(),
        "prediction": {},
        "model_confidence": 0.0,
        "news": [],
        "analysis": {},
        "communication_status": "" # Add the new field
    }
    final_state = app.invoke(initial_state)

    print("\n---WORKFLOW COMPLETE---")
    print(f"Final email status: {final_state.get('communication_status')}")




# This block ensures the code runs only when the script is executed directly
if __name__ == "__main__":
    build_and_run_graph()