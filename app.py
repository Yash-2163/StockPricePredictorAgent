# In app.py
import streamlit as st
import os
import io
import contextlib

# Import our main agent function from the other file
from stock_agent import run_full_analysis

# --- Page Configuration ---
st.set_page_config(
    page_title="AlphaStrat: AI Stock Strategist",
    page_icon="📈",
    layout="centered"
)

# --- UI Elements ---
st.title("AlphaStrat 📈")
st.write(
    "Your personal AI stock strategist. Provide a stock ticker and your email, "
    "and AlphaStrat will perform a comprehensive analysis and email you the report."
)

with st.form("input_form"):
    ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, NVDA)", value="GOOGL")
    email_input = st.text_input("Enter Your Email Address")
    submit_button = st.form_submit_button("Run Analysis")

# This block executes when the user clicks the button
if submit_button:
    if not ticker_input:
        st.error("Please enter a stock ticker.")
    elif not email_input:
        st.error("Please enter your email address.")
    else:
        # Show a spinner and create a placeholder for the log output
        with st.spinner(f"Running comprehensive analysis for {ticker_input.upper()}... This can take a minute."):
            log_stream = io.StringIO()
            
            # Redirect all print statements to our in-app log window
            with contextlib.redirect_stdout(log_stream):
                try:
                    # Call the main agent function from stock_agent.py
                    final_state = run_full_analysis(ticker_input, email_input)
                    status = final_state.get('communication_status', 'Status not available')
                    
                except Exception as e:
                    status = f"An error occurred: {e}"
            
            # Display the final status and the captured log
            if "SUCCESS" in status:
                st.success(f"Analysis complete! Final status: {status}")
            else:
                st.error(f"Analysis failed. Status: {status}")

            st.subheader("Agent Activity Log:")
            st.text_area("Log", log_stream.getvalue(), height=300)