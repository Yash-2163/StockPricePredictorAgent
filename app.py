import streamlit as st
import os
import io
import contextlib
import traceback


from dotenv import load_dotenv
load_dotenv() # This line reads the .env file and loads the variables

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
    "and AlphaStrat will perform a comprehensive analysis, email you the report, "
    "and show a preview below."
)

with st.form("input_form"):
    ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, NVDA)", value="GOOGL")
    email_input = st.text_input("Enter Your Email Address")
    submit_button = st.form_submit_button("Run Analysis")

if submit_button:
    if not ticker_input:
        st.error("Please enter a stock ticker.")
    elif not email_input:
        st.error("Please enter your email address.")
    else:
        # Show a spinner and create a placeholder for the log output
        with st.spinner(f"Running comprehensive analysis for {ticker_input.upper()}... This can take a minute."):
            log_stream = io.StringIO()

            with contextlib.redirect_stdout(log_stream), contextlib.redirect_stderr(log_stream):
                try:
                    result = run_full_analysis(ticker_input, email_input)
                    final_state = result["state"]
                    html_preview = result.get("html_report")
                    status = final_state.get('communication_status', 'Status not available')
                except Exception as e:
                    traceback.print_exc(file=log_stream)
                    status = f"An error occurred: {e}"
                    html_preview = None

            if "SUCCESS" in status:
                st.success(f"Analysis complete! Final status: {status}")
            else:
                st.error(f"Analysis failed. Status: {status}")

            st.subheader("Agent Activity Log:")
            st.text_area("Log", log_stream.getvalue(), height=300)

            if html_preview:
                st.subheader("📄 Report Preview")
                st.markdown(html_preview, unsafe_allow_html=True)
