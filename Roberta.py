import torch
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# Load the sentiment analysis pipeline
analyzer = pipeline("text-classification", model="bhadresh-savani/roberta-base-emotion")

def sentiment_analyzer(review):
    sentiment = analyzer(review)
    return sentiment[0]['label']

def sentiment_bar_chart(df):
    sentiment_counts = df['Sentiment'].value_counts()

    # Create a pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    sentiment_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=['green', 'red', 'blue', 'orange', 'purple', 'yellow'])
    ax.set_title('Review Sentiment Distribution')
    ax.set_ylabel('')  # Remove the y-label for better visualization
    return fig

def read_reviews_and_analyze_sentiment(file_object):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_object)

    # Check if 'Reviews' column is in the DataFrame
    if 'Reviews' not in df.columns:
        raise ValueError("Excel file must contain a 'Reviews' column.")

    # Apply the sentiment_analyzer function to each review in the DataFrame
    df['Sentiment'] = df['Reviews'].apply(sentiment_analyzer)

    # Generate the sentiment distribution chart
    chart_object = sentiment_bar_chart(df)

    return df, chart_object

# Create the Gradio interface
demo = gr.Interface(
    fn=read_reviews_and_analyze_sentiment,
    inputs=gr.File(file_types=[".xlsx"], label="Upload your review comment file"),
    outputs=[
        gr.Dataframe(label="Sentiments"),
        gr.Plot(label="Sentiment Analysis Chart")
    ],
    title="@GenAILearniverse Project 3: Sentiment Analyzer",
    description="This application analyzes sentiments based on the file uploaded. The file should contain a 'Reviews' column."
)

# Launch the Gradio app
demo.launch()
