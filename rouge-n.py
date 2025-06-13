import torch
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from rouge_score import rouge_scorer

# Load the sentiment analysis pipeline
analyzer = pipeline("text-classification",
                    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

def sentiment_analyzer(review):
    sentiment = analyzer(review)
    return sentiment[0]['label']

def sentiment_bar_chart(df):
    sentiment_counts = df['Sentiment'].value_counts()

    # Create a pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    sentiment_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=['green', 'red'])
    ax.set_title('Review Sentiment Distribution')
    ax.set_ylabel('')  # Remove the y-label for better visualization
    return fig

def compute_rouge_scores(reviews, reference):
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = []

    for review in reviews:
        scores = scorer.score(review, reference)
        rouge_scores.append({
            "ROUGE-1": scores["rouge1"].fmeasure,
            "ROUGE-2": scores["rouge2"].fmeasure,
            "ROUGE-L": scores["rougeL"].fmeasure
        })

    return pd.DataFrame(rouge_scores)

def read_reviews_and_analyze_sentiment(file_object, reference_text):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_object)

    # Check if 'Reviews' column is in the DataFrame
    if 'Reviews' not in df.columns:
        raise ValueError("Excel file must contain a 'Reviews' column.")

    # Apply the sentiment_analyzer function to each review in the DataFrame
    df['Sentiment'] = df['Reviews'].apply(sentiment_analyzer)

    # Generate the sentiment distribution chart
    chart_object = sentiment_bar_chart(df)

    # Compute ROUGE scores
    rouge_df = compute_rouge_scores(df['Reviews'], reference_text)

    # Merge ROUGE scores into the original DataFrame
    result_df = pd.concat([df, rouge_df], axis=1)

    # Save results to Excel
    output_file = "sentiment_analysis_results.xlsx"
    result_df.to_excel(output_file, index=False)

    return result_df, chart_object, output_file

# Create the Gradio interface
def process_file(file, reference):
    return read_reviews_and_analyze_sentiment(file, reference)

demo = gr.Interface(
    fn=process_file,
    inputs=[
        gr.File(file_types=[".xlsx"], label="Upload your review comment file"),
        gr.Textbox(label="Reference Text", placeholder="Enter the reference text for ROUGE analysis")
    ],
    outputs=[
        gr.Dataframe(label="Sentiments and ROUGE Scores"),
        gr.Plot(label="Sentiment Analysis Chart"),
        gr.File(label="Download Results as Excel")
    ],
    title="@GenAILearniverse Project 3: Sentiment and ROUGE Analyzer",
    description="This application analyzes sentiments and computes ROUGE scores based on the file uploaded. The file should contain a 'Reviews' column."
)

# Launch the Gradio app
demo.launch()
