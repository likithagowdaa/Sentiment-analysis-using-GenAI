import torch
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Load the pipelines
analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


# Sentiment analysis function with Neutral handling
def sentiment_analyzer(review):
    sentiment = analyzer(review)
    confidence_threshold = 0.7  # Define a threshold for Neutral classification
    label = sentiment[0]['label']
    confidence = sentiment[0]['score']

    # Map labels and add Neutral based on confidence threshold
    if confidence < confidence_threshold:
        label = "Neutral"
    else:
        label = "Positive" if label == "POSITIVE" else "Negative"

    return label, confidence


# Cluster reviews using SentenceTransformer embeddings
def cluster_reviews(reviews, num_clusters=3):
    embeddings = embedding_model.encode(reviews)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    return clusters


# Generate a sentiment distribution chart
def sentiment_bar_chart(df):
    sentiment_counts = df['Sentiment'].value_counts()

    # Create a pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    sentiment_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=['green', 'red', 'blue'])
    ax.set_title('Review Sentiment Distribution')
    ax.set_ylabel('')  # Remove the y-label for better visualization
    return fig


# Compute ROUGE scores for reviews against a reference text
def compute_rouge_scores(reviews, reference):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    individual_scores = []
    total_rouge1, total_rouge2, total_rougeL = 0, 0, 0

    for review in reviews:
        scores = scorer.score(review, reference)
        individual_scores.append({
            "ROUGE-1": scores["rouge1"].fmeasure,
            "ROUGE-2": scores["rouge2"].fmeasure,
            "ROUGE-L": scores["rougeL"].fmeasure
        })
        total_rouge1 += scores["rouge1"].fmeasure
        total_rouge2 += scores["rouge2"].fmeasure
        total_rougeL += scores["rougeL"].fmeasure

    # Calculate averages
    avg_rouge1 = total_rouge1 / len(reviews)
    avg_rouge2 = total_rouge2 / len(reviews)
    avg_rougeL = total_rougeL / len(reviews)

    return individual_scores, {
        "Average ROUGE-1": avg_rouge1,
        "Average ROUGE-2": avg_rouge2,
        "Average ROUGE-L": avg_rougeL
    }


# Process uploaded reviews file
def read_reviews_and_analyze_sentiment(file_object, reference_text, num_clusters):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_object)

    # Check if 'Reviews' column is in the DataFrame
    if 'Reviews' not in df.columns:
        raise ValueError("Excel file must contain a 'Reviews' column.")

    # Apply the sentiment analyzer function to each review
    df[['Sentiment', 'Confidence']] = df['Reviews'].apply(
        lambda review: pd.Series(sentiment_analyzer(review))
    )

    # Perform clustering on reviews
    df['Cluster'] = cluster_reviews(df['Reviews'].tolist(), num_clusters)

    # Generate the sentiment distribution chart
    chart_object = sentiment_bar_chart(df)

    # Compute ROUGE scores for the dataset and individual reviews
    individual_scores, rouge_aggregate = compute_rouge_scores(df['Reviews'], reference_text)

    # Add individual ROUGE scores to the DataFrame
    rouge_df = pd.DataFrame(individual_scores)
    result_df = pd.concat([df, rouge_df], axis=1)

    # Save results to Excel
    output_file = "sentiment_analysis_results_with_features.xlsx"
    result_df.to_excel(output_file, index=False)

    # Format aggregate ROUGE scores for display
    rouge_display = (
        f"Aggregate ROUGE Scores for Dataset:\n\n"
        f"- Average ROUGE-1: {rouge_aggregate['Average ROUGE-1']:.4f}\n"
        f"- Average ROUGE-2: {rouge_aggregate['Average ROUGE-2']:.4f}\n"
        f"- Average ROUGE-L: {rouge_aggregate['Average ROUGE-L']:.4f}"
    )

    return result_df, chart_object, rouge_display, output_file


# Create the Gradio interface
def process_file(file, reference, num_clusters):
    return read_reviews_and_analyze_sentiment(file, reference, num_clusters)


demo = gr.Interface(
    fn=process_file,
    inputs=[
        gr.File(file_types=[".xlsx"], label="Upload your review comment file"),
        gr.Textbox(label="Reference Text", placeholder="Enter the reference text for ROUGE analysis"),
        gr.Slider(2, 10, value=3, step=1, label="Number of Clusters for Sentiment Clustering"),
    ],
    outputs=[
        gr.Dataframe(label="Sentiments, Clusters, and ROUGE Scores"),
        gr.Plot(label="Sentiment Analysis Chart"),
        gr.Markdown(label="ROUGE Scores Summary"),
        gr.File(label="Download Results as Excel")
    ],
    title="Enhanced Sentiment and ROUGE Analyzer",
    description="This application analyzes sentiments, clusters reviews, and computes ROUGE scores. The file should contain a 'Reviews' column."
)

# Launch the Gradio app
demo.launch(share=True)