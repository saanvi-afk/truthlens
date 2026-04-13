import os
import sys
import plotly.graph_objects as go

# Ensure src in path to import the trainer module
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from src.trainer import train_and_save_all

def create_grouped_bar_chart():
    print("Fetching evaluation metrics. Models will be evaluated (and trained if missing)...")
    
    # train_and_save_all returns a dictionary of evaluation dictionaries 
    results = train_and_save_all(force_retrain=False)
    
    # The keys returned by train_and_save_all for the models
    model_keys = [
        ('fn_lr', 'LR (Fake News)'),
        ('fn_nb', 'NB (Fake News)'),
        ('fn_rf', 'RF (Fake News)'),
        ('pr_lr', 'LR (Propaganda)'),
        ('pr_nb', 'NB (Propaganda)'),
        ('pr_rf', 'RF (Propaganda)')
    ]
    
    # Separate the keys, labels, and metrics
    labels = [label for key, label in model_keys]
    keys = [key for key, label in model_keys]
    
    accuracies = [results[k]['accuracy'] for k in keys]
    precisions = [results[k]['precision'] for k in keys]
    recalls = [results[k]['recall'] for k in keys]

    # Create the Plotly figure
    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=labels, y=accuracies, marker_color='#A7F4BC'),
        go.Bar(name='Precision', x=labels, y=precisions, marker_color='#A7D4F4'),
        go.Bar(name='Recall', x=labels, y=recalls, marker_color='#F4D7A7')
    ])

    # Update layout to match TruthLens UI
    fig.update_layout(
        barmode='group',
        title="TruthLens Model Performance: Accuracy, Precision, and Recall",
        yaxis_title="Score (0.0 to 1.0)",
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
        font=dict(color='#e0e0e0', family='Space Grotesk, sans-serif'),
        legend=dict(x=1.01, y=1, bgcolor='rgba(0,0,0,0)'),
        yaxis=dict(range=[0, 1.1])
    )
    
    # Save the chart as an interactive HTML file directly in the project
    output_path = os.path.join(PROJECT_DIR, 'model_performance_chart.html')
    fig.write_html(output_path)
    
    print(f"\n==============================================")
    print(f"Interactive Chart successfully saved to: {output_path}")
    print(f"You can open this HTML file directly in your browser.")
    print(f"==============================================")

if __name__ == '__main__':
    create_grouped_bar_chart()
