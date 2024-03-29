from transformers import AutoModelForSequenceClassification, AutoTokenizer
import gradio as gr

# Load model and tokenizer
model_name = "checkpoint-2500/"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = logits.softmax(dim=1)
    sentiment = "Positive" if probabilities[0][1] > 0.5 else "Negative"
    return sentiment

# Create Gradio interface
text_input = gr.Textbox(lines=7, label="Input Text", placeholder="Enter your text here...")
output_text = gr.Textbox(label="Predicted Sentiment")

# Author information
author = "Ajeetkumar Ukande"

# Create Gradio interface
interface = gr.Interface(predict_sentiment, text_input, output_text, 
             title="<div style='color: #336699; font-size: 24px; font-weight: bold; border: 2px solid #336699; padding: 10px; border-radius: 10px;'>Sentiment-Analysis-FineTuned-DistilBERT</div>", 
             description=f"""<div style='color: #666666; font-family: Arial, sans-serif;'>
                             <p style='margin-top: 10px;'>This model predicts the sentiment of text.</p>
                             <p>It uses a fine-tuned DistilBERT model trained on IMDb movie reviews dataset.</p>
                             <p>The sentiment is classified as Positive if the probability of positive sentiment is greater than 0.5, otherwise it's classified as Negative.</p>
                             <p>Developed by <span style='color: #336699; font-weight: bold;'>{author}</span>.</p>
                             </div>""", 
             theme="huggingface",
             allow_flagging=False,
             )

interface.launch(share=True)
