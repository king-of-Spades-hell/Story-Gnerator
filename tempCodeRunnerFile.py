from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Load models
generation_model_name = "gpt2"  # Replace with your fine-tuned model if available
generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
generation_model = AutoModelForCausalLM.from_pretrained(generation_model_name)
generation_tokenizer.pad_token = generation_tokenizer.eos_token

# Define action validation function
def validate_actions(context, action):
    inputs = generation_tokenizer(context, action, return_tensors="pt", truncation=True, padding=True)
    generation_model.eval()
    with torch.no_grad():
        outputs = generation_model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    labels = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
    return labels[prediction]

# Define story expansion function
def expand_story(story_context, user_action, max_length=250):
    prompt = f"{story_context} User Action: {user_action} Next Phase:"
    inputs = generation_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    generation_model.eval()
    with torch.no_grad():
        outputs = generation_model.generate(
            **inputs,
            max_length=max_length + len(inputs['input_ids'][0]),
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            no_repeat_ngram_size=2,
            eos_token_id=generation_tokenizer.eos_token_id
        )
    generated_text = generation_tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_story_part = generated_text[len(prompt):].strip()
    meaningful_end_index = max(new_story_part.rfind('.'), new_story_part.rfind('?'), new_story_part.rfind('!'))
    if meaningful_end_index != -1:
        new_story_part = new_story_part[:meaningful_end_index + 1]
    return new_story_part

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    story_context = request.form.get("story_context")
    user_action = request.form.get("user_action")

    if not story_context or not user_action:
        return jsonify({"error": "Both story context and user action are required."})

    validation_result = validate_actions(story_context, user_action)
    if validation_result == "Contradiction":
        return jsonify({"error": "Action rejected: This action contradicts the current story. Please modify your action."})
    else:
        story = expand_story(story_context, user_action)
        return jsonify({
            "validation_result": validation_result,
            "new_story": story
        })

if __name__ == "__main__":
    app.run(debug=True)
