# Interactive Story Expansion with Validation

This project implements a Flask-based web application that allows users to expand stories interactively. The system validates user-provided actions against the story context using a fine-tuned validation model and generates story continuations using a pre-trained language model.

## 🚀 Features
- **Action Validation**: Ensures user-provided actions align with the story context using a sequence classification model.
- **Story Expansion**: Generates coherent and meaningful story continuations based on user input.
- **Interactive Web Interface**: Provides a simple web interface for input and output.
- **Customizable Models**: Supports swapping in different pre-trained or fine-tuned models.

## 📂 Models Used
1. **Text Generation Model**:
   - Pre-trained model: `gpt2`
   - Generates story continuations.
2. **Validation Model**:
   - Fine-tuned model for validating user actions based on context.
   - Assumes three classes: Entailment, Neutral, Contradiction.

## 🛠 Setup

### Prerequisites
- Python 3.x
- Flask
- PyTorch
- Transformers (Hugging Face library)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository.git
   cd your-repository
   ```
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the fine-tuned validation model and tokenizer in the specified directories.
4. Ensure the GPT-2 model is available for text generation.

### File Structure
- `app.py`: Main Flask application.
- `templates/index.html`: Web interface template.
- `models/`: Directory for storing fine-tuned models.

## ▶️ Usage
1. Start the Flask app:
   ```bash
   python app.py
   ```
2. Open a web browser and navigate to `http://127.0.0.1:5000/`.
3. Enter the story context and a user action in the interface.
4. Submit the form to receive validation results and a generated continuation of the story.

## 📚 Key Functions

### Action Validation
- Validates user actions using the fine-tuned sequence classification model.
- Returns one of three labels:
  - **Entailment**: Action aligns with the story.
  - **Neutral**: Action neither strongly aligns nor contradicts.
  - **Contradiction**: Action conflicts with the story context.

### Story Expansion
- Generates story continuations using GPT-2.
- Accepts user action and story context as input.
- Produces meaningful continuations by sampling from the language model.

## 🔧 Customization
- **Models**: Replace `gpt2` or the fine-tuned validation model with your custom models.
- **Routes**: Modify `/process` to include or exclude validation as needed.
- **Generation Parameters**: Adjust `max_length`, `temperature`, `top_k`, and `top_p` for better story quality.

## 🤝 Contributions
Contributions are welcome! Feel free to fork the repository and submit a pull request.

