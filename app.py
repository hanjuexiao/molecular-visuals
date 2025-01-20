# Import necessary libraries
from flask import Flask, request, jsonify
from rdkit import Chem
from transformers import BertTokenizer, BertModel

# Initialize the Flask app
app = Flask(__name__)

# Define the AI model for prediction
def predict_property(smiles):
    # Load the ChemBERTa tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('deepchem/ChemBERTa-77M-MLM')
    model = BertModel.from_pretrained('deepchem/ChemBERTa-77M-MLM')

    # Tokenize the SMILES string
    inputs = tokenizer(smiles, return_tensors="pt")
    outputs = model(**inputs)

    # Generate a dummy property (replace with a real model later)
    embedding = outputs.last_hidden_state.mean(dim=1).item()
    return f"Reactivity Score: {embedding:.2f}"

# Define the API route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    smiles = data.get('smiles')

    if not smiles:
        return jsonify({'error': 'No SMILES string provided'}), 400

    # Parse the molecule and compute hybridization
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return jsonify({'error': 'Invalid SMILES string'}), 400

    hybridizations = [atom.GetHybridization().name for atom in molecule.GetAtoms()]

    # Use AI to predict properties
    predicted_property = predict_property(smiles)

    # Return the results
    return jsonify({
        'hybridization': hybridizations,
        'predicted_property': predicted_property
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
