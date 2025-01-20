from flask import Flask, request, jsonify
from rdkit import Chem
from transformers import BertTokenizer, BertModel

app = Flask(__name__)

# Load
