# 🧠 Two-Stage Domain-Adaptive RoBERTa for IT Ticket Classification

---

## 📌 Overview

This project implements a **two-stage RoBERTa-based pipeline** to classify IT service desk tickets.

### 🧩 Approach

- **Stage 1 (MLM)**: Domain adaptation using unlabeled ticket data.
- **Stage 2 (Classification)**: Supervised fine-tuning for ticket categorization.

✅ Both stages are fully implemented and executed in the **Kaggle Notebook**.

---

## 🚀 Architecture

```text
Dataset (ticket_description.csv)
       ↓
Data Preprocessing
       ↓
Stage 1: MLM (Domain Adaptation)
       ↓
Stage 2: Classification Fine-Tuning
       ↓
Trained Model (roberta_it_ticket_classifier/)
       ↓
Streamlit App (Inference)
```

---

## 📂 Repository Structure

Based on the project directory (excluding items in `.gitignore`):

```text
.
├── app.py                     # Streamlit web application & Inference logic
├── config.yaml                # Configuration settings
├── requirements.txt           # Project dependencies
├── test_runner.py             # Internal tests and validation script
├── dataset/                   # Dataset directory
│   └── ticket_description.csv # Ticket dataset
├── notebook/                  # Full ML training pipeline
├── docs/                      # Documentation
└── README.md                  # Project information
```

*(Note: Directories like `__pycache__`, `itroberta_output_files/`, and `test_csvs/` are ignored per `.gitignore` and are generated locally during execution.)*

---

## 🧾 Dataset

The dataset is expected to be placed under:
`dataset/ticket_description.csv`

### Required Columns:
- `ticket_description`
- `ticket_category`

---

## 🧑‍💻 How to Run the Project

### 🔹 STEP 1: Train Model (Kaggle / Colab)

⚠️ **Training is recommended on GPU environments (not locally)**

#### 👉 Open Notebook
- Upload the notebook file from the `notebook/` folder to **Kaggle** or **Google Colab**.

#### 👉 Upload Dataset
- Upload the required dataset: `dataset/ticket_description.csv`

#### 👉 Run All Cells
The notebook performs the complete pipeline:
- ✅ Data preprocessing  
- ✅ Stage 1: MLM training (domain adaptation)  
- ✅ Stage 2: Classification fine-tuning  
- ✅ Model evaluation (Accuracy, F1, Confusion Matrix)  
- ✅ Saves the trained model  

#### 📦 Output
After training, you will get a model folder: 
`roberta_it_ticket_classifier/`

👉 **This folder is mandatory for local inference.**

---

### 🔹 STEP 2: Run Streamlit App (Local)

#### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

#### 2️⃣ Create a Virtual Environment

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

#### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4️⃣ Add the Trained Model
Copy the trained model folder (`roberta_it_ticket_classifier/`) from Kaggle/Colab and paste it into your project root:
```text
project_root/
├── app.py
├── roberta_it_ticket_classifier/   ✅ REQUIRED
└── ...
```

#### 5️⃣ Run the Application
```bash
streamlit run app.py
```

---

## 🔮 Inference Example

If you want to use the predictor programmatically:

```python
from app import load_predictor, predict_ticket

# Load the model and tokenizer
tokenizer, model, id2label = load_predictor()

# Make a prediction
result = predict_ticket(
    "VPN not connecting after password reset",
    tokenizer,
    model,
    id2label
)

print(result)
```

**📊 Example Output:**
```json
{
  "predicted_category": "access",
  "confidence_score": 0.92,
  "top_3_predictions": [
    {"label": "access", "score": 0.92},
    {"label": "network", "score": 0.05},
    {"label": "hardware", "score": 0.02}
  ]
}
```

---

## ⚠️ Common Issues

| Problem | Solution |
|---------|----------|
| **Model not found** | Ensure `roberta_it_ticket_classifier/` exists in the root directory. |
| **Notebook not running** | Enable GPU acceleration in Kaggle/Colab notebook settings. |
| **Missing columns** | Ensure your dataset contains `ticket_description` and `ticket_category`. |
| **Slow inference** | Run on hardware with a GPU or optimize the model. |

---

## 🧩 Key Concepts

- 📓 **Notebook** → Full ML pipeline (training + evaluation)
- 🖥️ **app.py** → Model inference & Streamlit UI logic
- 📁 **Model folder** → Bridge between remote training and local deployment

---

## 🛠️ Future Improvements
- [ ] API deployment (FastAPI/Flask)
- [ ] HuggingFace model hosting
- [ ] Real-time ticket routing
- [ ] Active learning pipeline
- [ ] MLOps integration