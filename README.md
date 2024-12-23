# Documentation for Fine-Tuning Vulnerability Detection Model

## **Overview**
This documentation provides a detailed explanation of the resources used, approach taken, and requirements for fine-tuning a pre-trained model (e.g., CodeBERT) to classify vulnerabilities in smart contracts. The fine-tuning process is focused on detecting two specific vulnerabilities: **Reentrancy** and **Time-Dependency**. The dataset is derived from Slither-audited smart contracts.

---

## **Resources Used**
### **Hardware**
- **GPU**: A CUDA-enabled GPU (e.g., NVIDIA GTX 1650 or higher) for efficient fine-tuning.

### **Software**
- **Python**: Version 3.9 or later.
- **Transformers Library**: For utilizing pre-trained models like CodeBERT.
- **Hugging Face Datasets**: For handling the dataset and tokenization.
- **PyTorch**: Backend for model training.
- **Slither Static Analyzer**: For extracting vulnerabilities and generating labeled datasets.
- **CUDA Toolkit**: Version 11.8 (for GPU acceleration).

---




## **Dataset**
### **Source**
- Hugging Face Dataset: `mwritescode/slither-audited-smart-contracts`
- SmartBugs: https://github.com/smartbugs/smartbugs-wild.git

### **Structure**
The dataset contains the following fields:
- `address`: Contract address.
- `source_code`: Solidity code of the contract.
- `vulnerability_type`: Labeled vulnerabilities for each contract.
- `code_snippet`: Extracted function-level code from contracts.

### **Preprocessing**
1. **Data Cleaning**:
   - Remove comments, whitespace, and unnecessary characters from the source code.
2. **Function Extraction**:
   - Use Slither source mapping to extract function-level code associated with vulnerabilities.
3. **Label Mapping**:
   - Normalize labels to match desired categories: `"Reentrancy"` and `"Time-Dependency"`.
4. **Tokenization**:
   - Use the CodeBERT tokenizer to tokenize the code snippets, ensuring a maximum length of 512 tokens.
5. **Data Split**:
   - Split the dataset into **90% training** and **10% testing**.

---



## **Fine-Tuning Approach**
### **Model**
- **Base Model**: `microsoft/codebert-base` (pre-trained on programming languages).

### **Training Pipeline**
1. **Input Format**:
   - Each record includes `"code_snippet"` and `"vulnerability_type"`.
2. **Tokenization**:
   - Ensure tokenized inputs are padded to 512 tokens.
   - Truncate inputs exceeding the maximum token length.
3. **Training Arguments**:
   - Learning Rate: `2e-5`
   - Batch Size: `8`
   - Epochs: `3`
   - Mixed Precision: Enabled (if supported by the GPU).
4. **Loss Function**:
   - Use cross-entropy loss for classification.
5. **Evaluation**:
   - Compute metrics like **accuracy**, **precision**, **recall**, and **F1-score** on the test set.

---

## **Requirements**
### **Hardware**
- NVIDIA GPU with CUDA support.

### **Software Packages**
Install the following Python packages using `pip`:
```bash
pip install torch transformers datasets scikit-learn slither-analyzer
```

### **Additional Tools**
- **Solidity Compiler**:
  - Install using `solc-select` to manage multiple versions.
- **Slither Static Analyzer**:
  - Install using: `pip install slither-analyzer`

---

## **Output**
### **Fine-Tuned Model**
- Saved in the `./fine_tuned_model` directory.
- Includes the model weights, tokenizer, and label mapping (`label_mapping.json`).

### **Metrics**
- Final metrics after training are logged, including accuracy, precision, recall, and F1-score.

### **Inference**
- The fine-tuned model can classify new Solidity code snippets as `"Reentrancy"`, `"Time-Dependency"`.

---

## **Limitations and Future Work**
1. **Dataset Size**:
   - The current dataset contains limited examples. Expanding the dataset can improve performance.
2. **Class Imbalance**:
   - Address imbalance by oversampling underrepresented classes or using weighted loss functions.
3. **Vulnerability Coverage**:
   - Extend the fine-tuning to include additional vulnerabilities like **Unchecked External Calls** and **tx.origin Authentication**.
4. **Explainability**:
   - Integrate tools to provide explanations for model predictions.
5. **Balanced Dataset**:
   - A more balanced dataset can assure greater meterics
     

---

## **Conclusion**
This fine-tuning pipeline is designed to provide a proof of concept for detecting vulnerabilities in smart contracts. The approach leverages pre-trained models and static analysis to classify vulnerabilities accurately. With more data and optimization, this pipeline can be scaled to production-level systems.
