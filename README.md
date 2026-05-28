# End-to-End Document Understanding Pipeline
### CSE 655 - Deep Learning Final Project

> **📂 Data:** [Google Drive](https://drive.google.com/drive/folders/1BTx6A4gxQEnKDt6rq4URQlAIzPPnK3wO?usp=drive_link) | Local: `data/rvlcdip_split_v3`
>
> ⚠️ **Not:** Data klasörü Teams yükleme sınırını aştığı için proje dosyalarından çıkarılmıştır. Veri setini indirmek için yukarıdaki Google Drive bağlantısını kullanabilir veya [`notebooks/document-classification.ipynb`](notebooks/document-classification.ipynb) dosyasındaki indirme talimatlarına bakabilirsiniz.
>
> **🚀 Run Project Directly:** [Google Drive - Project Notebooks](https://drive.google.com/drive/folders/1DsUKZ1QlYlGXSEMjYSAIfu6O7Dbcy7UA)

This project implements an intelligent document processing pipeline that classifies documents (e.g., invoices, emails, forms), extracts text using OCR, and structures the information using an LLM.

---

## 1. Hardware Requirements
**Critical Note:** This pipeline utilizes the **Chandra OCR** model and **Swin Transformer**, which are computationally intensive.

* **Platform:** Google Colab Pro / Pro+ (Recommended).
* **GPU:** **NVIDIA A100** (Highly Recommended for inference).
    * *Note:* Using T4 GPUs may result in Out-of-Memory (OOM) errors or significantly slower performance during the OCR stage.
* **RAM:** High-RAM runtime (24GB+) is preferred.
* **Storage:** ~5GB free space for model weights.

---

## 2. Google Colab Quick Access

All notebooks are available in this repository. To run them on Google Colab:

https://drive.google.com/drive/folders/1DsUKZ1QlYlGXSEMjYSAIfu6O7Dbcy7UA?usp=sharing 

Alternatively, you can upload the notebooks directly from your local machine to Google Colab.

---

## 3. Installation Instructions

### Option A: Google Colab (Easiest)
No manual installation is required beforehand. The notebooks contain cell blocks at the beginning to install necessary libraries dynamically.
1. Upload the `.ipynb` files to your Google Drive or clone the repository.
2. Open the notebook via Google Colab.
3. **Select A100 GPU** from *Runtime > Change runtime type*.
4. Run the "Setup & Imports" cell to install dependencies automatically.

### Option B: Local Environment
If you prefer running locally, ensure you have CUDA drivers installed and run:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all dependencies
pip install -r requirements.txt
```

**requirements.txt** includes:
- Core: `torch`, `torchvision`, `torchaudio`
- Transformers: `transformers`, `accelerate`, `timm`
- Image Processing: `pillow`
- Data Science: `numpy`, `pandas`, `scikit-learn`
- OCR: `chandra-ocr`
- LLM: `groq`
- Utilities: `einops`, `addict`, `easydict`, `safetensors`
- Visualization: `matplotlib`, `seaborn`, `tqdm`
- Optional: `kagglehub` (for dataset download)

---

## 4. File Structure & Descriptions

| File Name | Description |
|-----------|-------------|
| `pipeline.ipynb` | **Main Entry Point.** The end-to-end inference script. It combines Classification, OCR, and LLM modules to process a raw image into structured JSON. |
| `document-classification.ipynb` | **Training Script (Experiment).** Contains code for fine-tuning the Swin Transformer on the RVL-CDIP dataset. |
| `ocr.ipynb` | **OCR Experiments (Comparison).** Benchmarks different OCR models (DeepSeek, Hunyuan, Chandra) to demonstrate why Chandra was selected. |
| `llm.ipynb` | **LLM Prompts (Experiment).** Isolated environment to test different prompt engineering strategies for JSON parsing. |

---

## 5. How to Run

### Step 1: Training Classification Model (Optional)
Perform this step only if you want to re-train the classifier from scratch.
1. Open `document-classification.ipynb`.
2. Ensure the dataset path is correct.
3. Run all cells to fine-tune the Swin Transformer.
4. The best model weights will be saved as `rvlcdip_swin_v2.pth`.

### Step 2: OCR Model Comparison (Optional)
Perform this step if you want to see the performance difference between OCR models.
1. Open `ocr.ipynb`.
2. Run the cells to compare DeepSeek, Hunyuan, and Chandra models on sample images.
3. Observe the output quality to understand the selection rationale.

### Step 3: LLM Prompt Testing (Optional)
Perform this step if you want to experiment with the prompt engineering logic.
1. Open `llm.ipynb`.
2. Input raw text (simulated OCR output).
3. Run the cells to see how the LLM structures the data into JSON based on the document type (Email vs. Form).

### Step 4: End-to-End Inference (Main Demo)
**Run this step to see the final product in action.**
1. Open `pipeline.ipynb`.
2. Make sure the model weights (`rvlcdip_swin_v2.pth`) are present in the working directory.
3. Upload a test image (e.g., an invoice or email image).
4. Run the `run_pipeline(image_path)` function.
5. **Output:** The system will display the predicted class, extracted text, and the final JSON object.

---

## 6. Project Workflow

```
Input Image → Classification (Swin) → OCR (Chandra) → LLM (Structuring) → JSON Output
```

1. **Classification:** Identifies document type using fine-tuned Swin Transformer
2. **OCR:** Extracts text from the image using Chandra OCR model
3. **LLM Processing:** Structures extracted text into JSON format based on document type
4. **Output:** Returns structured data ready for downstream applications

---

## 7. Dependencies

All dependencies are listed in `requirements.txt`. 

## 8. Notes

- The pipeline is optimized for documents like invoices, emails, and forms
- For best results, use high-quality input images
- The A100 GPU requirement is crucial for the Chandra OCR model performance
- Model weights must be downloaded or trained before running the inference pipeline

---

## 9. License

This project is created as part of CSE 655 - Deep Learning coursework.

---

## 10. Contact

For questions or issues, please refer to the course materials or contact the project maintainers.