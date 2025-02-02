# Document Structure Analysis

## Overview
This project focuses on **Document Structure Analysis**, utilizing image processing and deep learning techniques to analyze and extract meaningful structures from scanned documents or digital text images. The goal is to segment, classify, and extract document components such as paragraphs, tables, headings, and figures.

## Features
- **Text and Layout Extraction**: Identifies and extracts text blocks, tables, and images.
- **Document Segmentation**: Separates document components into distinct regions.
- **Optical Character Recognition (OCR)**: Converts extracted text regions into machine-readable text.
- **Deep Learning Integration**: Uses advanced models like Detectron2 or Mask R-CNN for structure detection.
- **Preprocessing and Enhancement**: Improves image quality and text readability.

## Installation
```bash
# Clone the repository
git clone https://github.com/Ridh1234/Document_Structure_analysis.git
cd Document_Structure_analysis

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
python analyze_document.py --input path/to/document.jpg
```
Options:
- `--input`: Path to the input document image.
- `--output`: Path to save the analyzed output (optional).

## Dataset
The project can work with publicly available datasets like **PRImA Layout Analysis Dataset** or custom-labeled document images.

## Model Training
To train a custom model for document analysis:
```bash
python train_model.py --dataset path/to/dataset --epochs 20
```

## Results
- Outputs bounding boxes around text, tables, and figures.
- Extracted text stored in structured format (JSON, CSV, or TXT).

## Future Improvements
- Enhance OCR accuracy using Tesseract or deep learning models.
- Fine-tune models for domain-specific document types.
- Implement real-time document analysis with a web-based interface.

## Contributors
- **Hridyansh Sharma**
- **Team HELIX**

## Acknowledgments
Special thanks to open-source libraries such as OpenCV, Detectron2, and PyTesseract for enabling document analysis.
