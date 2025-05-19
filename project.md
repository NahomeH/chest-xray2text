# CXR Report Summarization

## Goal
Summarize the "Findings" section of chest X-ray radiology reports using vision-language models.
This project implements a retrieval-based baseline before moving to generative models like BLIP-2.

---

## Baseline: Retrieval-Based Summarization

### Pipeline:
1. **Preprocess**
   - Load MIMIC-CXR dataset.
   - Extract "Findings" section from free-text reports.
   - Resize and normalize chest X-ray images to 224×224.

2. **Encode Training Images**
   - Use a pretrained vision encoder (CLIP or BLIP).
   - Store normalized feature vectors and associated text.

3. **Retrieve Nearest Neighbor**
   - Encode the test image.
   - Find closest match using cosine similarity.
   - Return its "Findings" section as the summary.

4. **Evaluate**
   - Compare generated vs. ground-truth summaries.
   - Metrics: ROUGE-L, BLEU-4.

---

## Future Direction
Once the baseline is complete, we'll fine-tune BLIP-2 on the same dataset to generate summaries directly from images.

---

## Files
- `src/preprocess.py`: Extracts text and processes images.
- `src/encode_images.py`: Generates and stores image embeddings.
- `src/retrieve.py`: Performs similarity search.
- `src/evaluate.py`: Calculates metrics.
