# Handwritten Character Recognition (CNN + CRNN)

```
pip install -r requirements.txt
python train_cnn.py --split balanced --epochs 10      # characters  -> cnn_balanced.pt  (~90% on 47 classes)
python train_crnn.py --epochs 15 --samples 20000      # words       -> crnn.pt          (use a GPU if possible)
python predict_text.py letter.png --mode char
python predict_text.py page.png                       # words/sentences with CRNN
python app.py                                         # draw-and-recognise web demo
```
Files: models.py (CNN, CRNN, CTC decoding) | preprocess.py (OpenCV binarise, crop, line/word/char segmentation) |
train_cnn.py | train_crnn.py (synthetic words stitched from EMNIST) | predict_text.py | app.py

Notes
- EMNIST downloads automatically (~0.5 GB) into ./data.
- For digits only use torchvision MNIST (10 classes) instead of EMNIST in train_cnn.py.
- Synthetic-word CRNN works best on neat, separated block letters. For real cursive, train on IAM and
  replace SyntheticWords with a Dataset yielding (32-px-high image, text).
- Report CER / WER for the sequence model, accuracy for the CNN.
