**ABSTRACT**

Handwriting recognition has long been a vital component of human-computer interaction, with applications ranging from digitizing historical documents to enabling assistive technologies. This project 
presents a deep learning-based approach to offline handwriting recognition using Microsoft's pre-trained 
TrOCR (Transformer-based Optical Character Recognition) model. The model is fine-tuned on the IAM 
Handwriting Database, which provides high-quality handwritten English sentences. The pipeline includes 
image preprocessing, tokenization, model fine-tuning with mixed precision for optimized performance, 
and evaluation on validation data. To demonstrate its effectiveness, a lightweight application was 
developed where users can write on a canvas, and the system converts the input into editable text in real 
time. The results highlight the capability of transformer-based models in achieving high accuracy and 
generalization on complex handwriting data. This work lays the foundation for multilingual and 
multimodal extensions in future development.



**OBJECTIVES** 

1. To fine-tune a pre-trained TrOCR (Transformer-based Optical Character Recognition) model on the 
IAM handwriting dataset to enhance its performance on handwritten English text. 

2. To evaluate the fine-tuned model using relevant performance metrics such as training/validation loss 
and accuracy to ensure effective learning and generalization. 

3. To optimize the training process by addressing challenges such as memory limitations, mixed precision 
training, and gradient accumulation on a local machine. 

4. To develop a lightweight and user-friendly frontend application that includes a digital canvas for users 
to input handwritten text. 

5. To integrate the trained handwriting recognition model with the frontend interface to enable real-time 
conversion of handwritten strokes to digital text. 

6. To ensure the seamless transition of a machine learning model from a research/training environment to 
a production-ready application.



**DATASET** 

For this project, we used the **IAM Handwriting Dataset** hosted on Kaggle under the repository 
“**changheonkim/iam-trocr**”. This dataset is a curated version specifically formatted for use with the 
TrOCR model. 

It contains:

● 2915 handwritten text line images in total. 

● A metadata file named gt_test.txt that serves as the annotation file. 

The gt_test.txt file maps each image file name to its corresponding ground truth transcription using 
tab-separated values. Each entry in the file is structured as: 
filename text 

This structure enables easy loading of paired image-text data for training and evaluation of handwriting 
recognition models. 


**MODEL ARCHITECTURE**

For this project, we utilize Microsoft’s TrOCR (Transformer-based Optical Character Recognition) model, 
specifically the pretrained trocr-base-handwritten variant. 
TrOCR follows a Vision-to-Text architecture that combines the strengths of computer vision and natural 
language processing using Transformer models.

**Key components of the architecture:** 
1. Vision Encoder (ViT – Vision Transformer):
 
● The encoder takes raw input images and converts them into a sequence of embeddings. 

● It is based on a ViT model pretrained on large image datasets, allowing it to capture rich spatial 
and contextual features from handwriting. 

● The image is divided into patches which are then linearly embedded and passed through 
Transformer layers. 

3. Text Decoder (GPT2-like Transformer Decoder): 

● The decoder is a Transformer model similar to GPT2, designed to autoregressively generate text. 

● It takes the encoder’s output and generates the corresponding transcriptions token-by-token. 

● This part of the model learns the language modeling task and helps reconstruct the written text. 

4. Processor (TrOCRProcessor): 

● A wrapper that combines a feature extractor for the vision encoder and a tokenizer for the text 
decoder. 

● Handles all preprocessing of images and postprocessing of text.



Why TrOCR? 

● Pretrained on large handwriting datasets like IAM, TrOCR can generalize well on unseen 
handwriting styles. 

● The end-to-end architecture eliminates the need for separate text line detection, segmentation, and 
recognition modules — everything is handled in a single model. 



This architecture enables highly accurate and scalable handwriting recognition in multilingual and 
multimodal contexts. 
