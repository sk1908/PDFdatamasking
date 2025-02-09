from PIL import Image
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_path
import re
import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from rapidfuzz import process, fuzz
import img2pdf
from io import BytesIO
import os

# Ensure pytesseract uses the correct path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = 
poppler_path = # add poppler executable path

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Load BERT tokenizer and model for NER
bert_tokenizer = AutoTokenizer.from_pretrained('dslim/bert-large-NER')
bert_model = AutoModelForTokenClassification.from_pretrained('dslim/bert-large-NER')

# Initialize the NER pipeline
ner_pipeline = pipeline('ner', model=bert_model, tokenizer=bert_tokenizer)

def process_file(pdf_path):
    # Convert PDF to images
    pages = convert_from_path(pdf_path)
    all_words = []

    for page in pages:
        # Convert each page to text using Tesseract
        text = pytesseract.image_to_string(page)

        # Process the text with spaCy
        doc = nlp(text)

        # Extract sentences and preprocess
        for sent in doc.sents:
            # Remove newline characters and extra spaces
            sentence = sent.text.replace('\n', ' ').strip()

            # Tokenize and remove stopwords and punctuation
            tokens = [token.text for token in nlp(sentence) if not token.is_stop and not token.is_punct]

            # Capitalize each token and add it to the list
            capitalized_tokens = [token.capitalize() for token in tokens]
            all_words.extend(capitalized_tokens)

    # Join the list of words into a single string
    names_string = ' '.join(all_words)

    # Perform NER on the input text
    ner_list = ner_pipeline(names_string)

    def extract_text_from_pdf(pdf_path):
        # Convert PDF to images
        pages = convert_from_path(pdf_path)
        texts = []
        for page in pages:
            text = pytesseract.image_to_string(page)
            texts.append(text)
        return texts, pages

    def extract_words_with_boxes(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, lang='eng')
        n_boxes = len(data['level'])
        words_with_boxes = []
        for i in range(n_boxes):
            if int(data['conf'][i]) > 20:  # Adjust confidence threshold as needed
                word = data['text'][i]
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                words_with_boxes.append((word, (x, y, w, h)))
        return words_with_boxes

    def blur_specific_word(image, words_with_boxes, target_words):
        for word, box in words_with_boxes:
            if word in target_words:
                x, y, w, h = box
                image[y:y+h, x:x+w] = cv2.GaussianBlur(image[y:y+h, x:x+w], (51, 51), 30)
        return image

    def blur_faces(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            image[y:y+h, x:x+w] = cv2.GaussianBlur(image[y:y+h, x:x+w], (21, 21), 11)
        return image

    def find_target_words(texts):
        pattern = r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}|\+\d{2}-\d{10}|\d{10}|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        target_words = []
        for text in texts:
            doc = nlp(text)
            for match in re.findall(pattern, doc.text):
                target_words.append(match)
        return target_words

    def extract_person_words(ner_output, original_text):
        person_words = []
        current_word = ""
        current_end = -1

        for token in ner_output:
            if token['entity'] in ['B-PER', 'I-PER', 'I-ORG', 'B-ORG']:
                if current_word:
                    if token['start'] != current_end:
                        person_words.append(current_word)
                        current_word = original_text[token['start']:token['end']]
                    else:
                        current_word += original_text[token['start']:token['end']]
                else:
                    current_word = original_text[token['start']:token['end']]
                current_end = token['end']
            else:
                if current_word:
                    person_words.append(current_word)
                    current_word = ""
                    current_end = -1

        if current_word:
            person_words.append(current_word)

        individual_person_words = []
        for word in person_words:
            individual_person_words.extend(word.split())

        return [word.capitalize() for word in individual_person_words],individual_person_words

    def fuzzy_match_lists(list1, list2, threshold=80):
        matches = []

        for word1 in list1:
            best_match, score, index = process.extractOne(word1, list2, scorer=fuzz.ratio)

            if score >= threshold:
                matches.append(word1)

        return matches

    names,capital_names = extract_person_words(ner_list, names_string)

    # Apply fuzzy matching
    matched_words = fuzzy_match_lists(all_words, names)

    # Extract text from PDF and convert to images
    texts, pages = extract_text_from_pdf(pdf_path)

    # Process each page to blur specific words and save the modified images
    processed_images = []

    # Find target words
    target_words_list = find_target_words(texts)

    pdf_images = []

    for i, page in enumerate(pages):
        image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        words_with_boxes = extract_words_with_boxes(image)
        image = blur_specific_word(image, words_with_boxes, target_words_list)
        image = blur_specific_word(image, words_with_boxes, matched_words)
        image = blur_specific_word(image, words_with_boxes, capital_names)
        image = blur_faces(image)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image_buffer = BytesIO()
        pil_image.save(image_buffer, format='JPEG')
        pdf_images.append(image_buffer.getvalue())

    pdf_bytes = img2pdf.convert(pdf_images)

    # Save the final output PDF
    processed_pdf_path = os.path.join(os.path.dirname(pdf_path), "final_output.pdf")
    with open(processed_pdf_path, "wb") as f:
        f.write(pdf_bytes)
    
    print(f"Processed file saved to: {processed_pdf_path}")# Debug line
     
    return processed_pdf_path
