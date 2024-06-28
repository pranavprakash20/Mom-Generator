import moviepy.editor as mp
from transformers import BartForConditionalGeneration, BartTokenizer
from os import system
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import nltk
import re


class MomGenerator:
    def __init__(self, meeting_recording):
        self.meeting_recording = meeting_recording
        self.video_content = None
        self.audio_content = None
        self.audio_file = "converted_audio.wav"
        self.summary = ""

    def create_mom(self):
        # Load the recording
        self._load_video()

        # Convert the recording to audio
        self._convert_to_audio()

        # Create transcript using whisper
        self.create_transcript()

        # Generate summary
        self._generate_summary()
        return self.summary

    def _generate_summary(self, model):
        # Load your text file
        with open('transcript.txt', 'r') as file:
            text = file.read()

        if model == "bart":
            # Load the model and tokenizer
            model_name = "facebook/bart-large-cnn"
            model = BartForConditionalGeneration.from_pretrained(model_name)
            tokenizer = BartTokenizer.from_pretrained(model_name)

            # Tokenize and summarize the text
            inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                         early_stopping=True)
            self.summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        elif model == "nltk":
            sentence_list = nltk.sent_tokenize(text)
        text = re.sub(r'\[[0-9]*\]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        # Removing special characters and digits
        formatted_article_text = re.sub('[^a-zA-Z]', ' ', text )
        formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
        stopwords = nltk.corpus.stopwords.words('english')
        
        word_frequencies = {}
        for word in nltk.word_tokenize(formatted_article_text):
            if word not in stopwords:
                if word not in word_frequencies.keys():
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1
            maximum_frequncy = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
            sentence_scores = {}
        for sent in sentence_list:
            for word in nltk.word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if len(sent.split(' ')) < 30:
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word]
                        else:
                            sentence_scores[sent] += word_frequencies[word]
        summary_sentences = heapq.nlargest(100, sentence_scores, key=sentence_scores.get)
        
        self.summary = ' '.join(summary_sentences)
        # Else proceed with spacy
        else:
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(text)
            tokens = [token.text for token in doc]
            word_frequencies = {}
            per = 0.05
            for word in doc:
                if word.text.lower() not in list(STOP_WORDS):
                    if word.text.lower() not in punctuation:
                        if word.text not in word_frequencies.keys():
                            word_frequencies[word.text] = 1
                        else:
                            word_frequencies[word.text] += 1
            max_frequency = max(word_frequencies.values())
            for word in word_frequencies.keys():
                word_frequencies[word] = word_frequencies[word] / max_frequency
            sentence_tokens = [sent for sent in doc.sents]
            sentence_scores = {}
            for sent in sentence_tokens:
                for word in sent:
                    if word.text.lower() in word_frequencies.keys():
                        if sent not in sentence_scores.keys():
                            sentence_scores[sent] = word_frequencies[word.text.lower()]
                        else:
                            sentence_scores[sent] += word_frequencies[word.text.lower()]
            select_length = int(len(sentence_tokens) * per)
            summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
            final_summary = [word.text for word in summary]
            self.summary = ''.join(final_summary)

    def create_transcript(self):
        # Create the transcript using whisper
        cmd = f"whisper --model base.en {self.audio_file} >> transcript.txt"
        system(cmd)

    def _convert_to_audio(self):
        self.audio_content = self.video_content.audio.write_audiofile(self.audio_file)

    def _load_video(self):
        # Load the video
        self.video_content = mp.VideoFileClip(self.meeting_recording)


mom = MomGenerator("rhgs-ci.mp4")
print(mom.create_mom())

