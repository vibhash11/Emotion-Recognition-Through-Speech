Emotion Recognition through speech Project
------------------------------------------

Trained.pickle is an LGBM model that has been trained to classify whether an audio clip conveys "Happy" or "Sad" emotion.
Running recording.py will first record audio (using active mic) for a small amount of time and save it as a wav file.
After that prediction.py will load the pickle, play the recorded audio file and display whether that audio clip conveys "Happy" or "Sad" emotion. 
