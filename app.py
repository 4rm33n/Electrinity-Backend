from flask import Flask, render_template, request
import joblib
import pandas as pd
import traceback

# Initialize Flask app
app = Flask(__name__)

# Load models and encoders
rf_model = joblib.load('electrinity_model_rf_unbalanced.pkl')  # 🌟 Main model
sgd_model = joblib.load('electrinity_model_sgd_unbalanced.pkl')  # 🧠 Optional fallback
le = joblib.load('electrinity_labelencoder_unbalanced.pkl')
key_encoder = joblib.load('electrinity_key_encoder_unbalanced.pkl')
camelot_encoder = joblib.load('electrinity_camelot_encoder_unbalanced.pkl')
scaler = joblib.load('electrinity_scaler_unbalanced.pkl')
features = joblib.load('electrinity_features_unbalanced.pkl')
feature_means = joblib.load('electrinity_feature_means_unbalanced.pkl')  # Load default values for missing features

@app.route('/')
def home():
    key_names = list(key_encoder.classes_)
    camelot_names = list(camelot_encoder.classes_)
    return render_template('Genre.html', key_names=key_names, camelot_names=camelot_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        bpm = float(request.form['BPM'])
        dance = float(request.form['Dance'])
        energy = float(request.form['Energy'])
        duration = float(request.form['Duration_sec'])
        key_input = request.form['Key']
        camelot_input = request.form['Camelot']

        # Encode
        encoded_key = key_encoder.transform([key_input])[0]
        encoded_camelot = camelot_encoder.transform([camelot_input])[0]

        # Prepare input
        input_dict = feature_means.copy()
        input_dict.update({
            'BPM': bpm,
            'Dance': dance,
            'Energy': energy,
            'Key_encoded': encoded_key,
            'Duration_sec': duration,
            'Camelot_encoded': encoded_camelot
        })

        new_song_df = pd.DataFrame([input_dict])
        scaled_array = scaler.transform(new_song_df)
        scaled_song_df = pd.DataFrame(scaled_array, columns=new_song_df.columns)
        scaled_song_df = scaled_song_df[features]

        # 🌟 Predict with RandomForest
        predicted_encoded = rf_model.predict(scaled_song_df)[0]
        predicted_genre = le.inverse_transform([predicted_encoded])[0]

        # 🧠 Optional: Predict with SGD (for comparison/debugging)
        predicted_encoded_sgd = sgd_model.predict(scaled_song_df)[0]
        predicted_genre_sgd = le.inverse_transform([predicted_encoded_sgd])[0]

        # ⚠️ Override logic if RF says 'Other' but it's high-energy, fast BPM
        if predicted_genre == "Other" and bpm >= 140 and energy >= 0.9:
            print("⚠️ Overriding RF with SGD due to Hard Techno signal.")
            predicted_genre = predicted_genre_sgd

        print("🎿 RF Prediction:", predicted_genre)
        print("🧠 SGD Prediction:", predicted_genre_sgd)

        return render_template('Genre.html',
                               genre=predicted_genre,
                               genre_sgd=predicted_genre_sgd,
                               key_names=key_encoder.classes_,
                               camelot_names=camelot_encoder.classes_)

    except Exception as e:
        traceback.print_exc()
        return render_template('Genre.html',
                               error=f"🔥 Error: {str(e)}",
                               genre=None,
                               key_names=key_encoder.classes_,
                               camelot_names=camelot_encoder.classes_)

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        bpm = float(request.form['BPM'])
        dance = float(request.form['Dance'])
        energy = float(request.form['Energy'])
        duration = float(request.form['Duration_sec'])
        key_input = request.form['Key']
        camelot_input = request.form['Camelot']
        true_genre = request.form['TrueGenre']

        encoded_key = key_encoder.transform([key_input])[0]
        encoded_camelot = camelot_encoder.transform([camelot_input])[0]
        genre_encoded = le.transform([true_genre])[0]

        input_dict = feature_means.copy()
        input_dict.update({
            'BPM': bpm,
            'Dance': dance,
            'Energy': energy,
            'Key_encoded': encoded_key,
            'Duration_sec': duration,
            'Camelot_encoded': encoded_camelot
        })

        new_song_df = pd.DataFrame([input_dict])
        scaled_array = scaler.transform(new_song_df)
        scaled_song_df = pd.DataFrame(scaled_array, columns=new_song_df.columns)
        scaled_song_df = scaled_song_df[features]

        sgd_model.partial_fit(scaled_song_df, [genre_encoded])
        joblib.dump(sgd_model, 'electrinity_model_sgd.pkl')

        # ✅ Save feedback to CSV for retraining later
        feedback_row = {
            'BPM': bpm,
            'Dance': dance,
            'Energy': energy,
            'Key': key_input,
            'Duration_sec': duration,
            'Camelot': camelot_input,
            'TrueGenre': true_genre
        }
        pd.DataFrame([feedback_row]).to_csv('SGDFeedback_Log.csv'
        , mode='a', header=not pd.io.common.file_exists('SGDFeedback_Log.csv'), index=False)

        return "✅ Feedback received and model updated!"
    except Exception as e:
        traceback.print_exc()
        return f"🔥 Feedback error: {str(e)}"

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 10000)

#redeploying render
