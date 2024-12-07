from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

GB_MODEL_PATH = 'gradient_boosting_model.pkl'
XGB_MODEL_PATH = 'xgboost_model.pkl'
ENCODERS_PATH = 'label_encoders.pkl'
USER_DATA_PATH = 'train_users_2.csv'


GB_FEATURES = ['age', 'gender', 'signup_method', 'signup_flow', 'language',
               'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
               'signup_app', 'first_device_type', 'first_browser']
XGB_FEATURES = ['age', 'gender', 'signup_method', 'signup_flow', 'language',
                'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
                'signup_app', 'first_device_type', 'first_browser']

CATEGORICAL_COLS = ['gender', 'signup_method', 'language', 'affiliate_channel',
                    'affiliate_provider', 'first_affiliate_tracked', 'signup_app',
                    'first_device_type', 'first_browser', 'country_destination']


try:
    with open(GB_MODEL_PATH, 'rb') as f:
        gb_model = pickle.load(f)
    with open(XGB_MODEL_PATH, 'rb') as f:
        xgb_model = pickle.load(f)
    with open(ENCODERS_PATH, 'rb') as f:
        label_encoders = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading model or encoders: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_id = data.get('user_id')
    model_choice = data.get('model')

    if not user_id:
        return jsonify({"error": "User ID is required."}), 400

    try:
        
        user_data = pd.read_csv(USER_DATA_PATH)
        user_data = user_data[user_data['id'] == user_id]

        if user_data.empty:
            return jsonify({"result": "No data found for the given User ID."})

        
        model = gb_model
        features = GB_FEATURES

        
        user_features = user_data[features]

        
        for col in CATEGORICAL_COLS:
            if col in user_features.columns:
                user_features.loc[:, col] = label_encoders[col].transform(user_features[col].fillna('Unknown'))

        
        user_features.fillna(0, inplace=True)

        
        y_pred_proba = model.predict_proba(user_features)
        top_5_indices = np.argsort(y_pred_proba, axis=1)[0, -5:][::-1]
        top_5_countries = label_encoders['country_destination'].inverse_transform(top_5_indices)
        top_5_probs = y_pred_proba[0][top_5_indices]

        result = [{"country": country, "probability": prob} for country, prob in zip(top_5_countries, top_5_probs)]
        return jsonify({"result": result})

    except KeyError as e:
        return jsonify({"error": f"Missing required feature: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
