from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import cv2
import joblib
import os

app = Flask(__name__)

# Load the trained soil classification model
soil_model = tf.keras.models.load_model('soil_classification_Updated_model_vgg19.h5')

# Load or train and save the crop recommendation model
crop_model_path = 'crop_recommendation_model.pkl'
if os.path.exists(crop_model_path):
    crop_model = joblib.load(crop_model_path)
else:
    df = pd.read_csv('Crop_recommendation.csv')
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
    crop_model.fit(X, y)
    joblib.dump(crop_model, crop_model_path)

# Soil class mapping - Updated with correct order
soil_classes = ['Alluvial soil', 'Black Soil', 'Clay soil', 'NotSoil', 'Red soil', 'Yellow Soil']

# Crop suitability mapping based on soil type
soil_crop_mapping = {
    'Alluvial soil': ['wheat', 'rice', 'maize', 'sugarcane', 'mangoes', 'banana', 'pulses', 'oilseeds', 'cotton',
                      'jute'],
    'Black Soil': ['cotton', 'pulses', 'millets', 'linseed', 'tobacco', 'sugarcane', 'wheat', 'vegetables',
                   'citrus fruits', 'Mangoes', 'banana'],
    'Clay soil': ['rice', 'wheat', 'cotton', 'broccoli', 'beans', 'brussels sprouts', 'sugarcane', 'leafy greens'],
    'Red soil': ['wheat', 'Rice', 'cotton', 'jowar', 'pulses', 'tobacco', 'millets', 'oilseeds', 'potato', 'maize',
                 'groundnut',
                 'orchards'],
    'Yellow Soil': ['wheat', 'cotton', 'rice', 'pulses', 'mangoes', 'oranges', 'tobacco', 'millets', 'oilseeds',
                    'potato', 'maize', 'groundnut',
                    'orchards']
}

# NEW: Soil parameters information - water and fertilizer recommendations
soil_parameters = {
    'Alluvial soil': {
        'water_requirements': 'Moderate to high water requirements. Good drainage capacity with medium water retention.',
        'fertilizer_recommendations': 'Balanced NPK fertilizers with moderate nitrogen. Responds well to organic matter addition.'
    },
    'Black Soil': {
        'water_requirements': 'Low to moderate water requirements. High water retention capacity with poor drainage.',
        'fertilizer_recommendations': 'Less nitrogen, moderate phosphorus and higher potassium. Benefits from organic amendments to improve structure.'
    },
    'Clay soil': {
        'water_requirements': 'Low water requirements due to high water retention. Prone to waterlogging, requires good drainage.',
        'fertilizer_recommendations': 'Lower nitrogen doses, add organic matter to improve texture. May need gypsum for structure improvement.'
    },
    'Red soil': {
        'water_requirements': 'High water requirements due to low water retention. Requires frequent irrigation in dry periods.',
        'fertilizer_recommendations': 'Higher doses of phosphatic fertilizers. Benefits from regular organic matter addition to improve water retention.'
    },
    'Yellow Soil': {
        'water_requirements': 'High water requirements. Poor water retention capacity requiring proper irrigation management.',
        'fertilizer_recommendations': 'Balanced NPK with focus on micronutrients. Benefits from organic manure to improve soil structure.'
    }
}

# NEW: Crop growing months by soil type
crop_growing_months = {
    'Alluvial soil': {
        'wheat': 'October-December (sowing), March-April (harvesting)',
        'rice': 'June-July (sowing), September-October (harvesting)',
        'maize': 'Kharif: June-July, Rabi: October-November, Summer: January-February',
        'sugarcane': 'October-November (autumn), February-March (spring)',
        'pulses': 'October-November (sowing), February-April (harvesting)',
        'oilseeds': 'Varies by type: Mustard (Oct-Nov), Groundnut (June-July)'
    },
    'Black Soil': {
        'cotton': 'June-July (sowing), October onwards (harvesting)',
        'pulses': 'October-November (sowing), February-April (harvesting)',
        'millets': 'June-August (sowing), September-November (harvesting)',
        'linseed': 'October-November (sowing), February-March (harvesting)',
        'tobacco': 'September-November (transplanting)',
        'sugarcane': 'October-November (autumn), February-March (spring)',
        'vegetables': 'Year-round with seasonal variations',
        'citrus fruits': 'Planting: July-August, Fruiting varies by variety'
    },
    'Clay soil': {
        'rice': 'June-July (sowing), September-October (harvesting)',
        'wheat': 'October-December (sowing), March-April (harvesting)',
        'cotton': 'June-July (sowing), October onwards (harvesting)',
        'sugarcane': 'October-November (autumn), February-March (spring)',
        'leafy greens': 'August-November (sowing), cooler months harvesting',
        'broccoli': 'September-November (sowing), winter harvesting',
        'brussels sprouts': 'September-November (sowing), winter harvesting'
    },
    'Red soil': {
        'wheat': 'October-December (sowing), March-April (harvesting)',
        'cotton': 'June-July (sowing), October onwards (harvesting)',
        'pulses': 'October-November (sowing), February-April (harvesting)',
        'tobacco': 'September-November (transplanting)',
        'millets': 'June-August (sowing), September-November (harvesting)',
        'oilseeds': 'Varies by type: Mustard (Oct-Nov), Groundnut (June-July)',
        'potato': 'October-November (planting), January-March (harvesting)',
        'maize': 'Kharif: June-July, Rabi: October-November',
        'groundnut': 'Kharif: June-July, Rabi: November-December (in some regions)',
        'orchards': 'Planting: June-July (monsoon), Fruiting varies by species'
    },
    'Yellow Soil': {
        'wheat': 'October-December (sowing), March-April (harvesting)',
        'cotton': 'June-July (sowing), October onwards (harvesting)',
        'pulses': 'October-November (sowing), February-April (harvesting)',
        'tobacco': 'September-November (transplanting)',
        'millets': 'June-August (sowing), September-November (harvesting)',
        'oilseeds': 'Varies by type: Mustard (Oct-Nov), Groundnut (June-July)',
        'potato': 'October-November (planting), January-March (harvesting)',
        'maize': 'Kharif: June-July, Rabi: October-November',
        'groundnut': 'Kharif: June-July, Rabi: November-December (in some regions)',
        'orchards': 'Planting: June-July (monsoon), Fruiting varies by species'
    }
}

# NEW: Crop water requirements information
crop_water_info = {
    'wheat': 'Needs consistent moisture during sowing, tillering, and grain-filling stages. Typically requires 3-4 irrigations depending on rainfall.',
    'rice': 'Water-intensive crop, especially during transplanting and vegetative growth. Often requires standing water in fields.',
    'maize': 'Moderate and consistent irrigation needed, especially during tasseling and silking stages. Typically needs 5-6 irrigations.',
    'cotton': 'Regular irrigation needed, especially during boll formation and development. Can require 5-8 irrigations in drier areas.',
    'sugarcane': 'Significant water needed throughout its long growth period, especially during early growth and grand growth period.',
    'pulses': 'Relatively less water needed compared to cereals. Typically 2-3 irrigations at flowering and pod development stages.',
    'oilseeds': 'Water needs vary by type. Generally 2-4 irrigations needed depending on specific oilseed and rainfall.',
    'millets': 'Drought-tolerant crops requiring less water. 1-3 irrigations at early growth, flowering, and grain filling stages.',
    'linseed': 'Limited irrigation needed. Usually 1-2 irrigations at flowering and grain-filling stages.',
    'tobacco': 'Regular irrigation needed, especially during early growth and flowering stages.',
    'vegetables': 'Water requirements vary by vegetable type. Generally need consistent moisture.',
    'citrus fruits': 'Regular irrigation needed, especially during flowering and fruit development.',
    'leafy greens': 'Consistent moisture needed for rapid growth. Frequent, light irrigation preferred.',
    'broccoli': 'Consistent moisture needed, especially during head formation. About 1-1.5 inches of water per week.',
    'brussels sprouts': 'Consistent moisture needed throughout growth, especially during sprout development.',
    'potato': 'Regular watering needed, especially during tuber initiation and development. 7-15 day irrigation intervals.',
    'groundnut': 'Critical irrigation during flowering, pegging, and pod development stages.',
    'orchards': 'Varies by tree type. Young trees need regular watering. Mature trees need adequate irrigation during flowering and fruiting.'
}

# Language translation for soil types
soil_translations = {
    'en': {
        'Alluvial soil': 'Alluvial soil',
        'Black Soil': 'Black Soil',
        'Clay soil': 'Clay soil',
        'Red soil': 'Red soil',
        'Yellow Soil': 'Yellow Soil',
        'NotSoil': 'Not Soil'
    },
    'kn': {
        'Alluvial soil': 'ಮಣಲು ಮಣ್ಣು',
        'Black Soil': 'ಕಪ್ಪು ಮಣ್ಣು',
        'Clay soil': 'ಜೇಡಿ ಮಣ್ಣು',
        'Red soil': 'ಕೆಂಪು ಮಣ್ಣು',
        'Yellow Soil': 'ಹಳದಿ ಮಣ್ಣು',
        'NotSoil': 'ಮಣ್ಣು ಅಲ್ಲ'
    },
    'hi': {
        'Alluvial soil': 'जलोढ़ मिट्टी',
        'Black Soil': 'काली मिट्टी',
        'Clay soil': 'चिकनी मिट्टी',
        'Red soil': 'लाल मिट्टी',
        'Yellow Soil': 'पीली मिट्टी',
        'NotSoil': 'मिट्टी नहीं'
    }
}

# Crop translation dictionary
crop_translations = {
    'en': {
        'wheat': 'Wheat', 'rice': 'Rice', 'maize': 'Maize', 'cotton': 'Cotton',
        'sugarcane': 'Sugarcane', 'pulses': 'Pulses', 'oilseeds': 'Oilseeds',
        'millets': 'Millets', 'linseed': 'Linseed', 'tobacco': 'Tobacco',
        'vegetables': 'Vegetables', 'citrus fruits': 'Citrus Fruits',
        'leafy greens': 'Leafy Greens', 'broccoli': 'Broccoli', 'brussels sprouts': 'Brussels Sprouts',
        'potato': 'Potato', 'groundnut': 'Groundnut', 'orchards': 'Orchards',
        'jute': 'Jute', 'mangoes': 'Mangoes', 'banana': 'Banana', 'beans': 'Beans'
    },
    'kn': {
        'wheat': 'ಗೋಧಿ', 'rice': 'ಅಕ್ಕಿ', 'maize': 'ಮೆಕ್ಕೆಜೋಳ', 'cotton': 'ಹತ್ತಿ',
        'sugarcane': 'ಕಬ್ಬು', 'pulses': 'ಬೇಳೆಕಾಳುಗಳು', 'oilseeds': 'ಎಣ್ಣೆಕಾಳುಗಳು',
        'millets': 'ಸಿರಿಧಾನ್ಯಗಳು', 'linseed': 'ಅಗಸೆ', 'tobacco': 'ತಂಬಾಕು',
        'vegetables': 'ತರಕಾರಿಗಳು', 'citrus fruits': 'ನಿಂಬೆ ಹಣ್ಣುಗಳು',
        'leafy greens': 'ಸೊಪ್ಪು ತರಕಾರಿಗಳು', 'broccoli': 'ಬ್ರೊಕ್ಕೋಲಿ', 'brussels sprouts': 'ಬ್ರಸೆಲ್ಸ್ ಸ್ಪ್ರೌಟ್ಸ್',
        'potato': 'ಆಲೂಗಡ್ಡೆ', 'groundnut': 'ಕಡಲೇಕಾಯಿ', 'orchards': 'ತೋಟಗಳು',
        'jute': 'ಚೆಣೆ', 'mangoes': 'ಮಾವಿನ ಹಣ್ಣುಗಳು', 'banana': 'ಬಾಳೆಹಣ್ಣು', 'beans': 'ಹುರಳಿಕಾಯಿ'
    },
    'hi': {
        'wheat': 'गेहूं', 'rice': 'चावल', 'maize': 'मक्का', 'cotton': 'कपास',
        'sugarcane': 'गन्ना', 'pulses': 'दालें', 'oilseeds': 'तिलहन',
        'millets': 'बाजरा', 'linseed': 'अलसी', 'tobacco': 'तंबाकू',
        'vegetables': 'सब्जियां', 'citrus fruits': 'खट्टे फल',
        'leafy greens': 'पत्तेदार सब्जियां', 'broccoli': 'ब्रोकोली', 'brussels sprouts': 'ब्रसेल्स स्प्राउट्स',
        'potato': 'आलू', 'groundnut': 'मूंगफली', 'orchards': 'बागान',
        'jute': 'जूट', 'mangoes': 'आम', 'banana': 'केला', 'beans': 'फलियां'
    }
}

# NEW: Translation for soil parameter information
soil_param_translations = {
    'en': {
        'water_requirements': 'Water Requirements',
        'fertilizer_recommendations': 'Fertilizer Recommendations',
        'growing_months': 'Best Growing Months'
    },
    'kn': {
        'water_requirements': 'ನೀರಿನ ಅವಶ್ಯಕತೆಗಳು',
        'fertilizer_recommendations': 'ಗೊಬ್ಬರ ಶಿಫಾರಸುಗಳು',
        'growing_months': 'ಉತ್ತಮ ಬೆಳೆಯುವ ತಿಂಗಳುಗಳು'
    },
    'hi': {
        'water_requirements': 'पानी की आवश्यकताएँ',
        'fertilizer_recommendations': 'उर्वरक सिफारिशें',
        'growing_months': 'सर्वोत्तम उगाने के महीने'
    }
}


def process_image(image):
    try:
        img = cv2.resize(image, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        raise ValueError(f"Image processing failed: {str(e)}")


def validate_params(n, p, k, temp, humidity, ph, rainfall):
    params = {'N': n, 'P': p, 'K': k, 'Temperature': temp, 'Humidity': humidity, 'pH': ph, 'Rainfall': rainfall}
    for name, value in params.items():
        if not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"{name} must be a non-negative number")
    if ph < 0 or ph > 14:
        raise ValueError("pH must be between 0 and 14")
    if humidity > 100:
        raise ValueError("Humidity must be <= 100%")
    return np.array([[n, p, k, temp, humidity, ph, rainfall]])


@app.route('/')
def home():
    return render_template('index3.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get language preference
        lang = request.form.get('lang', 'en')
        if lang not in ['en', 'kn', 'hi']:
            lang = 'en'  # Default to English if invalid language

        # Check if soil image is provided
        if 'soil_image' not in request.files:
            return jsonify({'error': 'No soil image provided'}), 400
        file = request.files['soil_image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Process image
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image file'}), 400
        processed_img = process_image(img)

        # Soil classification
        soil_pred = soil_model.predict(processed_img)
        max_confidence = float(np.max(soil_pred)) * 100
        soil_idx = np.argmax(soil_pred)
        soil_type_en = soil_classes[soil_idx]

        # Check if NotSoil is detected
        if soil_type_en == 'NotSoil':
            # Return message to upload a soil image without crop recommendations
            not_soil_message = "Please upload a soil image and try again"
            translated_message = {
                'en': not_soil_message,
                'kn': "ದಯವಿಟ್ಟು ಮಣ್ಣಿನ ಚಿತ್ರವನ್ನು ಅಪ್ಲೋಡ್ ಮಾಡಿ ಮತ್ತು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ",
                'hi': "कृपया मिट्टी की तस्वीर अपलोड करें और फिर से प्रयास करें"
            }

            return jsonify({
                'is_soil': False,
                'message': translated_message[lang]
            }), 200

        # If confidence is too low, it's likely not a good soil image
        if max_confidence < 30:  # You can adjust this threshold
            return jsonify({
                'error': 'Low confidence in soil classification',
                'message': 'Please upload a clearer soil image'
            }), 400

        # Translate soil type based on selected language
        soil_type = soil_translations[lang].get(soil_type_en, soil_type_en)
        soil_confidence = max_confidence

        # Get and validate parameters
        try:
            n = float(request.form['nitrogen'])
            p = float(request.form['phosphorus'])
            k = float(request.form['potassium'])
            temp = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            rainfall = float(request.form['rainfall'])
            crop_features = validate_params(n, p, k, temp, humidity, ph, rainfall)
        except KeyError as e:
            return jsonify({'error': f'Missing parameter: {str(e)}'}), 400
        except ValueError as e:
            return jsonify({'error': str(e)}), 400

        # Crop recommendation with probabilities
        crop_probs = crop_model.predict_proba(crop_features)[0]
        crop_names = crop_model.classes_
        crop_prob_dict = dict(zip(crop_names, crop_probs))

        # Filter suitable crops based on soil type
        suitable_crops = soil_crop_mapping[soil_type_en]  # Use English soil type for mapping
        suitable_crop_probs = {crop: prob for crop, prob in crop_prob_dict.items() if crop in suitable_crops}

        # Sort by probability and get top crops
        top_crops = sorted(suitable_crop_probs.items(), key=lambda x: x[1], reverse=True)

        # Translate crop names based on selected language
        top_crops_with_confidence = [
            {
                'crop': crop_translations[lang].get(crop, crop),  # Translate crop name
                'confidence': f"{prob * 100:.2f}%",
                'water_info': crop_water_info.get(crop, "Information not available."),
                'growing_months': crop_growing_months.get(soil_type_en, {}).get(crop, "Information not available.")
            }
            for crop, prob in top_crops[:3]  # Top 3 crops
        ]

        # Highlight the most efficient crop (highest confidence)
        recommended_crop_en = top_crops[0][0] if top_crops else None
        recommended_crop = crop_translations[lang].get(recommended_crop_en,
                                                       recommended_crop_en) if recommended_crop_en else None
        crop_confidence = f"{top_crops[0][1] * 100:.2f}%" if top_crops else "N/A"

        # Get soil parameter information
        soil_param_info = {
            'water_requirements': soil_parameters[soil_type_en]['water_requirements'],
            'fertilizer_recommendations': soil_parameters[soil_type_en]['fertilizer_recommendations'],
        }

        # Translate parameter labels
        soil_param_labels = {
            key: soil_param_translations[lang].get(key, key)
            for key in soil_param_info.keys()
        }

        return jsonify({
            'is_soil': True,
            'soil_type': soil_type,
            'soil_confidence': f"{soil_confidence:.2f}%",
            'recommended_crop': recommended_crop,
            'crop_confidence': crop_confidence,
            'suitable_crops': top_crops_with_confidence,
            'soil_param_info': soil_param_info,
            'soil_param_labels': soil_param_labels,
            'growing_months_label': soil_param_translations[lang].get('growing_months', 'Best Growing Months')
        })

    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)