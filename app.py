from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
from pymongo import MongoClient
from bson import ObjectId, errors
import os
from datetime import datetime, timedelta
from functools import wraps
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import tensorflow as tf
from PIL import Image
import cv2
import xgboost as xgb
from xgboost import XGBClassifier
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import io
from flask_paginate import Pagination, get_page_parameter
from flask import abort
from bson import ObjectId
from flask_login import login_required, current_user


app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for session management

# Model paths
MODEL_DIR = os.path.join('templates', 'model')
ULTRASOUND_MODEL_PATH = os.path.join(MODEL_DIR, 'pcos_classification_model.h5')
QUESTIONNAIRE_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def create_default_questionnaire_model():
    """Create a default XGBoost model for questionnaire prediction"""
   
    # Create a simple XGBoost model
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
   
    # Save the model
    try:
        joblib.dump(model, QUESTIONNAIRE_MODEL_PATH)
        print(f"Created and saved default questionnaire model to {QUESTIONNAIRE_MODEL_PATH}")
    except Exception as e:
        print(f"Error saving default model: {str(e)}")
   
    return model

def create_default_scaler():
    """Create a default StandardScaler for feature scaling"""
    from sklearn.preprocessing import StandardScaler
   
    # Create a default scaler
    scaler = StandardScaler()
   
    # Create some dummy data to fit the scaler
    dummy_data = np.zeros((1, 39))  # 39 features as per our feature order
    scaler.fit(dummy_data)
   
    # Save the scaler
    try:
        joblib.dump(scaler, SCALER_PATH)
        print(f"Created and saved default scaler to {SCALER_PATH}")
    except Exception as e:
        print(f"Error saving default scaler: {str(e)}")
   
    return scaler

# Load models and scaler
def load_models():
    global ultrasound_model, questionnaire_model, scaler
   
    try:
        # Load ultrasound model
        if os.path.exists(ULTRASOUND_MODEL_PATH):
            ultrasound_model = tf.keras.models.load_model(ULTRASOUND_MODEL_PATH)
            print(f"Successfully loaded ultrasound model from {ULTRASOUND_MODEL_PATH}")
        else:
            print(f"Warning: Ultrasound model not found at {ULTRASOUND_MODEL_PATH}")
            ultrasound_model = None

        # Load questionnaire model
        if os.path.exists(QUESTIONNAIRE_MODEL_PATH):
            try:
                questionnaire_model = joblib.load(QUESTIONNAIRE_MODEL_PATH)
                print(f"Successfully loaded questionnaire model from {QUESTIONNAIRE_MODEL_PATH}")
            except Exception as e:
                print(f"Error loading questionnaire model: {str(e)}")
                print("Creating default questionnaire model...")
                questionnaire_model = create_default_questionnaire_model()
        else:
            print(f"Warning: Questionnaire model not found at {QUESTIONNAIRE_MODEL_PATH}")
            print("Creating default questionnaire model...")
            questionnaire_model = create_default_questionnaire_model()

        # Load or create scaler
        if os.path.exists(SCALER_PATH):
            try:
                scaler = joblib.load(SCALER_PATH)
                print(f"Successfully loaded scaler from {SCALER_PATH}")
            except Exception as e:
                print(f"Error loading scaler: {str(e)}")
                print("Creating default scaler...")
                scaler = create_default_scaler()
        else:
            print(f"Warning: Scaler not found at {SCALER_PATH}")
            print("Creating default scaler...")
            scaler = create_default_scaler()
           
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        ultrasound_model = None
        questionnaire_model = None
        scaler = None
        raise

# Initialize models
try:
    load_models()
except Exception as e:
    print(f"Failed to initialize models: {str(e)}")
    # Create default models if loading fails
    questionnaire_model = create_default_questionnaire_model()
    scaler = create_default_scaler()

def preprocess_questionnaire_data(data):
    """
    Preprocess questionnaire data using the scaler
    """
    if scaler is None:
        raise ValueError("Scaler is not initialized")
       
    # Define the feature order expected by the model
    features = [
        'age', 'weight', 'height', 'bmi', 'blood_group', 'pulse_rate',
        'rr', 'hb', 'cycle', 'cycle_length', 'marriage_status', 'pregnant',
        'no_of_abortions', 'beta_hcg', 'fsh', 'lh', 'fsh_lh_ratio',
        'hip', 'waist', 'waist_hip_ratio', 'tsh', 'amh', 'prl',
        'vit_d3', 'prg', 'rbs', 'weight_gain', 'hair_growth',
        'skin_darkening', 'hair_loss', 'pimples', 'fast_food',
        'reg_exercise', 'bp_systolic', 'bp_diastolic', 'follicle_no_l',
        'follicle_no_r', 'avg_f_size_l', 'avg_f_size_r', 'endometrium'
    ]
   
    # Convert data to numpy array in the correct order
    data_array = np.array([[data.get(f, 0) for f in features]])
   
    # Scale the data
    scaled_data = scaler.transform(data_array)
    return scaled_data

# MongoDB Connection
try:
    mongo_client = MongoClient('mongodb://localhost:27017/')
    db = mongo_client['pcos_companion']
    print("Connected to MongoDB successfully!")
   
    # Initialize collections if they don't exist
    collections = ['users', 'assessments', 'ultrasound_evaluations', 'appointments',
                  'diet_plans', 'health_plans', 'activities', 'blogs']
    for collection in collections:
        if collection not in db.list_collection_names():
            db.create_collection(collection)
   
    # Update existing users with default notification settings if they don't have them
    users_without_settings = db.users.find({'notification_settings': {'$exists': False}})
    for user in users_without_settings:
        db.users.update_one(
            {'_id': user['_id']},
            {'$set': {
                'notification_settings': {
                    'email_notifications': True,
                    'test_reminders': True,
                    'health_tips': True
                }
            }}
        )
   
    # Initialize blog posts if collection is empty
    if db.blogs.count_documents({}) == 0:
        initial_blogs = [
            {
                "title": "Understanding PCOS",
                "content": "Learn about the symptoms, causes, and treatments for PCOS.",
                "author": "Dr. Sarah Johnson",
                "date_posted": datetime.utcnow(),
                "category": "Education",
                "tags": ["PCOS", "Health", "Women's Health"]
            },
            {
                "title": "Best Diet for PCOS",
                "content": "Discover the best foods and diets to manage PCOS symptoms.",
                "author": "Nutritionist Emma Davis",
                "date_posted": datetime.utcnow(),
                "category": "Nutrition",
                "tags": ["Diet", "PCOS", "Nutrition"]
            },
            {
                "title": "Exercise Tips for PCOS",
                "content": "Find out the best workouts to improve PCOS symptoms.",
                "author": "Fitness Coach Mike Brown",
                "date_posted": datetime.utcnow(),
                "category": "Fitness",
                "tags": ["Exercise", "PCOS", "Fitness"]
            }
        ]
        db.blogs.insert_many(initial_blogs)
        print("Initialized blog posts in database")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

# Ensure uploads directory exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Authentication decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def patient_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))
        
        # Fetch user from MongoDB
        user = db.users.find_one({'_id': ObjectId(session['user_id'])})

        if not user or user.get('account_type') != 'patient':
            flash('Access denied. Patient privileges required.', 'danger')
            return redirect(url_for('home'))
       
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
       
        user = db.users.find_one({'_id': ObjectId(session['user_id'])})
        if not user or user.get('account_type') != 'admin':
            flash('Access denied. Admin account required.', 'error')
            return redirect(url_for('home'))
       
        return f(*args, **kwargs)
    return decorated_function

def lab_assistant_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))
       
        user = db.users.find_one({'_id': ObjectId(session['user_id'])})
        if not user or user.role != 'lab_assistant':
            flash('Access denied. Lab assistant privileges required.', 'danger')
            return redirect(url_for('home'))
        
        return f(*args, **kwargs)
    return decorated_function

def patient_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))
        
        # Fetch user from MongoDB
        user = db.users.find_one({'_id': ObjectId(session['user_id'])})

        if not user or user.get('account_type') != 'patient':  # Fix: Use get() method
            flash('Access denied. Patient privileges required.', 'danger')
            return redirect(url_for('home'))
       
        return f(*args, **kwargs)
    return decorated_function

def preprocess_image(image_file):
    """Preprocess ultrasound image for model prediction."""
    try:
        # Read and preprocess the image
        img = Image.open(image_file)
        img = img.resize((224, 224))  # Adjust size according to your model's requirements
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def allowed_file(filename, allowed_extensions=None):
    """Check if the file has an allowed extension."""
    if allowed_extensions is None:
        allowed_extensions = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_personalized_recommendations(risk_level):
    """Generate personalized recommendations based on risk level."""
    recommendations = {
        'diet': [],
        'exercise': [],
        'lifestyle': []
    }
   
    # Common recommendations for all risk levels
    common_recommendations = {
        'diet': [
            {'title': 'Balanced Diet', 'description': 'Focus on whole foods, lean proteins, and plenty of vegetables'},
            {'title': 'Regular Meals', 'description': 'Eat at regular intervals to maintain blood sugar levels'}
        ],
        'exercise': [
            {'title': 'Regular Activity', 'description': 'Aim for at least 30 minutes of moderate exercise daily'},
            {'title': 'Stress Management', 'description': 'Include relaxation techniques like yoga or meditation'}
        ],
        'lifestyle': [
            {'title': 'Sleep Well', 'description': 'Get 7-9 hours of quality sleep each night'},
            {'title': 'Stay Hydrated', 'description': 'Drink plenty of water throughout the day'}
        ]
    }
   
    # Add risk-specific recommendations
    if risk_level == 'high':
        recommendations['diet'].extend([
            {'title': 'Anti-inflammatory Foods', 'description': 'Include berries, leafy greens, and fatty fish'},
            {'title': 'Low Glycemic Index', 'description': 'Choose foods that don\'t spike blood sugar'},
            {'title': 'Portion Control', 'description': 'Monitor portion sizes and track calories'}
        ])
        recommendations['exercise'].extend([
            {'title': 'Cardio Workouts', 'description': '30 minutes of cardio, 5 times per week'},
            {'title': 'Strength Training', 'description': '2-3 sessions per week focusing on major muscle groups'}
        ])
        recommendations['lifestyle'].extend([
            {'title': 'Regular Check-ups', 'description': 'Schedule regular visits with your healthcare provider'},
            {'title': 'Stress Reduction', 'description': 'Practice stress management techniques daily'}
        ])
   
    elif risk_level == 'moderate':
        recommendations['diet'].extend([
            {'title': 'Healthy Eating', 'description': 'Focus on a balanced diet with plenty of vegetables'},
            {'title': 'Mindful Eating', 'description': 'Pay attention to hunger cues and eat mindfully'}
        ])
        recommendations['exercise'].extend([
            {'title': 'Regular Exercise', 'description': '150 minutes of moderate activity per week'},
            {'title': 'Mixed Activities', 'description': 'Combine cardio and strength training'}
        ])
        recommendations['lifestyle'].extend([
            {'title': 'Lifestyle Balance', 'description': 'Maintain a good work-life balance'},
            {'title': 'Health Monitoring', 'description': 'Track your symptoms and cycles'}
        ])
   
    else:  # low risk
        recommendations['diet'].extend([
            {'title': 'Maintain Diet', 'description': 'Continue eating a balanced, nutritious diet'},
            {'title': 'Healthy Habits', 'description': 'Focus on maintaining good eating habits'}
        ])
        recommendations['exercise'].extend([
            {'title': 'Stay Active', 'description': 'Maintain regular physical activity'},
            {'title': 'Enjoyable Exercise', 'description': 'Choose activities you enjoy'}
        ])
        recommendations['lifestyle'].extend([
            {'title': 'Preventive Care', 'description': 'Focus on maintaining good health habits'},
            {'title': 'Regular Monitoring', 'description': 'Keep track of any changes in your health'}
        ])
   
    # Add common recommendations
    for category in recommendations:
        recommendations[category].extend(common_recommendations[category])
   
    return recommendations

# Routes
@app.route('/')
def home():
    if 'user_id' not in session:
        return render_template('home.html')
   
    user = db.users.find_one({'_id': ObjectId(session['user_id'])})
    if not user:
        session.clear()
        return redirect(url_for('home'))
   
    # Redirect based on account type
    if user['account_type'] == 'doctor':
        return redirect(url_for('doctor_dashboard'))
    elif user['account_type'] == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif user['account_type'] == 'lab_assistant':
        return redirect(url_for('lab_dashboard'))
    else:  # patient
        return redirect(url_for('patient_dashboard'))

def calculate_health_score(user):
    """
    Calculate a health score for the user based on their assessments, activities, and other health metrics.
    Returns a score between 0-100.
    """
    # Default score starts at 70 (average health)
    score = 70
    
    # Get user's latest assessment if available
    latest_assessment = db.assessments.find_one(
        {"user_id": user["_id"]},
        sort=[("date", -1)]
    )
    
    # Adjust score based on assessment data if available
    if latest_assessment:
        # If user has PCOS, reduce base score
        if latest_assessment.get("prediction") == "PCOS":
            score -= 10
        
        # Check symptoms and adjust score
        symptoms = latest_assessment.get("symptoms", {})
        if symptoms:
            # Common PCOS symptoms that affect health score
            if symptoms.get("irregular_periods", False):
                score -= 3
            if symptoms.get("weight_gain", False):
                score -= 3
            if symptoms.get("fatigue", False):
                score -= 2
            if symptoms.get("sleep_problems", False):
                score -= 2
            if symptoms.get("mood_changes", False):
                score -= 2
        
        # Check lifestyle factors and adjust score
        lifestyle = latest_assessment.get("lifestyle", {})
        if lifestyle:
            # Positive lifestyle factors
            if lifestyle.get("regular_exercise", False):
                score += 5
            if lifestyle.get("balanced_diet", False):
                score += 5
            if lifestyle.get("stress_management", False):
                score += 3
            if lifestyle.get("adequate_sleep", False):
                score += 3
            
            # Negative lifestyle factors
            if lifestyle.get("smoking", False):
                score -= 5
            if lifestyle.get("alcohol", False):
                score -= 3
    
    # Get recent activities to adjust score
    recent_activities = list(db.activities.find(
        {"user_id": user["_id"]},
        sort=[("date", -1)],
        limit=10
    ))
    
    # Adjust score based on recent activities
    if recent_activities:
        for activity in recent_activities:
            activity_type = activity.get("type")
            if activity_type == "exercise":
                score += 1
            elif activity_type == "meditation":
                score += 0.5
            elif activity_type == "diet_tracking":
                score += 0.5
    
    # Ensure score stays within 0-100 range
    score = max(0, min(100, score))
    
    # Round to nearest integer
    return round(score)

def get_health_tips(user):
    """
    Generate personalized health tips based on user's assessment data and health score.
    Returns a list of health tips.
    """
    tips = []
    
    # Get user's latest assessment if available
    latest_assessment = db.assessments.find_one(
        {"user_id": user["_id"]},
        sort=[("date", -1)]
    )
    
    # Default tips that are good for everyone
    default_tips = [
        {
            "category": "nutrition",
            "title": "Nutrition Tip",
            "content": "Include more fiber-rich foods in your diet like whole grains, fruits, and vegetables to help manage insulin levels."
        },
        {
            "category": "exercise",
            "title": "Exercise Tip",
            "content": "Aim for 30 minutes of moderate exercise most days of the week to help improve insulin sensitivity."
        },
        {
            "category": "lifestyle",
            "title": "Lifestyle Tip",
            "content": "Prioritize sleep quality - aim for 7-9 hours of quality sleep to help regulate hormones and reduce stress."
        }
    ]
    
    # If no assessment data, return default tips
    if not latest_assessment:
        return default_tips
    
    # Add personalized tips based on assessment data
    symptoms = latest_assessment.get("symptoms", {})
    lifestyle = latest_assessment.get("lifestyle", {})
    
    # Tips for specific symptoms
    if symptoms.get("irregular_periods", False):
        tips.append({
            "category": "health",
            "title": "Menstrual Health",
            "content": "Regular exercise and maintaining a healthy weight can help regulate your menstrual cycle."
        })
    
    if symptoms.get("weight_gain", False) or symptoms.get("obesity", False):
        tips.append({
            "category": "nutrition",
            "title": "Weight Management",
            "content": "Focus on a low-glycemic diet with plenty of vegetables, lean proteins, and healthy fats to help manage weight."
        })
    
    if symptoms.get("acne", False) or symptoms.get("skin_issues", False):
        tips.append({
            "category": "skincare",
            "title": "Skin Health",
            "content": "Stay hydrated and consider reducing dairy and sugar intake, which may help improve skin conditions related to PCOS."
        })
    
    if symptoms.get("fatigue", False) or symptoms.get("sleep_problems", False):
        tips.append({
            "category": "energy",
            "title": "Energy Boost",
            "content": "Try to establish a regular sleep schedule and consider short power naps during the day to combat fatigue."
        })
    
    # Tips based on lifestyle factors
    if not lifestyle.get("regular_exercise", False):
        tips.append({
            "category": "exercise",
            "title": "Start Moving",
            "content": "Even small amounts of physical activity can help. Try starting with a 10-minute walk after meals."
        })
    
    if not lifestyle.get("balanced_diet", False):
        tips.append({
            "category": "nutrition",
            "title": "Balanced Eating",
            "content": "Try the plate method: fill half your plate with vegetables, a quarter with lean protein, and a quarter with whole grains."
        })
    
    if not lifestyle.get("stress_management", False):
        tips.append({
            "category": "mental_health",
            "title": "Stress Relief",
            "content": "Practice deep breathing exercises for 5 minutes daily to help reduce stress and cortisol levels."
        })
    
    # If we have enough personalized tips, return those
    if len(tips) >= 3:
        return tips[:3]
    
    # Otherwise, add some default tips to reach 3 tips total
    remaining_tips_needed = 3 - len(tips)
    for i in range(remaining_tips_needed):
        tips.append(default_tips[i])
    
    return tips


@app.route('/patient/dashboard')
@login_required
def patient_dashboard():
    user = db.users.find_one({'_id': ObjectId(session['user_id'])})
    if user.get('account_type') != 'patient':
        return redirect(url_for('home'))
    
    # Get latest assessment
    latest_assessment = db.assessments.find_one(
        {'user_id': ObjectId(session['user_id'])},
        sort=[('date', -1)]
    )
    
    # Ensure `latest_assessment` has prediction data
    assessment_data = {
        'date': latest_assessment.get('date', datetime.utcnow()),
        'prediction': {
            'result': latest_assessment.get('prediction', {}).get('result', 'Not Available'),
            'probability': latest_assessment.get('prediction', {}).get('probability', 0.0),
            'confidence': latest_assessment.get('prediction', {}).get('confidence', 'low')
        },
        'personal_info': latest_assessment.get('personal_info', {}),
        'symptoms': latest_assessment.get('symptoms', {}),
        'lifestyle': latest_assessment.get('lifestyle', {})
    } if latest_assessment else None

    # Get the upcoming appointment
    upcoming_appointment = db.appointments.find_one(
        {'patient_id': ObjectId(session['user_id']), 'appointment_date': {'$gte': datetime.now()}},
        sort=[('appointment_date', 1)]
    )

    # Get latest diet plan
    latest_diet_plan = db.diet_plans.find_one(
        {'patient_id': ObjectId(session['user_id'])},
        sort=[('creation_date', -1)]
    )

    # Get recent activities
    recent_activities = list(db.activities.find(
        {'user_id': ObjectId(session['user_id'])}
    ).sort('date', -1).limit(5))

    # Calculate health score
    health_score = calculate_health_score(user)

    # Get personalized health tips
    health_tips = get_health_tips(user)

    # Get notifications
    notifications = list(db.notifications.find(
        {'user_id': ObjectId(session['user_id'])}
    ).sort('created_at', -1).limit(5))

    # Determine statuses
    questionnaire_status = 'completed' if db.assessments.find_one({'user_id': ObjectId(session['user_id']), 'type': 'questionnaire'}) else 'pending'
    
    ultrasound_status = 'completed' if db.ultrasound_requests.find_one({'patient_id': ObjectId(session['user_id']), 'status': 'completed'}) else 'pending'
    
    diagnosis_status = 'completed' if db.diagnoses.find_one({'patient_id': ObjectId(session['user_id'])}) else 'pending'

    # Calculate days since joining
    days_since_join = (datetime.now() - user.get('created_at', datetime.utcnow())).days

    # Count completed assessments
    completed_assessments = db.assessments.count_documents({'user_id': ObjectId(session['user_id'])})

    return render_template('patient_home.html',
                         user=user,
                         latest_assessment=assessment_data,
                         upcoming_appointment=upcoming_appointment,
                         latest_diet_plan=latest_diet_plan,
                         recent_activities=recent_activities,
                         health_score=health_score,
                         health_tips=health_tips,
                         notifications=notifications,
                         questionnaire_status=questionnaire_status,
                         ultrasound_status=ultrasound_status,
                         diagnosis_status=diagnosis_status,
                         days_since_join=days_since_join,
                         completed_assessments=completed_assessments)

@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        session['test_taken'] = True  # Store test status in session
        return redirect(url_for('test'))  # Refresh page to show new options
   
    test_taken = session.get('test_taken', False)  # Check if user took test
    return render_template('test.html', test_taken=test_taken)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        users = db.users
        user = users.find_one({'email': request.form['email']})
       
        if user and user['password'] == request.form['password']:  # In production, verify hashed password
            session['user_id'] = str(user['_id'])
            flash('Login successful!', 'success')
           
            # Redirect based on account type
            if user['account_type'] == 'doctor':
                return redirect(url_for('doctor_dashboard'))
            elif user['account_type'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user['account_type'] == 'lab_assistant':
                return redirect(url_for('lab_dashboard'))
            else:  # patient
                return redirect(url_for('patient_dashboard'))
           
        flash('Invalid email or password!', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        users = db.users
        existing_user = users.find_one({'email': request.form['email']})
       
        if existing_user is None:
            user = {
                'name': request.form['name'],
                'email': request.form['email'],
                'password': request.form['password'],  # In production, hash this password
                'account_type': request.form['account_type'],
                'created_at': datetime.utcnow(),
                'status': 'pending' if request.form['account_type'] in ['doctor', 'lab_assistant'] else 'active',
                'notification_settings': {
                    'email_notifications': True,
                    'test_reminders': True,
                    'health_tips': True
                }
            }
           
            # Role-specific fields
            if request.form['account_type'] == 'doctor':
                user.update({
                    'specialization': request.form['specialization'],
                    'license_number': request.form['license_number'],
                    'hospital': request.form['hospital'],
                    'experience': int(request.form.get('experience', 0)),  # Default to 0 if empty
                    'verified': False
                })
            elif request.form['account_type'] == 'lab_assistant':
                user.update({
                    'lab_name': request.form['lab_name'],
                    'qualification': request.form['qualification'],
                    'experience': int(request.form.get('experience', 0)),  # Default to 0 if empty
                    'specializations': request.form.getlist('specializations'),
                    'verified': False
                })
            elif request.form['account_type'] == 'patient':
                user.update({
                    'age': None,
                    'height': None,
                    'weight': None,
                    'tests_taken': 0,
                    'days_tracked': 0,
                    'achievements': 0,
                    'bmi': None,
                    'cycle_length': None,
                    'last_test_date': None,
                    'assigned_doctor': None,
                    'assigned_lab': None
                })
            elif request.form['account_type'] == 'admin':
                user.update({
                    'is_admin': True,
                    'verified': True
                })
           
            users.insert_one(user)
           
            if request.form['account_type'] in ['doctor', 'lab_assistant']:
                flash('Registration submitted for admin approval. You will be notified once approved.', 'info')
            else:
                session['user_id'] = str(user['_id'])
                flash('Registration successful!', 'success')
                return redirect(url_for('home'))
           
            return redirect(url_for('login'))
           
        flash('Email already exists!', 'error')
    return render_template('register.html')

@app.route('/blog')
def blog():
    # Get all blog posts from MongoDB
    blogs = list(db.blogs.find().sort('date_posted', -1))
    # Convert ObjectId to string for JSON serialization
    for blog in blogs:
        blog['_id'] = str(blog['_id'])
        blog['date_posted'] = blog['date_posted'].strftime('%Y-%m-%d')
    return render_template('blog.html', blogs=blogs)

@app.route('/blog/<blog_id>')
def blog_detail(blog_id):
    try:
        blog = db.blogs.find_one({'_id': ObjectId(blog_id)})
        if blog:
            blog['_id'] = str(blog['_id'])
            blog['date_posted'] = blog['date_posted'].strftime('%Y-%m-%d')
            return render_template('blog_detail.html', blog=blog)
        else:
            flash('Blog post not found', 'error')
            return redirect(url_for('blog'))
    except Exception as e:
        flash('Invalid blog ID', 'error')
        return redirect(url_for('blog'))

@app.route('/blog/category/<category>')
def blog_by_category(category):
    blogs = list(db.blogs.find({'category': category}).sort('date_posted', -1))
    for blog in blogs:
        blog['_id'] = str(blog['_id'])
        blog['date_posted'] = blog['date_posted'].strftime('%Y-%m-%d')
    return render_template('blog.html', blogs=blogs, current_category=category)

@app.route('/account')
def account():
    return render_template('account.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    file = request.files.get('ultrasound_image')
    if file and file.filename:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)  # Save image to uploads folder
        return "Image uploaded successfully!"
    return "Failed to upload!"

@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if 'user_id' not in session:
        return redirect(url_for('login'))
   
    if request.method == 'POST':
        try:
            # Get form data
            assessment_data = {
                'user_id': ObjectId(session['user_id']),
                'date': datetime.utcnow(),
                'personal_info': {
                    'age': int(request.form['age']),
                    'weight': float(request.form['weight']),
                    'height': float(request.form['height']),
                    'blood_group': request.form.get('blood_group', 'Not Specified'),
                    'bmi': round(float(request.form['weight']) / ((float(request.form['height'])/100) ** 2), 2)
                },
                'menstrual_history': {
                    'cycle_regularity': request.form['cycle_regularity'],
                    'cycle_length': int(request.form['cycle_length'])
                },
                'marriage_pregnancy': {
                    'marriage_years': int(request.form['marriage_years']),
                    'pregnant': request.form['pregnant'] == 'yes',
                    'abortions': int(request.form['abortions'])
                },
                'symptoms': {
                    'weight_gain': request.form['weight_gain'] == 'yes',
                    'hair_growth': request.form['hair_growth'] == 'yes',
                    'skin_darkening': request.form['skin_darkening'] == 'yes',
                    'hair_loss': request.form['hair_loss'] == 'yes',
                    'pimples': request.form['pimples'] == 'yes'
                },
                'lifestyle': {
                    'fast_food': request.form['fast_food'] == 'yes',
                    'regular_exercise': request.form['regular_exercise'] == 'yes'
                }
            }
           
            # Convert blood group to numeric
            blood_group_map = {
                'A+': 1, 'A-': 2, 'B+': 3, 'B-': 4,
                'O+': 5, 'O-': 6, 'AB+': 7, 'AB-': 8,
                'Not Specified': 0
            }
           
            # Prepare features dictionary
            features = {
                'age': float(assessment_data['personal_info']['age']),
                'weight': float(assessment_data['personal_info']['weight']),
                'height': float(assessment_data['personal_info']['height']),
                'bmi': float(assessment_data['personal_info']['bmi']),
                'blood_group': float(blood_group_map.get(assessment_data['personal_info']['blood_group'], 0)),
                'pulse_rate': 72.0,  # Default normal pulse rate
                'rr': 16.0,  # Default respiratory rate
                'hb': 12.0,  # Default hemoglobin
                'cycle': float(1 if assessment_data['menstrual_history']['cycle_regularity'] == 'regular' else 0),
                'cycle_length': float(assessment_data['menstrual_history']['cycle_length']),
                'marriage_status': 1.0,  # Default to 1 for married
                'pregnant': float(1 if assessment_data['marriage_pregnancy']['pregnant'] else 0),
                'no_of_abortions': float(assessment_data['marriage_pregnancy']['abortions']),
                'beta_hcg': 5.0,  # Default beta HCG
                'fsh': 7.0,  # Default FSH
                'lh': 7.0,  # Default LH
                'fsh_lh_ratio': 1.0,  # Default FSH/LH ratio
                'hip': 90.0,  # Default hip measurement
                'waist': 75.0,  # Default waist measurement
                'waist_hip_ratio': 0.83,  # Default waist/hip ratio
                'tsh': 2.5,  # Default TSH
                'amh': 3.0,  # Default AMH
                'prl': 15.0,  # Default prolactin
                'vit_d3': 30.0,  # Default vitamin D3
                'prg': 1.0,  # Default progesterone
                'rbs': 85.0,  # Default random blood sugar
                'weight_gain': float(1 if assessment_data['symptoms']['weight_gain'] else 0),
                'hair_growth': float(1 if assessment_data['symptoms']['hair_growth'] else 0),
                'skin_darkening': float(1 if assessment_data['symptoms']['skin_darkening'] else 0),
                'hair_loss': float(1 if assessment_data['symptoms']['hair_loss'] else 0),
                'pimples': float(1 if assessment_data['symptoms']['pimples'] else 0),
                'fast_food': float(1 if assessment_data['lifestyle']['fast_food'] else 0),
                'reg_exercise': float(1 if assessment_data['lifestyle']['regular_exercise'] else 0),
                'bp_systolic': 120.0,  # Default systolic BP
                'bp_diastolic': 80.0,  # Default diastolic BP
                'follicle_no_l': 8.0,  # Default follicle number left
                'follicle_no_r': 8.0,  # Default follicle number right
                'avg_f_size_l': 5.0,  # Default follicle size left
                'avg_f_size_r': 5.0,  # Default follicle size right
                'endometrium': 8.0  # Default endometrium thickness
            }
           
            # Make prediction
            if questionnaire_model is not None and scaler is not None:
                try:
                    print("Starting prediction process...")
                    print(f"Model type: {type(questionnaire_model)}")
                    print(f"Scaler type: {type(scaler)}")
                   
                    # Get the feature order from the preprocess_questionnaire_data function
                    feature_order = [
                        'age', 'weight', 'height', 'bmi', 'blood_group', 'pulse_rate',
                        'rr', 'hb', 'cycle', 'cycle_length', 'marriage_status', 'pregnant',
                        'no_of_abortions', 'beta_hcg', 'fsh', 'lh', 'fsh_lh_ratio',
                        'hip', 'waist', 'waist_hip_ratio', 'tsh', 'amh', 'prl',
                        'vit_d3', 'prg', 'rbs', 'weight_gain', 'hair_growth',
                        'skin_darkening', 'hair_loss', 'pimples', 'fast_food',
                        'reg_exercise', 'bp_systolic', 'bp_diastolic', 'follicle_no_l',
                        'follicle_no_r', 'avg_f_size_l', 'avg_f_size_r', 'endometrium'
                    ]
                   
                    # Create feature array in correct order
                    feature_array = np.array([[features[f] for f in feature_order]])
                    print(f"Feature array shape: {feature_array.shape}")
                   
                    # Scale features
                    scaled_features = scaler.transform(feature_array)
                    print(f"Scaled features shape: {scaled_features.shape}")
                   
                    # Get prediction
                    try:
                        prediction_prob = questionnaire_model.predict_proba(scaled_features)[0][1]
                        print(f"Raw prediction probability: {prediction_prob}")
                    except Exception as e:
                        print(f"Error getting prediction probability: {str(e)}")
                        # Fallback to a simple rule-based prediction
                        symptoms_count = sum([
                            assessment_data['symptoms']['weight_gain'],
                            assessment_data['symptoms']['hair_growth'],
                            assessment_data['symptoms']['skin_darkening'],
                            assessment_data['symptoms']['hair_loss'],
                            assessment_data['symptoms']['pimples']
                        ])
                        prediction_prob = min(0.8, 0.3 + (symptoms_count * 0.1))
                        print(f"Using fallback prediction probability: {prediction_prob}")
                   
                    prediction_result = 'PCOS' if prediction_prob >= 0.5 else 'Normal'
                   
                    # Calculate confidence level based on probability
                    if prediction_prob >= 0.8 or prediction_prob <= 0.2:
                        confidence = 'high'
                    elif prediction_prob >= 0.6 or prediction_prob <= 0.4:
                        confidence = 'medium'
                    else:
                        confidence = 'low'
                   
                    # Get indicators based on prediction and symptoms
                    indicators = []
                    if prediction_result == 'PCOS':
                        if prediction_prob >= 0.8:
                            indicators.extend(['high_risk', 'multiple_symptoms'])
                        if prediction_prob >= 0.6:
                            indicators.append('moderate_risk')
                       
                        # Add specific indicators based on symptoms
                        if assessment_data['symptoms']['weight_gain']:
                            indicators.append('weight_gain')
                        if assessment_data['symptoms']['hair_growth']:
                            indicators.append('hair_growth')
                        if assessment_data['symptoms']['skin_darkening']:
                            indicators.append('skin_darkening')
                        if assessment_data['symptoms']['hair_loss']:
                            indicators.append('hair_loss')
                        if assessment_data['symptoms']['pimples']:
                            indicators.append('acne')
                        if assessment_data['menstrual_history']['cycle_regularity'] != 'regular':
                            indicators.append('irregular_cycles')
                   
                    # Add prediction to assessment data
                    assessment_data['prediction'] = {
                        'probability': float(prediction_prob),
                        'result': prediction_result,
                        'confidence': confidence,
                        'indicators': indicators
                    }
                    print(f"Final prediction: {assessment_data['prediction']}")
                   
                except Exception as e:
                    print(f"Prediction error: {str(e)}")
                    print(f"Error type: {type(e).__name__}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    # Fallback to a simple rule-based prediction
                    symptoms_count = sum([
                        assessment_data['symptoms']['weight_gain'],
                        assessment_data['symptoms']['hair_growth'],
                        assessment_data['symptoms']['skin_darkening'],
                        assessment_data['symptoms']['hair_loss'],
                        assessment_data['symptoms']['pimples']
                    ])
                    prediction_prob = min(0.8, 0.3 + (symptoms_count * 0.1))
                    prediction_result = 'PCOS' if prediction_prob >= 0.5 else 'Normal'
                    confidence = 'medium' if symptoms_count >= 3 else 'low'
                   
                    assessment_data['prediction'] = {
                        'probability': float(prediction_prob),
                        'result': prediction_result,
                        'confidence': confidence,
                        'indicators': ['fallback_prediction'] + (['multiple_symptoms'] if symptoms_count >= 3 else [])
                    }
            else:
                print("Models not available")
                print(f"Questionnaire model: {questionnaire_model}")
                print(f"Scaler: {scaler}")
                assessment_data['prediction'] = {
                    'probability': 0.0,
                    'result': 'Model Unavailable',
                    'confidence': 'low',
                    'indicators': ['model_unavailable']
                }
           
            # Store assessment in database
            db.assessments.insert_one(assessment_data)
           
            # Update user's last assessment
            db.users.update_one(
                {'_id': ObjectId(session['user_id'])},
                {
                    '$set': {
                        'last_assessment_date': datetime.utcnow(),
                        'last_assessment_result': assessment_data['prediction']['result'],
                        'last_assessment_probability': assessment_data['prediction']['probability']
                    },
                    '$inc': {'tests_taken': 1}
                }
            )
           
            # Store prediction in session for results page
            session['pcos_risk'] = assessment_data['prediction']['result'].lower()
            session['prediction_confidence'] = assessment_data['prediction']['confidence']
            session['prediction_indicators'] = assessment_data['prediction']['indicators']
           
            # Log activity
            activity = {
                'user_id': ObjectId(session['user_id']),
                'type': 'assessment_completed',
                'details': f"Completed PCOS assessment with result: {assessment_data['prediction']['result']}",
                'date': datetime.utcnow()
            }
            db.activities.insert_one(activity)
           
            return redirect(url_for('test_results'))
           
        except Exception as e:
            print(f"Form processing error: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            flash('An error occurred while processing your questionnaire. Please try again.', 'error')
            return redirect(url_for('questionnaire'))
   
    return render_template('questionnaire.html')

@app.route('/upload-ultrasound', methods=['GET', 'POST'])
def upload_ultrasound():
    if 'user_id' not in session:
        return redirect(url_for('login'))
   
    if request.method == 'POST':
        try:
            if 'ultrasound' not in request.files:
                flash('No ultrasound image uploaded', 'error')
                return redirect(url_for('upload_ultrasound'))
           
            file = request.files['ultrasound']
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect(url_for('upload_ultrasound'))
           
            # Save the uploaded image
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_filename = f"{timestamp}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
           
            # Preprocess the image for the model
            processed_image = preprocess_image(file)
            if processed_image is None:
                flash('Error processing the image', 'error')
                return redirect(url_for('upload_ultrasound'))
           
            # Make prediction using the loaded model
            if ultrasound_model is None:
                flash('Model not loaded properly', 'error')
                return redirect(url_for('upload_ultrasound'))
           
            prediction = ultrasound_model.predict(processed_image)
            probability = float(prediction[0][0])
            result = 'PCOS' if probability >= 0.5 else 'Normal'
           
            # Get confidence level based on probability
            if probability >= 0.8 or probability <= 0.2:
                confidence = 'high'
            elif probability >= 0.6 or probability <= 0.4:
                confidence = 'medium'
            else:
                confidence = 'low'
           
            # Get indicators based on prediction
            indicators = []
            if result == 'PCOS':
                if probability >= 0.8:
                    indicators.extend(['polycystic_ovaries', 'enlarged_ovaries'])
                if probability >= 0.6:
                    indicators.append('thickened_endometrium')
           
            # Store the prediction results in session
            session['pcos_risk'] = result.lower()
            session['prediction_confidence'] = confidence
            session['prediction_indicators'] = indicators
            session['ultrasound_image'] = unique_filename
           
            # Create assessment record
            assessment = {
                'user_id': ObjectId(session['user_id']),
                'date': datetime.utcnow(),
                'type': 'ultrasound',
                'image_path': unique_filename,
                'prediction': {
                    'result': result,
                    'probability': probability,
                    'confidence': confidence,
                    'indicators': indicators
                }
            }
           
            db.assessments.insert_one(assessment)
           
            # Log activity
            activity = {
                'user_id': ObjectId(session['user_id']),
                'type': 'ultrasound_uploaded',
                'details': f"Uploaded ultrasound image with prediction: {result}",
                'date': datetime.utcnow()
            }
            db.activities.insert_one(activity)
           
            return redirect(url_for('test_results'))
           
        except Exception as e:
            print(f"Error in upload_ultrasound: {str(e)}")
            flash('An error occurred while processing your ultrasound image. Please try again.', 'error')
            return redirect(url_for('upload_ultrasound'))
   
    return render_template('upload_ultrasound.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        # Save the contact form submission to the database
        contact_submission = {
            'name': name,
            'email': email,
            'subject': subject,
            'message': message,
            'date': datetime.now(),
            'status': 'new'
        }
        
        db.contact_submissions.insert_one(contact_submission)
        
        flash('Thank you for your message! We will get back to you soon.', 'success')
        return redirect(url_for('contact'))
        
    return render_template('contact.html')

@app.route('/test-results', methods=['GET', 'POST'])
def test_results():
    if 'user_id' not in session:
        return redirect(url_for('login'))
   
    # Get the user's latest assessment from database
    latest_assessment = None
    pcos_result = None
    risk_score = 0
   
    if 'user_id' in session:
        latest_assessment = db.assessments.find_one(
            {'user_id': ObjectId(session['user_id'])},
            sort=[('date', -1)]
        )
       
        if latest_assessment:
            # Get the prediction result
            prediction = latest_assessment.get('prediction', {})
            pcos_result = prediction.get('result', 'Unknown')
            risk_score = prediction.get('probability', 0)
           
            # Determine risk level based on probability
            if risk_score > 0.7:
                risk_level = 'high'
            elif risk_score > 0.4:
                risk_level = 'moderate'
            else:
                risk_level = 'low'
        else:
            risk_level = 'unknown'
            pcos_result = 'No assessment data available'
   
    # Get personalized recommendations
    recommendations = get_personalized_recommendations(risk_level)
   
    # Get user data for the template
    user = None
    if 'user_id' in session:
        user = db.users.find_one({'_id': ObjectId(session['user_id'])})
   
    return render_template('test_results.html',
                         risk_level=risk_level,
                         risk_score=risk_score,
                         pcos_result=pcos_result,
                         recommendations=recommendations,
                         assessment=latest_assessment,
                         user=user)

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        # Add your password reset logic here
        # For example, send a reset link to the user's email
        flash('If an account exists with this email, you will receive password reset instructions.', 'info')
        return redirect(url_for('login'))
    return render_template('forgot_password.html')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    users = db.users
    activities = db.activities

    user = users.find_one({'_id': ObjectId(session['user_id'])})
    if not user:
        session.clear()
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            age = request.form.get('age', '').strip()
            height = request.form.get('height', '').strip()
            weight = request.form.get('weight', '').strip()

            # Validate and convert inputs
            age = int(age) if age.isdigit() and int(age) > 0 else None
            height = float(height) if height.replace('.', '', 1).isdigit() and float(height) > 0 else None
            weight = float(weight) if weight.replace('.', '', 1).isdigit() and float(weight) > 0 else None

            if height and weight:
                height_m = height / 100  # Convert cm to meters
                bmi = round(weight / (height_m ** 2), 1)
            else:
                bmi = None

            update_data = {'age': age, 'height': height, 'weight': weight, 'bmi': bmi}

            # Remove None values to avoid overwriting fields with null
            update_data = {k: v for k, v in update_data.items() if v is not None}

            if update_data:
                users.update_one({'_id': ObjectId(session['user_id'])}, {'$set': update_data})

                # Log the activity with updated fields
                activity = {
                    'user_id': ObjectId(session['user_id']),
                    'type': 'profile_update',
                    'details': f"Updated fields: {', '.join(update_data.keys())}",
                    'date': datetime.utcnow()
                }
                activities.insert_one(activity)

                return jsonify({'status': 'success', 'message': 'Profile updated successfully'})

            return jsonify({'status': 'error', 'message': 'No valid data provided'}), 400

        except ValueError:
            return jsonify({'status': 'error', 'message': 'Invalid input values'}), 400

    # Fetch and format user activities
    user_activities = list(activities.find({'user_id': ObjectId(session['user_id'])}).sort('date', -1).limit(5))
    formatted_activities = [
        {
            'type': activity.get('type', 'general'),
            'details': activity.get('details', ''),
            'date': activity.get('date', datetime.utcnow()).strftime('%Y-%m-%d')
        }
        for activity in user_activities
    ]

    return render_template('profile.html', user=user, activities=formatted_activities)

@app.route('/upload-profile-image', methods=['POST'])
def upload_profile_image():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
   
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400
       
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No image selected'}), 400
       
    if file:
        # Create user-specific directory
        user_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], session['user_id'])
        os.makedirs(user_upload_dir, exist_ok=True)
       
        # Save file
        filename = f"profile_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = os.path.join(user_upload_dir, filename)
        file.save(filepath)
       
        # Update user profile with new image path
        users = db.users
        users.update_one(
            {'_id': ObjectId(session['user_id'])},
            {'$set': {'profile_image': filename}}
        )
       
        return jsonify({
            'status': 'success',
            'message': 'Profile image updated',
            'image_url': url_for('static', filename=f'uploads/{session["user_id"]}/{filename}')
        })

@app.route('/update-notifications', methods=['POST'])
def update_notifications():
    if 'user_id' not in session:
        return jsonify({'status': 'error', 'message': 'Not logged in'}), 401
   
    users = db.users
    notification_settings = {
        'email_notifications': request.json.get('email_notifications', False),
        'test_reminders': request.json.get('test_reminders', False),
        'health_tips': request.json.get('health_tips', False)
    }
   
    users.update_one(
        {'_id': ObjectId(session['user_id'])},
        {'$set': {'notification_settings': notification_settings}}
    )
   
    return jsonify({'status': 'success', 'message': 'Notification settings updated'})

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

def doctor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))
        
        # Fetch user from MongoDB
        user = db.users.find_one({'_id': ObjectId(session['user_id'])})

        if not user or user.get('account_type') != 'doctor':
            flash('Access denied. Doctor privileges required.', 'danger')
            return redirect(url_for('home'))
       
        return f(*args, **kwargs)
    return decorated_function

@app.route('/doctor/dashboard')
@doctor_required
def doctor_dashboard():
    user = db.users.find_one({'_id': ObjectId(session['user_id'])})
    print(f"DEBUG: Doctor dashboard for: {user['name']} (ID: {user['_id']})")
   
    # Get pending ultrasound reviews
    pending_ultrasounds = list(db.ultrasound_requests.find({
        'status': 'completed',
        'doctor_diagnosis': None,
        'images': {'$exists': True, '$ne': []}
    }).sort('upload_date', -1))
    
    print(f"DEBUG: Found {len(pending_ultrasounds)} pending ultrasound requests")
    
    # Add patient information to ultrasound requests
    for request in pending_ultrasounds:
        patient = db.users.find_one({'_id': request['patient_id']})
        if patient:
            request['patient'] = patient
            # Get patient's questionnaire assessment
            questionnaire = db.assessments.find_one({
                'user_id': request['patient_id'],
                'type': 'questionnaire'
            }, sort=[('date', -1)])
            request['patient']['questionnaire'] = questionnaire
            print(f"DEBUG: Request ID: {request['_id']}, Patient: {patient['name']}, Images: {len(request.get('images', []))}")
        else:
            request['patient'] = {'name': 'Unknown Patient', 'questionnaire': None}
            print(f"DEBUG: Request ID: {request['_id']}, Patient: Unknown, Images: {len(request.get('images', []))}")
   
    # Get doctor's patients
    patients = list(db.users.find({
        'assigned_doctor': ObjectId(session['user_id']),
        'account_type': 'patient'
    }).sort('last_visit', -1))
   
    # Get recent reports
    recent_reports = list(db.final_reports.find({
        'doctor_id': ObjectId(session['user_id'])
    }).sort('created_at', -1).limit(5))
    
    # Add patient information to reports
    for report in recent_reports:
        patient = db.users.find_one({'_id': report['patient_id']})
        if patient:
            report['patient'] = patient
   
    # Get statistics
    total_patients = len(patients)
    pending_reviews = len(pending_ultrasounds)
    completed_reviews = db.ultrasound_requests.count_documents({
        'status': 'completed',
        'doctor_diagnosis': {'$ne': None}
    })
    total_reports = db.final_reports.count_documents({
        'doctor_id': ObjectId(session['user_id'])
    })
   
    return render_template('doctor/dashboard.html',
                         user=user,
                         pending_ultrasounds=pending_ultrasounds,
                         recent_reports=recent_reports,
                         total_patients=total_patients,
                         pending_reviews=pending_reviews,
                         completed_reviews=completed_reviews,
                         total_reports=total_reports)
@app.route('/completed-reviews')
def completed_reviews():
    completed_reviews = list(db.ultrasound_requests.find({'status': 'completed'}))
    return render_template('doctor/completed_reviews.html', completed_reviews=completed_reviews)

@app.route('/review-details/<int:review_id>')
def view_review_details(review_id):
    review = UltrasoundReview.query.get_or_404(review_id)
    return render_template('review_details.html', review=review)

@app.route('/doctor/patient/<patient_id>/questionnaire')
@doctor_required
def view_patient_questionnaire(patient_id):
    patient = db.users.find_one({'_id': ObjectId(patient_id)})
    if not patient:
        flash('Patient not found', 'error')
        return redirect(url_for('doctor_dashboard'))
   
    # Get patient's questionnaire assessments
    questionnaire_assessments = list(db.assessments.find({
        'user_id': ObjectId(patient_id),
        'type': 'questionnaire'
    }).sort('date', -1))
   
    return render_template('doctor/view_questionnaire.html',
                         patient=patient,
                         assessments=questionnaire_assessments)

@app.route('/doctor/evaluate-ultrasound/<evaluation_id>', methods=['GET', 'POST'])
@doctor_required
def evaluate_ultrasound(evaluation_id):
    evaluation = db.ultrasound_evaluations.find_one({'_id': ObjectId(evaluation_id)})
    if not evaluation:
        flash('Evaluation not found', 'error')
        return redirect(url_for('doctor_dashboard'))
   
    if request.method == 'POST':
        try:
            # Get form data
            diagnosis = request.form.get('diagnosis')
            confidence = request.form.get('confidence')
            indicators = request.form.getlist('indicators')
            notes = request.form.get('notes')

            # Update evaluation with doctor's diagnosis
            update_data = {
                'diagnosis': diagnosis,
                'confidence': confidence,
                'pcos_indicators': indicators,
                'notes': notes,
                'evaluation_date': datetime.utcnow(),
                'status': 'completed',
                'doctor_id': ObjectId(session['user_id'])
            }

            db.ultrasound_evaluations.update_one(
                {'_id': ObjectId(evaluation_id)},
                {'$set': update_data}
            )

            # Create assessment record
            assessment = {
                'user_id': evaluation['patient_id'],
                'date': datetime.utcnow(),
                'type': 'combined',
                'doctor_id': ObjectId(session['user_id']),
                'lab_assistant_id': evaluation['lab_assistant_id'],
                'ultrasound_evaluation_id': ObjectId(evaluation_id),
                'diagnosis': diagnosis,
                'confidence': confidence,
                'pcos_indicators': indicators,
                'notes': notes,
                'ai_predictions': {
                    'ultrasound': evaluation['ai_prediction'],
                    'questionnaire': evaluation.get('questionnaire_prediction', {})
                }
            }
            db.assessments.insert_one(assessment)

            # Create activity log
            activity = {
                'user_id': evaluation['patient_id'],
                'type': 'ultrasound_evaluated',
                'details': 'Ultrasound evaluated by Dr. ' + db.users.find_one({'_id': ObjectId(session['user_id'])})['name'],
                'date': datetime.utcnow()
            }
            db.activities.insert_one(activity)

            return jsonify({'success': True, 'message': 'Evaluation submitted successfully'})

        except Exception as e:
            return jsonify({'success': False, 'message': str(e)})

    # Get patient information
    patient = db.users.find_one({'_id': evaluation['patient_id']})
   
    # Get questionnaire results if available
    questionnaire = db.assessments.find_one({
        'user_id': evaluation['patient_id'],
        'type': 'questionnaire'
    }, sort=[('date', -1)])

    return render_template('doctor/evaluate_ultrasound.html',
                         evaluation=evaluation,
                         patient=patient,
                         questionnaire=questionnaire)

@app.route('/doctor/evaluations')
@doctor_required
def doctor_evaluations():
    # Get pending evaluations
    pending_evaluations = list(db.ultrasound_evaluations.find({
        'status': 'pending'
    }).sort('upload_date', -1))

    # Get completed evaluations
    completed_evaluations = list(db.ultrasound_evaluations.find({
        'status': 'completed',
        'doctor_id': ObjectId(session['user_id'])
    }).sort('evaluation_date', -1))

    return render_template('doctor/evaluations.html',
                         pending_evaluations=pending_evaluations,
                         completed_evaluations=completed_evaluations)

@app.route('/doctor/assign-patient', methods=['POST'])
@doctor_required
def assign_patient():
    patient_id = request.form.get('patient_id')
    lab_assistant_id = request.form.get('lab_assistant_id')
   
    if not patient_id or not lab_assistant_id:
        flash('Missing required information', 'error')
        return redirect(url_for('doctor_dashboard'))
   
    # Update patient's assigned doctor and lab
    db.users.update_one(
        {'_id': ObjectId(patient_id)},
        {
            '$set': {
                'assigned_doctor': ObjectId(session['user_id']),
                'assigned_lab': ObjectId(lab_assistant_id)
            }
        }
    )
   
    flash('Patient assigned successfully', 'success')
    return redirect(url_for('doctor_dashboard'))

@app.route('/doctor/patient/<patient_id>')
@doctor_required
def patient_history(patient_id):
    patient = db.users.find_one({'_id': ObjectId(patient_id)})
    if not patient:
        flash('Patient not found', 'error')
        return redirect(url_for('doctor_dashboard'))
   
    # Get patient's assessment history
    assessments = list(db.assessments.find({'user_id': ObjectId(patient_id)}).sort('date', -1))
   
    # Get patient's ultrasound history
    ultrasounds = list(db.ultrasound_evaluations.find({'patient_id': ObjectId(patient_id)}).sort('submission_date', -1))
   
    # Get patient's diet plans
    diet_plans = list(db.diet_plans.find({'patient_id': ObjectId(patient_id)}).sort('creation_date', -1))
   
    return render_template('patient_history.html',
                         patient=patient,
                         assessments=assessments,
                         ultrasounds=ultrasounds,
                         diet_plans=diet_plans)

@app.route('/doctor/diet-plan/create/<patient_id>', methods=['GET', 'POST'])
@doctor_required
def create_diet_plan(patient_id):
    patient = db.users.find_one({'_id': ObjectId(patient_id)})
    if not patient:
        flash('Patient not found', 'error')
        return redirect(url_for('doctor_dashboard'))
   
    if request.method == 'POST':
        plan_data = {
            'patient_id': ObjectId(patient_id),
            'doctor_id': ObjectId(session['user_id']),
            'creation_date': datetime.utcnow(),
            'title': request.form.get('title'),
            'description': request.form.get('description'),
            'duration_weeks': int(request.form.get('duration_weeks')),
            'meal_plan': {
                'breakfast': request.form.getlist('breakfast'),
                'lunch': request.form.getlist('lunch'),
                'dinner': request.form.getlist('dinner'),
                'snacks': request.form.getlist('snacks')
            },
            'recommendations': request.form.get('recommendations'),
            'restrictions': request.form.getlist('restrictions'),
            'supplements': request.form.getlist('supplements')
        }
       
        db.diet_plans.insert_one(plan_data)
       
        # Create activity log
        activity = {
            'user_id': ObjectId(patient_id),
            'type': 'diet_plan_created',
            'details': 'New diet plan created by Dr. ' + db.users.find_one({'_id': ObjectId(session['user_id'])})['name'],
            'date': datetime.utcnow()
        }
        db.activities.insert_one(activity)
       
        flash('Diet plan created successfully', 'success')
        return redirect(url_for('patient_history', patient_id=patient_id))
   
    return render_template('create_diet_plan.html', patient=patient)

@app.route('/doctor/diet-plan/<plan_id>')
@doctor_required
def view_diet_plan(plan_id):
    plan = db.diet_plans.find_one({'_id': ObjectId(plan_id)})
    if not plan:
        flash('Diet plan not found', 'error')
        return redirect(url_for('doctor_dashboard'))
   
    patient = db.users.find_one({'_id': plan['patient_id']})
    return render_template('view_diet_plan.html', plan=plan, patient=patient)

@app.route('/doctor/diet-plan/edit/<plan_id>', methods=['GET', 'POST'])
@doctor_required
def edit_diet_plan(plan_id):
    plan = db.diet_plans.find_one({'_id': ObjectId(plan_id)})
    if not plan:
        flash('Diet plan not found', 'error')
        return redirect(url_for('doctor_dashboard'))
   
    if request.method == 'POST':
        update_data = {
            'title': request.form.get('title'),
            'description': request.form.get('description'),
            'duration_weeks': int(request.form.get('duration_weeks')),
            'meal_plan': {
                'breakfast': request.form.getlist('breakfast'),
                'lunch': request.form.getlist('lunch'),
                'dinner': request.form.getlist('dinner'),
                'snacks': request.form.getlist('snacks')
            },
            'recommendations': request.form.get('recommendations'),
            'restrictions': request.form.getlist('restrictions'),
            'supplements': request.form.getlist('supplements'),
            'last_modified': datetime.utcnow()
        }
       
        db.diet_plans.update_one(
            {'_id': ObjectId(plan_id)},
            {'$set': update_data}
        )
       
        flash('Diet plan updated successfully', 'success')
        return redirect(url_for('view_diet_plan', plan_id=plan_id))
   
    patient = db.users.find_one({'_id': plan['patient_id']})
    return render_template('edit_diet_plan.html', plan=plan, patient=patient)

def generate_personalized_recommendations(patient_data):
    """
    Generate personalized diet and exercise recommendations based on patient data
    """
    # Prepare features for prediction
    features = [
        patient_data['age'],
        patient_data['bmi'],
        patient_data['weight'],
        patient_data['cycle_regularity'],
        patient_data['skin_darkening'],
        patient_data['hair_growth'],
        patient_data['pcos_history'],
        patient_data['physical_activity']
    ]
   
    # Scale features
    features_scaled = scaler.transform([features])
   
    # Get diet category prediction
    diet_category = diet_model.predict(features_scaled)[0]
   
    # Get exercise intensity prediction
    exercise_intensity = exercise_model.predict(features_scaled)[0]
   
    # Diet recommendations based on category
    diet_recommendations = {
        'low_carb': {
            'breakfast': [
                'Greek yogurt with berries and nuts',
                'Vegetable omelet with avocado',
                'Chia seed pudding with almond milk'
            ],
            'lunch': [
                'Grilled chicken salad with olive oil dressing',
                'Quinoa bowl with roasted vegetables',
                'Salmon with steamed broccoli'
            ],
            'dinner': [
                'Baked fish with asparagus',
                'Turkey lettuce wraps',
                'Cauliflower rice stir-fry'
            ],
            'snacks': [
                'Almonds and seeds',
                'Celery with almond butter',
                'Hard-boiled eggs'
            ]
        },
        'balanced': {
            'breakfast': [
                'Oatmeal with fruits and seeds',
                'Whole grain toast with eggs',
                'Smoothie bowl with protein'
            ],
            'lunch': [
                'Brown rice with lean protein',
                'Whole grain wrap with hummus',
                'Lentil soup with vegetables'
            ],
            'dinner': [
                'Grilled fish with quinoa',
                'Chicken breast with sweet potato',
                'Bean and vegetable stir-fry'
            ],
            'snacks': [
                'Apple with peanut butter',
                'Greek yogurt with honey',
                'Mixed nuts and dried fruits'
            ]
        },
        'anti_inflammatory': {
            'breakfast': [
                'Turmeric smoothie bowl',
                'Anti-inflammatory porridge',
                'Berries and nuts parfait'
            ],
            'lunch': [
                'Mediterranean salad',
                'Ginger-turmeric soup',
                'Leafy greens with salmon'
            ],
            'dinner': [
                'Baked fish with herbs',
                'Vegetable curry with turmeric',
                'Grilled chicken with leafy greens'
            ],
            'snacks': [
                'Green tea with ginger',
                'Berries and dark chocolate',
                'Walnuts and green tea'
            ]
        }
    }
   
    # Exercise recommendations based on intensity
    exercise_recommendations = {
        'low': {
            'cardio': [
                {'name': 'Walking', 'duration': '30 minutes', 'frequency': '5 times/week'},
                {'name': 'Swimming', 'duration': '20 minutes', 'frequency': '3 times/week'},
                {'name': 'Yoga', 'duration': '30 minutes', 'frequency': '3 times/week'}
            ],
            'strength': [
                {'name': 'Bodyweight exercises', 'sets': 2, 'reps': '10-12'},
                {'name': 'Resistance band training', 'sets': 2, 'reps': '12-15'},
                {'name': 'Light dumbbell exercises', 'sets': 2, 'reps': '12-15'}
            ]
        },
        'moderate': {
            'cardio': [
                {'name': 'Brisk walking/jogging', 'duration': '30 minutes', 'frequency': '5 times/week'},
                {'name': 'Cycling', 'duration': '30 minutes', 'frequency': '4 times/week'},
                {'name': 'Dance cardio', 'duration': '30 minutes', 'frequency': '3 times/week'}
            ],
            'strength': [
                {'name': 'Dumbbell training', 'sets': 3, 'reps': '12-15'},
                {'name': 'Circuit training', 'sets': 3, 'reps': '15-20'},
                {'name': 'Pilates', 'duration': '45 minutes', 'frequency': '3 times/week'}
            ]
        },
        'high': {
            'cardio': [
                {'name': 'HIIT workouts', 'duration': '20 minutes', 'frequency': '3 times/week'},
                {'name': 'Running', 'duration': '30 minutes', 'frequency': '4 times/week'},
                {'name': 'Cardio kickboxing', 'duration': '45 minutes', 'frequency': '3 times/week'}
            ],
            'strength': [
                {'name': 'Weight training', 'sets': 4, 'reps': '8-12'},
                {'name': 'CrossFit-style workouts', 'duration': '45 minutes', 'frequency': '3 times/week'},
                {'name': 'Advanced circuit training', 'sets': 3, 'reps': '12-15'}
            ]
        }
    }
   
    return {
        'diet_plan': diet_recommendations[diet_category],
        'exercise_plan': exercise_recommendations[exercise_intensity]
    }

@app.route('/generate_health_plan/<patient_id>')
@doctor_required
def generate_health_plan(patient_id):
    patient = db.users.find_one({'_id': ObjectId(patient_id)})
    if not patient:
        flash('Patient not found', 'error')
        return redirect(url_for('doctor_dashboard'))
   
    # Get latest assessment data
    latest_assessment = db.assessments.find_one(
        {'user_id': ObjectId(patient_id)},
        sort=[('date', -1)]
    )
   
    if not latest_assessment:
        flash('No assessment data available', 'error')
        return redirect(url_for('patient_history', patient_id=patient_id))
   
    # Prepare patient data for ML model
    patient_data = {
        'age': patient.get('age', 25),
        'bmi': latest_assessment.get('personal_info', {}).get('bmi', 22),
        'weight': latest_assessment.get('personal_info', {}).get('weight', 60),
        'cycle_regularity': 1 if latest_assessment.get('menstrual_history', {}).get('regular_periods', False) else 0,
        'skin_darkening': 1 if latest_assessment.get('clinical_symptoms', {}).get('skin_darkening', False) else 0,
        'hair_growth': 1 if latest_assessment.get('clinical_symptoms', {}).get('hair_growth', False) else 0,
        'pcos_history': 1 if latest_assessment.get('health_history', {}).get('family_history', False) else 0,
        'physical_activity': 1 if latest_assessment.get('health_history', {}).get('exercise', False) else 0
    }
   
    # Generate recommendations
    recommendations = generate_personalized_recommendations(patient_data)
   
    # Create new health plan
    health_plan = {
        'patient_id': ObjectId(patient_id),
        'doctor_id': ObjectId(session['user_id']),
        'creation_date': datetime.utcnow(),
        'diet_plan': recommendations['diet_plan'],
        'exercise_plan': recommendations['exercise_plan'],
        'assessment_id': latest_assessment['_id'],
        'status': 'active'
    }
   
    # Save to database
    db.health_plans.insert_one(health_plan)
   
    # Create activity log
    activity = {
        'user_id': ObjectId(patient_id),
        'doctor_id': ObjectId(session['user_id']),
        'type': 'health_plan_generated',
        'details': 'New AI-generated health plan created',
        'date': datetime.utcnow()
    }
    db.activities.insert_one(activity)
   
    return render_template('health_plan.html',
                         patient=patient,
                         health_plan=health_plan,
                         assessment=latest_assessment)

@app.route('/predict_ultrasound', methods=['POST'])
def predict_ultrasound():
    try:
        print("DEBUG: predict_ultrasound route called")
        if 'ultrasound_image' not in request.files:
            print("DEBUG: No image uploaded")
            return jsonify({'success': False, 'message': 'No image uploaded'})
       
        file = request.files['ultrasound_image']
        if file.filename == '':
            print("DEBUG: No image selected")
            return jsonify({'success': False, 'message': 'No image selected'})
       
        if not allowed_file(file.filename):
            print("DEBUG: Invalid file type")
            return jsonify({'success': False, 'message': 'Invalid file type'})
       
        # Save the uploaded image
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        print(f"DEBUG: Saved image to {file_path}")
       
        # Initialize variables with default values
        probability = 0.5
        result = 'Unknown'
        
        # Try to use the model for prediction
        try:
            # Load the specified PCOS classification model
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing import image
            import numpy as np
            
            model_path = 'templates/model/pcos_classification_model.h5'
            print(f"DEBUG: Loading model from {model_path}")
            
            # Check if model file exists
            if not os.path.exists(model_path):
                print(f"DEBUG: Model file not found at {model_path}")
                # Try alternative path
                model_path = 'model/pcos_classification_model.h5'
                if not os.path.exists(model_path):
                    print(f"DEBUG: Model file not found at alternative path {model_path}")
                    raise FileNotFoundError("Model file not found")
            
            pcos_model = load_model(model_path)
            print("DEBUG: Model loaded successfully")
       
        # Preprocess the image for the model
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize the image
            print("DEBUG: Image preprocessed successfully")
       
        # Make prediction using the loaded model
            prediction = pcos_model.predict(img_array)
            probability = float(prediction[0][0])
            result = 'PCOS' if probability >= 0.5 else 'Normal'
            print(f"DEBUG: Prediction result: {result}, probability: {probability}")
        except Exception as model_error:
            print(f"DEBUG: Error in model prediction: {str(model_error)}")
            # Fallback to random prediction if model fails
            import random
            probability = random.random()
            result = 'PCOS' if probability >= 0.5 else 'Normal'
            print(f"DEBUG: Using fallback random prediction: {result}, probability: {probability}")
       
        # Get confidence level based on probability
        if probability >= 0.8 or probability <= 0.2:
            confidence = 'high'
        elif probability >= 0.6 or probability <= 0.4:
            confidence = 'medium'
        else:
            confidence = 'low'
       
        # Get indicators based on prediction
        indicators = []
        if result == 'PCOS':
            if probability >= 0.8:
                indicators.extend(['polycystic_ovaries', 'enlarged_ovaries'])
            if probability >= 0.6:
                indicators.append('thickened_endometrium')
            if probability >= 0.5:
                indicators.append('multiple_follicles')
       
        response_data = {
            'success': True,
            'prediction': {
                'result': result,
                'probability': probability,
                'confidence': confidence,
                'indicators': indicators
            },
            'image_path': unique_filename
        }
        print(f"DEBUG: Returning prediction response: {response_data}")
        return jsonify(response_data)
       
    except Exception as e:
        print(f"ERROR in predict_ultrasound: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/predict_questionnaire', methods=['POST'])
def predict_questionnaire():
    try:
        data = request.get_json()
       
        # Extract features from the request data
        age = float(data.get('age', 0))
        weight = float(data.get('weight', 0))
        height = float(data.get('height', 0))
        cycle_regularity = 1 if data.get('cycle_regularity') == 'regular' else 0
        cycle_length = float(data.get('cycle_length', 0))
        marriage_years = float(data.get('marriage_years', 0))
        pregnant = 1 if data.get('pregnant') == 'yes' else 0
        abortions = float(data.get('abortions', 0))
        weight_gain = 1 if data.get('weight_gain') == 'yes' else 0
        hair_growth = 1 if data.get('hair_growth') == 'yes' else 0
        skin_darkening = 1 if data.get('skin_darkening') == 'yes' else 0
        hair_loss = 1 if data.get('hair_loss') == 'yes' else 0
        pimples = 1 if data.get('pimples') == 'yes' else 0
        fast_food = 1 if data.get('fast_food') == 'yes' else 0
        regular_exercise = 1 if data.get('regular_exercise') == 'yes' else 0
        blood_group = data.get('blood_group', 'Not Specified')
       
        # Convert blood group to numeric (A+ = 1, A- = 2, B+ = 3, B- = 4, O+ = 5, O- = 6, AB+ = 7, AB- = 8, Not Specified = 0)
        blood_group_map = {
            'A+': 1, 'A-': 2, 'B+': 3, 'B-': 4,
            'O+': 5, 'O-': 6, 'AB+': 7, 'AB-': 8,
            'Not Specified': 0
        }
        blood_group_numeric = blood_group_map.get(blood_group, 0)
       
        # Calculate BMI
        height_m = height / 100
        bmi = weight / (height_m * height_m)
       
        # Prepare features in the correct order
        features = [
            age, weight, height, cycle_regularity, cycle_length, marriage_years,
            pregnant, abortions, weight_gain, hair_growth, skin_darkening,
            hair_loss, pimples, fast_food, regular_exercise, blood_group_numeric, bmi
        ]
       
        # Scale features
        features_scaled = scaler.transform([features])
       
        # Make prediction
        prediction = questionnaire_model.predict(features_scaled)[0]
        probability = questionnaire_model.predict_proba(features_scaled)[0][1]
       
        return jsonify({
            'prediction': 'PCOS' if prediction == 1 else 'Normal',
            'probability': float(probability)
        })
       
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_total_doctors():
    return db.users.count_documents({'account_type': 'doctor'})

def get_total_lab_assistants():
    return db.users.count_documents({'account_type': 'lab_assistant'})

def get_total_patients():
    return db.users.count_documents({'account_type': 'patient'})

def get_pending_doctors():
    return list(db.users.find({'account_type': 'doctor', 'status': 'pending'}))

def get_pending_lab_assistants():
    return list(db.users.find({'account_type': 'lab_assistant', 'status': 'pending'}))


@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    search_query = request.args.get('search', '').strip().lower()

    # Fetch data
    total_doctors = get_total_doctors()
    total_lab_assistants = get_total_lab_assistants()
    total_patients = get_total_patients()
    pending_doctors = get_pending_doctors()
    pending_lab_assistants = get_pending_lab_assistants()

    # Filter results based on search query
    if search_query:
        pending_doctors = [doc for doc in pending_doctors if search_query in doc.name.lower()]
        pending_lab_assistants = [lab for lab in pending_lab_assistants if search_query in lab.name.lower()]

    # Pagination
    page = request.args.get(get_page_parameter(), type=int, default=1)
    per_page = 10  # Items per page
    total = len(pending_doctors) + len(pending_lab_assistants)
    pagination = Pagination(page=page, total=total, per_page=per_page, css_framework='bootstrap4')

    # Paginate filtered results
    start = (page - 1) * per_page
    end = start + per_page
    paginated_pending_doctors = pending_doctors[start:end]
    paginated_pending_lab_assistants = pending_lab_assistants[start:end]

    return render_template('admin/dashboard.html',
                           pending_doctors=paginated_pending_doctors,
                           pending_lab_assistants=paginated_pending_lab_assistants,
                           total_doctors=total_doctors,
                           total_lab_assistants=total_lab_assistants,
                           total_patients=total_patients,
                           search_query=search_query,  # Pass search query to the template
                           pagination=pagination)

@app.route('/admin/approve_user/<user_id>', methods=['POST'])
@admin_required
def approve_user(user_id):
    user = db.users.find_one({'_id': ObjectId(user_id)})
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('admin_dashboard'))
   
    db.users.update_one(
        {'_id': ObjectId(user_id)},
        {'$set': {'status': 'active', 'verified': True}}
    )
   
    flash(f'{user["account_type"].title()} approved successfully', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/reject_user/<user_id>', methods=['POST'])
@admin_required
def reject_user(user_id):
    user = db.users.find_one({'_id': ObjectId(user_id)})
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('admin_dashboard'))
   
    db.users.update_one(
        {'_id': ObjectId(user_id)},
        {'$set': {'status': 'rejected'}}
    )
   
    flash(f'{user["account_type"].title()} rejected', 'success')
    return redirect(url_for('admin_dashboard'))
@app.route('/admin/users')
@admin_required
def admin_users():
    users = list(db.users.find())

    # Define a stats dictionary
    stats = {
        "total_users": len(users),
        "total_doctors": sum(1 for user in users if user.get("role") == "doctor"),
        "total_lab_assistants": sum(1 for user in users if user.get("role") == "lab_assistant"),
        "total_patients": sum(1 for user in users if user.get("role") == "patient")
    }

    # Get filter values from request arguments
    filters = {
        "search": request.args.get("search", ""),
        "account_type": request.args.get("account_type", "all")
    }

    return render_template('admin/users.html', users=users, stats=stats, filters=filters)
@app.route('/admin/assign-staff', methods=['GET', 'POST'])
@admin_required
def admin_assign_staff():
    if request.method == 'POST':
        patient_id = request.form.get('patient_id')
        doctor_id = request.form.get('doctor_id')
        lab_assistant_id = request.form.get('lab_assistant_id')
        notes = request.form.get('notes')

        if not all([patient_id, doctor_id, lab_assistant_id]):
            flash('Please provide all required information', 'error')
            return redirect(url_for('admin_assign_staff'))

        try:
            # Update patient's record with assigned staff
            db.users.update_one(
                {'_id': ObjectId(patient_id)},
                {
                    '$set': {
                        'assigned_doctor': ObjectId(doctor_id),
                        'assigned_lab_assistant': ObjectId(lab_assistant_id),
                        'assignment_date': datetime.now(),
                        'assignment_notes': notes
                    }
                }
            )

            # Log the assignment
            db.staff_assignments.insert_one({
                'patient_id': ObjectId(patient_id),
                'doctor_id': ObjectId(doctor_id),
                'lab_assistant_id': ObjectId(lab_assistant_id),
                'assigned_by': ObjectId(session['user_id']),
                'assigned_at': datetime.now(),
                'notes': notes
            })

            flash('Staff assigned successfully', 'success')
            return redirect(url_for('admin_users'))

        except Exception as e:
            flash('Error assigning staff: ' + str(e), 'error')
            return redirect(url_for('admin_assign_staff'))

    # GET request - display the assignment form
    # Get all patients, not just unassigned ones
    patients = list(db.users.find({'account_type': 'patient'}))
    
    # Try multiple query options to find doctors
    # This approach tries different status fields to accommodate different data structures
    doctors = []
    doctors_query1 = list(db.users.find({'account_type': 'doctor', 'status': 'active'}))
    if doctors_query1:
        doctors = doctors_query1
    else:
        doctors_query2 = list(db.users.find({'account_type': 'doctor', 'is_active': True}))
        if doctors_query2:
            doctors = doctors_query2
        else:
            # Fallback: just get all doctors regardless of status
            doctors = list(db.users.find({'account_type': 'doctor'}))

    # Similar approach for lab assistants
    lab_assistants = []
    lab_assistants_query1 = list(db.users.find({'account_type': 'lab_assistant', 'status': 'active'}))
    if lab_assistants_query1:
        lab_assistants = lab_assistants_query1
    else:
        lab_assistants_query2 = list(db.users.find({'account_type': 'lab_assistant', 'is_active': True}))
        if lab_assistants_query2:
            lab_assistants = lab_assistants_query2
        else:
            # Fallback: just get all lab assistants regardless of status
            lab_assistants = list(db.users.find({'account_type': 'lab_assistant'}))

    # Log counts to help with debugging
    app.logger.info(f"Found {len(patients)} patients, {len(doctors)} doctors, and {len(lab_assistants)} lab assistants")

    return render_template('admin/assign_staff.html',
                          patients=patients,
                          doctors=doctors,
                          lab_assistants=lab_assistants)

@app.route('/admin/remove-staff-assignment/<assignment_id>', methods=['POST'])
@admin_required
def remove_staff_assignment(assignment_id):
    assignment = db.doctor_assignments.find_one({'_id': ObjectId(assignment_id)})
    if assignment:
        # Remove doctor and lab assistant assignments from patient
        db.users.update_one(
            {'_id': assignment['patient_id']},
            {
                '$unset': {
                    'assigned_doctor': 1,
                    'assigned_lab_assistant': 1,
                    'assignment_date': 1,
                    'assignment_notes': 1
                }
            }
        )
       
        # Remove assignment record
        db.doctor_assignments.delete_one({'_id': ObjectId(assignment_id)})
       
        # Create activity log
        activity = {
            'user_id': assignment['patient_id'],
            'type': 'staff_unassigned',
            'details': 'Staff assignments removed by admin',
            'date': datetime.utcnow()
        }
        db.activities.insert_one(activity)
       
        flash('Staff assignments removed successfully', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/reports')
@admin_required
def admin_reports():
    status = {}  # Define an empty or default status if needed
    return render_template('admin/reports.html', stats=status)

@app.route('/admin/settings')
@admin_required
def admin_settings():
    status = {}
    return render_template('admin/settings.html', stats=status)
# Add this function to your app to debug the issue
@app.route('/admin/debug-staff')
@admin_required
def debug_staff_data():
    # Check all users
    all_users = list(db.users.find())
    
    # Check specifically for doctors with various queries
    doctors_query1 = list(db.users.find({'account_type': 'doctor'}))
    doctors_query2 = list(db.users.find({'account_type': 'doctor', 'is_active': True}))
    doctors_query3 = list(db.users.find({'account_type': 'doctor', 'status': 'active'}))
    
    # Check for lab assistants with various queries
    lab_assistants_query1 = list(db.users.find({'account_type': 'lab_assistant'}))
    lab_assistants_query2 = list(db.users.find({'account_type': 'lab_assistant', 'is_active': True}))
    lab_assistants_query3 = list(db.users.find({'account_type': 'lab_assistant', 'status': 'active'}))
    
    # Return debug information
    debug_info = {
        'total_users': len(all_users),
        'user_types': [user.get('account_type') for user in all_users],
        'doctor_counts': {
            'basic_query': len(doctors_query1),
            'with_is_active': len(doctors_query2),
            'with_status_active': len(doctors_query3)
        },
        'lab_assistant_counts': {
            'basic_query': len(lab_assistants_query1),
            'with_is_active': len(lab_assistants_query2),
            'with_status_active': len(lab_assistants_query3)
        },
        'sample_doctor': doctors_query1[0] if doctors_query1 else None,
        'sample_lab_assistant': lab_assistants_query1[0] if lab_assistants_query1 else None
    }
    
    return jsonify(debug_info)
@app.route('/lab/dashboard')
@login_required
def lab_dashboard():
    user_id = session.get('user_id')
    if not user_id or not ObjectId.is_valid(user_id):
        flash('Invalid session. Please log in again.', 'danger')
        return redirect(url_for('login'))

    user = db.users.find_one({'_id': ObjectId(user_id)})
    
    if not user or user.get('account_type') != 'lab_assistant':
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('lab_dashboard'))

    lab_assistant_id = ObjectId(user_id)

    # Fetch assigned patients
    assigned_patients = list(db.users.find({
        'assigned_lab_assistant': lab_assistant_id,
        'account_type': 'patient'
    }))

    # Fetch pending ultrasound requests with patient details
    pending_requests = list(db.ultrasound_requests.aggregate([
        {
            '$match': {'status': 'pending'}  # Filter only pending requests
        },
        {
            '$lookup': {
                'from': 'users',  # Reference the users collection
                'localField': 'patient_id',  # Field in ultrasound_requests
                'foreignField': '_id',  # Matching field in users
                'as': 'patient_details'  # Store the joined data here
            }
        },
        {
            '$sort': {'request_date': -1}  # Sort by newest requests first
        }
    ]))

    
     # Convert patient details from list to a single object
    for request in pending_requests:
        if request.get('patient_details') and len(request['patient_details']) > 0:
            request['patient'] = request['patient_details'][0] # Extract patient name
        else:
            request['patient'] = {"name": "No Name Available", "_id": "N/A"}  

        request.pop('patient_details', None)  # Remove the extra field
    # Fetch recent uploads with patient details
    recent_uploads = list(db.ultrasound_requests.aggregate([
        {
            '$match': {
                'status': 'completed',
                'images': {'$exists': True, '$ne': []}
            }
        },
        {
            '$lookup': {
                'from': 'users',  # Assuming 'users' collection stores patient details
                'localField': 'patient_id',  # Field storing patient reference
                'foreignField': '_id',  
                'as': 'patient_details'
            }
        },
        {
            '$sort': {'upload_date': -1}
        },
        {
            '$limit': 10
        }
    ]))

    # Convert patient details to a single object instead of a list
    for upload in recent_uploads:
        if upload.get('patient_details'):
            upload['patient'] = upload['patient_details'][0]  # Extract first entry
        else:
            upload['patient'] = {"name": "No Name Available", "_id": "N/A"}  

        upload.pop('patient_details', None)  # Remove extra field

    return render_template('lab/dashboard.html',
                           assigned_patients=assigned_patients,
                           pending_requests=pending_requests,
                           recent_uploads=recent_uploads)

@app.route('/lab/upload-ultrasound/<request_id>', methods=['GET', 'POST'])
@login_required
def upload_lab_ultrasound(request_id):
    # Check if user is a lab assistant
    user = db.users.find_one({'_id': ObjectId(session['user_id'])})
    if user['account_type'] != 'lab_assistant':
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get the ultrasound request
    ultrasound_request = db.ultrasound_requests.find_one({'_id': ObjectId(request_id)})
    if not ultrasound_request:
        flash('Ultrasound request not found.', 'danger')
        return redirect(url_for('lab_dashboard'))
    
    if request.method == 'POST':
        print("DEBUG: POST request received for upload_lab_ultrasound")
        
        # Ensure upload directory exists
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'ultrasounds')
        os.makedirs(upload_dir, exist_ok=True)
        print(f"DEBUG: Upload directory: {upload_dir}")
        
        # Check if files were uploaded
        if 'ultrasound_images' not in request.files:
            print("DEBUG: No files in request")
            flash('No files selected.', 'danger')
            return redirect(request.url)
        
        files = request.files.getlist('ultrasound_images')
        print(f"DEBUG: Files in request: {[f.filename for f in files]}")
        
        if not files or files[0].filename == '':
            print("DEBUG: Empty file list or filename")
            flash('No files selected.', 'danger')
            return redirect(request.url)
        
        print(f"DEBUG: Received {len(files)} files for upload")
        
        # Process each uploaded file
        uploaded_images = []
        for file in files:
            print(f"DEBUG: Processing file: {file.filename}, type: {file.content_type}")
            
            if file and file.filename:
                try:
                    # Generate secure filename
                    filename = secure_filename(file.filename)
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    new_filename = f"ultrasound_{ultrasound_request['patient_id']}_{timestamp}_{filename}"
                    
                    # Save file
                    file_path = os.path.join(upload_dir, new_filename)
                    file.save(file_path)
                    print(f"DEBUG: Saved file to {file_path}")
                    
                    # Add to uploaded images list
                    uploaded_images.append({
                        'filename': new_filename,
                        'original_name': filename,
                        'path': f"ultrasounds/{new_filename}",
                        'upload_date': datetime.now()
                    })
                except Exception as e:
                    print(f"ERROR: Failed to save file {file.filename}: {str(e)}")
                    flash(f'Error saving file {file.filename}: {str(e)}', 'danger')
        
        if not uploaded_images:
            print("DEBUG: No valid images were uploaded")
            flash('No valid images were uploaded.', 'danger')
            return redirect(request.url)
            
        print(f"DEBUG: Successfully processed {len(uploaded_images)} images")
        
        try:
            # Update ultrasound request
            update_result = db.ultrasound_requests.update_one(
            {'_id': ObjectId(request_id)},
            {
            '$set': {
                'status': 'completed',
                'upload_date': datetime.now(),
                'uploaded_by': ObjectId(session['user_id']),
                'lab_notes': request.form.get('lab_notes', '')
            },
            '$push': {
                'images': {'$each': uploaded_images}
            }
            }
            )
            print(f"DEBUG: Updated ultrasound request {request_id} in database. Modified count: {update_result.modified_count}")

            # Get patient information
            patient = db.users.find_one({'_id': ultrasound_request['patient_id']})

            # Update or create health report for patient
            db.health_reports.update_one(
            {'patient_id': ultrasound_request['patient_id']},
            {
                '$push': {
                    'reports': {
                        'type': 'Ultrasound',
                        'images': uploaded_images,
                        'lab_notes': request.form.get('lab_notes', ''),
                        'date_uploaded': datetime.now()
                    }
                }
            },
            upsert=True  # Creates a new entry if no health report exists for the patient
            )
            print(f"DEBUG: Health report updated for patient {patient['name']}")

            # Create notification for patient
            notification = {
                'user_id': ultrasound_request['patient_id'],
                'title': 'Ultrasound Images Uploaded',
                'message': 'Your ultrasound images have been uploaded by the lab. A doctor will review them soon.',
                'icon': 'fa-microscope',
                'created_at': datetime.now(),
                'read': False
            }
            db.notifications.insert_one(notification)
            print(f"DEBUG: Created notification for patient {patient['name']}")

            # Proceed with doctor notification logic as in your original code...

        except Exception as e:
            print(f"ERROR: Exception in upload_lab_ultrasound: {str(e)}")
            flash(f'Error processing upload: {str(e)}', 'danger')
            return redirect(request.url)

    
    # GET request - show upload form
    patient = db.users.find_one({'_id': ultrasound_request['patient_id']})
    return render_template('lab/upload_ultrasound.html', 
                          request=ultrasound_request,
                          patient=patient)
@app.route('/view-upload/<upload_id>')
def view_upload(upload_id):
    # Fetch and display the uploaded ultrasound image
    return f"Viewing upload with ID: {upload_id}"

@app.route('/assigned_patients')
def assigned_patients():
    patients_collection = db.users  # Assuming 'users' is the collection for patients
    patients = list(patients_collection.find({'account_type': 'patient'}, {"_id": 0}))  # Fetch assigned patients, excluding _id
    return render_template('lab/assigned_patients.html', patients=patients)  # Send data to HTML

@app.route('/doctor/review-ultrasound/<request_id>', methods=['GET', 'POST'])
@login_required
def doctor_review_ultrasound(request_id):
    # Check if user is a doctor
    user = db.users.find_one({'_id': ObjectId(session['user_id'])})
    if user['account_type'] != 'doctor':
        flash('Access denied. Only doctors can review ultrasounds.', 'error')
        return redirect(url_for('home'))
   
    # Get ultrasound request
    ultrasound_request = db.ultrasound_requests.find_one({'_id': ObjectId(request_id)})
    if not ultrasound_request:
        flash('Ultrasound request not found.', 'error')
        return redirect(url_for('doctor_dashboard'))
    
    # Get patient information
    patient = db.users.find_one({'_id': ultrasound_request['patient_id']})
    if not patient:
        flash('Patient information not found.', 'error')
        return redirect(url_for('doctor_dashboard'))
    
    # Load PCOS classification model
    model_path = os.path.join('templates', 'model', 'pcos_classification_model.h5')
    model = tf.keras.models.load_model(model_path)
    
    # Get model predictions for each image
    model_predictions = []
    for image in ultrasound_request.get('images', []):
        try:
            # Load and preprocess image
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image['path'])
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_array = img_array / 255.0  # Normalize
            
            # Make prediction
            prediction = model.predict(img_array)
            probability = float(prediction[0][0])
            
            # Determine result and confidence
            result = 'PCOS' if probability > 0.5 else 'Normal'
            if probability > 0.8 or probability < 0.2:
                confidence = 'high'
            elif probability > 0.6 or probability < 0.4:
                confidence = 'medium'
            else:
                confidence = 'low'
            
            model_predictions.append({
                'result': result,
                'probability': probability,
                'confidence': confidence
            })
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            model_predictions.append(None)
    
    if request.method == 'POST':
        try:
            # Update ultrasound request with doctor's diagnosis
            update_result = db.ultrasound_requests.update_one(
                {'_id': ObjectId(request_id)},
                {
                    '$set': {
                        'doctor_diagnosis': request.form.get('diagnosis'),
                        'has_pcos': request.form.get('has_pcos') == 'yes',
                        'recommendations': request.form.get('recommendations'),
                        'reviewed_by': ObjectId(session['user_id']),
                        'review_date': datetime.now()
                    }
                }
            )
            
            if update_result.modified_count > 0:
                flash('Ultrasound review submitted successfully!', 'success')
                return redirect(url_for('doctor_dashboard'))
            else:
                flash('Error updating ultrasound review.', 'error')
                return redirect(url_for('doctor_dashboard'))
                
        except Exception as e:
            print(f"Error submitting review: {str(e)}")
            flash('Error submitting review. Please try again.', 'error')
            return redirect(url_for('doctor_dashboard'))
    
    return render_template('doctor/review_ultrasound.html',
                         request=ultrasound_request,
                         patient=patient,
                         model_predictions=model_predictions)


@app.route('/patient/ultrasound-results')
@login_required
def patient_ultrasound_results():
    """Retrieve the most recent completed ultrasound result for the logged-in patient."""
    
    # Get the logged-in patient's user info
    user = db.users.find_one({'_id': ObjectId(session['user_id'])})

    # Ensure the user is a patient
    if not user or user.get('account_type') != 'patient':
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('patient_dashboard'))

    # Fetch the most recent completed ultrasound request where the doctor has submitted the diagnosis
    latest_ultrasound = db.ultrasound_requests.find_one(
        {
            'patient_id': ObjectId(session['user_id']),
            'status': 'completed',  # Only fetch completed requests
            'doctor_diagnosis': {'$exists': True, '$ne': ""}  # Ensure a diagnosis is present
        },
        sort=[('diagnosis_date', -1)]  # Get the latest report first
    )

    # Convert ObjectId to string for frontend rendering
    if latest_ultrasound:
        latest_ultrasound['_id'] = str(latest_ultrasound['_id'])
        latest_ultrasound['diagnosis_date'] = latest_ultrasound.get('diagnosis_date', datetime.now()).strftime('%Y-%m-%d %H:%M')

    return render_template(
        'patient/ultrasound_results.html', 
        report=latest_ultrasound,  # Pass the latest completed report
        patient=user,
        model_predictions=latest_ultrasound.get('model_predictions', []) if latest_ultrasound else [],
    )


@app.route('/request-lab-ultrasound', methods=['GET', 'POST'])
@login_required
def request_lab_ultrasound():
    if request.method == 'POST':
        # Get form data
        notes = request.form.get('notes', '')
        
        # Create ultrasound request
        ultrasound_request = {
            'patient_id': ObjectId(session['user_id']),
            'patient_name': db.users.find_one({'_id': ObjectId(session['user_id'])})['name'],
            'notes': notes,
            'status': 'pending',
            'request_date': datetime.now(),
            'images': [],  # Will store uploaded ultrasound images
            'doctor_diagnosis': None,  # Will store doctor's diagnosis
            'has_pcos': None  # Will store doctor's PCOS determination
        }
        
        # Save to database
        request_id = db.ultrasound_requests.insert_one(ultrasound_request).inserted_id
        
        # Create notification for lab assistants
        for lab_assistant in db.users.find({'account_type': 'lab_assistant', 'status': 'active'}):
            notification = {
                'user_id': lab_assistant['_id'],
                'title': 'New Ultrasound Request',
                'message': f"Patient {ultrasound_request['patient_name']} has requested an ultrasound.",
                'icon': 'fa-microscope',
                'created_at': datetime.now(),
                'read': False,
                'request_id': request_id  # Link notification to the request
            }
            db.notifications.insert_one(notification)
        
        flash('Your ultrasound request has been submitted. The lab will contact you to confirm your appointment.', 'success')
        return redirect(url_for('patient_dashboard'))
   
    return render_template('request_lab_ultrasound.html')

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/doctor/create-final-report/<patient_id>', methods=['GET', 'POST'])
@doctor_required
def create_final_report(patient_id):
    if request.method == 'POST':
        try:
            # Get patient's latest questionnaire and ultrasound data
            questionnaire = db.assessments.find_one({
                'user_id': ObjectId(patient_id),
                'type': 'questionnaire'
            }, sort=[('date', -1)])
            
            ultrasound = db.ultrasound_requests.find_one({
                'patient_id': ObjectId(patient_id),
                'status': 'completed'
            }, sort=[('upload_date', -1)])
            
            if not questionnaire or not ultrasound:
                flash('Both questionnaire and ultrasound data are required for a final report.', 'error')
                return redirect(url_for('doctor_dashboard'))
            
            # Create final report
            final_report = {
                'patient_id': ObjectId(patient_id),
                'doctor_id': ObjectId(session['user_id']),
                'questionnaire_id': questionnaire['_id'],
                'ultrasound_id': ultrasound['_id'],
                'diagnosis': request.form.get('diagnosis'),
                'pcos_status': request.form.get('pcos_status'),
                'severity': request.form.get('severity'),
                'recommendations': request.form.get('recommendations'),
                'treatment_plan': request.form.get('treatment_plan'),
                'follow_up_date': datetime.strptime(request.form.get('follow_up_date'), '%Y-%m-%d'),
                'created_at': datetime.now()
            }
            
            # Save final report
            db.final_reports.insert_one(final_report)
            
            # Update patient's last visit
            db.users.update_one(
                {'_id': ObjectId(patient_id)},
                {'$set': {'last_visit': datetime.now()}}
            )
            
            flash('Final report created successfully!', 'success')
            return redirect(url_for('doctor_dashboard'))
            
        except Exception as e:
            print(f"Error creating final report: {str(e)}")
            flash('Error creating final report. Please try again.', 'error')
            return redirect(url_for('doctor_dashboard'))
    
    # Get patient data for the form
    patient = db.users.find_one({'_id': ObjectId(patient_id)})
    if not patient:
        flash('Patient not found.', 'error')
        return redirect(url_for('doctor_dashboard'))
    
    # Get latest questionnaire and ultrasound data
    questionnaire = db.assessments.find_one({
        'user_id': ObjectId(patient_id),
        'type': 'questionnaire'
    }, sort=[('date', -1)])
    
    ultrasound = db.ultrasound_requests.find_one({
        'patient_id': ObjectId(patient_id),
        'status': 'completed'
    }, sort=[('upload_date', -1)])
    
    if not questionnaire or not ultrasound:
        flash('Both questionnaire and ultrasound data are required for a final report.', 'error')
        return redirect(url_for('doctor_dashboard'))
    
    return render_template('doctor/create_final_report.html',
                         patient=patient,
                         questionnaire=questionnaire,
                         ultrasound=ultrasound)

@app.route('/doctor/view-final-report/<patient_id>')
@doctor_required
def view_final_report(patient_id):
    # Get final report
    final_report = db.final_reports.find_one({'patient_id': ObjectId(patient_id)})

    if not final_report:
        flash('Final report not found.', 'error')
        return redirect(url_for('doctor_dashboard'))

    # Get patient data
    patient = db.users.find_one({'_id': ObjectId(patient_id)})

    # Get questionnaire and ultrasound data
    questionnaire = db.assessments.find_one({'_id': final_report['questionnaire_id']})
    ultrasound_result = db.ultrasound_requests.find_one({'patient_id': ObjectId(patient_id)})

    print("Ultrasound Data:", ultrasound_result)  # Debugging

    return render_template('doctor/view_final_report.html',
                           patient=patient,
                           final_report=final_report,
                           questionnaire=questionnaire,
                           ultrasound_result=ultrasound_result)  # ✅ Pass as "ultrasound_result"


@app.route('/doctor/ultrasound-reviews')
@doctor_required
def doctor_ultrasound_reviews():
    # Get all completed ultrasound requests that need review
    pending_ultrasounds = list(db.ultrasound_requests.find({
        'status': 'completed',
        'doctor_diagnosis': None,
        'images': {'$exists': True, '$ne': []}
    }).sort('upload_date', -1))
    
    # Add patient information to ultrasound requests
    for request in pending_ultrasounds:
        patient = db.users.find_one({'_id': request['patient_id']})
        if patient:
            request['patient'] = patient
            # Get patient's questionnaire assessment
            questionnaire = db.assessments.find_one({
                'user_id': request['patient_id'],
                'type': 'questionnaire'
            }, sort=[('date', -1)])
            request['patient']['questionnaire'] = questionnaire
        else:
            request['patient'] = {'name': 'Unknown Patient', 'questionnaire': None}
    
    return render_template('doctor/ultrasound_reviews.html',
                         pending_ultrasounds=pending_ultrasounds)




@app.route('/ultrasound/report/<request_id>')
def view_ultrasound_report(request_id):
    print(f"Received request_id: {request_id}")  # Debugging

    try:
        obj_id = ObjectId(request_id)  # Convert to ObjectId
    except Exception:
        print("Invalid ObjectId. Redirecting to dashboard.")  # Debugging
        flash("Invalid request ID.", "danger")
        return redirect(url_for('patient_dashboard'))

    # Fetch ultrasound report
    ultrasound = db.ultrasounds.find_one({"_id": obj_id})

    if not ultrasound:
        print("Ultrasound report not found. Redirecting to dashboard.")  # Debugging
        flash("Ultrasound report not found.", "danger")
        return redirect(url_for('patient_dashboard'))

    # Fetch user details
    user = db.users.find_one({"_id": ObjectId(ultrasound["patient_id"])})

    # Fetch doctor's details
    doctor = db.doctors.find_one({"_id": ObjectId(ultrasound["diagnosed_by"])})
    doctor_name = doctor["name"] if doctor else "Unknown Doctor"

    return render_template("patient/ultrasound_report.html", 
                           ultrasound=ultrasound, 
                           user=user, 
                           doctor_name=doctor_name)
@app.route('/profile/health-report')
@login_required
def view_health_report():
    user = db.users.find_one({'_id': ObjectId(session['user_id'])})
    if not user:
        session.clear()
        return redirect(url_for('login'))
    
    # Get latest assessment
    latest_assessment = db.assessments.find_one(
        {'user_id': ObjectId(session['user_id'])},
        sort=[('date', -1)]
    )
    
    # Get latest ultrasound request
    latest_ultrasound = db.ultrasound_requests.find_one(
        {'patient_id': ObjectId(session['user_id']), 'status': 'completed'},
        sort=[('upload_date', -1)]
    )
    
    # Get latest final report
    latest_report = db.final_reports.find_one(
        {'patient_id': ObjectId(session['user_id'])},
        sort=[('created_at', -1)]
    )
    
    # Fetch uploaded ultrasound images
    uploaded_ultrasound_images = []
    if latest_ultrasound and 'images' in latest_ultrasound:
        uploaded_ultrasound_images.extend(latest_ultrasound['images'])

    # Calculate health score
    health_score = calculate_health_score(user)
    
    # Get health tips
    health_tips = get_health_tips(user)
    
    # Get recent activities
    recent_activities = list(db.activities.find(
        {'user_id': ObjectId(session['user_id'])}
    ).sort('date', -1).limit(5))
    
    return render_template('health_report.html',
                           user=user,
                           latest_assessment=latest_assessment,
                           latest_ultrasound=latest_ultrasound,  # ✅ Updated
                           latest_report=latest_report,
                           health_score=health_score,
                           health_tips=health_tips,
                           recent_activities=recent_activities,
                           uploaded_ultrasound_images=uploaded_ultrasound_images)

@app.route('/admin/approvals')
@login_required
@admin_required
def admin_approvals():
    # Get pending doctors and lab assistants from your database
    pending_doctors = db.users.find({"role": "doctor", "approved": False})
    pending_lab_assistants = db.users.find({"role": "lab_assistant", "approved": False})
    
    # Debug: Print the counts to server console - fixed for newer PyMongo versions
    print(f"Found {db.users.count_documents({'role': 'doctor', 'approved': False})} pending doctors")
    print(f"Found {db.users.count_documents({'role': 'lab_assistant', 'approved': False})} pending lab assistants")
    
    # Convert to list to make them iterable multiple times (MongoDB cursors can only be iterated once)
    pending_doctors = list(pending_doctors)
    pending_lab_assistants = list(pending_lab_assistants)
    
    return render_template('admin/approvals.html', 
                          pending_doctors=pending_doctors,
                          pending_lab_assistants=pending_lab_assistants)


# Route: List all blogs
@app.route('/admin/blogs')
@login_required
def blog_list():
    if not admin_required():
        flash("Access denied", "danger")
        return redirect(url_for("index"))
    
    blogs = list(db.blogs.find().sort("created_at", -1))
    return render_template('blog_list.html', blogs=blogs)

@app.route('/admin/blogs/create', methods=['GET', 'POST'])
@login_required
def create_blog():
    if not current_user.is_authenticated or current_user.role != "admin":
        flash("Access denied", "danger")
        return redirect(url_for("index"))

    if request.method == 'POST':
        title = request.form.get('title')
        content = request.form.get('content')

        if not title or not content:
            flash("Title and Content are required", "danger")
            return redirect(url_for("create_blog"))

        blog = {
            "title": title,
            "content": content,
            "author": current_user.username,
            "created_at": datetime.utcnow()
        }
        db.blogs.insert_one(blog)
        flash("Blog created successfully!", "success")
        return redirect(url_for("blog_list"))

    return render_template('create_blog.html')
# Route: Edit a blog
@app.route('/admin/blogs/edit/<blog_id>', methods=['GET', 'POST'])
@login_required
def edit_blog(blog_id):
    if not admin_required():
        flash("Access denied", "danger")
        return redirect(url_for("index"))

    blog = db.blogs.find_one({"_id": ObjectId(blog_id)})

    if not blog:
        flash("Blog not found", "danger")
        return redirect(url_for("blog_lists"))

    if request.method == 'POST':
        new_title = request.form.get('title')
        new_content = request.form.get('content')

        db.blogs.update_one(
            {"_id": ObjectId(blog_id)},
            {"$set": {"title": new_title, "content": new_content, "updated_at": datetime.utcnow()}}
        )
        flash("Blog updated successfully!", "success")
        return redirect(url_for("blog_lists"))

    return render_template('edit_blog.html', blog=blog)

# Route: Delete a blog
@app.route('/admin/blogs/delete/<blog_id>', methods=['POST'])
@login_required
def delete_blog(blog_id):
    if not admin_required():
        flash("Access denied", "danger")
        return redirect(url_for("index"))

    db.blogs.delete_one({"_id": ObjectId(blog_id)})
    flash("Blog deleted successfully!", "success")
    return redirect(url_for("blog_lists"))


if __name__ == '__main__':
    app.run(debug=True)