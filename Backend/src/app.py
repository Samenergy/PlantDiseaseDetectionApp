import os
import shutil
import zipfile
import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends  # Added Depends import
from fastapi.responses import JSONResponse
from typing import List
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    predicted_disease = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Retraining(Base):
    __tablename__ = "retrainings"
    id = Column(Integer, primary_key=True, index=True)
    num_classes = Column(Integer, nullable=False)
    training_accuracy = Column(Float)
    validation_accuracy = Column(Float, nullable=True)
    class_metrics = Column(Text)  # Store JSON string
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
MODEL_PATH = "../models/plant_disease_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define initial class names for plant diseases
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

UPLOAD_DIR = "Data"
VISUALIZATION_DIR = "visualizations"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def preprocess_image(img_bytes: bytes):
    """Preprocess image for prediction."""
    img = image.load_img(io.BytesIO(img_bytes), target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/predict")
async def predict(file: UploadFile = File(...), db: SessionLocal = Depends(get_db)):
    img_bytes = await file.read()
    img = preprocess_image(img_bytes)
    predictions = model.predict(img)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    disease = CLASS_NAMES[predicted_index]
    
    # Save prediction to database
    prediction = Prediction(predicted_disease=disease, confidence=float(confidence))
    db.add(prediction)
    db.commit()
    
    return JSONResponse(content={"predicted_disease": disease, "confidence": float(confidence)})

def extract_zip(zip_path, extract_to):
    """Extract ZIP files."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def save_visualizations(y_true, y_pred_classes, target_names):
    """Save classification report and confusion matrix as PNG files."""
    class_report = classification_report(y_true, y_pred_classes, target_names=target_names)
    
    plt.figure(figsize=(10, len(target_names) * 0.5 + 2))
    plt.text(0.01, 0.99, class_report, {'fontsize': 10}, fontfamily='monospace')
    plt.axis('off')
    plt.title("Classification Report")
    plt.savefig(os.path.join(VISUALIZATION_DIR, "classification_report.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(max(10, len(target_names)), max(10, len(target_names))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, "confusion_matrix.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()

@app.post("/retrain")
async def retrain(files: List[UploadFile] = File(...), 
                 learning_rate: float = 0.0001, 
                 epochs: int = 10,
                 db: SessionLocal = Depends(get_db)):
    global model, CLASS_NAMES
    
    new_data_dir = os.path.join(UPLOAD_DIR, "new_data")
    os.makedirs(new_data_dir, exist_ok=True)
    
    try:
        # 1. Process uploaded files
        image_paths = []
        extracted_dirs = []
        
        for file in files:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            if file.filename.endswith(".zip"):
                extract_dir = os.path.join(UPLOAD_DIR, f"extract_{os.path.splitext(file.filename)[0]}")
                os.makedirs(extract_dir, exist_ok=True)
                extract_zip(file_path, extract_dir)
                extracted_dirs.append(extract_dir)
                os.remove(file_path)
                
                print(f"Extracted to: {extract_dir}")
                print(f"Contents of extract_dir: {os.listdir(extract_dir)}")
                
                for subdir in ['train', 'val']:
                    subdir_path = os.path.join(extract_dir, subdir)
                    if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                        print(f"Processing {subdir}: {os.listdir(subdir_path)}")
                        for item in os.listdir(subdir_path):
                            item_path = os.path.join(subdir_path, item)
                            if os.path.isdir(item_path) and item not in ['__MACOSX']:
                                target_dir = os.path.join(new_data_dir, item)
                                os.makedirs(target_dir, exist_ok=True)
                                has_images = False
                                for img in os.listdir(item_path):
                                    if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                                        shutil.copy(os.path.join(item_path, img), os.path.join(target_dir, img))
                                        has_images = True
                                if not has_images:
                                    print(f"Skipping empty folder: {item}")
                                    shutil.rmtree(target_dir)
            else:
                image_paths.append(file_path)
        
        # 2. Process loose image files
        for img_path in image_paths:
            try:
                img_array = preprocess_image(open(img_path, 'rb').read())
                prediction = model.predict(img_array)
                label_index = np.argmax(prediction)
                label = CLASS_NAMES[label_index]
                
                label_dir = os.path.join(new_data_dir, label)
                os.makedirs(label_dir, exist_ok=True)
                img_filename = os.path.basename(img_path)
                shutil.copy(img_path, os.path.join(label_dir, img_filename))
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
        
        # 3. Filter classes with sufficient data
        class_counts = {}
        for class_dir in os.listdir(new_data_dir):
            class_path = os.path.join(new_data_dir, class_dir)
            if os.path.isdir(class_path):
                image_count = len([f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if image_count >= 2:
                    class_counts[class_dir] = image_count
                else:
                    print(f"Skipping class {class_dir} with insufficient samples ({image_count})")
                    shutil.rmtree(class_path)
        
        if not class_counts:
            return JSONResponse(content={
                "error": "No valid classes with sufficient data found",
                "details": "Each class must have at least 2 images"
            }, status_code=400)
        
        print(f"Class counts: {class_counts}")
        
        # 4. Create data generators
        target_names = list(class_counts.keys())
        all_classes = list(set(CLASS_NAMES + target_names))
        use_validation = all(count >= 4 for count in class_counts.values())
        
        if use_validation:
            data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1
            )
            train_generator = data_generator.flow_from_directory(
                new_data_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical',
                classes=all_classes,
                subset='training',
                shuffle=True
            )
            validation_generator = data_generator.flow_from_directory(
                new_data_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical',
                classes=all_classes,
                subset='validation',
                shuffle=False
            )
        else:
            data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1
            )
            train_generator = data_generator.flow_from_directory(
                new_data_dir,
                target_size=(128, 128),
                batch_size=32,
                class_mode='categorical',
                classes=all_classes,
                shuffle=True
            )
            validation_generator = None
        
        # 5. Create a new model for fine-tuning
        temp_model_path = os.path.join(os.path.dirname(MODEL_PATH), "temp_model.keras")
        model.save(temp_model_path)
        working_model = tf.keras.models.load_model(temp_model_path)
        
        num_layers = len(working_model.layers)
        freeze_until = int(num_layers * 0.98)
        for i, layer in enumerate(working_model.layers):
            layer.trainable = i >= freeze_until
        
        working_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 6. Add callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if use_validation else 'loss',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if use_validation else 'loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7
            )
        ]
        
        # 7. Train the model
        if use_validation:
            history = working_model.fit(
                train_generator,
                validation_data=validation_generator,
                epochs=epochs,
                callbacks=callbacks,
                steps_per_epoch=max(1, len(train_generator)),
                validation_steps=max(1, len(validation_generator))
            )
        else:
            history = working_model.fit(
                train_generator,
                epochs=epochs,
                callbacks=callbacks,
                steps_per_epoch=max(1, len(train_generator))
            )
        
        # 8. Generate classification report and predictions
        if use_validation:
            validation_generator.reset()
            y_pred = working_model.predict(validation_generator)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = validation_generator.classes
        else:
            train_generator.reset()
            y_pred = working_model.predict(train_generator)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = train_generator.classes
        
        class_report = classification_report(
            y_true,
            y_pred_classes,
            target_names=target_names,
            output_dict=True
        )
        
        # 9. Save visualizations
        save_visualizations(y_true, y_pred_classes, target_names)
        
        # 10. Save the fine-tuned model
        fine_tuned_model_path = os.path.join(os.path.dirname(MODEL_PATH), "plant_disease_model.keras")
        working_model.save(fine_tuned_model_path)
        model = tf.keras.models.load_model(fine_tuned_model_path)
        
        CLASS_NAMES = all_classes
        with open(os.path.join(os.path.dirname(MODEL_PATH), "class_names.json"), "w") as f:
            json.dump(CLASS_NAMES, f)
        
        # 11. Prepare metrics and save to database
        class_metrics = {}
        for class_name in target_names:
            if class_name in class_report:
                class_metrics[class_name] = {
                    "precision": float(class_report[class_name]['precision']),
                    "recall": float(class_report[class_name]['recall']),
                    "f1_score": float(class_report[class_name]['f1-score']),
                    "support": int(class_report[class_name]['support'])
                }
        
        new_classes_added = [cls for cls in target_names if cls not in CLASS_NAMES]
        
        training_accuracy = float(history.history['accuracy'][-1]) if 'accuracy' in history.history else None
        validation_accuracy = float(history.history['val_accuracy'][-1]) if use_validation and 'val_accuracy' in history.history else None
        
        # Save retraining data to database
        retraining = Retraining(
            num_classes=len(CLASS_NAMES),
            training_accuracy=training_accuracy,
            validation_accuracy=validation_accuracy,
            class_metrics=json.dumps(class_metrics)
        )
        db.add(retraining)
        db.commit()
        
        # 12. Prepare response
        response_content = {
            "message": "Model fine-tuning successful with preserved knowledge!",
            "num_classes": len(CLASS_NAMES),
            "new_classes_added": new_classes_added,
            "class_counts": class_counts,
            "training_accuracy": training_accuracy,
            "class_metrics": class_metrics,
            "fine_tuned_model_path": fine_tuned_model_path,
            "visualization_files": {
                "classification_report": os.path.join(VISUALIZATION_DIR, "classification_report.png"),
                "confusion_matrix": os.path.join(VISUALIZATION_DIR, "confusion_matrix.png")
            },
            "retraining_id": retraining.id
        }
        
        if use_validation:
            response_content["validation_accuracy"] = validation_accuracy
        
        return JSONResponse(content=response_content)
        
    except Exception as e:
        import traceback
        return JSONResponse(content={
            "error": str(e),
            "details": traceback.format_exc()
        }, status_code=500)
    
    finally:
        for extract_dir in extracted_dirs:
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
        if os.path.exists(new_data_dir):
            shutil.rmtree(new_data_dir)
        temp_model_path = os.path.join(os.path.dirname(MODEL_PATH), "temp_model.keras")
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Plant Disease Prediction API!"}

@app.get("/prediction_history")
async def get_prediction_history(db: SessionLocal = Depends(get_db)):
    predictions = db.query(Prediction).order_by(Prediction.timestamp.desc()).all()
    return [{"id": p.id, "disease": p.predicted_disease, "confidence": p.confidence, "timestamp": p.timestamp.isoformat()} 
            for p in predictions]

@app.get("/retraining_history")
async def get_retraining_history(db: SessionLocal = Depends(get_db)):
    retrainings = db.query(Retraining).order_by(Retraining.timestamp.desc()).all()
    return [{"id": r.id, "num_classes": r.num_classes, "training_accuracy": r.training_accuracy,
             "validation_accuracy": r.validation_accuracy, "class_metrics": json.loads(r.class_metrics),
             "timestamp": r.timestamp.isoformat()} 
            for r in retrainings]