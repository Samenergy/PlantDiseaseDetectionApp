import os
import shutil
import zipfile
import io
import json
import warnings
import asyncio
from datetime import datetime, timedelta
from typing import List
from bson import ObjectId

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
from pymongo import MongoClient

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_URL = DATABASE_URL.replace("mysql://", "mysql+mysqlconnector://", 1)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client["plant_disease_db"]
visualizations_collection = mongo_db["visualizations"]

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

warnings.filterwarnings("ignore", category=UserWarning, module='urllib3')

# SQLAlchemy Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    predictions = relationship("Prediction", back_populates="user")
    retrainings = relationship("Retraining", back_populates="user")

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    predicted_disease = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="predictions")

class Retraining(Base):
    __tablename__ = "retrainings"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    num_classes = Column(Integer, nullable=False)
    training_accuracy = Column(Float)
    validation_accuracy = Column(Float, nullable=True)
    class_metrics = Column(Text)  # Store JSON string
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = relationship("User", back_populates="retrainings")

Base.metadata.create_all(bind=engine)

# Define base directory and paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "Data")
MODEL_PATH = os.path.join(BASE_DIR, "../models/plant_disease_model.keras")  # Updated to .keras format

# Create directories upfront
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
print(f"Model path: {MODEL_PATH}")
print(f"Does the model file exist? {os.path.exists(MODEL_PATH)}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
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

# Pydantic models for request/response
class UserCreate(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# WebSocket manager
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

ws_manager = WebSocketManager()

# Utility functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user(db, username=username)
    if user is None:
        raise credentials_exception
    
    return user

def preprocess_image(img_path: str):
    """Preprocess image for prediction."""
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Authentication endpoints
@app.post("/signup", response_model=Token)
async def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user(db, username=user.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    new_user = User(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user(db, username=form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# WebSocket endpoint
@app.websocket("/ws/retrain-progress")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection alive
    except Exception as e:
        print(f"WebSocket disconnected: {e}")
        ws_manager.disconnect(websocket)

# Protected endpoints
@app.post("/predict")
async def predict(file: UploadFile = File(...),
                 db: Session = Depends(get_db),
                 current_user: User = Depends(get_current_user)):
    img_bytes = await file.read()
    img = image.load_img(io.BytesIO(img_bytes), target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    disease = CLASS_NAMES[predicted_index]
    
    prediction = Prediction(
        user_id=current_user.id,
        predicted_disease=disease,
        confidence=float(confidence)
    )
    db.add(prediction)
    db.commit()
    
    return JSONResponse(content={"prediction": disease, "confidence": float(confidence)})

def extract_zip(zip_path, extract_to):
    """Extract ZIP files."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def save_visualizations_to_mongo(y_true, y_pred_classes, target_names, class_indices, history=None):
    """Save visualizations to MongoDB and return their IDs."""
    viz_ids = {}

    # 1. Classification Report
    class_report = classification_report(y_true, y_pred_classes, target_names=target_names, labels=class_indices, output_dict=True, zero_division=0)
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    rows = []
    for cls in target_names:
        if cls in class_report:
            rows.append([
                cls,
                f"{class_report[cls]['precision']:.2f}",
                f"{class_report[cls]['recall']:.2f}",
                f"{class_report[cls]['f1-score']:.2f}",
                f"{class_report[cls]['support']}"
            ])
    total_support = sum(class_report[cls]['support'] for cls in target_names if cls in class_report)
    correct = sum(1 for true, pred in zip(y_true, y_pred_classes) if true == pred and true in class_indices)
    accuracy = correct / total_support if total_support > 0 else 0.0
    rows.append(["Accuracy", "", "", f"{accuracy:.2f}", f"{total_support}"])

    fig, ax = plt.subplots(figsize=(12, len(target_names) * 0.6 + 2))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center',
                     colColours=['#4CAF50'] * len(headers), colWidths=[0.4, 0.15, 0.15, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4CAF50')
        else:
            cell.set_text_props(color='black')
            cell.set_facecolor('#F5F5F5' if row % 2 == 0 else '#FFFFFF')
        cell.set_edgecolor('#D3D3D3')
    plt.title("Classification Report", fontsize=18, weight='bold', pad=20, color='#333333')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    viz_ids["classification_report"] = str(visualizations_collection.insert_one({"image": buf.read(), "type": "classification_report"}).inserted_id)
    plt.close()
    buf.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes, labels=class_indices)
    plt.figure(figsize=(max(10, len(target_names)), max(10, len(target_names))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, cbar=True)
    plt.title("Confusion Matrix", fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    viz_ids["confusion_matrix"] = str(visualizations_collection.insert_one({"image": buf.read(), "type": "confusion_matrix"}).inserted_id)
    plt.close()
    buf.close()

    # 3. Training and Validation Loss
    if history and 'loss' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
        if 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
        plt.title('Training and Validation Loss', fontsize=16, pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        viz_ids["loss_plot"] = str(visualizations_collection.insert_one({"image": buf.read(), "type": "loss_plot"}).inserted_id)
        plt.close()
        buf.close()

    # 4. Training and Validation Accuracy
    if history and 'accuracy' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linewidth=2)
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linewidth=2)
        plt.title('Training and Validation Accuracy', fontsize=16, pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        buf.seek(0)
        viz_ids["accuracy_plot"] = str(visualizations_collection.insert_one({"image": buf.read(), "type": "accuracy_plot"}).inserted_id)
        plt.close()
        buf.close()

    return viz_ids

@app.post("/retrain")
async def retrain(files: List[UploadFile] = File(...),
                 learning_rate: float = 0.0001,
                 epochs: int = 10,
                 db: Session = Depends(get_db),
                 current_user: User = Depends(get_current_user)):
    global model, CLASS_NAMES
    
    new_data_dir = os.path.join(UPLOAD_DIR, "new_data")
    os.makedirs(new_data_dir, exist_ok=True)
    
    try:
        # 1. Process uploaded files
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
        
        await ws_manager.broadcast(json.dumps({"progress": 10}))  # Step 1 complete

        # 2. Organize data and classify images
        class_counts = {}
        initial_predictions = {"train": {}, "val": {}}
        
        for subdir in ['train', 'val']:
            subdir_path = os.path.join(extract_dir, subdir)
            if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                for class_name in os.listdir(subdir_path):
                    class_path = os.path.join(subdir_path, class_name)
                    if os.path.isdir(class_path) and class_name in CLASS_NAMES and class_name not in ['__MACOSX']:
                        target_dir = os.path.join(new_data_dir, subdir, class_name)
                        os.makedirs(target_dir, exist_ok=True)
                        image_count = 0
                        initial_predictions[subdir][class_name] = []
                        for img in os.listdir(class_path):
                            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                                img_path = os.path.join(class_path, img)
                                shutil.copy(img_path, os.path.join(target_dir, img))
                                image_count += 1
                                img_array = preprocess_image(img_path)
                                pred = model.predict(img_array)
                                pred_index = np.argmax(pred, axis=1)[0]
                                pred_confidence = float(np.max(pred))
                                initial_predictions[subdir][class_name].append({
                                    "image": img,
                                    "predicted_class": CLASS_NAMES[pred_index],
                                    "confidence": pred_confidence,
                                    "correct": CLASS_NAMES[pred_index] == class_name
                                })
                        if image_count >= 2:
                            class_counts[class_name] = class_counts.get(class_name, 0) + image_count
                        else:
                            shutil.rmtree(target_dir)
        
        if not class_counts:
            raise HTTPException(status_code=400, detail={
                "error": "No valid classes with sufficient data found",
                "details": "Each class must have at least 2 images"
            })
        
        target_names = list(class_counts.keys())
        class_indices = [CLASS_NAMES.index(cls) for cls in target_names]
        
        await ws_manager.broadcast(json.dumps({"progress": 30}))  # Step 2 complete

        # 3. Create data generators
        train_dir = os.path.join(new_data_dir, "train")
        val_dir = os.path.join(new_data_dir, "val")
        
        train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        ).flow_from_directory(
            train_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical',
            classes=CLASS_NAMES,
            shuffle=True
        )
        
        validation_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255
        ).flow_from_directory(
            val_dir,
            target_size=(128, 128),
            batch_size=32,
            class_mode='categorical',
            classes=CLASS_NAMES,
            shuffle=False
        )
        
        await ws_manager.broadcast(json.dumps({"progress": 40}))  # Step 3 complete

        # 4. Fine-tune the model
        temp_model_path = os.path.join(os.path.dirname(MODEL_PATH), "temp_model.keras")
        model.save(temp_model_path)
        working_model = tf.keras.models.load_model(temp_model_path)
        
        num_layers = len(working_model.layers)
        freeze_until = int(num_layers * 0.98)
        for i, layer in enumerate(working_model.layers):
            layer.trainable = i >= freeze_until
        
        working_model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate),  # Legacy optimizer for M1/M2 Macs
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, total_epochs):
                super().__init__()
                self.total_epochs = total_epochs

            async def broadcast_progress(self, progress):
                await ws_manager.broadcast(json.dumps({"progress": min(progress, 80)}))

            def on_epoch_end(self, epoch, logs=None):
                progress = 40 + ((epoch + 1) / self.total_epochs) * 40  # 40% to 80%
                asyncio.create_task(self.broadcast_progress(progress))  # Schedule async task
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7),
            ProgressCallback(epochs)
        ]
        
        history = working_model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            steps_per_epoch=max(1, len(train_generator)),
            validation_steps=max(1, len(validation_generator))
        )
        
        await ws_manager.broadcast(json.dumps({"progress": 80}))  # Training complete

        # 5. Evaluate on validation set
        validation_generator.reset()
        y_pred = working_model.predict(validation_generator)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = validation_generator.classes
        
        class_report = classification_report(
            y_true,
            y_pred_classes,
            target_names=target_names,
            labels=class_indices,
            output_dict=True,
            zero_division=0
        )
        
        # 6. Save visualizations
        viz_ids = save_visualizations_to_mongo(y_true, y_pred_classes, target_names, class_indices, history)
        
        await ws_manager.broadcast(json.dumps({"progress": 90}))  # Visualizations saved

        # 7. Save the fine-tuned model
        fine_tuned_model_path = os.path.join(os.path.dirname(MODEL_PATH), "plant_disease_model.keras")
        working_model.save(fine_tuned_model_path)
        model = tf.keras.models.load_model(fine_tuned_model_path)
        
        # 8. Prepare metrics and save to database
        class_metrics = {}
        for class_name in target_names:
            if class_name in class_report:
                class_metrics[class_name] = {
                    "precision": float(class_report[class_name].get('precision', 0.0)),
                    "recall": float(class_report[class_name].get('recall', 0.0)),
                    "f1_score": float(class_report[class_name].get('f1-score', 0.0)),
                    "support": int(class_report[class_name].get('support', 0))
                }
        
        training_accuracy = float(history.history['accuracy'][-1])
        validation_accuracy = float(history.history['val_accuracy'][-1])
        
        retraining = Retraining(
            user_id=current_user.id,
            num_classes=len(CLASS_NAMES),
            training_accuracy=training_accuracy,
            validation_accuracy=validation_accuracy,
            class_metrics=json.dumps(class_metrics)
        )
        db.add(retraining)
        db.commit()
        
        await ws_manager.broadcast(json.dumps({"progress": 100}))  # Complete

        # 9. Prepare response
        base_url = os.getenv("BASE_URL", "")
        visualization_files = {
            "classification_report": f"{base_url}/visualization/{viz_ids.get('classification_report')}",
            "confusion_matrix": f"{base_url}/visualization/{viz_ids.get('confusion_matrix')}",
            "loss_plot": f"{base_url}/visualization/{viz_ids.get('loss_plot')}" if "loss_plot" in viz_ids else None,
            "accuracy_plot": f"{base_url}/visualization/{viz_ids.get('accuracy_plot')}" if "accuracy_plot" in viz_ids else None
        }
        
        response_content = {
            "message": "Model fine-tuning successful!",
            "num_classes": len(CLASS_NAMES),
            "classes_in_zip": target_names,
            "class_counts": class_counts,
            "initial_predictions": initial_predictions,
            "training_accuracy": training_accuracy,
            "validation_accuracy": validation_accuracy,
            "class_metrics": class_metrics,
            "fine_tuned_model_path": fine_tuned_model_path,
            "visualization_files": visualization_files,
            "retraining_id": retraining.id,
            "user_id": current_user.id
        }
        
        return JSONResponse(content=response_content)
        
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        print(f"Error during retraining: {str(e)}")
        print(traceback.format_exc())
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
        if os.path.exists(temp_model_path):
            os.remove(temp_model_path)

@app.get("/visualization/{viz_id}")
async def get_visualization(viz_id: str):
    try:
        viz = visualizations_collection.find_one({"_id": ObjectId(viz_id)})
        if not viz or "image" not in viz:
            raise HTTPException(status_code=404, detail="Visualization not found")
        return StreamingResponse(io.BytesIO(viz["image"]), media_type="image/png")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid visualization ID")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Plant Disease Prediction API!"}

@app.get("/prediction_history")
async def get_prediction_history(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    predictions = db.query(Prediction).filter(Prediction.user_id == current_user.id).order_by(Prediction.timestamp.desc()).all()
    return [{"id": p.id, "text": f"Predicted disease: {p.predicted_disease}", "confidence": p.confidence, "date": p.timestamp.isoformat()}
            for p in predictions]

@app.get("/retraining_history")
async def get_retraining_history(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    retrainings = db.query(Retraining).filter(Retraining.user_id == current_user.id).order_by(Retraining.timestamp.desc()).all()
    return [{"id": r.id, "text": f"Retrained model with {r.num_classes} classes", "training_accuracy": r.training_accuracy,
             "validation_accuracy": r.validation_accuracy, "class_metrics": json.loads(r.class_metrics),
             "date": r.timestamp.isoformat()}
            for r in retrainings]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)