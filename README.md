# LeafSense: Plant Disease Detection App

![Plant Disease Detection](https://img.shields.io/badge/AI-Plant%20Disease%20Detection-brightgreen)
![React](https://img.shields.io/badge/Frontend-React-blue)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green)
![TensorFlow](https://img.shields.io/badge/ML-TensorFlow-orange)

LeafSense is an advanced plant disease detection application using deep learning to identify plant diseases from leaf images. It helps farmers, gardeners, and agricultural professionals detect plant diseases early and prevent crop losses.

## 🔗 Links

- **Frontend**: [https://leafsense.vercel.app](https://leafsense.vercel.app)
- **Backend API**: [https://appdeploy-production.up.railway.app](https://appdeploy-production.up.railway.app)
- **API Documentation**: [https://appdeploy-production.up.railway.app/docs](https://plant-disease-backend.onrender.com/docs)

## Features

- **Real-time Disease Detection**: Upload leaf images to instantly identify diseases
- **High Accuracy**: Powered by a CNN model trained on a large dataset of plant diseases
- **User Accounts**: Create accounts to track disease detection history
- **Custom Model Retraining**: Upload your own data to retrain the model
- **Visualization Tools**: View confusion matrices, accuracy graphs, and other metrics

## Supported Plants & Diseases

The model can detect diseases in 14 different plant species, including Apple, Blueberry, Cherry, Corn/Maize, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato with various disease conditions.

## Technology Stack

### Frontend
- React 19
- TypeScript
- TailwindCSS
- React Router

### Backend
- Python 3.9+
- FastAPI
- TensorFlow 2.15
- SQLAlchemy (MySQL)
- MongoDB (for visualizations)
- JWT Authentication

## Installation

### Clone the repository
```bash
git clone https://github.com/Samenergy/PlantDiseaseDetectionApp.git
cd PlantDiseaseDetectionApp
```

### Backend Setup
```bash
cd Backend
pip install -r requirements.txt

# Set up environment variables (.env file)
# DATABASE_URL=mysql://username:password@localhost/plant_disease_db
# SECRET_KEY=your_secret_key
# MONGO_URI=mongodb://localhost:27017

# Run the backend
cd src
uvicorn app:app --reload
```

### Frontend Setup
```bash
cd Frontend
npm install
npm run dev
```

## Testing

For testing the application without collecting your own plant images, you can use the sample images provided:

```bash
cd Backend/Data
```

This directory contains sample leaf images organized by plant type and disease condition that you can use to test the disease detection functionality.

## Usage

1. Create an account or login
2. Upload an image of a plant leaf
3. View the disease detection results and recommendations
4. Check your prediction history
5. (Optional) Upload a dataset to retrain the model for improved accuracy

## API Endpoints

- `POST /signup` - Create a new user account
- `POST /token` - Get authentication token
- `POST /predict` - Submit an image for disease prediction
- `POST /retrain` - Retrain the model with new data
- `GET /visualization/{viz_id}` - Get visualizations for training runs
- `GET /prediction_history` - Get user prediction history
- `GET /retraining_history` - Get user retraining history
- `WebSocket /ws/retrain-progress` - Real-time updates during retraining

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [PlantVillage Dataset](https://www.kaggle.com/code/imtkaggleteam/plant-diseases-detection-pytorch/input)
- TensorFlow and Keras for the ML framework
- FastAPI for the backend framework
- React Vite for the frontend framework
