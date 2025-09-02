import os
# Optimize TensorFlow loading
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for faster startup

import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Class names for the brain tumor classification
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Global variable to store the model
model = None

def load_model():
    """Load the brain tumor classification model"""
    global model
    try:
        model = tf.keras.models.load_model('brain_tumor_model.keras')
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def preprocess_image(image):
    """
    Preprocess the uploaded image for model prediction
    
    Args:
        image: PIL Image object
    
    Returns:
        Preprocessed image array ready for model prediction
    """
    # Convert to RGB if image has different mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to model's expected input size
    image = image.resize((224, 224))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize pixel values to [0, 1]
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def classify_brain_tumor(image):
    """
    Classify brain tumor from uploaded image
    
    Args:
        image: PIL Image object from Gradio interface
    
    Returns:
        Dictionary with class probabilities for Gradio output
    """
    if model is None:
        return {"Error": 1.0}
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Convert predictions to dictionary format for Gradio
        results = {}
        for i, class_name in enumerate(CLASS_NAMES):
            results[class_name] = float(predictions[0][i])
        
        return results
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {"Error": 1.0}

def create_interface():
    """Create and configure the Gradio interface"""
    
    # Custom CSS for styling
    css = """
    .gradio-container {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .gr-button {
        color: white;
        background-color: #007bff;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .gr-button:hover {
        background-color: #0056b3;
    }
    """
    
    # Create the interface
    with gr.Blocks(css=css, title="Brain Tumor Classification") as demo:
        
        # Header
        gr.Markdown(
            """
            # 🧠 Brain Tumor Classification
            
            Upload an MRI brain scan image to classify the type of tumor using a deep learning model.
            
            **Classes:**
            - **Glioma**: A type of tumor that occurs in the brain and spinal cord
            - **Meningioma**: A tumor that arises from the meninges
            - **No Tumor**: Normal brain scan without tumor
            - **Pituitary**: A tumor in the pituitary gland
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Image input
                image_input = gr.Image(
                    label="Upload Brain MRI Image",
                    type="pil",
                    height=400
                )
                
                # Predict button
                predict_btn = gr.Button(
                    "🔍 Classify Tumor",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Output
                output = gr.Label(
                    label="Prediction Results",
                    num_top_classes=4,
                    show_label=True
                )
                
                # Additional info
                gr.Markdown(
                    """
                    ### 📊 How to interpret results:
                    - Higher percentage = Higher confidence
                    - The model shows probabilities for all classes
                    - Results are for educational purposes only
                    """
                )
        
        # Examples section
        gr.Markdown("### 📋 Example Images")
        gr.Markdown("Click on any example below to try the classifier:")
        
        # You can add example images here if you have them
        # gr.Examples(
        #     examples=[
        #         ["example_glioma.jpg"],
        #         ["example_meningioma.jpg"],
        #         ["example_notumor.jpg"],
        #         ["example_pituitary.jpg"]
        #     ],
        #     inputs=image_input,
        #     outputs=output,
        #     fn=classify_brain_tumor,
        #     cache_examples=True
        # )
        
        # Medical disclaimer
        gr.Markdown(
            """
            ---
            ⚠️ **Medical Disclaimer:** This tool is for educational and research purposes only. 
            It should not be used as a substitute for professional medical diagnosis. 
            Always consult with qualified healthcare professionals for medical decisions.
            
            **Model Info:** Trained on brain MRI images | Input: 224x224 RGB | Framework: TensorFlow
            """
        )
        
        # Set up the prediction function
        predict_btn.click(
            fn=classify_brain_tumor,
            inputs=image_input,
            outputs=output,
            show_progress=True
        )
        
        # Also allow prediction on image change
        image_input.change(
            fn=classify_brain_tumor,
            inputs=image_input,
            outputs=output,
            show_progress=True
        )
    
    return demo

# Load model on startup
print("Loading brain tumor classification model...")
model_loaded = load_model()

if not model_loaded:
    print("Failed to load model. Please ensure 'brain_tumor_model.keras' is in the same directory.")

# Create and launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )