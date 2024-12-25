import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import matplotlib.pyplot as plt
import io

@st.cache_resource
def load_model():
    """Load saved model and labels"""
    try:
        model = tf.keras.models.load_model('src/moduls6/ResNet_fine_tuned_model.keras')
        labels = ['Normal', 'Pneumonia-Bacterial', 'Pneumonia-Viral', 'COVID-19']
        return model, labels
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def make_gradcam_heatmap(img_array, model, pred_index=None):
    """Generate Grad-CAM heatmap with improved error handling"""
    try:
        # First, find the last conv layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        
        if last_conv_layer is None:
            print("No convolutional layer found")
            return None

        # Create gradient model
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [last_conv_layer.output, model.output]
        )

        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        # Handle potential gradient computation errors
        try:
            grads = tape.gradient(class_channel, conv_outputs)
            if grads is None:
                print("Gradient computation failed")
                return None
                
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            # Normalize between 0 and 1
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + tf.keras.backend.epsilon())
            return heatmap.numpy()
            
        except Exception as e:
            print(f"Error in gradient computation: {e}")
            return None
            
    except Exception as e:
        print(f"Error in GradCAM generation: {e}")
        return None

def apply_gradcam(image, heatmap, alpha=0.4):
    """Apply Grad-CAM heatmap with robust error handling"""
    try:
        # Ensure image is in correct format
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image = np.array(image)
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Add debug prints
        print(f"Image shape: {image.shape}")
        print(f"Heatmap shape before resize: {heatmap.shape}")
        
        # Resize heatmap to match image dimensions
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        print(f"Heatmap shape after resize: {heatmap.shape}")
        
        # Apply colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Blend images
        superimposed = cv2.addWeighted(image, 1-alpha, heatmap, alpha, 0)
        superimposed = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
        
        return superimposed
        
    except Exception as e:
        print(f"Error in heatmap application: {e}")
        return image

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to match training size
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        
        # Convert to array and preprocess
        img_array = np.array(image)
        img_array = preprocess_input(img_array)
        
        if len(img_array.shape) != 3 or img_array.shape[-1] != 3:
            raise ValueError(f"Invalid image shape: {img_array.shape}")
            
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
        
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None

def pneumonia_classification():
    # Set page config
    st.set_page_config(
        page_title="Pneumonia X-Ray Classifier",
        page_icon="🫁",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stAlert {
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .css-1v0mbdj.etr89bj1 {
            margin-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title('🫁 Pneumonia X-Ray Classification')
    st.markdown("""
        This application analyzes chest X-ray images to detect and classify pneumonia cases.
        Upload a chest X-ray image to get started.
    """)
    
    # Load model
    model, labels = load_model()
    
    if model is None or labels is None:
        st.error("Could not load model. Please ensure model file is available.")
        return

    # Initialize session state
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear chest X-ray image for analysis"
        )
        
        if uploaded_file is None:
            st.session_state.predictions = None
            st.info("Please upload an X-ray image to begin analysis")
    
    if uploaded_file is not None:
        try:
            # Read and display original image
            image = Image.open(uploaded_file)
            col1.image(image, caption='Uploaded X-Ray Image', use_container_width=True)
            
            # Add predict button
            if col1.button('Analyze X-Ray', type='primary'):
                with st.spinner('Analyzing image...'):
                    # Preprocess and predict
                    processed_image = preprocess_image(image)
                    if processed_image is not None:
                        # Make prediction
                        predictions = model.predict(processed_image, verbose=0)
                        st.session_state.predictions = predictions
                        
                        # Get predicted class
                        pred_class = labels[np.argmax(predictions[0])]
                        
                        # Generate Grad-CAM heatmap
                        heatmap = make_gradcam_heatmap(processed_image, model)
                        
                        # Convert PIL image to numpy array for OpenCV processing
                        img_array = np.array(image)
                        
                        # Apply Grad-CAM visualization
                        if heatmap is not None and pred_class != "Normal":
                            highlighted_image = apply_gradcam(img_array, heatmap)
                        else:
                            highlighted_image = img_array
                        
                        # Display results in second column
                        with col2:
                            st.subheader("Analysis Results")
                            
                            # Display highlighted image
                            st.image(highlighted_image, caption='Analysis Visualization', use_container_width=True)
                            
                            # Show prediction results
                            top_3_indices = predictions[0].argsort()[-3:][::-1]
                            top_3_labels = [labels[idx] for idx in top_3_indices]
                            top_3_probs = predictions[0][top_3_indices]
                            
                            # Display main prediction with blue background
                            st.markdown(f"""
                                <div style='padding: 1rem; background: #001F3F; 
                                    border-radius: 0.5rem; margin-bottom: 1rem;'>
                                    <h3 style='margin: 0;'>Primary Diagnosis: {pred_class}</h3>
                                    <p style='margin: 0.5rem 0 0 0;'>Confidence: {top_3_probs[0] * 100:.1f}%</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Display additional predictions
                            st.subheader("Detailed Analysis")
                            chart_data = pd.DataFrame({
                                'Condition': top_3_labels,
                                'Probability': top_3_probs * 100
                            })
                            st.bar_chart(chart_data.set_index('Condition'))
                            
                            # Add explanation for abnormal cases
                            if pred_class != "Normal":
                                st.info("""
                                    **Highlighted Areas**: The red-yellow overlay indicates regions 
                                    that the model finds most significant for the diagnosis. 
                                    Brighter colors indicate stronger activation.
                                """)
                                
        except Exception as e:
            st.error(f"Error processing image: {e}")
    
    # About section
    st.markdown('---')
    st.info("""
        ### About This Tool
        - Uses a fine-tuned ResNet50V2 model for pneumonia classification
        - Can detect: Normal, Bacterial Pneumonia, Viral Pneumonia, and COVID-19
        - Uses Grad-CAM visualization to highlight relevant regions
        - For research and educational purposes only
        - Not intended for clinical diagnosis
    """)

if __name__ == '__main__':
    pneumonia_classification()