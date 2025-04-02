const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const axios = require('axios');
const sharp = require('sharp');
const fs = require('fs').promises;

const app = express();
const port = 3000;

app.use(express.json());

// Load labels from labels.txt
async function loadLabels() {
    const labelsText = await fs.readFile('labels.txt', 'utf8');
    return labelsText.split('\n').filter(line => line.trim() !== '');
}

// Load the model
let model;
async function loadModel() {
    model = await tf.loadLayersModel('file://./model/model.json');
    console.log('Model loaded');
}

// Preprocess the image using sharp
async function preprocessImage(imageUrl) {
    try {
        // Download the image
        const response = await axios.get(imageUrl, { responseType: 'arraybuffer' });
        const imageBuffer = Buffer.from(response.data);

        // Use sharp to process the image
        const image = await sharp(imageBuffer)
            .resize(224, 224, { fit: 'fill' }) // Resize to 224x224, similar to ImageOps.fit
            .raw() // Get raw pixel data
            .toBuffer({ resolveWithObject: true });

        const { data, info } = image;
        if (info.channels !== 3) {
            throw new Error('Image must have 3 channels (RGB)');
        }

        // Convert to Float32Array and normalize
        const imageArray = new Float32Array(224 * 224 * 3);
        for (let i = 0; i < data.length; i++) {
            imageArray[i] = (data[i] / 127.5) - 1; // Normalize: (pixel / 127.5) - 1
        }

        // Reshape to [1, 224, 224, 3]
        return tf.tensor4d(imageArray, [1, 224, 224, 3]);
    } catch (error) {
        console.error('Error in preprocessImage:', error);
        throw error;
    }
}

// API endpoint to predict
app.post('/predict', async (req, res) => {
    const { imageUrl } = req.body;

    if (!imageUrl) {
        return res.status(400).json({ error: 'Image URL is required' });
    }

    try {
        if (!model) await loadModel();
        const classNames = await loadLabels();

        const tensor = await preprocessImage(imageUrl);
        const prediction = model.predict(tensor);
        const predictionArray = prediction.dataSync();
        const index = prediction.argMax(1).dataSync()[0];
        const className = classNames[index].substring(2); // Remove numbering
        const confidenceScore = predictionArray[index];

        tensor.dispose();
        prediction.dispose();

        res.json({
            class: className,
            confidenceScore: confidenceScore
        });
    } catch (error) {
        console.error('Prediction error:', error);
        res.status(500).json({ error: 'An error occurred during prediction' });
    }
});

// Start the server
app.listen(port, async () => {
    console.log(`Server running on port ${port}`);
    await loadModel();
});