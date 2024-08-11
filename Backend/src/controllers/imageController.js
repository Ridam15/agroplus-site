const Image = require("../models/imageModel");

// Function to store object details in the database
const storeData = async (req, res) => {
    try {
        const { user_id, category, image_url, disease, cure, details } = req.body;

        // Create a new image document
        const newImage = new Image({
            user_id: user_id, // Assuming user_id is passed as a string
            category: category,
            image_url: image_url,
            disease: disease,
            cure: cure,
            details: details,
        });

        // Save the image document to the database
        const savedImage = await newImage.save();

        // Respond with success and the saved image data
        return res.status(200).json({
            success: true,
            message: 'Data stored successfully',
            data: savedImage,
        });
    } catch (error) {
        // Handle errors and respond with an error message
        return res.status(500).json({
            success: false,
            message: 'Error storing data',
            error: error.message,
        });
    }
};

// Function to get data by user ID
const getDataByUserId = async (req, res) => {
    try {
        const userId = req.params.userId; // Assuming user ID is passed as a URL parameter

        // Find all images associated with the given user ID
        const images = await Image.find({ user_id: userId });

        if (!images || images.length === 0) {
            // If no images are found, respond with a not found message
            return res.status(404).json({
                success: false,
                message: 'No data found for this user',
            });
        }

        // Respond with the found images
        return res.status(200).json({
            success: true,
            data: images,
        });
    } catch (error) {
        // Handle errors and respond with an error message
        return res.status(500).json({
            success: false,
            message: 'Error retrieving data',
            error: error.message,
        });
    }
};

module.exports = {
    storeData,
    getDataByUserId,
};
