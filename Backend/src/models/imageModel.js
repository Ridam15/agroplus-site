const mongoose = require("mongoose");

// Define the schema for the object inside each category
const imageObjectSchema = new mongoose.Schema({
    category: {
        type: String, // Category of the image
        required: true,
    },
    image_url: {
        type: String, // Stores the URL of the image
        required: true,
    },
    disease: {
        type: String, // Name of the disease associated with the image
        default:"No Disease",
    },
    cure: {
        type: String, // Cure for the disease
        default:"No Cure Needed",
    },
    details: {
        type: String, // Additional details about the disease and cure
    },
    user_id: {
        type: String, // Store user_id as a string
        required: true,
    }
});

module.exports = mongoose.model("Image", imageObjectSchema);
