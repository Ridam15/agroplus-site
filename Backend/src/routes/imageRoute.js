const express = require("express");
const image_route = express.Router();
const image_controller = require("../controllers/imageController");

// Route to store image data
image_route.post('/storeData', image_controller.storeData);

// Route to get image data by user ID
image_route.get('/getDataByUserId/:userId', image_controller.getDataByUserId);

module.exports = image_route;