const express = require("express");
const user_route = express.Router(); // Change express() to express.Router() for proper routing
const bodyParser = require("body-parser");
const multer = require("multer");
const path = require('path');

const user_controller = require("../controllers/userController");
const { upload, uploadImage } = require("./imageUpload");

// Middleware setup
user_route.use(bodyParser.json());
user_route.use(bodyParser.urlencoded({ extended: true }));
user_route.use(express.static('public'));

// Disk storage setup for local file uploads (if needed)
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        const dirPath = path.join(__dirname, '../public/userImages');
        cb(null, dirPath);
    },
    filename: function (req, file, cb) {
        const name = Date.now() + '-' + file.originalname;
        cb(null, name);
    }
});

const uploadDisk = multer({ storage: storage });

// User routes for registration and login
user_route.post('/register', user_controller.registerUser);
user_route.post('/userLogin', user_controller.userlogin);

// New routes for image upload and storing data
user_route.post('/uploadImage', upload.single('image'), uploadImage);

module.exports = user_route;
