// File: index.js

const express = require('express');
const cors = require('cors');
const cookieParser = require('cookie-parser');
require('dotenv').config();
const connectToMongo = require('./db/conn');

// Initialize MongoDB connection
connectToMongo();

const app = express();
const PORT = process.env.PORT || 5000;

// Importing image upload logic
const { upload, uploadImage } = require('./routes/imageUpload');

// Import routes
const user_route = require('./routes/userRoute');
const image_route = require('./routes/imageRoute');

// Middleware setup
app.use(cors());
app.use(cookieParser());

// Image upload endpoint
app.post('/upload', upload.single('image'), uploadImage);

// Additional routes
app.use('/api', user_route);
app.use('/api', image_route);

app.get('/', (req, res) => {
  res.send('Hello, World!');
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
