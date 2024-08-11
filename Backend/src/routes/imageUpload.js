// File: imageUpload.js
const mongoose = require('mongoose');
const multer = require('multer');
const cloudinary = require('cloudinary').v2;
const { CloudinaryStorage } = require('multer-storage-cloudinary');

// Configure Cloudinary with your credentials
cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME || 'dj6564anu',
  api_key: process.env.CLOUDINARY_API_KEY || '228688232979645',
  api_secret: process.env.CLOUDINARY_API_SECRET || '-DKb3iCdJeTW74F7bXazugD1nUI'
});

// Configure Cloudinary storage for multer
const storage = new CloudinaryStorage({
  cloudinary: cloudinary,
  params: {
    folder: 'uploads', // Folder where images will be stored in Cloudinary
    allowed_formats: ['jpg', 'png'],
  },
});

// Multer middleware to handle file upload
const upload = multer({ storage: storage });

// Image upload handler function
const uploadImage = (req, res) => {
  try {
      if (req.file) {
          return res.json({
              success: true,
              url: req.file.path,
          });
      } else {
          return res.status(400).json({
              success: false,
              message: 'Image upload failed. No file provided.',
          });
      }
  } catch (error) {
      return res.status(500).json({
          success: false,
          message: 'An error occurred during the image upload process.',
          error: error.message,
      });
  }
};


// Exporting the upload middleware and uploadImage function
module.exports = {
  upload,
  uploadImage,
};
