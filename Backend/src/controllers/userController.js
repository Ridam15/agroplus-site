const User = require("../models/userModel");
const bcryptjs = require("bcryptjs");

// Function to hash the password
const securePassword = async (password) => {
    try {
        const passwordhash = await bcryptjs.hash(password, 10);
        return passwordhash;
    } catch (error) {
        throw new Error('Error securing password');
    }
}

// Function to register a new user
const registerUser = async (req, res) => {
    try {
        const spassword = await securePassword(req.body.password);
        const user = new User({
            username: req.body.username,
            email: req.body.email,
            password: spassword,
        });

        const userData = await User.findOne({ email: req.body.email });

        if (userData) {
            return res.status(400).send({ success: false, msg: "User already exists" });
        } else {
            const user_data = await user.save();
            return res.status(200).send({ success: true, data: user_data });
        }

    } catch (error) {
        return res.status(400).send({ success: false, msg: error.message });
    }
}

// Function to log in a user
const userlogin = async (req, res) => {
    try {
        const email = req.body.email;
        const password = req.body.password;
        const userData = await User.findOne({ email: email });
        if (userData) {
            const passwordMatch = await bcryptjs.compare(password, userData.password);
            if (passwordMatch) {
                const userResult = {
                    _id: userData._id,
                    username: userData.username,
                    email: userData.email,
                    password: userData.password,
                }

                const response = {
                    success: true,
                    msg: "user details",
                    data: userResult,
                }

                return res.status(200).send(response);
            } else {
                return res.status(400).send({ success: false, msg: "Login details are incorrect (password incorrect)" });
            }
        } else {
            return res.status(400).send({ success: false, msg: "Login details are incorrect (Register First)" });
        }

    } catch (error) {
        return res.status(400).send({ success: false, msg: error.message });
    }
}

// Function to store object details in the database
const storeData = async (req, res) => {
    try {
        const { _id, categoryName, imageUrl, disease, cure, details } = req.body;

        const user = await User.findById(_id);
        if (!user) {
            return res.status(404).json({
                success: false,
                message: 'User not found',
            });
        }

        const newImageObject = {
            unique_id: new mongoose.Types.ObjectId(), // Ensure unique_id is generated
            image_url: imageUrl,
            disease: disease,
            cure: cure,
            details: details,
        };

        switch (categoryName) {
            case 'Potato':
                user.potato.images.push(newImageObject);
                break;
            case 'Tomato':
                user.tomato.images.push(newImageObject);
                break;
            case 'Pepper Bell':
                user.pepperBell.images.push(newImageObject);
                break;
            default:
                return res.status(400).json({
                    success: false,
                    message: 'Invalid category name',
                });
        }

        await user.save();

        return res.status(200).json({
            success: true,
            message: 'Data stored successfully',
            data: newImageObject,
        });
    } catch (error) {
        return res.status(500).json({
            success: false,
            message: 'Error storing data',
            error: error.message,
        });
    }
}

// Function to get Potato data
const getPotatoData = async (req, res) => {
    try {
        const userId = req.query._id; // Assuming user ID is passed as a query parameter

        const user = await User.findById(userId);
        if (!user) {
            return res.status(404).send({ success: false, msg: 'User not found' });
        }

        return res.status(200).send({ success: true, data: user.potato.images });
    } catch (error) {
        return res.status(500).send({ success: false, msg: 'Error fetching potato data', error: error.message });
    }
}

// Function to get Tomato data
const getTomatoData = async (req, res) => {
    try {
        const userId = req.query._id; // Assuming user ID is passed as a query parameter

        const user = await User.findById(userId);
        if (!user) {
            return res.status(404).send({ success: false, msg: 'User not found' });
        }

        return res.status(200).send({ success: true, data: user.tomato.images });
    } catch (error) {
        return res.status(500).send({ success: false, msg: 'Error fetching tomato data', error: error.message });
    }
}

// Function to get Pepper Bell data
const getPepperBellData = async (req, res) => {
    try {
        const userId = req.query._id; // Assuming user ID is passed as a query parameter

        const user = await User.findById(userId);
        if (!user) {
            return res.status(404).send({ success: false, msg: 'User not found' });
        }

        return res.status(200).send({ success: true, data: user.pepperBell.images });
    } catch (error) {
        return res.status(500).send({ success: false, msg: 'Error fetching pepper bell data', error: error.message });
    }
}

module.exports = {
    registerUser,
    userlogin,
    storeData,
    getPotatoData,
    getTomatoData,
    getPepperBellData,
}
