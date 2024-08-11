import React, { useState } from 'react';
import { useNavigate } from "react-router-dom";
import { toast } from 'react-toastify';

const AgroLogin = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const nav = useNavigate();

    const handleLogin = async (e) => {
        e.preventDefault();

        try {
            const response = await fetch('http://20.63.137.159:8000/api/userLogin', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ email, password }),
            });

            const result = await response.json();

            if (response.ok) {
                // Store the user ID in localStorage
                const userId = result.data._id; // Assuming this is the user ID in the response
                localStorage.setItem('userId', JSON.stringify(userId));

                // Show success toast notification
                toast.success('Login successful!', {
                    position: "top-right",
                    autoClose: 5000,
                    hideProgressBar: false,
                    closeOnClick: true,
                    pauseOnHover: true,
                    draggable: true,
                    progress: undefined,
                });

                // Redirect the user after successful login
                nav('/plant-check'); // Adjust the path to your desired route

            } else {
                // Show error toast notification
                toast.error(result.msg || 'Login failed. Please check your credentials.', {
                    position: "top-right",
                    autoClose: 5000,
                    hideProgressBar: false,
                    closeOnClick: true,
                    pauseOnHover: true,
                    draggable: true,
                    progress: undefined,
                });
            }
        } catch (err) {
            // Show error toast notification for network/server errors
            toast.error('An error occurred while trying to log in. Please try again.', {
                position: "top-right",
                autoClose: 5000,
                hideProgressBar: false,
                closeOnClick: true,
                pauseOnHover: true,
                draggable: true,
                progress: undefined,
            });
        }
    };

    return (
        <div className="auth-wrapper">
            <div className="auth-form-container">
                <h2 className="auth-title">Welcome back!</h2>
                <p className="auth-subtitle">Enter your Credentials to access your account</p>
                <form className="auth-form" onSubmit={handleLogin}>
                    <label htmlFor="login-email" className="auth-label">Email address</label>
                    <input
                        type="email"
                        id="login-email"
                        className="auth-input"
                        placeholder="Enter your email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                    />

                    <label htmlFor="login-password" className="auth-label">Password</label>
                    <input
                        type="password"
                        id="login-password"
                        className="auth-input"
                        placeholder="Enter your password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                    />

                    <div className="auth-remember-forgot">
                        <label className="auth-remember">
                            <input type="checkbox" />
                            Remember for 30 days
                        </label>
                    </div>

                    <button type="submit" className="auth-button">Login</button>
                </form>

                <p onClick={() => nav('/register')} className="auth-footer">Don't have an account? <a className="auth-footer-link">Sign Up</a></p>
            </div>
            <div className="auth-image-container">
                <img src="images/Login.jpg" alt="Leaf Image" className="auth-image" />
            </div>
        </div>
    );
};

export default AgroLogin;
