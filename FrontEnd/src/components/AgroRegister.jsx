import React, { useState } from 'react';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { useNavigate } from "react-router-dom";

const SignupForm = () => {
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [message, setMessage] = useState('');
    const [error, setError] = useState(null);
    const nav = useNavigate();

    const handleSignup = async (e) => {
        e.preventDefault();

        // Clear previous messages
        setMessage('');
        setError(null);

        try {
            const response = await fetch('http://20.63.137.159:8000/api/register', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ username, email, password }),
            });

            const result = await response.json();

            if (response.ok) {
                setMessage('Signup successful!');
                toast.success('Signup successful!', {
                    position: "top-right",
                    autoClose: 5000,
                    hideProgressBar: false,
                    closeOnClick: true,
                    pauseOnHover: true,
                    draggable: true,
                    progress: undefined,
                });
                setTimeout(() => {
                    nav('/login'); // Redirect to the login page or any other page
                }, 1000);
            } else {
                setError(result.msg || 'Signup failed');
                toast.error(result.msg || 'Signup failed', {
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
            setError('An error occurred while trying to sign up. Please try again.');
            toast.error('An error occurred while trying to sign up. Please try again.', {
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
            <ToastContainer />
            <div className="auth-form-container">
                <h2 className="auth-title">Get Started Now</h2>
                <form className="auth-form" onSubmit={handleSignup}>
                    <label htmlFor="signup-username" className="auth-label">Username</label>
                    <input
                        type="text"
                        id="signup-username"
                        className="auth-input"
                        placeholder="Enter your username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                    />

                    <label htmlFor="signup-email" className="auth-label">Email address</label>
                    <input
                        type="email"
                        id="signup-email"
                        className="auth-input"
                        placeholder="Enter your email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                    />

                    <label htmlFor="signup-password" className="auth-label">Password</label>
                    <input
                        type="password"
                        id="signup-password"
                        className="auth-input"
                        placeholder="Enter your password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                    />

                    <button type="submit" className="auth-button">Signup</button>
                </form>

                <p onClick={() => nav('/login')} className="auth-footer">
                    Already have an account? <a className="auth-footer-link">Login here</a>
                </p>
            </div>

            <div className="auth-image-container">
                <img src="images/Login.jpg" alt="Leaf Image" className="auth-image" />
            </div>
        </div>
    );
};

export default SignupForm;
