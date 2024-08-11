import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from "react-router-dom";

const Header = () => {
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const nav = useNavigate();
    const location = useLocation();

    useEffect(() => {
        // Check if the user ID exists in localStorage
        const userId = localStorage.getItem('userId');
        if (userId) {
            setIsLoggedIn(true);
        } else {
            setIsLoggedIn(false);
        }
    }, []);

    const handleLogout = () => {
        // Clear user data from localStorage and update the state
        localStorage.removeItem('userId');
        setIsLoggedIn(false);
        nav('/'); // Redirect to the home page after logging out
    };

    // Determine the header style based on the current URL path
    const headerStyle = location.pathname !== '/' 
        ? { backgroundColor: '#138113', color:'white', opacity:'0.9' } 
        : {};

    return (
        <nav className="home-navbar" style={headerStyle}>
            <div className="home-logo">AgroPlus</div>
            <ul className="home-nav-links">
                <li><div onClick={() => nav('/')}>Home</div></li>
                <li><div onClick={() => nav('/plant-check')}>Check Plant</div></li>
                <li><div onClick={() => nav('/about')}>About Us</div></li>
                {isLoggedIn ? (
                    <>
                        <li><div onClick={handleLogout}>Logout</div></li>
                    </>
                ) : (
                    <>
                        <li><div onClick={() => nav('/register')}>Sign Up</div></li>
                        <li><div onClick={() => nav('/login')}>Login</div></li>
                    </>
                )}
            </ul>
        </nav>
    );
};

export default Header;
