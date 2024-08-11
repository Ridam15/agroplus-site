import React, { useRef } from 'react';
import { useNavigate } from "react-router-dom";
import Header from './Header'; // Import the Header component

const AgroHome = () => {
    const nav = useNavigate();
    const aboutSectionRef = useRef(null); // Create a ref to the about section

    const scrollToAboutSection = () => {
        if (aboutSectionRef.current) {
            aboutSectionRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    };

    return (
        <>
            {/* Hero Section */}
            <section className="home-container">
                <Header /> {/* Use the Header component here */}
                <div className="home-sub_container">
                    <h1>Predicting Plant Health for a Sustainable Future</h1>
                    <p>Leveraging AI-Powered Disease Detection to Safeguard Crops and Boost Agricultural Productivity.</p>
                    <div className="home-container-buttons">
                        <a onClick={scrollToAboutSection} className="home-btn-secondary">GET STARTED</a>
                    </div>
                </div>
            </section>

            {/* About Section */}
            <section ref={aboutSectionRef} className="home-about"> {/* Attach the ref here */}
                <h2>How to Use Our Website?</h2>
                <div className="crop-detection-container">
                    <div className="steps-container">
                        <div className="step step1">
                            <div className="step-image">
                                <img src="images/cut.png" alt="Pulling leaf" />
                            </div>
                            <div className="step-text">
                                <h3>Carefully pull the leaf you found most affected by a disease.</h3>
                            </div>
                        </div>

                        <div className="step step2">
                            <div className="step-text">
                                <h3>Place the leaf in a bright background and take a clear photo.</h3>
                            </div>
                            <div className="step-image">
                                <img src="images/sun.png" alt="Leaf on bright background" />
                            </div>
                        </div>

                        <div className="step step3">
                            <div className="step-image">
                                <img src="images/take.png" alt="Upload image" />
                            </div>
                            <div className="step-text">
                                <h3>Upload the image using our 'Check Plant's Disease' feature & Check your Plant's Disease.</h3>
                            </div>
                        </div>
                    </div>

                    <div className="button-container">
                        <button onClick={() => nav('/plant-check')} className="check-plant-button">Check the Plant</button>
                    </div>
                </div>
            </section>

            {/* Footer Section */}
            <footer className="home-footer">
                <div className="home-footer-content">
                    <div className="home-footer-logo">AgroPlus</div>
                    <div className="home-footer-links">
                        <ul>
                            <li><a href="#">About Us</a></li>
                            <li><a href="#">Products</a></li>
                            <li><a href="#">Media</a></li>
                            <li><a href="#">Checkout</a></li>
                        </ul>
                    </div>
                    <div className="home-footer-newsletter">
                        <p>Contact with us for Future</p>
                        <input type="email" placeholder="Enter your email" />
                        <button><i className="fas fa-paper-plane"></i> Enquiry Now</button>
                    </div>
                    <div className="home-footer-social">
                        <a href="#"><i className="fab fa-facebook-f"></i> Facebook</a>
                        <a href="#"><i className="fab fa-linkedin-in"></i> LinkedIn</a>
                        <a href="#"><i className="fab fa-twitter"></i> Twitter</a>
                    </div>
                </div>
                <div className="home-footer-bottom">
                    <p>&copy; 2024 AgroPlus. All rights reserved.</p>
                </div>
            </footer>
        </>
    );
};

export default AgroHome;
