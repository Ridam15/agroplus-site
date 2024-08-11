import React from 'react';
import Header from './Header'; // Assuming you have a Header component

const AboutUs = () => {
    return (
        <>
            <Header />

                <section className="about-content">
                    <div className="container">
                        <h2>Our Commitment to Farmers</h2>
                        <p>
                            At AgroPlus, we are dedicated to empowering farmers with innovative technology to protect their crops and improve yields. 
                            Our platform, <strong>Crop Disease Detection and Management Using Machine Learning</strong>, utilizes advanced AI to provide quick 
                            and accurate identification of plant diseases, helping you take timely action to safeguard your fields.
                        </p>

                        <h2>Our Mission</h2>
                        <p>
                            Our mission is simple: to make disease detection easy and accessible for everyone in agriculture. By leveraging cutting-edge machine learning 
                            models, we enable you to upload a photo of an affected plant and receive instant diagnoses, along with practical solutions for treatment and prevention.
                        </p>

                        <h2>Why Choose AgroPlus?</h2>
                        <ul>
                            <li><strong>Instant Disease Detection:</strong> Upload a photo and get an immediate diagnosis.</li>
                            <li><strong>Practical Solutions:</strong> Receive actionable advice for treatment and prevention.</li>
                            <li><strong>Innovative Technology:</strong> We use state-of-the-art AI to ensure accuracy and reliability.</li>
                            <li><strong>Sustainable Farming:</strong> Helping you secure a better future through smarter farming practices.</li>
                        </ul>

                        <h2>Join Us in Revolutionizing Agriculture</h2>
                        <p>
                            With AgroPlus, you're not just managing crops—you're securing a better future for farming. Our platform empowers you to stay ahead of crop 
                            diseases and optimize your yields, all while promoting sustainable agricultural practices.
                        </p>
                    </div>
                </section>
        </>
    );
};

export default AboutUs;
