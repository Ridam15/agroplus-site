import React, { useState, useEffect } from 'react';
import Header from './Header';
import { useNavigate, useLocation } from 'react-router-dom';

const AgroPlantDetailsTable = () => {
    const [requests, setRequests] = useState([]);
    const [showModal, setShowModal] = useState(false);
    const [imageUrl, setImageUrl] = useState(''); // State to store the uploaded image URL
    const [category, setCategory] = useState('');
    const [predictionData, setPredictionData] = useState({ predictedClass: '', confidence: 0 }); // State to store prediction data
    const navigate = useNavigate();
    const location = useLocation();

    useEffect(() => {
        const userId = localStorage.getItem('userId');
        if (!userId) {
            navigate('/login');
            return;
        }

        if (location.state && location.state.category) {
            const category = location.state.category;
            setCategory(category);
            fetchData(userId,category);
        }
    }, [location.state?.category, navigate,requests]);

    const fetchData = async (userId, category) => {
        // Clean the userId by removing any surrounding quotes
        const cleanUserId = userId.replace(/^"|"$/g, '');
        console.log(cleanUserId); // Log the cleaned userId to ensure it's correct
    
        try {
            // Use the cleaned userId in the fetch request
            const response = await fetch(`http://20.63.137.159:8000/api/getDataByUserId/${cleanUserId}`);
            const result = await response.json();
    
            if (response.ok) {
                // Filter the fetched data based on the category
                const filteredData = result.data.filter(item => item.category === category);
    
                const fetchedData = filteredData.map((item, index) => ({
                    id: index + 1,
                    imageUrl: item.image_url,
                    requestDate: item.request_date || new Date().toISOString().split('T')[0],
                    status: item.status || 'Success',
                    disease: item.disease, // Add disease field
                    cure: item.cure,       // Add cure field
                    details: item.details, // Add details field
                    category:item.category,
                }));
                setRequests(fetchedData);
                console.log(fetchedData);
            } else {
                console.error('Error fetching data:', result.msg);
            }
        } catch (error) {
            console.error('Error fetching data:', error);
        }
    };
    
    
    
    const handleFileChange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
    
        const formData = new FormData();
        formData.append('image', file);
    
        try {
            const uploadResponse = await fetch('http://20.63.137.159:8000/api/uploadImage', {
                method: 'POST',
                body: formData,
            });
    
            const uploadResult = await uploadResponse.json();
    
            if (uploadResponse.ok) {
                const imageUrl = uploadResult.url;
                setImageUrl(imageUrl); // Assuming the API returns the image URL as 'url'
                console.log("Image uploaded successfully:", imageUrl);
    
                // Now call the external API with the uploaded image URL
                const predictionResponse = await fetch('http://20.63.137.159:5000/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        vegetable: category.toLowerCase(), // assuming the category is the vegetable type
                        image_url: imageUrl,
                    }),
                });
    
                const predictionResult = await predictionResponse.json();
    
                if (predictionResponse.ok) {
                    console.log("Prediction data received:", predictionResult);
                    // Update the state with predictionResult
                    setPredictionData({
                        predictedClass: predictionResult.predicted_class,
                        confidence: predictionResult.confidence,
                    });
                } else {
                    console.error('Error getting prediction:', predictionResult.message);
                }
            } else {
                console.error('Error uploading image:', uploadResult.msg);
            }
        } catch (error) {
            console.error('Error uploading image or getting prediction:', error);
        }
    };
    
    const handleAddRequest = async (e) => {
        e.preventDefault();
        const userId = localStorage.getItem('userId');
        if (!imageUrl || !userId || !category) {
            console.error("Missing data to submit the request.");
            return;
        }
    
        // Ensure that userId is a string without extra quotes
        const cleanUserId = userId.replace(/\"/g, '');
    
        const data = {
            user_id: cleanUserId,
            category: category,
            image_url: imageUrl,
            disease: predictionData.predictedClass, // Use predicted class from the prediction API
            cure: "", // Replace with actual data or logic to determine cure
            details: `Confidence: ${predictionData.confidence}`, // Include confidence in details
        };
    
        try {
            const response = await fetch(`http://20.63.137.159:8000/api/storeData`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data),
            });
    
            const result = await response.json();
    
            if (response.ok) {
                const newRequest = {
                    id: requests.length + 1,
                    imageUrl: result.data.image_url,
                    requestDate: new Date().toISOString().split('T')[0],
                    status: 'Pending',
                    disease: result.data.disease,
                    cure: result.data.cure,
                    details: result.data.details,
                };
                // Update the state with the new request
                setRequests([...requests, newRequest]);
                setShowModal(false);
            } else {
                console.error('Error storing data:', result.msg);
            }
        } catch (error) {
            console.error('Error storing data:', error);
        }
    };
    
    

    return (
        <>
            <Header />
            <button className="plant-back-button" onClick={() => navigate(-1)}>← Back</button>
            <div className="plant-request-container">
                <h1>{category} Request History</h1>
                <button className="add-request-btn" onClick={() => {
                    setShowModal(true);
                    setImageUrl(''); // Reset image URL when opening the modal
                    setPredictionData({ predictedClass: '', confidence: 0 }); // Reset prediction data
                }}>+ Add New Request</button>
                
                <table className="plant-table">
                    <thead>
                        <tr>
                            <th>Sr. No.</th>
                            <th>Image</th>
                            <th>Request Date</th>
                            <th>Status</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {requests.map((request) => (
                            <tr key={request.id}>
                                <td>{request.id}</td>
                                <td><img src={request.imageUrl} alt={category} /></td>
                                <td>{request.requestDate}</td>
                                <td>{request.status}</td>
                                <td><button onClick={()=>navigate('/plant-details',{state:request})}>Check</button></td>
                            </tr>
                        ))}
                    </tbody>
                </table>

                {showModal && (
                    <div className="modal">
                        <div className="modal-content">
                            <h2>Add New {category} Request</h2>
                            <form onSubmit={handleAddRequest}>
                                <div>
                                    <label>Upload Image:</label>
                                    <input 
                                        type="file" 
                                        onChange={handleFileChange} 
                                        required 
                                    />
                                </div>
                                {imageUrl && (
                                    <div>
                                        <img src={imageUrl} alt="Uploaded Preview" style={{ width: '400px', height: '400px' }} />
                                        {predictionData.predictedClass && (
                                            <p>Prediction: {predictionData.predictedClass} (Confidence: {predictionData.confidence})</p>
                                        )}
                                    </div>
                                )}
                                <button type="submit">Submit</button>
                                <button type="button" onClick={() => setShowModal(false)}>Close</button>
                            </form>
                        </div>
                    </div>
                )}
            </div>
        </>
    );
};

export default AgroPlantDetailsTable;
