import React from 'react';
import Header from './Header';
import { useNavigate } from "react-router-dom";

const AgroPlantCheck = () => {
    const nav=useNavigate();
    
    return (
        <>
        <Header/>
        <div className="card-container">
            <div className="cards">
                <div className="card">
                    <img src="images/potato.jpg" alt="Potato" />
                    <h2>Potato</h2>
                    <p>This is a potato plant. It is a staple food in many parts of the world.</p>
                    <button onClick={()=>nav('/plant-details-table',{state:{category:"Potato"}})}>CHECK</button>
                </div>
                <div className="card">
                    <img src="images/tomato.jpg" alt="Tomato" />
                    <h2>Tomato</h2>
                    <p>This is a tomato plant. Tomatoes are widely used in cooking around the world.</p>
                    <button onClick={()=>nav('/plant-details-table',{state:{category:"Tomato"}})}>CHECK</button>
                </div>
                <div className="card">
                    <img src="images/pepper.jpg" alt="Pepper" />
                    <h2>Pepper</h2>
                    <p>This is a pepper plant. Peppers can be sweet or spicy and are used in many dishes.</p>
                    <button onClick={()=>nav('/plant-details-table',{state:{category:"Pepper"}})}>CHECK</button>
                </div>
            </div>
        </div>
        </>
    );
};

export default AgroPlantCheck;
