import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import Header from './Header';
import { useNavigate } from "react-router-dom";

// Function to get detailed disease information
const getPlantDiseaseDetails = (category, disease) => {
    const details = {
        Tomato: {
            "Bacterial Spot": {
                cause: "Xanthomonas campestris pv. vesicatoria, a bacteria.",
                symptoms: "Small, water-soaked spots on leaves, stems, and fruit, which may turn dark and scabby.",
                treatment: {
                    culturalPractices: [
                        "Start with disease-free seeds or transplants.",
                        "Avoid overhead irrigation to reduce leaf wetness.",
                        "Remove and destroy infected plant debris.",
                        "Rotate crops every 2-3 years with non-host plants (e.g., corn, beans)."
                    ],
                    chemicalControl: [
                        "Apply copper-based bactericides regularly, starting as soon as symptoms appear.",
                        "Combine copper with mancozeb or other fungicides to improve effectiveness."
                    ],
                    prevention: [
                        "Implement resistant varieties when available.",
                        "Keep the garden area weed-free to reduce alternate hosts."
                    ]
                }
            },
            "Early Blight": {
                cause: "Fungus Alternaria solani.",
                symptoms: "Circular, dark brown spots with concentric rings, often appearing on older leaves first.",
                treatment: {
                    culturalPractices: [
                        "Remove and destroy infected leaves and plant debris.",
                        "Use drip irrigation or water at the base of the plant to keep foliage dry.",
                        "Mulch around plants to prevent soil from splashing onto leaves."
                    ],
                    chemicalControl: [
                        "Apply fungicides like chlorothalonil, mancozeb, or copper-based products starting early in the season.",
                        "Repeat applications every 7-14 days, especially in humid or wet conditions."
                    ],
                    prevention: [
                        "Use resistant varieties when available.",
                        "Rotate crops annually, avoiding the planting of tomatoes or related crops (e.g., potatoes, peppers) in the same spot."
                    ]
                }
            },
            "Late Blight": {
                cause: "Fungus-like organism Phytophthora infestans.",
                symptoms: "Large, irregular, water-soaked, and grayish-green spots on leaves; white fuzzy growth on the underside of leaves; rapid decay of fruit.",
                treatment: {
                    culturalPractices: [
                        "Immediately remove and destroy infected plants.",
                        "Avoid overhead watering and ensure plants are spaced well to improve air circulation.",
                        "Mulch to reduce soil splashing."
                    ],
                    chemicalControl: [
                        "Apply fungicides like chlorothalonil, copper-based products, or mancozeb as soon as symptoms are detected.",
                        "Regular applications (every 7-10 days) are essential, especially in cool, wet weather."
                    ],
                    prevention: [
                        "Use late blight-resistant tomato varieties.",
                        "Avoid planting tomatoes and potatoes in close proximity, as they can both host the disease.",
                        "Practice crop rotation and remove all plant debris after the season ends."
                    ]
                }
            },
            "Leaf Mold": {
                cause: "Fungus Passalora fulva.",
                symptoms: "Pale yellow spots on upper leaf surfaces; olive-green to brown mold on the underside of leaves.",
                treatment: {
                    culturalPractices: [
                        "Improve ventilation by spacing plants properly and pruning excess foliage.",
                        "Water at the base of plants and avoid wetting leaves.",
                        "Keep the environment dry, especially in greenhouses, by reducing humidity."
                    ],
                    chemicalControl: [
                        "Apply sulfur-based fungicides or copper-based products.",
                        "Organic options like potassium bicarbonate can also be effective."
                    ],
                    prevention: [
                        "Use resistant tomato varieties if available.",
                        "Keep greenhouse humidity below 85% if growing indoors."
                    ]
                }
            },
            "Septoria Leaf Spot": {
                cause: "Fungus Septoria lycopersici.",
                symptoms: "Small, circular spots with dark borders and grayish centers on leaves, usually starting on the lower leaves.",
                treatment: {
                    culturalPractices: [
                        "Remove and destroy infected leaves as soon as symptoms appear.",
                        "Avoid overhead watering and mulch around plants to prevent soil splashing.",
                        "Space plants adequately to improve air circulation."
                    ],
                    chemicalControl: [
                        "Apply fungicides like copper-based products, chlorothalonil, or mancozeb.",
                        "Start treatments early in the season and reapply as needed, especially during wet periods."
                    ],
                    prevention: [
                        "Rotate crops with non-host plants and avoid planting tomatoes in the same spot for at least 2 years.",
                        "Clean up and destroy all plant debris at the end of the season."
                    ]
                }
            },
            "Spider Mites": {
                cause: "Tetranychus urticae (tiny arachnids that feed on plant sap).",
                symptoms: "Fine webbing on leaves; leaves may appear stippled or speckled, and eventually turn yellow or bronze.",
                treatment: {
                    culturalPractices: [
                        "Regularly spray plants with a strong stream of water to knock off mites.",
                        "Keep plants well-watered to reduce stress, as dry plants are more susceptible.",
                        "Remove severely infested leaves."
                    ],
                    chemicalControl: [
                        "Apply insecticidal soaps or horticultural oils like neem oil.",
                        "For severe infestations, miticides like abamectin or bifenthrin may be necessary."
                    ],
                    biologicalControl: [
                        "Introduce natural predators such as ladybugs, predatory mites (Phytoseiulus persimilis), or lacewing larvae to control mite populations."
                    ],
                    prevention: [
                        "Avoid using broad-spectrum insecticides that can kill beneficial predators.",
                        "Regularly inspect plants for early signs of mite infestations."
                    ]
                }
            },
            "Target Spot": {
                cause: "Fungus Corynespora cassiicola.",
                symptoms: "Brown, round lesions with concentric rings on leaves, stems, and fruit.",
                treatment: {
                    culturalPractices: [
                        "Remove and destroy affected plant parts.",
                        "Space plants adequately to improve air circulation and reduce leaf wetness.",
                        "Mulch to prevent soil splashing onto leaves."
                    ],
                    chemicalControl: [
                        "Apply fungicides such as chlorothalonil or mancozeb.",
                        "Begin applications early in the season and continue regularly, particularly in wet conditions."
                    ],
                    prevention: [
                        "Rotate crops and avoid planting tomatoes in the same location for several years.",
                        "Use resistant varieties when available.",
                        "Ensure proper spacing and trellising of plants to reduce moisture retention on leaves."
                    ]
                }
            },
            "Yellow Leaf Curl Virus": {
                cause: "Tomato yellow leaf curl virus (TYLCV) transmitted by whiteflies.",
                symptoms: "Yellowing and curling of leaves, stunted growth, and reduced fruit production.",
                treatment: {
                    culturalPractices: [
                        "Remove and destroy infected plants to prevent the spread of the virus.",
                        "Implement strict whitefly control measures, including using yellow sticky traps and reflective mulches.",
                        "Practice good garden hygiene, such as removing weeds that can host whiteflies."
                    ],
                    chemicalControl: [
                        "Apply insecticidal soaps or neem oil to control whitefly populations.",
                        "Consider using systemic insecticides in severe cases."
                    ],
                    prevention: [
                        "Use virus-resistant tomato varieties.",
                        "Cover plants with fine mesh netting to exclude whiteflies.",
                        "Avoid planting tomatoes near whitefly host plants like beans and cucumbers."
                    ]
                }
            },
            "Mosaic Virus": {
                cause: "Several viruses, including Tobacco Mosaic Virus (TMV) and Tomato Mosaic Virus (ToMV), often transmitted by aphids, contaminated tools, or plant material.",
                symptoms: "Mottled or mosaic-like patterns on leaves, distorted leaf growth, and sometimes stunted plant growth.",
                treatment: {
                    culturalPractices: [
                        "Remove and destroy infected plants as soon as symptoms are noticed.",
                        "Sterilize tools and equipment regularly to prevent the spread of the virus.",
                        "Avoid handling plants when they are wet, as the virus can spread more easily."
                    ],
                    chemicalControl: [
                        "There are no chemical treatments for viral infections.",
                        "Focus on controlling aphid vectors with insecticidal soaps or neem oil."
                    ],
                    prevention: [
                        "Use resistant varieties where available.",
                        "Avoid smoking near tomato plants and wash hands thoroughly before handling them, as TMV can be transmitted through tobacco products.",
                        "Practice good crop rotation and avoid planting in contaminated soil."
                    ]
                }
            },
        },
        Potato: {
            "Early Blight": {
                cause: "Alternaria solani (a fungus)",
                symptoms: "Small, dark brown or black lesions with concentric rings on the leaves, typically appearing on older foliage first.",
                treatment: {
                    culturalPractices: [
                        "Use disease-free seed potatoes.",
                        "Practice crop rotation, avoiding planting potatoes or related crops in the same area for at least two years.",
                        "Remove and destroy infected plant debris to reduce the source of the fungus.",
                        "Water plants at the base to avoid wetting foliage, as wet conditions promote fungal growth."
                    ],
                    chemicalControl: [
                        "Apply fungicides such as chlorothalonil, mancozeb, or copper-based products.",
                        "Begin fungicide applications early in the season, especially during periods of wet or humid weather.",
                        "Repeat treatments every 7-14 days, depending on weather conditions and disease pressure."
                    ],
                    prevention: [
                        "Plant resistant or tolerant potato varieties if available.",
                        "Ensure proper plant spacing to improve air circulation and reduce humidity around the plants."
                    ]
                }
            },
            "Late Blight": {
                cause: "Phytophthora infestans (a fungus-like organism)",
                symptoms: "Large, irregular, water-soaked, and grayish-green spots on leaves.",
                treatment: {
                    culturalPractices: [
                        "Immediately remove and destroy infected plants to prevent the spread of the disease.",
                        "Avoid overhead watering and ensure adequate spacing between plants to improve air circulation.",
                        "Mulch around the base of plants to prevent soil splashing."
                    ],
                    chemicalControl: [
                        "Apply fungicides such as chlorothalonil, mancozeb, or copper-based products as soon as symptoms are detected.",
                        "Regular applications every 7-10 days are essential, particularly in cool, wet weather.",
                        "Use systemic fungicides for better protection during high-risk periods."
                    ],
                    prevention: [
                        "Plant late blight-resistant potato varieties.",
                        "Avoid planting potatoes near tomatoes, as they can both host late blight.",
                        "Practice good crop rotation and remove all plant debris at the end of the growing season."
                    ]
                }
            },
        },
        Pepper: {
            "Bacterial Spot": {
                cause: "Xanthomonas campestris pv. vesicatoria (a bacteria)",
                symptoms: "Small, water-soaked spots on leaves, which may enlarge and become dark and scabby.",
                treatment: {
                    culturalPractices: [
                        "Start with disease-free seeds or transplants to prevent introducing the bacteria into your garden.",
                        "Avoid overhead watering to minimize leaf wetness, which can promote bacterial spread.",
                        "Remove and destroy infected plant debris to reduce the source of the bacteria.",
                        "Rotate crops every 2-3 years with non-host plants (e.g., corn, beans) to prevent buildup of the bacteria in the soil."
                    ],
                    chemicalControl: [
                        "Apply copper-based bactericides regularly, starting as soon as symptoms appear.",
                        "Combine copper with mancozeb or other fungicides to enhance effectiveness, especially during wet conditions."
                    ],
                    prevention: [
                        "Plant resistant varieties of bell peppers if available.",
                        "Keep the garden area weed-free to reduce alternate hosts that could harbor the bacteria.",
                        "Practice good garden hygiene, including disinfecting tools and avoiding working with plants when they are wet."
                    ]
                }
            },
        },
    };

    // Normalize disease name to match keys (capitalize the first letter of each word)
    const normalizedDisease = disease
        .toLowerCase()
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');

    return details[category]?.[normalizedDisease] || {
        cause: "Unknown",
        symptoms: "No symptoms available",
        treatment: {
            culturalPractices: ["No cultural practices available"],
            chemicalControl: ["No chemical control available"],
            biologicalControl: [],
            prevention: ["No prevention steps available"]
        }
    };
};

const AgroPlantDetails = () => {
    const location = useLocation();

    const [plant, setPlant] = useState({
        id: 1,
        plantName: 'Unknown',
        category: 'Unknown',
        imageUrl: 'https://via.placeholder.com/300',
        disease: 'Unknown',
        details: 'No details available',
        cure: 'No cure available',
        cause: 'Unknown',
        symptoms: 'No symptoms available',
        culturalPractices: [],
        chemicalControl: [],
        biologicalControl: [],
        prevention: [],
    });

    useEffect(() => {
        if (location.state) {
            const diseaseDetails = getPlantDiseaseDetails(location.state.category, location.state.disease);
            setPlant({
                id: location.state.id,
                plantName: location.state.category || 'Unknown',
                category: location.state.category || 'Unknown',
                imageUrl: location.state.imageUrl || 'https://via.placeholder.com/300',
                disease: location.state.disease || 'Unknown',
                details: location.state.details || 'No details available',
                cure: location.state.cure || 'No cure available',
                cause: diseaseDetails.cause,
                symptoms: diseaseDetails.symptoms,
                culturalPractices: diseaseDetails.treatment.culturalPractices || [],
                chemicalControl: diseaseDetails.treatment.chemicalControl || [],
                biologicalControl: diseaseDetails.treatment.biologicalControl || [],
                prevention: diseaseDetails.treatment.prevention || [],
            });
        }
    }, [location.state]);

    const nav=useNavigate();
    const handleBackClick = () => {
        nav('/plant-check');
        console.log('Back button clicked');
    };

    const formatConfidence = (confidenceString) => {
        // Check if confidenceString is defined and contains a valid number
        if (!confidenceString || !confidenceString.match(/[\d.]+/)) {
            return "0.0000"; // Return 0 formatted as 4 decimal places
        }
    
        // Extract the numeric part from the string
        const confidenceValue = parseFloat(confidenceString.match(/[\d.]+/)[0]);
    
        // Multiply by 100 and format to 4 decimal places
        const formattedValue = (confidenceValue * 100).toFixed(4);
    
        return formattedValue;
    };

    return (
        <>
        <Header />
        <button className="plant-back-button" onClick={handleBackClick}>← Back</button>
        <img src={plant.imageUrl} alt={plant.plantName} className="plant-image" />
        <table className="plant-details-table">
            <tbody>
                {plant.category && (
                    <tr>
                        <td className="plant-detail-label"><strong>Category:</strong></td>
                        <td className="plant-detail-value">{plant.category}</td>
                    </tr>
                )}
                {plant.disease && (
                    <tr>
                        <td className="plant-detail-label"><strong>Disease:</strong></td>
                        <td className="plant-detail-value">{plant.disease}</td>
                    </tr>
                )}
                {plant.cure && (
                    <tr>
                        <td className="plant-detail-label"><strong>Cure:</strong></td>
                        <td className="plant-detail-value">{plant.cure}</td>
                    </tr>
                )}
                {plant.details && (
                    <tr>
                        <td className="plant-detail-label"><strong>Extra Details:</strong></td>
                        <td className="plant-detail-value">{formatConfidence(plant.details)}</td>
                    </tr>
                )}
                {plant.cause && (
                    <tr>
                        <td className="plant-detail-label"><strong>Cause:</strong></td>
                        <td className="plant-detail-value">{plant.cause}</td>
                    </tr>
                )}
                {plant.symptoms && (
                    <tr>
                        <td className="plant-detail-label"><strong>Symptoms:</strong></td>
                        <td className="plant-detail-value">{plant.symptoms}</td>
                    </tr>
                )}
                {plant.culturalPractices.length > 0 && (
                    <tr>
                        <td className="plant-detail-label"><strong>Cultural Practices:</strong></td>
                        <td className="plant-detail-value">
                            <ul>
                                {plant.culturalPractices.map((practice, index) => (
                                    <li key={index}>{practice}</li>
                                ))}
                            </ul>
                        </td>
                    </tr>
                )}
                {plant.chemicalControl.length > 0 && (
                    <tr>
                        <td className="plant-detail-label"><strong>Chemical Control:</strong></td>
                        <td className="plant-detail-value">
                            <ul>
                                {plant.chemicalControl.map((control, index) => (
                                    <li key={index}>{control}</li>
                                ))}
                            </ul>
                        </td>
                    </tr>
                )}
                {plant.biologicalControl.length > 0 && (
                    <tr>
                        <td className="plant-detail-label"><strong>Biological Control:</strong></td>
                        <td className="plant-detail-value">
                            <ul>
                                {plant.biologicalControl.map((control, index) => (
                                    <li key={index}>{control}</li>
                                ))}
                            </ul>
                        </td>
                    </tr>
                )}
                {plant.prevention.length > 0 && (
                    <tr>
                        <td className="plant-detail-label"><strong>Prevention:</strong></td>
                        <td className="plant-detail-value">
                            <ul>
                                {plant.prevention.map((step, index) => (
                                    <li key={index}>{step}</li>
                                ))}
                            </ul>
                        </td>
                    </tr>
                )}
            </tbody>
        </table>
    </>
    
    );
};

export default AgroPlantDetails;
