import { Route, Routes } from "react-router-dom";
import { ToastContainer } from "react-toastify";
import AgroHome from "./components/AgroHome";
import AgroLogin from "./components/AgroLogin";
import AgroRegister from "./components/AgroRegister";
import AgroPlantCheck from "./components/AgroPlantCheck";
import AgroPlantDetailsTable from "./components/AgroPlantDetailsTable";
import AgroPlantDetails from "./components/AgroPlantDetails";
import AboutUs from "./components/AboutUs";

function App() {
  return (
    <>
    <ToastContainer/>
      <Routes>
        <Route exact path="/" element={<AgroHome/>} />
        <Route exact path="/login" element={<AgroLogin/>} />
        <Route exact path="/register" element={<AgroRegister/>} />
        <Route exact path="/plant-check" element={<AgroPlantCheck/>} />
        <Route exact path="/plant-details-table" element={<AgroPlantDetailsTable/>} />
        <Route exact path="/plant-details" element={<AgroPlantDetails/>} />
        <Route exact path="/about" element={<AboutUs/>} />
      </Routes>
    </>
  );
}

export default App;
