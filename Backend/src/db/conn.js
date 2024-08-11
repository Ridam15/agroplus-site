const mongoose = require("mongoose");

const connectToMongo = async () => {
    try {
        // mongodb+srv://drashtidankhara7:<password>@cluster0.jcwxk8q.mongodb.net/test
        await mongoose.connect("mongodb+srv://tulsirathod9102:TDC5Nux07N2ofruy@atom.s3kvfrj.mongodb.net/?retryWrites=true&w=majority&appName=Atom");
        console.log("Connected To AgroPlus !!!");
    } catch (error) {
        console.log("Error in connection : ", error);
    }
}

module.exports = connectToMongo;