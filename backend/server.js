const runAI = require("./ai/aiService");

const express = require("express");

const multer = require("multer");

const mongoose = require("mongoose");

const connectDB = require("./config/database");

const DisasterReport = require("./models/DisasterReport");

const cloudinary = require("./config/cloudinary");

const app = express();


// Multer storage configuration
const storage = multer.diskStorage({

    destination: "uploads/",

    filename: (req, file, cb) => {

        cb(null, Date.now() + "-" + file.originalname);

    }

});

const upload = multer({ storage });

connectDB();


// Use Render's PORT when deployed
const PORT = process.env.PORT || 5000;

app.use(express.json());


// Health check
app.get("/api/health", (req, res) => {

    res.json({

        success: true,

        message: "DisasterVisionAI backend is running"

    });

});


// Analyze disaster images
app.post("/api/analyze", upload.array("images", 10), async (req, res) => {

    try {

        // Check whether images were uploaded
        if (!req.files || req.files.length === 0) {

            return res.status(400).json({

                success: false,

                message: "Please upload at least one image"

            });

        }


        // Upload images to Cloudinary
        const cloudinaryResults = await Promise.all(

            req.files.map(file =>

                cloudinary.uploader.upload(file.path, {

                    folder: "disasterVisionAI"

                })

            )

        );


        // Store Cloudinary URLs
        const imageName = cloudinaryResults

            .map(result => result.secure_url)

            .join(", ");


        // Run restoration + VLM analysis one image at a time
        const aiResults = [];

        for (const file of req.files) {

            const result = await runAI(file.path);

            aiResults.push(result);

        }


        // Combine VLM results
        const triageReport = aiResults

            .map(result => result.message)

            .join("\n\n");


        // Split numbered VLM answers
        const answer1Match = triageReport.match(

            /1\.\s*(.*?)(?=\s*2\.|$)/is

        );

        const answer2Match = triageReport.match(

            /2\.\s*(.*?)(?=\s*3\.|$)/is

        );

        const answer3Match = triageReport.match(

            /3\.\s*(.*?)(?=\s*4\.|$)/is

        );

        const answer4Match = triageReport.match(

            /4\.\s*(.*?)(?=\s*5\.|$)/is

        );

        const answer5Match = triageReport.match(

            /5\.\s*(.*)$/is

        );


        // Extract values
        const disasterType = answer1Match

            ? answer1Match[1].trim()

            : "Unknown";

        const peopleVisible = answer2Match

            ? answer2Match[1].trim()

            : "Unclear";

        const infrastructureDamage = answer3Match

            ? answer3Match[1].trim()

            : "Unclear";

        const roadPassable = answer4Match

            ? answer4Match[1].trim()

            : "Unclear";

        const majorHazards = answer5Match

            ? answer5Match[1].trim()

            : "No major hazards reported";


        const result = {

            // Cloudinary image URL
            imageName: imageName,

            disasterType: disasterType,

            peopleVisible: peopleVisible,

            infrastructureDamage: infrastructureDamage,

            roadPassable: roadPassable,

            // Existing fields
            severity: "High",

            priority: "Urgent",

            hazards: [

                majorHazards

            ],

            // Store complete VLM response
            triageReport: triageReport

        };


        // Save report to MongoDB
        const report = await DisasterReport.create(result);


        res.status(201).json({

            success: true,

            message: "Analysis saved successfully",

            reportId: report._id,

            report: report

        });


    } catch (error) {

        console.error("Analysis error:", error.message);

        res.status(500).json({

            success: false,

            message: "Failed to save analysis",

            error: error.message

        });

    }

});


// Get all disaster reports
app.get("/api/reports", async (req, res) => {

    try {

        const reports = await DisasterReport

            .find()

            .sort({ createdAt: -1 });


        res.json({

            success: true,

            reports: reports

        });

    } catch (error) {

        console.error("Reports error:", error.message);

        res.status(500).json({

            success: false,

            message: "Failed to fetch reports"

        });

    }

});


// Get one disaster report by ID
app.get("/api/reports/:id", async (req, res) => {

    try {

        // Check whether the MongoDB ID is valid
        if (!mongoose.Types.ObjectId.isValid(req.params.id)) {

            return res.status(400).json({

                success: false,

                message: "Invalid report ID"

            });

        }


        const report = await DisasterReport.findById(req.params.id);


        if (!report) {

            return res.status(404).json({

                success: false,

                message: "Report not found"

            });

        }


        res.json({

            success: true,

            report: report

        });

    } catch (error) {

        console.error("Report error:", error.message);

        res.status(500).json({

            success: false,

            message: "Failed to fetch report"

        });

    }

});


// Start server
app.listen(PORT, "0.0.0.0", () => {

    console.log(`Server running on port ${PORT}`);

});