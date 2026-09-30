const mongoose = require("mongoose");

const disasterReportSchema = new mongoose.Schema(
    {
        imageName: {
            type: String,
            required: true
        },

        disasterType: {
            type: String,
            required: true
        },

        peopleVisible: {
            type: String,
            required: true
        },

        infrastructureDamage: {
            type: String,
            required: true
        },

        roadPassable: {
            type: String,
            required: true
        },

        severity: {
            type: String,
            required: true
        },

        priority: {
            type: String,
            required: true
        },

        hazards: {
            type: [String],
            default: []
        },

        triageReport: {
            type: String,
            required: true
        }
    },
    {
        timestamps: true
    }
);

const DisasterReport = mongoose.model(
    "DisasterReport",
    disasterReportSchema
);

module.exports = DisasterReport;