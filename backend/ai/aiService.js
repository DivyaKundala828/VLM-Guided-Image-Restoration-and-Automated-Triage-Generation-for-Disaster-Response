const { spawn } = require("child_process");
const path = require("path");

const runPython = (script, imagePath) => {
    return new Promise((resolve, reject) => {

        const pythonProcess = spawn(
            "python",
            [script, imagePath]
        );

        let output = "";
        let error = "";

        pythonProcess.stdout.on("data", (data) => {
            output += data.toString();
        });

        pythonProcess.stderr.on("data", (data) => {
            error += data.toString();
        });

        pythonProcess.on("close", (code) => {

            if (code !== 0) {
                reject(new Error(error));
                return;
            }

            resolve(output.trim());
        });
    });
};


const runAI = async (imagePath) => {

    // Step 1: Restore image
    const restorationOutput = await runPython(
        "ai/ai_pipeline.py/restore_image.py",
        imagePath
    );

    console.log("Image restoration completed.");


    // Step 2: Find restored image
    const imageName = path.basename(imagePath);
    const imageStem = path.parse(imageName).name;

    const restoredImagePath = path.join(
        "restored_images",
        `restored_${imageStem}.png`
    );


    // Step 3: Run SmolVLM on restored image
    const vlmOutput = await runPython(
        "ai/ai_pipeline.py/vlm_analysis.py",
        restoredImagePath
    );


    console.log("VLM analysis completed.");


    return {
        message: vlmOutput
    };
};


module.exports = runAI;