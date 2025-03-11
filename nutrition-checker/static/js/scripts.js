document.getElementById("uploadForm").addEventListener("submit", async (event) => {
    event.preventDefault();

    const imageInput = document.getElementById("imageInput");
    const resultsDiv = document.getElementById("results");
    const nutritionDiv = document.getElementById("nutrition");
    const validationDiv = document.getElementById("validation");

    if (!imageInput.files[0]) {
        alert("Please select an image!");
        return;
    }

    // Prepare the form data
    const formData = new FormData();
    formData.append("image", imageInput.files[0]);

    try {
        // Send the image to the backend
        const response = await fetch("/upload", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Failed to upload and analyze the image.");
        }

        const data = await response.json();

        // Display results
        resultsDiv.classList.remove("hidden");
        nutritionDiv.innerHTML = "<h3>Nutritional Values:</h3>";
        validationDiv.innerHTML = "<h3>Validation Results:</h3>";

        // Show nutritional values
        for (const [key, value] of Object.entries(data.nutrition)) {
            nutritionDiv.innerHTML += `<p><strong>${key}:</strong> ${value} g</p>`;
        }

        // Show validation results
        for (const [key, isValid] of Object.entries(data.validation)) {
            const status = isValid ? "✅ Acceptable" : "❌ Exceeds limit";
            validationDiv.innerHTML += `<p><strong>${key}:</strong> ${status}</p>`;
        }
    } catch (error) {
        alert("Error: " + error.message);
    }
});
