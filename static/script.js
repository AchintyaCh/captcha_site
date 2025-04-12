document.getElementById('upload-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const formData = new FormData();
    const file = document.getElementById('file-input').files[0];
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status}\n${errorText}`);
        }

        const result = await response.json();
        document.getElementById('result').innerText =
            `Image: ${result.image_name} | Prediction: ${result.prediction}`;
    } catch (error) {
        document.getElementById('result').innerText = `Error: ${error.message}`;
        console.error("Prediction error:", error);
    }
});
