// Preview selected image
document.getElementById('file-input').addEventListener('change', function () {
    const file = this.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            const preview = document.getElementById('preview');
            preview.src = e.target.result;
            document.getElementById('preview-wrapper').style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

// Handle form submission
document.getElementById('upload-form').addEventListener('submit', async function (e) {
    e.preventDefault();
    const formData = new FormData();
    const file = document.getElementById('file-input').files[0];
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        if (result.prediction) {
            document.getElementById('result').innerText = `Prediction: ${result.prediction}`;
        } else {
            document.getElementById('result').innerText = `Error: ${result.error}`;
        }
    } catch (error) {
        document.getElementById('result').innerText = 'Prediction failed. Try again.';
    }
});
