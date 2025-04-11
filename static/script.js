document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const formData = new FormData();
    const imageFile = document.getElementById('imageInput').files[0];
    formData.append('image', imageFile);
  
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });
  
    const data = await response.json();
    document.getElementById('prediction').innerText = data.prediction;
  });
  