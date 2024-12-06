document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('form');
    const loadingIndicator = document.getElementById('loading');
    const resultsContainer = document.getElementById('results-container');
    const imageQueryInput = document.getElementById('image_query');
    const originalImageContainer = document.getElementById('original-image-container');
    const imagePreview = document.getElementById('image-preview');
    const resultsHeader = document.getElementById('results-header');
    const originalImageHeader = document.getElementById('original-image-header')
    const queryTypeSelect = document.getElementById('query-type');

    // Function to show the image preview when a user uploads an image
    function previewImage() {
        const file = imageQueryInput.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                originalImageContainer.style.display = 'block';  // Show the preview container

                // Update the results header to indicate "Original Image" is displayed
                originalImageHeader.innerHTML = '<h2>Original Image</h2>';
            };
            reader.readAsDataURL(file);
        }
    }

    // Listen for the file input change to preview the image
    imageQueryInput.addEventListener('change', previewImage);

    // Function to manage visibility of the original image container based on query type
    function manageOriginalImageDisplay(queryType) {
        if (queryType === 'text') {
            // Hide original image for text queries
            originalImageContainer.style.display = 'none';
            originalImageHeader.style.display = 'none';
        } else {
            // Show original image if the query type is image or hybrid
            if (imageQueryInput.files.length > 0) {
                originalImageContainer.style.display = 'block';
                originalImageHeader.style.display = 'block';
            }
        }
    }

    // Listen for changes in the query type selection
    queryTypeSelect.addEventListener('change', (e) => {
        manageOriginalImageDisplay(e.target.value);
    });

    // Listen for the file input change to preview the image
    imageQueryInput.addEventListener('change', previewImage);

    // Form submit event listener
    form.addEventListener('submit', (e) => {
        e.preventDefault(); // Prevents the form from refreshing the page

        loadingIndicator.style.display = 'block'; // Show loading indicator
        
        const formData = new FormData(form);
        const selectedQueryType = queryTypeSelect.value;

        // Manage visibility of original image based on query type
        manageOriginalImageDisplay(selectedQueryType);

        // Send the form data to the server via Fetch API (POST method)
        fetch('/search', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json()) // Assuming server sends back JSON
        .then(data => {
            // Hide loading indicator after receiving response
            loadingIndicator.style.display = 'none';
            
            // Clear the previous results
            resultsContainer.innerHTML = '';

            // Change the results header to show "Results"
            resultsHeader.style.display = 'block'; // Show the results header
            resultsHeader.innerHTML = '<h2>Results</h2>';
            
            // Check if results exist
            if (data.results && data.results.length > 0) {
                // Loop through and display images and their similarity scores
                data.results.forEach((result, index) => {
                    const resultDiv = document.createElement('div');
                    resultDiv.classList.add('result');
                    
                    // Create image element
                    const image = document.createElement('img');
                    image.src = result.image_url;

                    // Determine the label based on the query type
                    const labelText = selectedQueryType === 'pca' ? 'Distance' : 'Score';

                    // Handle image load
                    image.onload = () => {
                        const score = document.createElement('p');
                        // Display Distance for PCA or Score for others
                        score.textContent = `${labelText}: ${selectedQueryType === 'pca' ? result.distance.toFixed(2) : result.similarity_score.toFixed(2)}`;
                        
                        resultDiv.appendChild(image);
                        resultDiv.appendChild(score);

                        // Add the result to the container
                        resultsContainer.appendChild(resultDiv);
                    };

                    // Optional: Handle image error
                    image.onerror = () => {
                        resultsContainer.innerHTML = 'Error loading image.';
                    };
                });
            } else {
                resultsContainer.innerHTML = 'No results found.';
            }
        })
        .catch(error => {
            loadingIndicator.style.display = 'none';
            console.error('Error fetching search results:', error);
        });
    });
});
