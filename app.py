from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from assignment10 import image_to_image, text_to_image, hybrid_query, pca_first_k, clear_results_folder
from PIL import Image
import time

app = Flask(__name__)

# Path to store result images
RESULTS_FOLDER = os.path.abspath('./results')
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER

# Ensure the results folder exists
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Main route to render the home page
@app.route('/')
def home():
    # Serve the main HTML page
    return render_template('index.html')

# Route to handle search (POST request)
@app.route('/search', methods=['POST'])
def search():
    clear_results_folder()

    query_type = request.form.get('query_type')
    text_query = request.form.get('text_query')
    image_file = request.files.get('image_query')
    weight = request.form.get('weight', type=float)
    k_components = request.form.get('k_components', type=int)
    
    results = []
    print(f"Query Type: {query_type}")

    # Save the uploaded image to the results folder if an image is provided
    if image_file and image_file.filename:
        image_filename = os.path.join(RESULTS_FOLDER, image_file.filename)
        image_file.save(image_filename)

    # Handle the search logic based on the selected query type
    if query_type == 'hybrid' and text_query and image_file and weight:
        results = hybrid_query(image_filename, text_query, weight)
    elif query_type == 'text' and text_query:
        results = text_to_image(text_query)
    elif query_type == 'image' and image_file:
        results = image_to_image(image_filename)
    elif query_type == 'pca' and image_file and k_components:
        results = pca_first_k(image_filename, k_components)

    print(f"Results: {results}")

    # Modify the image URL to include a unique query parameter to prevent caching
    for result in results:
        result['image_url'] = f"{result['image_url']}?timestamp={int(time.time())}"

    for result in results:
        if query_type != 'pca':
            result['similarity_score'] = float(result['similarity_score'])
        else:
            result['distance'] = float(result['distance'])

    # Return the JSON response
    return jsonify({'results': results})


# Route to serve result images (after search)
@app.route('/results/<filename>')
def results(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True, port=3000)

