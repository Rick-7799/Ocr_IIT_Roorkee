# README

## Overview

This project is an Optical Character Recognition (OCR) web application that utilizes state-of-the-art technologies to recognize and extract text from images. Built using the **ColPali** implementation from the **Byaldi** library and the **Hugging Face Transformers** for the **Qwen2-VL** model, this application provides a user-friendly interface for uploading images, performing OCR, and searching for specific keywords within the recognized text.

### Tech Stack

#### 1. **Gradio**
Gradio is a Python library that allows you to quickly create web-based user interfaces for machine learning models. It simplifies the deployment process by providing a straightforward way to build interactive demos. With Gradio, users can upload files, input text, and view outputs directly in their web browser without needing extensive frontend development skills.

#### 2. **Byaldi Library**
The Byaldi library offers advanced implementations of various multi-modal models, including ColPali, which is specifically designed for OCR tasks. ColPali excels at recognizing text in images and can handle various fonts and layouts. By integrating this library, we leverage its capabilities to perform high-quality OCR, making it suitable for diverse applications ranging from document scanning to image analysis.

#### 3. **Hugging Face Transformers**
The Hugging Face Transformers library is a widely-used framework for natural language processing (NLP) and computer vision tasks. It provides pre-trained models that can be fine-tuned for specific tasks or used out-of-the-box. In this project, we utilize the Qwen2-VL model, which is optimized for image and text processing. This model enhances our application's ability to understand context and semantics in both visual and textual data.

#### 4. **Python**
Python serves as the primary programming language for this project due to its rich ecosystem of libraries and frameworks that facilitate machine learning and web development. Its simplicity and readability make it an ideal choice for rapid prototyping and deployment of data-driven applications.

### Application Workflow

1. **Image Upload:** Users can upload images containing text through the Gradio interface.
  
2. **Text Input:** Users can provide additional context or instructions in the form of text.

3. **Keyword Search:** Users can specify a keyword they wish to search for within the recognized text.

4. **OCR Processing:** The application processes the uploaded image using ColPali from the Byaldi library, extracting text content.

5. **Output Display:** The recognized text is displayed along with an indication of whether the specified keyword was found in the output.

### Key Features

- **User-Friendly Interface:** Built with Gradio, allowing seamless interaction without technical barriers.
  
- **High-Quality OCR:** Utilizes ColPali's advanced OCR capabilities to ensure accurate text recognition from images.
  
- **Keyword Search Functionality:** Enables users to quickly find specific information within the recognized text.
  
- **Multi-Modal Processing:** Combines image and text inputs effectively using Qwen2-VL, enhancing contextual understanding.

### Use Cases

This application can be applied in various scenarios such as:

- Digitizing printed documents or books.
- Extracting information from receipts or invoices.
- Analyzing images containing handwritten notes.
- Assisting visually impaired individuals by reading text aloud from images.

## Requirements

To set up the environment and run the application, ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package installer)

## Setting Up the Environment

1. **Clone the Repository**

   Open your terminal and clone this repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a Virtual Environment**

   It is recommended to use a virtual environment to manage dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Packages**

   Install the necessary libraries using pip:

   ```bash
   pip install gradio transformers byaldi
   ```

## Running the Web Application Locally

1. **Run the Application**

   Once all dependencies are installed, you can start the application with:

   ```bash
   python app.py  
   ```

2. **Access the Application**

   After running the command, you should see output indicating that Gradio is running. Open your web browser and navigate to `http://127.0.0.1:7860` to access the application.

3. **Using the Application**

   - Upload an image containing text.
   - Enter any accompanying text if needed.
   - Input a keyword to search for in the OCR output.
   - Click on "Submit" to see the recognized text and whether your keyword was found.

## Deployment Process

To deploy this application, you can choose from various platforms such as Heroku, AWS, or Google Cloud. Below are general steps for deploying on Heroku:

1. **Install Heroku CLI**

   Download and install the Heroku Command Line Interface (CLI) from [Heroku's website](https://devcenter.heroku.com/articles/heroku-cli).

2. **Create a Heroku App**

   In your terminal, log in to Heroku and create a new app:

   ```bash
   heroku login
   heroku create <app-name>
   ```

3. **Prepare for Deployment**

   Create a `requirements.txt` file if it does not exist:

   ```bash
   pip freeze > requirements.txt
   ```

4. **Add a Procfile**

   Create a file named `Procfile` in your project directory with the following content:

   ```
   web: python app.py
   ```

5. **Deploy Your Application**

   Add your changes to git, commit them, and push to Heroku:

   ```bash
   git add .
   git commit -m "Initial deployment"
   git push heroku master
   ```

6. **Open Your App**

   After deployment, open your app in a web browser:

   ```bash
   heroku open
   ```

## Conclusion

By leveraging cutting-edge technologies like Gradio, Byaldi's ColPali, and Hugging Face Transformers, this OCR web application provides an efficient and effective solution for recognizing and extracting text from images. It serves as a powerful tool for both personal and professional use cases, enhancing productivity and accessibility in various fields.

For any issues or further questions, feel free to raise an issue in this repository or reach out for support!
