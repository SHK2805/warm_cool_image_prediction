import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from logger.logger import logger
from wc_prediction import WarmCoolPredictor

app = Flask(__name__)


def allowed_images(filename):
    if '.' not in filename:
        return False
    allowed_extensions = ('png', 'jpg', 'jpeg', 'gif')
    return filename.rsplit('.')[-1].lower() in allowed_extensions


app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'images')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maximum file size: 16MB

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


@app.route('/')
def upload():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            raise ValueError('Image input is required in the form')

        file = request.files['image']

        if file.filename == '':
            raise ValueError('No image selected')

        if not allowed_images(file.filename):
            raise ValueError('Invalid image format, allowed formats are - png, jpg, jpeg, gif only')

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if os.path.exists(file_path):
            raise ValueError('Image with the same name already exists')

        file.save(file_path)
        logger.info('Image successfully uploaded')

        wc_predictor = WarmCoolPredictor()
        prediction = wc_predictor.predict_image(file_path)
        message = f"The image {filename} is: {prediction}"
        logger.info(message)
        return render_template('upload.html', message=message)

    except ValueError as e:
        logger.error(e)
        return render_template('upload.html', message=str(e))

    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        return render_template('error.html'), 500


@app.errorhandler(404)
def not_found_error(error):
    logger.error(f'404 error: {error}')
    return render_template('error.html'), 404


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='127.0.0.1', port=8080, debug=True)
