from flask import Blueprint

blueprint = Blueprint(
    'video_feed_blueprint',
    __name__,
    url_prefix=''
)
# In your __init__.py or where you create your Flask app
# from flask import Flask
# from .video_feed_blueprint import video_feed_blueprint

# def create_app():
#     app = Flask(__name__)
#     # ... other app configurations ...
    
#     app.register_blueprint(video_feed_blueprint, url_prefix='/video')
    
#     return app