# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from apps.home import blueprint
from flask import render_template, request, redirect, url_for
from flask_login import login_required
from jinja2 import TemplateNotFound
from flask_login import current_user
from apps.video_feed.dbmodels import YogaPoseData, YogaSession, HeartRateData

from apps.config import API_GENERATOR


# In your home/routes.py

@blueprint.route('/interface')
@login_required
def interface():
    default_pose = {
        'name': 'Yoga Pose',
        'pose_key': 'default',
        'is_last_pose': False
    }
    
    return render_template('home/interface.html', 
                         segment='interface', 
                         show_sideBar=True,
                         pose=default_pose,
                         )  # Add this parameter
                         
@blueprint.route('/profile')
@login_required
def profile():
    bpm_data = HeartRateData.query.filter_by(user_id=current_user.id).all() or []

    # Get the most recent yoga session and poses data
    last_session = YogaSession.query.filter_by(user_id=current_user.id)\
        .order_by(YogaSession.end_time.desc()).first()

    poses_data = []
    total_calories = 0
    if last_session:
        poses_data = YogaPoseData.query.filter_by(session_id=last_session.id).all()
        total_calories = last_session.total_calories

    return render_template(
        'home/profile.html',
        segment="profile",
        bpm_data=bpm_data,
        show_sideBar=True,
        total_calories=total_calories,
        poses_data=poses_data
    )

@blueprint.route('/last_session')
@login_required
def last_session():
    # Redirect to profile since we're combining the functionality
    return redirect(url_for('home_blueprint.profile'))
@blueprint.route('/index')
@login_required
def index():
    pose_data = [
        {"name": "Tree Pose (Vrikshasana)", "level": "Beginner", "color": "#63e5ff", "pose_key": "Tree", "next_pose_key": "Done"},
        {"name": "Chair Pose (Bhujangasana)", "level": "Beginner", "color": "#00fc7e", "pose_key": "Chair", "next_pose_key": "Tree"},
        {"name": "Cobra Pose (Bhujangasana)", "level": "Intermediate", "color": "#00fc7e", "pose_key": "Cobra", "next_pose_key": "Chair"},
    ]
    try:
        return render_template('home/index.html', segment='index', API_GENERATOR=len(API_GENERATOR), show_sideBar=True, poses=pose_data)
    except Exception as e:
        print(f"Error in rendering home: {e}")
        return render_template('home/page-500.html'), 500

@blueprint.route('/<template>')
@login_required
def route_template(template):
    try:
        if not template.endswith('.html'):
            template += '.html'

        segment = get_segment(request)

        if segment == 'interface':
            return render_template('home/interface.html', segment='interface', show_sideBar=True)
        elif segment != 'index':
            return render_template("home/" + template, segment=segment, API_GENERATOR=len(API_GENERATOR), show_sideBar=True)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404
    except:
        return render_template('home/page-500.html'), 500

def get_segment(request):
    try:
        segment = request.path.split('/')[-1]
        return segment if segment else 'index'
    except:
        return None