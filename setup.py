from setuptools import setup

setup(
    setup_requires=['setuptools_scm'],
    name='swin_server',
    entry_points={
        'console_scripts': [
            'swin-server=swin_segmentation_server.serve:serve'
        ],
    },
    install_requires=[
        "click",
        "click_completion",
        "logbook",
        "flask",
        "opencv-python",
    ],
)
