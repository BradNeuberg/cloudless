Setup
-----------
    brew install gdal
    Install virtualenv and virtualenvwrapper: https://jamie.curle.io/posts/installing-pip-virtualenv-and-virtualenvwrapper-on-os-x/
    mkvirtualenv annotate-django
    cd src/annotate
    pip install -r requirements.txt
    ./manage.py migrate
    echo 'PLANET_KEY="SECRET PLANET LABS KEY"' >> $VIRTUAL_ENV/bin/postactivate

Importing imagery
-------------------
1. Choose your lat/lng and buffer distance (meters) you want (this example is for San Fran) and the directory to download to, then run:

    python train/scripts/download_planetlabs.py 37.796105 -122.461349 --buffer 200 --dir ../../data/planetlab/images/

2. Chop up these raw images into 512x512 pixels and add them to the database

   ./manage.py runscript populate_db --script-args ../../data/planetlab/images/ 512

Annotating imagery
--------------------
1. Start the server running:

    ./manage.py runserver

2. Go to http://127.0.0.1:8000/train/annotate

3. Draw bounding boxes on the image.  Note - the checkboxes are ignored.  Max believes they are superfluous.

4. Hit the "Done" button to submit results to the server

5. Upon successful submission, the browser will load a new image to annotate

NOTE: If the image shows up as all white - that's an artifact of us not doing
a good job of eliminating blank image chunks.  Just reload the browser to get a new image

Exporting annotated imagery
-----------------------------
1. Writes out annotated.json and all the annotated images to a specified directory

    ./manage.py runscript export --script-args /Users/max/Desktop/annotated
