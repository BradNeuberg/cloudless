Setup
-----------
    mkvirtualenv annotate
    cd src/annotate
    pip install -r requirements.txt
    ,/manage.py migrate
    ./manage.py runserver
    echo 'SECRET PLANET LABS KEY' >> $VIRTUAL_ENV/bin/postactivate
