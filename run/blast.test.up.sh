docker exec -it blast-app_dev-1 bash -c 'coverage run manage.py test host.tests -v 2 &&
coverage report -i --omit=host/tests/*,host/migrations/*,app/*,host/urls.py,host/admin.py,host/apps.py,host/__init__.py,manage.py'
