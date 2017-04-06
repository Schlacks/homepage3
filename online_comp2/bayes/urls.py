from django.conf.urls import url
from bayes import views

app_name = 'basic_app'

urlpatterns=[

    url(r'^intro',views.intro,name='intro'),
    url(r'^create_and_update',views.create_and_update,name='create_and_update'),
    url(r'^predict',views.predict,name='predict'),

]
