from django.contrib import admin
from django.urls import path

from .views import *

urlpatterns = [
    path('add_missing', add_missing.as_view(), name="add_missing12"),
    path('registration',registration.as_view(),name="registration"),
    path('add_new_criminal', add_new_criminal.as_view(), name="add_new_criminal"),
    path('complaint',complaint.as_view(),name="complaint"),
    path('view_criminal',view_criminal.as_view(),name="view_criminal"),
    path('delete_criminal/<int:c_id>', delete_c.as_view(), name="delete_c"),
    path('view_user',view_user.as_view(),name="view_user"),
    path('delete_u/<int:u_id>', delete_u.as_view(), name="delete_u"),
    path('edit_criminal/<int:id>',Edit_criminal.as_view(),name='Edit_criminal'),
    path('add_fire',add_fire.as_view(),name="add_fire"),
    path('view_fire',view_fire.as_view(),name="view_fire"),
    path('delete_fire/<int:f_id>', delete_f.as_view(), name="delete_f"),
    path('edit_fire/<int:id>',Edit_fire.as_view(),name='Edit_fire'),
    
]
