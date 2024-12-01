from django.forms import ModelForm

from .models import CriminalTable, FireTable, RegisterTable


class RegForm(ModelForm):
    class Meta:
        model = RegisterTable
        fields = ['Username', 'Password','Name','Dob']


class crimeform(ModelForm):

    class Meta:
        model=CriminalTable
        fields = ['Criminalname','Image','Type','Details','Address']

class fireform(ModelForm):
    class Meta:
        model=FireTable
        fields =['Name','Address','Phone','Email','Location']
        
        
