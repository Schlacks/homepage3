from django.db import models
from django.forms import ModelForm
from django import forms


class Input(models.Model):
    m = models.FloatField(
        verbose_name=' Mean of the Normal Distribution: ', default=0.0)
    s = models.FloatField(
        verbose_name=' Standard Deviation of the Normal Distribution (strictly positive):   ', default=1.0)

class InputForm(ModelForm):


    class Meta:
        model = Input
        fields='__all__'



class Sample_Input(models.Model):
    n = models.FloatField(
        verbose_name=' Number of samples you wish to draw:', default=1)

class Sample_InputForm(ModelForm):
    class Meta:
        model=Sample_Input
        fields='__all__'

class Predict_Input(models.Model):
    p1 = models.FloatField(
        verbose_name=' Lower bound:  ', default=-1.0)
    p2 = models.FloatField(
        verbose_name=' Upper bound:  ', default=1.0)

class Predict_InputForm(ModelForm):
    class Meta:
        model=Predict_Input
        fields='__all__'
