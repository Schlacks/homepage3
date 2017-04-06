from django.shortcuts import render

from django.shortcuts import render_to_response
from django.template import RequestContext
from django.http import HttpResponse
from bayes.models import InputForm, Sample_InputForm,Predict_InputForm
from bayes.compute import gauss_bandit, pull_and_update#,initiate_gauss
import os


def index(request):
    #return render ( request, 'C:/Users/Jan/PycharmProjects/Newversion/untitled/online_comp/templates/bayes/index.html')
    return render ( request, 'bayes/index.html')


def intro(request):
    return render(request,'bayes/intro.html')

def create_and_update(request):

    global bandit
    global invgammaresult
    global predictresult
    samples=None
    prob_statement=None

    if request.method == 'POST' and 'button1' in request.POST:
        form = InputForm(request.POST)
        if form.is_valid():
            form2 = form.save(commit=False)
            bandit = gauss_bandit(form2.m, abs(form2.s))
            invgammaresult=bandit.first_img()
            invgammaresult=invgammaresult[7:]
            form3= Sample_InputForm()
            form4=Predict_InputForm()

            predictresult=None


    elif request.method == 'POST' and 'button2' in request.POST:
        form3 = Sample_InputForm(request.POST)
        if form3.is_valid():
            form2 = form3.save(commit=False)
            invgammaresult,outcome=pull_and_update(bandit,form2.n)
            invgammaresult=invgammaresult[7:]
            form = InputForm(request.POST)
            form4=Predict_InputForm()
            samples=outcome
            if len(samples)>100:
                samples=samples[:100]

    else:
        form = InputForm()
        form3= Sample_InputForm()
        form4=Predict_InputForm
        invgammaresult = None
        predictresult = None

    return render ( request, 'bayes/create_and_update.html',
            {'form': form,
             'result': invgammaresult,
             'form3':form3,
             'form4':form4,
             'predictresult':predictresult,
             'samples':samples,
             })

def predict(request):
    prob_statement=None

    if request.method == 'POST' and 'button3' in request.POST:
        form4 = Predict_InputForm(request.POST)
        if form4.is_valid():
            form2 = form4.save(commit=False)
            try:

                predictresult,prob_statement=bandit.predict(form2.p1,form2.p2)
                prob_statement=str(prob_statement)
                predictresult=predictresult[7:]
                form = InputForm(request.POST)
                form3= Sample_InputForm()
            except:
                prob_statement='First you need to generate a distribution. Please go back to "Create and Update". '
                predictresult = None
                print(prob_statement)
    else:
        form = InputForm()
        form3= Sample_InputForm()
        form4=Predict_InputForm
        invgammaresult = None
        predictresult = None

    return render ( request, 'bayes/predict.html',
            {'form4':form4,
             'predictresult':predictresult,

             'prob_statement':prob_statement
             })
