# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:57:11 2022
@author: Till Leissner, Mads Clausen Institute, University of Southern Denmark
@email: till@mci.sdu.dk

    Copyright (C) 2022  Till Leissner

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
#import datetime
import os, fnmatch
import lmfit
from lmfit import Model, Parameter #,Parameters 
#from lmfit.models import ExponentialGaussianModel, ExponentialModel, GaussianModel
import plotly.graph_objs as go
#from ipywidgets import widgets
#from IPython.display import display, clear_output, Image
from plotly.subplots import make_subplots
#import plotly.express as px
#import plotly.tools as tls
#from pandas.plotting import table 
import scipy
from scipy.ndimage.interpolation import shift
from scipy.signal import fftconvolve 


### Helper function
####Import / Export tools


def findFiles (path, filter):
    for root, dirs, files in os.walk(path):
        for file in fnmatch.filter(files, filter):
            yield os.path.join(file)
 
    
 
def file_array(path, filter):
    files = []
    for textFile in findFiles(path, filter):
                files.append(textFile)
    files = sorted(files)
    return files 



def load_datasets(datapath, datafilter):
    data = dict()
    files = file_array(datapath,datafilter)
    for i,file in enumerate(files):
        print(file)
        timestamp = os.path.getmtime(os.path.join(datapath,file))
        trace = pd.read_csv(os.path.join(datapath,file), delim_whitespace= True, header=0, index_col=0)
        data.update({os.path.splitext(file)[0]: {'trace': trace, 'timestamp':timestamp}})
    return data



def init_datasets(data):
    for key in data: 
        trace = data[key]['trace']
        decay = trace.sum(axis=1)
        spectrum = trace.sum(axis=0)
        data[key]['spectrum'], data[key]['decay'] = spectrum, decay
        data[key]['tmin'], data[key]['tmax'], data[key]['dt'] = decay.index[0], decay.index[-1], decay.index[1]-decay.index[0]
        data[key]['lmin'], data[key]['lmax'], data[key]['dl'] = np.float(spectrum.index[0]), np.float(spectrum.index[-1]), np.float(spectrum.index[1])-np.float(spectrum.index[0])
        data[key]['tpeaki'], data[key]['tpeak'] = np.argmax(decay), decay.index[np.argmax(decay)]
    return data



def plot_datasets(data):
    fig = make_subplots(rows=1, cols=3)
    for key in data:
        fig.add_trace(go.Scatter(x=data[key]['spectrum'].keys(), y=data[key]['spectrum'].iloc[:], 
                                  mode='lines',name='Spectrum '+ key),row=1, col=1)
    
    for key in data:
        fig.add_trace(go.Scatter(x=data[key]['decay'].keys(), y=data[key]['decay'].iloc[:],
                                  mode='lines',name='Decay ' + key),row=1, col=2)
    
    fig.add_trace(go.Heatmap(
                    y=data[next(iter(data))]['decay'].keys(),
                    x=data[next(iter(data))]['spectrum'].keys(),
                    z=data[next(iter(data))]['trace'],
                    colorscale='Viridis', showscale = False),
                    row=1, col=3)
    fig
    fig.update_layout(height=400, width=1000)
    fig.update_layout(legend_orientation="h")
    fig.show()
    return fig


def plot_spectra(data):
    fig = make_subplots(rows=1, cols=1)
    for key in data:
        fig.add_trace(go.Scatter(x=data[key]['spectrum'].keys(), y=data[key]['spectrum'].iloc[:], 
                                  mode='lines',name='Spectrum '+ key),row=1, col=1)
    fig
    fig.update_layout(height=400, width=1000)
    fig.update_layout(legend_orientation="h")
    fig.show()
    return fig

def plot_decaycurves(data):
    fig = make_subplots(rows=1, cols=1)
    for key in data:
        fig.add_trace(go.Scatter(x=data[key]['decay'].keys(), y=data[key]['decay'].iloc[:],
                                  mode='lines',name='Decay ' + key),row=1, col=1)
    fig
    fig.update_layout(height=400, width=1000)
    fig.update_layout(legend_orientation="h")
    fig.show()
    return fig

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def check_options(data, options):
    for option in ['tstart', 'tstop']: 
        if options[option] == [0]:
            options[option] = []
            for i, key in enumerate(data):
                if option == 'tstart':
                    options[option].append(data[key]['tpeak']+options['tstart_offset']*data[key]['dt'])
                elif option == 'tstop':
                    options[option].append(data[key]['tmax'])
        elif len(options[option]) == 1:
            options[option]=list(np.repeat(np.array(options[option]),len(data)))  
        elif (len(options[option]) == len(data)):
            pass
        else:
            print("Error")    
    if options['convolution'] == True:
        options['tstart'] = list(np.repeat([0],len(data))) 
                
    for option in ['lstart', 'lstop']:
         if options[option] == [0]:
             options[option] = []
             for i, key in enumerate(data):
                 if option == 'lstart':
                     options[option].append(data[key]['lmin'])
                 elif option == 'lstop':
                     options[option].append(data[key]['lmax'])
         if (len(options[option]) == 1):
             options[option]=list(np.repeat(np.array(options[option]),len(data)))
         elif (len(options[option]) == len(data)):
             pass
         else:
             print("Error")     
                 
    for i,key in enumerate(data):    
        tstopi, tstop = find_nearest(data[key]['decay'].index, options['tstop'][i])
        tstarti, tstart = find_nearest(data[key]['decay'].index, options['tstart'][i])
        lstarti, lstart = find_nearest(data[key]['spectrum'].index.astype('float'), np.float(options['lstart'][i]))
        lstopi, lstop = find_nearest(data[key]['spectrum'].index.astype('float'), np.float(options['lstop'][i]))
        data[key]['options'] =  {'tstart': tstart, 'tstop': tstop, 'lstart':lstart, 'lstop': lstop, 'tstarti': tstarti, 'tstopi': tstopi, 'lstarti':lstarti, 'lstopi': lstopi}
        data[key]['options']['threshold'] = options['threshold']
        data[key]['options']['convolution'] = options['convolution']
        
        if options['exponents'] == 1 and options['convolution'] == True:
            data[key]['fitmodel']=Model(fitmodel_1exp_conv)
        elif options['exponents'] == 1 and options['convolution'] == False:
            data[key]['fitmodel']=Model(fitmodel_1exp)
        elif options['exponents'] == 2 and options['convolution'] == True:
            data[key]['fitmodel']=Model(fitmodel_2exp_conv)
        elif options['exponents'] == 2 and options['convolution'] == False:
            data[key]['fitmodel']=Model(fitmodel_2exp)
        elif options['exponents'] == 3 and options['convolution'] == True:
            data[key]['fitmodel']=Model(fitmodel_3exp_conv)
        elif options['exponents'] == 3 and options['convolution'] == False:
            data[key]['fitmodel']=Model(fitmodel_3exp)
        elif options['exponents'] == 4 and options['convolution'] == True:
            data[key]['fitmodel']=Model(fitmodel_4exp_conv)
        elif options['exponents'] == 4 and options['convolution'] == False:
            data[key]['fitmodel']=Model(fitmodel_4exp)
            
    return data



def make_weights(fitdata, threshold, t1i, t2i):
    #weights = np.ones_like(fitdata)
    weights = np.zeros_like(fitdata)
    weights = np.array(1 / np.sqrt(fitdata))
    weights[fitdata < threshold]  = 0 
    weights[np.isnan(weights)] = 0
    weights[np.greater(weights,1)] = 1        
    weights[np.isinf(weights)] = 0
    weights[:t1i] = 0
    weights[t2i:] = 0
    return weights#/weights.max()



def init_fitdata(data):
    for key in data:
        t1i, t2i, tpeaki = data[key]['options']['tstarti'], data[key]['options']['tstopi'], data[key]['tpeaki']
        l1i, l2i = data[key]['options']['lstarti'], data[key]['options']['lstopi']
        #data[key]['fitdata'] = np.sum(data[key]['trace'].iloc[t1i:t2i+1,l1i:l2i+1],axis=1)
        data[key]['fitdata'] = np.sum(data[key]['trace'].iloc[:,l1i:l2i+1],axis=1)
        data[key]['tail'] = np.sum(data[key]['trace'].iloc[tpeaki:t2i,l1i:l2i],axis=1)
        data[key]['weights'] = make_weights(data[key]['fitdata'], data[key]['options']['threshold'], t1i, t2i)
        #print(data[key]['weights'])
    return data

def init_parameters(data):
    for key in data:
        tail = data[key]['tail'] 
        params = data[key]['fitmodel'].make_params()
        params['y0'] = Parameter(name='y0', value=np.average(np.asarray(data[key]['fitdata'])[0:10]), vary=False)
        data[key]['moment'] = np.sum((tail-params['y0'])*tail.index.values,axis=0)/np.sum(tail-params['y0'],axis=0)
        params['tau1'] = Parameter(name='tau1', value=data[key]['moment']/10, vary=True, min=0.001, max=1000)
        params['tau'] = Parameter(name='tau', expr='tau1')
        params['a1'] = Parameter(name='a1', value=np.max(tail), vary=True, min=0.001, max=1e6)
        params['x0'] = Parameter(name='x0', value=0, vary=True, min=-1.5, max=1.5)
        if 'tau2' in params: 
            params['a2'] = Parameter(name='a2', value=np.max(tail)/10, vary=True, min=0.001, max=1e6)
            params['tau2'] = Parameter(name='tau2', value=1.0*data[key]['moment'], vary=True, min=0.001, max=1000)
            params['tau'] = Parameter(name='tau', expr='a1/(a1+a2)*tau1+a2/(a1+a2)*tau2')
        if 'tau3' in params:
            params['a3'] = Parameter(name='a3', value=np.max(tail)/40, vary=True, min=0.001, max=1e6)
            params['tau3'] = Parameter(name='tau3', value=2.0*data[key]['moment'], vary=True, min=0.001, max=1000)
            params['tau'] = Parameter(name='tau', expr='a1/(a1+a2+a3)*tau1+a2/(a1+a2+a3)*tau2+a3/(a1+a2+a3)*tau3')
        if 'tau4' in params: 
            params['a4'] = Parameter(name='a4', value=np.max(tail)/80, vary=True, min=0.001, max=1e6)
            params['tau4'] = Parameter(name='tau4', value=3.0*data[key]['moment'], vary=True, min=0.001, max=1000)
            params['tau'] = Parameter(name='tau', expr='a1/(a1+a2+a3+a4)*tau1+a2/(a1+a2+a3+a4)*tau2+a3/(a1+a2+a3+a4)*tau3+a4/(a1+a2+a3+a4)*tau4')
        data[key]['init_params'] = params
    return data



def fit_irf(data, irffile, show=False):
    trace = pd.read_csv(irffile, delim_whitespace= True, header=0, index_col=0)
    decay = trace.sum(axis=1)
    spectrum = trace.sum(axis=0)
    y=decay
    x=decay.index
    irfmod=Model(exGaussian)
    pars = irfmod.make_params(tau=0.15, mu=0.08, sig=0.01, ampl=np.max(y))
    out  = irfmod.fit(y, pars, x=x)
    if show:
        print(out.fit_report(min_correl=0.25))
        plt.plot(x, y,         'bo')
        plt.plot(x, out.init_fit, 'k--')
        plt.plot(x, out.best_fit, 'r-')
        plt.show()  
    
    for key in data:
        data[key]['tpeaki_roi'], data[key]['tpeak_roi'] = np.argmax(data[key]['fitdata']), decay.index[np.argmax(data[key]['fitdata'])]
        irf_x_shift = data[key]['tpeaki_roi']-np.argmax(out.eval(x=data[key]['decay'].index))
        #print(irf_x_shift)
        irf_shifted = shift(out.eval(x=data[key]['decay'].index),irf_x_shift,cval=0.0)
        data[key]['irf']={'irf_x_shift': irf_x_shift, 'irf_shifted': irf_shifted, 'trace': trace, 'decay': decay, 'spectrum': spectrum, 'fit': out}
    #for key in data:    
    #    plt.plot(data[key]['decay'].index, minmax(data[key]['irf']['fit'].eval(x=data[key]['decay'].index)))
    #    plt.plot(data[key]['decay'].index, minmax(data[key]['irf']['irf_shifted']))
    return data


def minmax(X):
    xmin =  X.min(axis=0)
    return (X - xmin) / (X.max(axis=0) - xmin)

### Define fitting models and related functions
def fitmodel_1exp(x, a1, tau1, y0, x0):
        return a1*np.exp(-(x-x0)/tau1)+y0

def fitmodel_2exp(x, a1, tau1, a2, tau2, y0, x0):
        return a1*np.exp(-(x-x0)/tau1)+a2*np.exp(-(x-x0)/tau2)+y0
    
def fitmodel_3exp(x, a1, tau1, a2, tau2, a3, tau3, y0, x0):
        return a1*np.exp(-(x-x0)/tau1)+a2*np.exp(-(x-x0)/tau2)+a3*np.exp(-(x-x0)/tau3)+y0

def fitmodel_4exp(x, a1, tau1, a2, tau2, a3, tau3, a4, tau4,  y0, x0):
        return a1*np.exp(-(x-x0)/tau1)+a2*np.exp(-(x-x0)/tau2)+a3*np.exp(-(x-x0)/tau3)+a4*np.exp(-(x-x0)/tau4)+y0

def fitmodel_1exp_conv(x, a1, tau1, y0, x0, irf_shifted=None):
    tmp = fftconvolve(a1*np.exp(-(x)/tau1)+y0, irf_shifted/np.max(irf_shifted), mode='full')
    tmp = shift(tmp,x0,cval=0.0)
    return tmp[0:len(irf_shifted)]

def fitmodel_2exp_conv(x, a1, tau1, a2, tau2, y0, x0, irf_shifted=None):
    tmp = fftconvolve(a1*np.exp(-(x)/tau1)+a2*np.exp(-(x)/tau2)+y0, irf_shifted/np.max(irf_shifted), mode='full')
    tmp = shift(tmp,x0,cval=0.0)
    return tmp[0:len(irf_shifted)]
    
def fitmodel_3exp_conv(x, a1, tau1, a2, tau2, a3, tau3, y0, x0, irf_shifted=None):
    tmp = fftconvolve(a1*np.exp(-(x)/tau1)+a2*np.exp(-(x)/tau2)+a3*np.exp(-(x)/tau3)+y0, irf_shifted/np.max(irf_shifted), mode='full')
    tmp = shift(tmp,x0,cval=0.0)
    return tmp[0:len(irf_shifted)]

def fitmodel_4exp_conv(x, a1, tau1, a2, tau2, a3, tau3, a4, tau4,  y0, x0, irf_shifted=None):
    tmp = fftconvolve(a1*np.exp(-(x)/tau1)+a2*np.exp(-(x)/tau2)+a3*np.exp(-(x)/tau3)+a4*np.exp(-(x)/tau4)+y0, irf_shifted/np.max(irf_shifted), mode='full')
    tmp = shift(tmp,x0,cval=0.0)
    return tmp[0:len(irf_shifted)]

def exGaussian(x, tau, mu, sig, ampl):
    #Model for fitting and simulating IRF
    #Call exGaussian(x, tau, mu, sig, ampl)
    lam = 1./tau
    return ampl * np.exp(0.5*lam * (2*mu + lam*(sig**2) - 2*x)) *\
           scipy.special.erfc((mu + lam*(sig**2) - x)/(np.sqrt(2)*sig)) 

def to_df(data):
    df = pd.DataFrame()
    for key, value in data.items():
        df = df.append(data[key]['fitresult'].params.valuesdict(),ignore_index=True)
    for i, key in enumerate(data):
        df = df.rename(index={i:key})
    df.style.format("{:.2%}")
    return df


def do_fitting(data, show=False):
    for key in data:
        print(key)
        fitmodel = data[key]['fitmodel']
        fitdata = data[key]['fitdata']
        weights = data[key]['weights']
        init_params = data[key]['init_params']
        
        if data[key]['options']['convolution'] == True:
            irf_shifted=minmax(data[key]['irf']['irf_shifted'])
            irf_shifted=irf_shifted[0:len(fitdata)]
            data[key]['fitresult'] = fitmodel.fit(fitdata, x=fitdata.index.values, params=init_params, weights=weights, irf_shifted=irf_shifted)
            if show:
                #plt.plot(data[key]['fitresult'].eval(x=fitdata.index.values))
                plt.plot(data[key]['fitdata'].index, data[key]['fitdata'], '--', label='data')
                plt.plot(data[key]['fitdata'].index, data[key]['fitresult'].init_fit, '--', label='initial fit')
                plt.plot(data[key]['fitdata'].index, data[key]['fitresult'].best_fit, '-', label='best fit')
                plt.plot(data[key]['fitdata'].index, np.max(data[key]['fitdata'])*irf_shifted, '-', label='irf')
                #data[key]['fitresult'].plot()
                plt.ylim((0,np.max(data[key]['fitdata'])))
                plt.draw()
            lmfit.report_fit(data[key]['fitresult'].params, min_correl=0.5)
        else:
           data[key]['fitresult'] = fitmodel.fit(fitdata, init_params , x=fitdata.index.values, weights=weights)
           if show:
                plt.plot(data[key]['fitdata'].index, data[key]['fitdata'], '--', label='data')
                plt.plot(data[key]['fitdata'].index, data[key]['fitresult'].init_fit, '--', label='initial fit')
                plt.plot(data[key]['fitdata'].index, data[key]['fitresult'].best_fit, '-', label='best fit')
                #data[key]['fitresult'].plot()
                plt.ylim((0,np.max(data[key]['fitdata'])))
                plt.draw()
           lmfit.report_fit(data[key]['fitresult'].params, min_correl=0.5)
            
    return data

def calc_differential_lifetime(data):
        for key in data:
            print(key)
            x=np.array(data[key]['fitdata'].index)
            yfit = data[key]['fitresult'].best_fit
            dtau = np.gradient(np.log(yfit), x)
            dtau = -np.nan_to_num(dtau, copy=False, nan=0, posinf=1, neginf=1)**(-1)
            data[key]['dtau'] = dtau
        return data

def plot_differential_lifetime(data):
        fig = make_subplots(rows=1, cols=1)
        for i,key in enumerate(data):
                fig.add_trace(go.Scatter(x=data[key]['decay'].keys(), y=data[key]['dtau'], 
                                        mode='lines',name=key),row=1, col=1)
                if (i==0):
                    fig.add_vline(x=data[key]['tpeak'],annotation_text="Excitation", 
                    annotation_position="top right")
        fig
        fig.update_layout(height=400, width=1000)
        fig.update_layout(legend_orientation="v")
        fig.update_layout(
            xaxis_title="t (ns)",
            yaxis_title="-(d ln(tau)/dt)^(-1)"
        )
        return fig


def show_fitresults(data):
    fig = make_subplots(rows=3, cols=1, row_heights=(0.7, 0.15, 0.15), subplot_titles=('Data and fits', 'Weights', 'Residuals'))
    for key in data:
        #,x = data[key]['decay'].index.values
        #y = data[key]['decay']
        
        #print(data[key]['fitresult'].eval)
        #print(data[key]['fitresult'].best_fit)
        #x0 = data[key]['fitresult'].params['x0']
        
        if data[key]['options']['convolution'] == True:
            fig.add_trace(go.Scatter(x=data[key]['fitdata'].index, y=data[key]['fitdata'],name=key),row=1,col=1)
            fig.add_trace(go.Scatter(x=data[key]['fitdata'].index, y=np.max(data[key]['fitdata'])*(data[key]['irf']['irf_shifted']/np.max(data[key]['irf']['irf_shifted'])),mode='lines',name=key+'_irf'),row=1,col=1)
            fig.add_trace(go.Scatter(x=data[key]['fitdata'].index, y=data[key]['fitresult'].best_fit,mode='lines',name=key+'_fit'),row=1,col=1)
        else:
            fig.add_trace(go.Scatter(x=data[key]['fitdata'].index, y=data[key]['fitdata'],name=key),row=1,col=1)
            fig.add_trace(go.Scatter(x=data[key]['fitdata'].index, y=data[key]['fitresult'].best_fit,mode='lines',name=key+'_fit'),row=1,col=1)
        fig.add_trace(go.Scatter(x=data[key]['fitdata'].index, y=data[key]['weights'] ,name=key+'_weights'),row=2,col=1)
        fig.add_trace(go.Scatter(x=data[key]['fitdata'].index, y=data[key]['fitresult'].residual ,name=key+'_residual'),row=3,col=1)
    fig.add_vline(x=data[key]['options']['tstart'], line_width=3, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_vline(x=data[key]['options']['tstop'], line_width=3, line_dash="dash", line_color="green", row=2, col=1)
    #fig.add_hrect(y0=data[key]['options']['lstart'], y1=data[key]['options']['lstop'],row=4,col=1, line_width=0, fillcolor="red", opacity=0.2)
    #fig.add_trace(go.Heatmap(
    #                y=data[key]['decay'].keys(),
    #                x=data[key]['spectrum'].keys(),
    #                z=data[key]['trace'],
    #                colorscale='Viridis', showscale = False),
    #                row=4, col=1)
    fig
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(range=[-3, 1], type="log",  row=2, col=1)
    fig.update_yaxes(range=[-4, 4], type="log", row=3, col=1)
    #fig.update_yaxes(type="linear", row=4,col=1)
    #fig.update_layout(width=800)
    return fig