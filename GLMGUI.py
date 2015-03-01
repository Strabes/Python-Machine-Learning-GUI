from Tkinter import *
import numpy as np
import statsmodels.api as sm
import patsy as pt
import pandas as pd
from scipy import stats
import matplotlib.pyplot as pl
#from mpl_toolkits.mplot3d import axes3d, Axes3D
#from matplotlib.collections import PolyCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

class ScrollCanvas:
	'''Create a scrollable canvas class using Tkinter widgets
	Author:		Greg Strabel'''
	def __init__(self, root):
		self.outerFrame = Frame(root, relief=SUNKEN)
		self.outerFrame.pack(fill=BOTH,expand=1)
		self.outerFrame.grid_rowconfigure(0, weight=1)
		self.outerFrame.grid_columnconfigure(0, weight=1)
		self.hscrollbar=Scrollbar(self.outerFrame, orient='horizontal')#, command
		self.vscrollbar=Scrollbar(self.outerFrame, orient='vertical')
		self.canvas=Canvas(self.outerFrame,xscrollcommand=self.hscrollbar.set,yscrollcommand=self.vscrollbar.set, bg='white')
		self.canvas.grid(row=0,column=0, sticky=N+E+S+W)
		self.canvas.configure(scrollregion=self.canvas.bbox("all"))
		self.hscrollbar.configure(command=self.canvas.xview)
		self.vscrollbar.configure(command=self.canvas.yview)
		self.hscrollbar.grid(row=1,column=0,sticky=E+W)
		self.vscrollbar.grid(row=0,column=1,sticky=N+S)
		
	def destroy(self):
		self.outerFrame.destroy()

class GLMGUI:
	'''
	Author					Greg Strabel
	GLMGUI has two required inputs, both instances of pandas.DataFrame:
	endog_var:	a pandas.DataFrame with a single column containing the target values
	exog_var:	a pandas.DataFrame of regressors
	GLMGUI has a few optional inputs at this point:
	rootWindow:	an instance of class Tkinter.Tk. Defaults to None,
	resulting in the creation of a master window.
	formula:	a string representing a valid patsy formula. If the formula is
	not valid, the program will err. Defaults to the empty string, resulting in fitting
	only an intercept
	family:	an instance of statsmodels.api.families. Defaults to
	statsmodels.api.families.Gaussian()
	contVarMaxLevel:	# of levels to use in grouping numeric data
	for the purpose of plotting. Only used for plotting not for fitting.
	Can be changed from the GUI menu. Defaults to 10.
	bin_method: method for binning the numeric variables for plotting.
	Defaults to 'uniform' bins. Only other option is 'quantile'. Binning is done
	using pandas.cut and pandas.qcut
	'''
	
	def __init__(self, endog_var, exog_var, rootWindow=None, formula='', family=sm.families.Gaussian(), contVarMaxLevel=10, bin_method = 'uniform'):
		
		self.formula = formula
		self.endog_var = endog_var
		self.exog_var=exog_var
		if rootWindow is None:
			self.rootWindow = Tk()
		else:
			self.rootWindow = rootWindow
		self.family=family
		self.contVarMaxLevel = contVarMaxLevel
		self.bin_method = bin_method
		self.masterFrame = Frame(self.rootWindow)
		self.masterFrame.pack(side=TOP, fill=BOTH, expand=1)
		self.masterFrameTop = Frame(self.masterFrame)
		self.masterFrameTop.pack(side=TOP, fill=X, expand=0)
		self.masterFrameBottom = Frame(self.masterFrame)
		self.masterFrameBottom.pack(side=BOTTOM, fill=BOTH, expand=1)

		# Create Menu
		self.menubar = Menu(self.rootWindow)
		self.familyMenu = Menu(self.menubar)
		self.plotInterMenu = Menu(self.menubar)
		self.menubar.add_command(label="Summary", command=self.regressSummary)
		self.menubar.add_command(label="Refit", command=self.refit)
		self.menubar.add_cascade(label="Plot Interactions", menu=self.plotInterMenu)
		self.plotInterMenu.add_command(label='Standard Plot', command=self.standardInterPlot)
		self.plotInterMenu.add_command(label='Plot Surface', command=self.surfacePlot)
		self.menubar.add_cascade(label="Family", menu=self.familyMenu)
		self.familyMenu.add_command(label="Gaussian", command=self.fitGaussian)
		self.familyMenu.add_command(label="Binomial", command=self.fitBinomial)
		self.familyMenu.add_command(label="Gamma", command=self.fitGamma)
		self.menubar.add_command(label="Variable Options", command=self.variableOptions)
		self.menubar.add_command(label="Quit", command=self.masterFrame.destroy)
		self.rootWindow.config(menu=self.menubar)
		
		# create Listbox for feature_names
		self.leftFrame=Frame(self.masterFrameBottom)
		self.leftFrame.pack(side=LEFT,fill=Y,expand=0)
		self.listbox=Listbox(self.leftFrame, bg='gray90')
		self.listbox.grid(row=0,column=0, sticky=N+S+E+W)
		#self.listbox.insert(END, "Regression Summary")
		vscrollbar = Scrollbar(self.leftFrame, orient='vertical')
		vscrollbar.grid(row=0,column=1,sticky=N+S)
		vscrollbar.configure(command=self.listbox.yview)
		self.listbox.configure(yscrollcommand=vscrollbar.set)
		self.leftFrame.grid_rowconfigure(0, weight=1)
		self.leftFrame.grid_columnconfigure(0, weight=1)
		
		self.endog_var_name = endog_var.columns.tolist()
		self.endog_var_name = self.endog_var_name[0]
		
		# Add the names of the regressors to the Listbox
		for item in exog_var.columns.tolist():
			self.listbox.insert(END,item)
		
		self.rightFrame=Frame(self.masterFrameBottom)
		self.rightFrame.pack(side=RIGHT,fill=BOTH,expand=1)

		self.rightFrameTop=ScrollCanvas(self.rightFrame)
		#self.rightFrameTop.outerFrame.pack(side=TOP,fill=BOTH,expand=1)
		
		self.rightFrameBottom=Frame(self.rightFrame)
		self.rightFrameBottom.pack(side=BOTTOM,fill=X,expand=0)		
		
		self.formulaLabel = Label(self.rightFrameBottom,text='Edit formula here: ')
		#self.formulaLabel.pack(side=LEFT)
		self.formulaLabel.grid(row=0,column=0)
		self.formulaEntry=Entry(self.rightFrameBottom,width=100)
		self.formulaEntry.grid(row=0,column=1)   # .pack(side=LEFT,expand=1)
		self.formulaEntry.delete(0, last='end')
		self.formulaEntry.insert(0,self.formula)
	
		self.design_mat = pt.dmatrix(formula,data=exog_var)
	
		self.arg = sm.GLM(endog_var,self.design_mat,family=self.family)
	
		self.mode_exog_var = exog_var.mode()[0:1]
		self.median_exog_var = exog_var.median()
		self.the_base = self.mode_exog_var.fillna(value=self.median_exog_var.to_dict())

		self.GLMResult = self.arg.fit()   #    fit the GLM
		#print self.GLMResult.summary()   #    print the GLM summary
	
		self.paramNames=self.design_mat.design_info.column_names
	
		self.GLMResultSummary = self.GLMResult.summary(xname=self.paramNames).as_text()
	
		self.data_and_pred = pd.concat([exog_var,pd.DataFrame({'Predicted':self.GLMResult.mu}), pd.DataFrame({'Actual':self.arg.endog})],axis=1)

		GLMSummarytext_id = self.rightFrameTop.canvas.create_text(0,0,text=self.GLMResultSummary, anchor=NW,justify=LEFT,font=('Courier',10))
		self.rightFrameTop.canvas.config(scrollregion=self.rightFrameTop.canvas.bbox(GLMSummarytext_id))
		
		self.listbox.bind("<Double-Button-1>", self.plot_a_graph)
		w, h = self.rootWindow.winfo_screenwidth(), self.rootWindow.winfo_screenheight()
		self.rootWindow.geometry("%dx%d+0+0" % (w, h))
		self.rootWindow.mainloop()

	def regressSummary(self,*args):
		self.rightFrameTop.destroy()
		self.rightFrameTop=ScrollCanvas(self.rightFrame)
		GLMSummarytext_id = self.rightFrameTop.canvas.create_text(0,0,text=self.GLMResultSummary, anchor=NW,justify=LEFT,font=('Courier',10))
		self.rightFrameTop.canvas.config(scrollregion=self.rightFrameTop.canvas.bbox(GLMSummarytext_id))
		
	def plot_a_graph(self,*args):
		chosen=map(int, self.listbox.curselection())
		pl.close("all")
		self.variable_list=self.exog_var.columns.tolist()
		self.var_to_plot=self.variable_list[chosen[0]]
		self.rightFrameTop.destroy()
		self.rightFrameTop=Frame(self.rightFrame)
		self.rightFrameTop.pack(side=TOP,fill=BOTH,expand=1)

		self.fig=pl.figure(figsize=(5,4), dpi=100)
		ax1 = pl.axes()
		pl.grid(b=True, axis='both')
		ax1.patch.set_facecolor('white')
		ax2 = pl.twinx()			
			
		if self.var_to_plot in self.exog_var.select_dtypes(include=[np.number]).columns.tolist()					\
		and len(np.unique(self.exog_var[self.var_to_plot].values)) > self.contVarMaxLevel:
			if self.bin_method == 'uniform':
				self.grouped=self.data_and_pred.groupby													\
				(pd.cut(self.data_and_pred[self.var_to_plot], bins=self.contVarMaxLevel))
			elif self.bin_method == 'quantile':
				self.grouped=self.data_and_pred.groupby													\
				(pd.qcut(self.data_and_pred[self.var_to_plot], self.contVarMaxLevel))
		else:
			self.grouped=self.data_and_pred.groupby(self.var_to_plot)
			
		line_to_graph=self.grouped.mean()[['Actual','Predicted']]
		line_to_graph.plot(ax=ax1)			
		bar_to_graph=self.grouped.count()['Predicted']/self.exog_var.shape[0]
		bar_to_graph.plot(kind='bar',ax=ax2,alpha=0.3)
			
		pl.title(self.endog_var_name + ' vs. ' + self.var_to_plot)
		ax2.set_ylabel('Weights')
		ax1.set_xlabel(self.var_to_plot)
		ax1.set_xmargin(0.2)
		ax1.set_ylabel(self.endog_var_name) 
		pl.setp(ax1.get_xticklabels(), rotation=30, horizontalalignment='right')
		#pl.tight_layout()
		self.fig.subplots_adjust(top=.9,left=.15,right=.9,bottom=.25)
			
		self.canvas = FigureCanvasTkAgg(self.fig, master=self.rightFrameTop)
		self.toolbar = NavigationToolbar2TkAgg( self.canvas, self.rightFrameTop )
		self.toolbar.update()
		self.toolbar.pack(side=TOP)
		self.canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1) #side was TOP

		self.canvas.show()				
		
	
			
	def refit(self,*args):
		self.newFormula=self.formulaEntry.get()
		self.masterFrame.destroy()
		self.__init__(self.endog_var, self.exog_var, formula=self.newFormula,
			rootWindow=self.rootWindow, family=self.family,contVarMaxLevel=self.contVarMaxLevel)

	def interactionPlot(self,*args):
		self.varX_num = self.interListboxPrimary.curselection()[0]
		self.varX=self.exog_var.columns[self.varX_num]
		self.varY_num = self.interListboxSecondary.curselection()[0]
		self.varY=self.exog_var.columns[self.varY_num]
		self.InteractionPopUp.destroy()

		if self.varX in self.exog_var.select_dtypes(include=[np.number]).columns.tolist()					\
		and len(np.unique(self.exog_var[self.varX].values)) > self.contVarMaxLevel:
		
			self.grouped_X=pd.cut(self.data_and_pred[self.varX], bins=self.contVarMaxLevel)
		else:
			self.grouped_X=self.varX
		
		if self.varY in self.exog_var.select_dtypes(include=[np.number]).columns.tolist()					\
		and len(np.unique(self.exog_var[self.varY].values)) > self.contVarMaxLevel:
		
			self.grouped_Y=pd.cut(self.data_and_pred[self.varY], bins=self.contVarMaxLevel)
		else:
			self.grouped_Y=self.varY
		
		self.grouped = self.data_and_pred.groupby([self.grouped_X,self.grouped_Y]).mean()[['Predicted','Actual']]
		
		pl.close("all")
		self.standardPlotfig=pl.figure()
		self.rightFrameTop.destroy()
		self.rightFrameTop=Frame(self.rightFrame)
		self.rightFrameTop.pack(side=TOP,fill=BOTH,expand=1)
		self.rightFrameTopLeft = Frame(self.rightFrameTop)
		self.rightFrameTopLeft.pack(side=LEFT,fill=Y,expand=0)
		self.YvalsListbox = Listbox(self.rightFrameTopLeft,selectmode=MULTIPLE)
		vscrollbar = Scrollbar(self.rightFrameTopLeft, orient='vertical')
		#if self.
		for item in self.grouped.index.levels[1].tolist():
			self.YvalsListbox.insert(END,item)
		self.YvalsListbox.grid(row=0,column=0,sticky=N+S)
		vscrollbar.grid(row=0,column=1,sticky=N+S)
		vscrollbar.configure(command=self.YvalsListbox.yview)
		self.rightFrameTopLeft.grid_rowconfigure(0, weight=1)
		self.rightFrameTopLeft.grid_columnconfigure(0, weight=1)
		self.YvalsListbox.configure(yscrollcommand=vscrollbar.set)
		self.YvalsPlotButton = Button(self.rightFrameTopLeft, text='Plot')
		self.YvalsPlotButton.bind("<Button-1>", self.plotChosenYLevels)
		self.YvalsPlotButton.grid(row=1,column=0,sticky=N+E+S+W)
		self.rightFrameTopRight = Frame(self.rightFrameTop)
		self.rightFrameTopRight.pack(side=RIGHT,fill=BOTH,expand=1)
		self.canvas = FigureCanvasTkAgg(self.standardPlotfig, master=self.rightFrameTopRight)
		self.toolbar = NavigationToolbar2TkAgg( self.canvas, self.rightFrameTopRight )
		self.toolbar.update()
		self.toolbar.pack(side=TOP)
		self.canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1) #side was TOP
		self.standardPlotax1 = pl.axes()
		self.canvas.show()
		
	def plotChosenYLevels(self,*args):
		YValPos=self.YvalsListbox.curselection()
		Yindices = self.grouped.index.levels[1].tolist()
		self.standardPlotax1.cla()
		for i in YValPos:
			levelToPlot = Yindices[i]
			self.grouped.iloc[self.grouped.index.get_level_values(self.varY)==levelToPlot]['Actual'] \
			.plot(label=levelToPlot+' - Actual', ls='-')
			self.grouped.iloc[self.grouped.index.get_level_values(self.varY)==levelToPlot]['Predicted'] \
			.plot(label=levelToPlot+' - Predicted', ls='--')
		self.standardPlotax1.set_xticks(range(len(self.grouped.index.levels[0].tolist())))	
		self.standardPlotax1.set_xticklabels(self.grouped.index.levels[0].tolist(),rotation=30)
		pl.legend(loc='best')
		self.standardPlotax1.set_xlabel(self.varX)
		self.standardPlotax1.set_ylabel(self.endog_var_name)
		self.standardPlotax1.set_title('Predicted ' + self.endog_var_name
				+ ' by ' + self.varX + ' and ' + self.varY)
		self.canvas.show()

	
	def surfacePlot(self, event=None):
		# The surface interaction plot has already been selected
		self.plotType = 'Plot Surface'
		self.ChoosePrimaryVariable()
		
	def standardInterPlot(self, event=None):
		self.plotType = 'Standard Plot'
		self.ChoosePrimaryVariable()

	def variableOptions(self,*args):
		self.variableOptionsPopUp = Toplevel()
		contVarOption1 = Label(self.variableOptionsPopUp,
			text = 'Maximum number of levels for plotting')
		contVarOption1.grid(row=0,column=0)
		self.contVarMaxLevel_entry = Entry(self.variableOptionsPopUp)
		self.contVarMaxLevel_entry.grid(row=0,column=1)
		self.contVarMaxLevel_entry.insert(0, self.contVarMaxLevel)
		contVarOption2 = Label(self.variableOptionsPopUp,
			text = 'Binning Method for continuous variables')
		contVarOption1.grid(row=1,column=0)
		self.bin_method_choice=StringVar(self.variableOptionsPopUp)
		self.bin_method_choice.set(self.bin_method)
		self.bin_method_menu = OptionMenu(self.variableOptionsPopUp,self.bin_method_choice,'uniform','quantile')
		self.bin_method_menu.grid(row=1,column=1)
		ApplyButton = Button(self.variableOptionsPopUp, text='Apply')
		ApplyButton.grid(row=3,column=1)
		ApplyButton.bind("<Button-1>", self.variableOptionsSelected)
		self.variableOptionsPopUp.wait_window(self.variableOptionsPopUp)
		
	def variableOptionsSelected(self,*args):
		self.contVarMaxLevel = int(self.contVarMaxLevel_entry.get())
		self.bin_method = self.bin_method_choice.get()
		self.variableOptionsPopUp.destroy()
		
	def ChoosePrimaryVariable(self,event=None):
		self.InteractionPopUp = Toplevel()
		self.InteractionPopUpFrame = Frame(self.InteractionPopUp)
		self.InteractionPopUpFrame.pack(fill=BOTH, expand=1)
		self.InteractionPopUpFrame.grid_rowconfigure(1, weight=1)
		self.InteractionPopUpFrame.grid_columnconfigure(0, weight=1)
		self.InteractionPopUpFrame.grid_columnconfigure(2, weight=1)
		self.primaryLabel = Label(self.InteractionPopUpFrame, text='Select Primary Variable')
		self.primaryLabel.grid(row=0,column=0,sticky=E+W)
		self.secondaryLabel = Label(self.InteractionPopUpFrame, text='Select Secondary Variable')
		self.secondaryLabel.grid(row=0,column=2,sticky=E+W)
		self.interListboxPrimary = Listbox(self.InteractionPopUpFrame,exportselection=0)
		self.interListboxSecondary = Listbox(self.InteractionPopUpFrame,exportselection=0)
		vscrollbarPrimary = Scrollbar(self.InteractionPopUpFrame, orient='vertical')
		vscrollbarPrimary.grid(row=1,column=1,sticky=N+S)
		vscrollbarSecondary = Scrollbar(self.InteractionPopUpFrame, orient='vertical')
		vscrollbarSecondary.grid(row=1,column=3,sticky=N+S)
		for item in self.exog_var.columns.tolist():
			self.interListboxPrimary.insert(END,item)
			self.interListboxSecondary.insert(END,item)
		self.plotInterButton = Button(self.InteractionPopUpFrame, text='Plot Interactions')
		self.plotInterButton.grid(row=2,column=0)
		self.plotInterButton.bind("<Button-1>", self.interactionPlot)
		self.interListboxPrimary.grid(row=1,column=0)
		self.interListboxSecondary.grid(row=1,column=2)
		vscrollbarPrimary.configure(command=self.interListboxPrimary.yview)
		vscrollbarSecondary.configure(command=self.interListboxSecondary.yview)
		self.interListboxPrimary.configure(yscrollcommand=vscrollbarPrimary.set)
		self.interListboxSecondary.configure(yscrollcommand=vscrollbarSecondary.set)
		self.InteractionPopUp.wait_window(self.InteractionPopUp)
		
	def fitGaussian(self, *args):
		self.newFormula=self.formulaEntry.get()
		self.masterFrame.destroy()
		self.__init__(self.endog_var, self.exog_var, formula=self.formula,
		rootWindow=self.rootWindow, family=sm.families.Gaussian(),contVarMaxLevel=self.contVarMaxLevel)
	def fitBinomial(self,*args):
		self.newFormula=self.formulaEntry.get()
		self.masterFrame.destroy()
		self.__init__(self.endog_var, self.exog_var, formula=self.formula,
		rootWindow=self.rootWindow, family=sm.families.Binomial(),contVarMaxLevel=self.contVarMaxLevel)
	def fitGamma(self,*args):
		self.newFormula=self.formulaEntry.get()
		self.masterFrame.destroy()
		self.__init__(self.endog_var, self.exog_var, formula=self.formula,
		rootWindow=self.rootWindow, family=sm.families.Gamma(),contVarMaxLevel=self.contVarMaxLevel)
