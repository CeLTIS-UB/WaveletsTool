from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import tkinter
import tkinter.messagebox
import logging
import gdal
import ogr
import osr
import os
import numpy as np
import math
import sys
import pandas as pd
import tempfile
import json
from scipy import spatial

gdal.AllRegister()  # register all gdal drivers
ogr.UseExceptions()

# create logger instance
module_logger = logging.getLogger(__name__)

# --------------------------------------------------- #
###
RUN_RESULT = 0

### Tool Frontend ###
class GUI(object):

	def __init__(self, master):  # master = root

		self.master = master
		self.master.title('Wavelet Instrument')
		self.master.geometry("1425x755+0+0")
		self.raster_par = StringVar()
		self.approach_par = StringVar()
		self.predefined_pattern_par = StringVar()
		self.input_pattern_folder_par = StringVar()
		self.point_matrix_size_par = IntVar()
		self.point_matrix_size_par.set(3)
		self.input_point_vectors_par = StringVar()
		self.mapping_field_par = StringVar()
		self.move_to_maximum_par = BooleanVar()
		self.move_to_maximum_dist_par = IntVar()
		self.move_to_maximum_dist_par.set('')
		self.mh_iteration_par = BooleanVar()
		self.mh_dilation_value_par = DoubleVar()
		self.mh_dilation_value_par.set('')
		self.mh_dilation_value_start_par = DoubleVar()
		self.mh_dilation_value_start_par.set('')
		self.mh_dilation_value_stop_par = DoubleVar()
		self.mh_dilation_value_stop_par.set('')
		self.mh_dilation_step_par = DoubleVar()
		self.mh_dilation_step_par.set('')
		self.transform_par = StringVar()
		self.size_of_the_cell_par = IntVar()
		self.size_of_the_cell_par.set('')
		self.out_sim_matrix_par = StringVar()
		self.out_table_par = StringVar()
		self.out_raster_wksp_par = StringVar()
		self.status_par = IntVar()
		self.status_par.set(2)
		self.status_checker = []

		# frames
		self.frame_labels = ttk.Frame(master, width=500, height=700, relief=SUNKEN, style="My.TFrame")
		self.frame_labels.grid(row=1, column=1, padx=10, pady=10, sticky='news', rowspan=2)

		self.frame_log = ttk.Frame(master, width=200, height=100, relief=SUNKEN, style='My.TFrame')
		self.frame_log.grid(row=1, column=3, padx=0, pady=10, sticky='news')

		self.credits_frame = ttk.Frame(master, width=200, height=500, relief=SUNKEN)
		self.credits_frame.grid(row=2, column=3, padx=0, pady=10, sticky='news')

		# Log area
		self.scrollbar = tkinter.Scrollbar(self.frame_log)
		self.mytext = tkinter.Text(self.frame_log, width=70, height=35, wrap="word", yscrollcommand=self.scrollbar.set,
								   borderwidth=2, highlightthickness=0, state=DISABLED)
		self.scrollbar.config(command=self.mytext.yview)
		self.scrollbar.pack(side="right", fill="y")
		self.mytext.pack(side="left", fill="both", expand=True)

		# Credits area
		self.credits_par = StringVar()
		self.credits_par.set ("The authors equally share the copyright.")

		self.credits_text = ttk.Label(self.credits_frame, textvariable=self.credits_par, width=80, font=('arial', 10))
		self.credits_text.grid(row=1, column=1, padx=4.5, pady=4.5, sticky=W)

		# Labels
		self.raster_l = ttk.Label(self.frame_labels, text='Input raster',
								  borderwidth=1, relief='solid', font=('arial', 12),
								  justify='left', style='MY.TLabel')
		self.approach_l = ttk.Label(self.frame_labels, text='Approach',
									borderwidth=1, relief='solid', font=('arial', 12),
									justify='left', style='MY.TLabel')
		self.predefined_pattern_l = ttk.Label(self.frame_labels, text='Predefined pattern',
											  borderwidth=1, relief='solid', font=('arial', 12),
											  justify='left', style='MY.TLabel', state='disabled')
		self.input_pattern_folder_l = ttk.Label(self.frame_labels, text='Input pattern folder',
												borderwidth=1, relief='solid', font=('arial', 12),
												justify='left', style='MY.TLabel', state='disabled')
		self.point_matrix_size_l = ttk.Label(self.frame_labels, text='Point Matrix Size',
											 borderwidth=1, relief='solid', font=('arial', 12),
											 justify='left', style='MY.TLabel')
		self.input_point_vectors_l = ttk.Label(self.frame_labels, text='Input Point Vectors',
											   borderwidth=1, relief='solid', font=('arial', 12),
											   justify='left', style='MY.TLabel')
		self.mapping_field_l = ttk.Label(self.frame_labels, text='Mapping Field',
										 borderwidth=1, relief='solid', font=('arial', 12),
										 justify='left', style='MY.TLabel')
		self.move_to_maximum_l = ttk.Label(self.frame_labels, text='Move to maximum',
										   borderwidth=1, relief='solid', font=('arial', 12),
										   justify='left', style='MY.TLabel', state='disabled')
		self.move_to_maximum_dist_l = ttk.Label(self.frame_labels, text='Move to maximum distance',
												borderwidth=1, relief='solid', font=('arial', 12),
												justify='left', style='MY.TLabel', state='disabled')
		self.mh_iteration_l = ttk.Label(self.frame_labels, text='Mexican Hat Iteration',
										borderwidth=1, relief='solid', font=('arial', 12),
										justify='left', style='MY.TLabel', state='disabled')
		self.mh_dilation_value_l = ttk.Label(self.frame_labels, text='Mexican Hat dilation value',
											 borderwidth=1, relief='solid', font=('arial', 12),
											 justify='left', style='MY.TLabel', state='disabled')
		self.mh_dilation_value_start_l = ttk.Label(self.frame_labels, text='Mexican Hat dilation value start',
												   borderwidth=1, relief='solid', font=('arial', 12),
												   justify='left', style='MY.TLabel', state='disabled')
		self.mh_dilation_value_stop_l = ttk.Label(self.frame_labels, text='Mexican Hat dilation value stop',
												  borderwidth=1, relief='solid', font=('arial', 12),
												  justify='left', style='MY.TLabel', state='disabled')
		self.mh_dilation_step_l = ttk.Label(self.frame_labels, text='Mexican Hat dilation step',
											borderwidth=1, relief='solid', font=('arial', 12),
											justify='left', style='MY.TLabel', state='disabled')
		self.transform_l = ttk.Label(self.frame_labels, text='Transform',
								  borderwidth=1, relief='solid', font=('arial', 12),
								  justify='left', style='MY.TLabel')
		self.size_of_cell_l = ttk.Label(self.frame_labels, text='Size of the cell',
										borderwidth=1, relief='solid', font=('arial', 12),
										justify='left', style='MY.TLabel', state='disabled')
		self.out_sim_matrix_l = ttk.Label(self.frame_labels, text='Output similarity matrix',
										  borderwidth=1, relief='solid', font=('arial', 12),
										  justify='left', style='MY.TLabel')
		self.out_table_l = ttk.Label(self.frame_labels, text='Output table',
									 borderwidth=1, relief='solid', font=('arial', 12),
									 justify='left', style='MY.TLabel', state='disabled')
		self.out_raster_wksp_l = ttk.Label(self.frame_labels, text='Output raster workspace',
										   borderwidth=1, relief='solid', font=('arial', 12),
										   justify='left', style='MY.TLabel', state='disabled')
		self.status_val_l = ttk.Label(self.frame_labels, text='Status',
								  borderwidth=1, relief='solid', font=('arial', 12),
								  style='MY.TLabel')

		# parameters
		self.input_raster_ent = ttk.Entry(self.frame_labels, textvariable=self.raster_par, state='readonly',
										  font=('arial', 12),
										  width=45)
		self.rast_browse_button = ttk.Button(self.frame_labels, text='Browse raster...', command=self.open_raster_file, style="Default.TButton")

		self.approach_combo_box = ttk.Combobox(self.frame_labels, textvariable=self.approach_par,
											   state='readonly', font=('arial', 12),
											   width=45)
		self.approach_combo_box['value'] = ('Locations in the DEM generated from field observations',
											'Locations in the DEM versus pre-defined pattern',
											'Seek occurrence of pre-defined pattern in the DEM')
		self.approach_combo_box.current(0)
		self.approach_combo_box.bind("<<ComboboxSelected>>", self.approach_combo_fun)

		self.predefined_pattern_combo_box = ttk.Combobox(self.frame_labels, textvariable=self.predefined_pattern_par,
														 state='disabled', font=('arial', 12),
														 width=45)
		self.predefined_pattern_combo_box['value'] = ('Mexican Hat wavelet', 'Custom pattern', '')
		self.predefined_pattern_combo_box.current(2)
		self.predefined_pattern_combo_box.bind("<<ComboboxSelected>>", self.predefined_pattern_fun)

		self.input_pattern_folder_ent = ttk.Entry(self.frame_labels, textvariable=self.input_pattern_folder_par,
												  state='readonly',
												  font=('arial', 12), width=45)
		self.input_pattern_button = ttk.Button(self.frame_labels, text='Browse folder...', command=self.open_directory,
											   state='disabled', style="Default.TButton")

		self.point_matrix_size_ent = ttk.Entry(self.frame_labels, textvariable=self.point_matrix_size_par,
											   font=('arial', 12),
											   width=45)

		self.input_point_vectors_ent = ttk.Entry(self.frame_labels, textvariable=self.input_point_vectors_par,
												 font=('arial', 12), width=45, state='readonly')
		self.input_point_vectors_but = ttk.Button(self.frame_labels, text='Browse point vector...',
												  command=self.open_point_file, style="Default.TButton")

		self.mapping_field_combo_box = ttk.Combobox(self.frame_labels, textvariable=self.mapping_field_par,font=('arial', 12),
														 width=45 ,state='readonly')
		self.mapping_field_combo_box['value'] = ([''])
		self.mapping_field_combo_box.current(0)

		self.move_to_maximum_checkbox = ttk.Checkbutton(self.frame_labels, variable=self.move_to_maximum_par,
														offvalue=0, onvalue=1, command=self.move_to_max_check_fun)

		self.move_to_maximum_dist_ent = ttk.Entry(self.frame_labels, textvariable=self.move_to_maximum_dist_par,
												  font=('arial', 12),
												  width=45, state='disabled')

		self.mh_iteration_checkbox = ttk.Checkbutton(self.frame_labels, variable=self.mh_iteration_par,
													 offvalue=0, onvalue=1, command=self.mh_iteration_check_fun,
													 state='disabled')

		self.mh_dilation_value_ent = ttk.Entry(self.frame_labels, textvariable=self.mh_dilation_value_par,
											   font=('arial', 12),
											   width=45, state='disabled')

		self.mh_dilation_value_start_ent = ttk.Entry(self.frame_labels, textvariable=self.mh_dilation_value_start_par,
													 font=('arial', 12),
													 width=45, state='disabled')

		self.mh_dilation_value_stop_ent = ttk.Entry(self.frame_labels, textvariable=self.mh_dilation_value_stop_par,
													font=('arial', 12),
													width=45, state='disabled')

		self.mh_dilation_step_ent = ttk.Entry(self.frame_labels, textvariable=self.mh_dilation_step_par,
											  font=('arial', 12),
											  width=45, state='disabled')

		self.transform_combo_box = ttk.Combobox(self.frame_labels, textvariable=self.transform_par,
											 state='readonly', font=('arial', 12),
											 width=45)
		self.transform_combo_box['value'] = ('Work directly on the elevation matrix',
										  'Perform a local translation',
										  'Compute slopes',
										  'Compute slopes and perform local translation')
		self.transform_combo_box.current(0)
		self.transform_combo_box.bind("<<ComboboxSelected>>", self.transform_combo_fun)

		self.size_of_cell_ent = ttk.Entry(self.frame_labels, textvariable=self.size_of_the_cell_par, font=('arial', 12),
										  width=45, state='disabled')

		self.out_sim_matrix_ent = ttk.Entry(self.frame_labels, textvariable=self.out_sim_matrix_par,
											font=('arial', 12), width=45)
		self.out_sim_matrix_but = ttk.Button(self.frame_labels, text='Save matrix...', command=self.save_sim_matrix, style="Default.TButton")

		self.out_table_ent = ttk.Entry(self.frame_labels, textvariable=self.out_table_par, font=('arial', 12), state='disabled',
									   width=45)
		self.out_table_but = ttk.Button(self.frame_labels, text='Save table...', command=self.save_table, state='disabled',
										style="Default.TButton")

		self.out_raster_wksp_ent = ttk.Entry(self.frame_labels, textvariable=self.out_raster_wksp_par,
											 font=('arial', 12), state='disabled',
											 width=45)
		self.out_raster_wksp_but = ttk.Button(self.frame_labels, text='Browse...', command=self.out_raster_wksp, state='disabled', style="Default.TButton")
		self.run_but = ttk.Button(self.frame_labels, text='Run Tool', command=self.run_tool, state='disabled', style="Default.TButton")
		self.status_l = ttk.Label(self.frame_labels, text='Status',
									  borderwidth=1, relief='solid', font=('arial', 12),
									  style='MY.TLabel')
		self.status_val_l = ttk.Label(self.frame_labels, text='Not ready to run!',
									  borderwidth=1, relief='solid', font=('arial', 12),
									  style='RedStatus.TLabel')

		# params gridding
		self.input_raster_ent.grid(row=0, column=1, padx=5, pady=5, sticky=W)
		self.rast_browse_button.grid(row=0, column=2, ipadx=0, ipady=0, sticky=W)
		self.approach_combo_box.grid(row=1, column=1, padx=5, pady=5, sticky=W)
		self.predefined_pattern_combo_box.grid(row=2, column=1, padx=5, pady=5, sticky=W)
		self.input_pattern_folder_ent.grid(row=3, column=1, padx=5, pady=5, sticky=W)
		self.input_pattern_button.grid(row=3, column=2, padx=5, pady=5, sticky=W)
		self.point_matrix_size_ent.grid(row=4, column=1, padx=5, pady=5, sticky=W)
		self.input_point_vectors_ent.grid(row=5, column=1, padx=5, pady=5, sticky=W)
		self.input_point_vectors_but.grid(row=5, column=2, padx=5, pady=5, sticky=W)
		self.mapping_field_combo_box.grid(row=6, column=1, padx=5, pady=5, sticky=W)
		self.move_to_maximum_checkbox.grid(row=7, column=1, padx=8, pady=8, sticky=W)
		self.move_to_maximum_dist_ent.grid(row=8, column=1, padx=5, pady=5, sticky=W)
		self.mh_iteration_checkbox.grid(row=9, column=1, padx=8, pady=8, sticky=W)
		self.mh_dilation_value_ent.grid(row=10, column=1, padx=5, pady=5, sticky=W)
		self.mh_dilation_value_start_ent.grid(row=11, column=1, padx=5, pady=5, sticky=W)
		self.mh_dilation_value_stop_ent.grid(row=12, column=1, padx=5, pady=5, sticky=W)
		self.mh_dilation_step_ent.grid(row=13, column=1, padx=5, pady=5, sticky=W)
		self.transform_combo_box.grid(row=14, column=1, padx=5, pady=5, sticky=W)
		self.size_of_cell_ent.grid(row=15, column=1, padx=5, pady=5, sticky=W)
		self.out_sim_matrix_ent.grid(row=16, column=1, padx=5, pady=5, sticky=W)
		self.out_sim_matrix_but.grid(row=16, column=2, padx=5, pady=5, sticky=W)
		self.out_table_ent.grid(row=17, column=1, padx=5, pady=5, sticky=W)
		self.out_table_but.grid(row=17, column=2, padx=5, pady=5, sticky=W)
		self.out_raster_wksp_ent.grid(row=18, column=1, padx=5, pady=5, sticky=W)
		self.out_raster_wksp_but.grid(row=18, column=2, padx=5, pady=5, sticky=W)
		self.run_but.grid(row=19, column=2, padx=5, pady=10, sticky=W)
		self.status_val_l.grid(row=19, column=1, padx=5, pady=10, sticky=W)

		# label gridding
		self.raster_l.grid(row=0, column=0, padx=4.5, pady=4.5, sticky=W)
		self.approach_l.grid(row=1, column=0, padx=4.5, pady=4.5, sticky=W)
		self.predefined_pattern_l.grid(row=2, column=0, padx=4.5, pady=4.5, sticky=W)
		self.input_pattern_folder_l.grid(row=3, column=0, padx=4.5, pady=4.5, sticky=W)
		self.point_matrix_size_l.grid(row=4, column=0, padx=4.5, pady=4.5, sticky=W)
		self.input_point_vectors_l.grid(row=5, column=0, padx=4.5, pady=4.5, sticky=W)
		self.mapping_field_l.grid(row=6, column=0, padx=4.5, pady=4.5, sticky=W)
		self.move_to_maximum_l.grid(row=7, column=0, padx=4.5, pady=4.5, sticky=W)
		self.move_to_maximum_dist_l.grid(row=8, column=0, padx=4.5, pady=4.5, sticky=W)
		self.mh_iteration_l.grid(row=9, column=0, padx=4.5, pady=4.5, sticky=W)
		self.mh_dilation_value_l.grid(row=10, column=0, padx=4.5, pady=4.5, sticky=W)
		self.mh_dilation_value_start_l.grid(row=11, column=0, padx=4.5, pady=4.5, sticky=W)
		self.mh_dilation_value_stop_l.grid(row=12, column=0, padx=4.5, pady=4.5, sticky=W)
		self.mh_dilation_step_l.grid(row=13, column=0, padx=4.5, pady=4.5, sticky=W)
		self.transform_l.grid(row=14, column=0, padx=4.5, pady=4.5, sticky=W)
		self.size_of_cell_l.grid(row=15, column=0, padx=4.5, pady=4.5, sticky=W)
		self.out_sim_matrix_l.grid(row=16, column=0, padx=4.5, pady=4.5, sticky=W)
		self.out_table_l.grid(row=17, column=0, padx=4.5, pady=4.5, sticky=W)
		self.out_raster_wksp_l.grid(row=18, column=0, padx=4.5, pady=4.5, sticky=W)
		self.status_l.grid(row=19, column=0, padx=10, pady=5, sticky=W)
		self.status_val_l.grid(row=19, column=1, padx=10, pady=5, sticky=W)

	# Backend
	def move_to_max_check_fun(self):
		if self.move_to_maximum_par.get() == 0:
			self.move_to_maximum_dist_ent.config(state='disabled')
			self.move_to_maximum_dist_l.config(state='disabled')
			self.move_to_maximum_l.config(state='disabled')
			self.move_to_maximum_dist_par.set('')
		else:
			self.move_to_maximum_dist_ent.config(state='enabled')
			self.move_to_maximum_dist_l.config(state='enabled')
			self.move_to_maximum_l.config(state='enabled')
			self.move_to_maximum_dist_par.set(3)

	def mh_iteration_check_fun(self):
		if self.mh_iteration_par.get() == 0:
			self.mh_iteration_l.config(state='disabled')
			self.mh_dilation_value_ent.config(state='enabled')
			self.mh_dilation_value_l.config(state='enabled')
			self.mh_dilation_value_start_ent.config(state='disabled')
			self.mh_dilation_value_start_l.config(state='disabled')
			self.mh_dilation_value_stop_ent.config(state='disabled')
			self.mh_dilation_value_stop_l.config(state='disabled')
			self.mh_dilation_step_ent.config(state='disabled')
			self.mh_dilation_step_l.config(state='disabled')
			self.mh_dilation_value_par.set(1)
			self.mh_dilation_value_start_par.set('')
			self.mh_dilation_value_stop_par.set('')
			self.mh_dilation_step_par.set('')
		else:
			self.mh_iteration_l.config(state='enabled')
			self.mh_dilation_value_ent.config(state='disabled')
			self.mh_dilation_value_l.config(state='disabled')
			self.mh_dilation_value_start_ent.config(state='enabled')
			self.mh_dilation_value_start_l.config(state='enabled')
			self.mh_dilation_value_stop_ent.config(state='enabled')
			self.mh_dilation_value_stop_l.config(state='enabled')
			self.mh_dilation_step_ent.config(state='enabled')
			self.mh_dilation_step_l.config(state='enabled')
			self.mh_dilation_value_par.set('')
			self.mh_dilation_value_start_par.set(0.1)
			self.mh_dilation_value_stop_par.set(1)
			self.mh_dilation_step_par.set(0.1)

	def approach_combo_fun(self, event):
		if self.approach_par.get() == 'Locations in the DEM generated from field observations':
			self.point_matrix_size_ent.config(state='enabled')
			self.input_point_vectors_ent.config(state='enabled')
			self.input_point_vectors_but.config(state='enabled')
			self.input_point_vectors_l.config(state='enabled')
			self.mapping_field_combo_box.config(state='enabled')
			self.mapping_field_l.config(state='enabled')
			self.out_sim_matrix_ent.config(state='enabled')
			self.out_sim_matrix_l.config(state='enabled')
			self.out_sim_matrix_but.config(state='enabled')

			self.predefined_pattern_l.config(state='disabled')
			self.predefined_pattern_combo_box.config(state='disabled')
			self.predefined_pattern_combo_box['value'] = ([''])
			self.predefined_pattern_combo_box.current(0)
			self.input_pattern_folder_ent.config(state='disabled')
			self.input_pattern_button.config(state='disabled')
			self.input_pattern_folder_l.config(state='disabled')

			self.move_to_maximum_dist_par.set('')

			self.out_table_l.config(state='disabled')
			self.out_table_ent.config(state='disabled')
			self.out_table_par.set('')
			self.out_table_but.config(state='disabled')
			self.out_raster_wksp_l.config(state='disabled')
			self.out_raster_wksp_but.config(state='disabled')
			self.out_raster_wksp_ent.config(state='disabled')
			self.out_raster_wksp_par.set('')

		elif self.approach_par.get() == 'Locations in the DEM versus pre-defined pattern':
			self.point_matrix_size_ent.config(state='enabled')
			self.input_point_vectors_ent.config(state='enabled')
			self.input_point_vectors_but.config(state='enabled')
			self.input_point_vectors_l.config(state='enabled')
			self.mapping_field_combo_box.config(state='enabled')
			self.mapping_field_l.config(state='enabled')
			self.predefined_pattern_combo_box['value'] = ('Mexican Hat wavelet', 'Custom pattern')
			self.predefined_pattern_combo_box.current(0)
			self.predefined_pattern_combo_box.config(state='readonly')
			self.predefined_pattern_l.config(state='enabled')
			self.out_table_l.config(state='enabled')
			self.out_table_ent.config(state='enabled')
			self.out_table_but.config(state='enabled')
			self.out_table_par.set('')

			self.out_sim_matrix_ent.config(state='disabled')
			self.out_sim_matrix_l.config(state='disabled')
			self.out_sim_matrix_but.config(state='disabled')
			self.out_sim_matrix_par.set('')
			self.out_raster_wksp_l.config(state='disabled')
			self.out_raster_wksp_but.config(state='disabled')
			self.out_raster_wksp_ent.config(state='disabled')
			self.out_raster_wksp_par.set('')

			if self.predefined_pattern_par.get() == 'Mexican Hat wavelet':
				self.mh_iteration_checkbox.config(state='enabled')
				self.mh_dilation_value_l.config(state='enabled')
				self.mh_dilation_value_ent.config(state='enabled')
				self.mh_dilation_value_par.set(1)

				self.input_pattern_folder_l.config(state='disabled')
				self.input_pattern_button.config(state='disabled')
				self.input_pattern_folder_par.set('')

			elif self.predefined_pattern_par.get() == 'Custom pattern':
				self.mh_iteration_checkbox.config(state='disabled')
				self.mh_dilation_value_l.config(state='disabled')
				self.mh_dilation_value_ent.config(state='disabled')
				self.mh_dilation_value_par.set('')

				self.input_pattern_folder_l.config(state='enabled')
				self.input_pattern_button.config(state='enabled')

		elif self.approach_par.get() == 'Seek occurrence of pre-defined pattern in the DEM':
			self.predefined_pattern_combo_box['value'] = ('Mexican Hat wavelet', 'Custom pattern')
			self.predefined_pattern_combo_box.current(0)
			self.predefined_pattern_combo_box.config(state='readonly')
			self.predefined_pattern_l.config(state='enabled')
			self.out_raster_wksp_l.config(state='enabled')
			self.out_raster_wksp_but.config(state='enabled')
			self.out_raster_wksp_ent.config(state='enabled')

			if self.predefined_pattern_par.get() == 'Mexican Hat wavelet':
				self.mh_iteration_checkbox.config(state='enabled')
				self.mh_dilation_value_l.config(state='enabled')
				self.mh_dilation_value_ent.config(state='enabled')
				self.mh_dilation_value_par.set(1)

			self.out_sim_matrix_ent.config(state='disabled')
			self.out_sim_matrix_l.config(state='disabled')
			self.out_sim_matrix_but.config(state='disabled')
			self.out_sim_matrix_par.set('')
			self.input_point_vectors_ent.config(state='disabled')
			self.input_point_vectors_but.config(state='disabled')
			self.input_point_vectors_par.set('')
			self.input_point_vectors_l.config(state='disabled')
			self.out_table_l.config(state='disabled')
			self.out_table_ent.config(state='disabled')
			self.out_table_but.config(state='disabled')
			self.out_table_par.set('')
			self.mapping_field_combo_box.config(state='disabled')
			self.mapping_field_l.config(state='disabled')
			self.mapping_field_par.set('')


	def predefined_pattern_fun(self, event):
		if self.predefined_pattern_par.get() == 'Mexican Hat wavelet':
			self.mh_dilation_value_par.set(1)
			self.mh_dilation_value_l.config(state='enabled')
			self.mh_dilation_value_ent.config(state='enabled')
			self.mh_iteration_checkbox.config(state='enabled')

			self.input_pattern_folder_l.config(state='disabled')
			self.input_pattern_button.config(state='disabled')
			self.input_pattern_folder_par.set('')

		elif self.predefined_pattern_par.get() == 'Custom pattern':  # it is predefined pattern

			self.mh_iteration_checkbox.config(state='disabled')
			self.mh_dilation_value_l.config(state='disabled')
			self.mh_dilation_value_ent.config(state='disabled')
			self.mh_dilation_value_par.set('')

			self.input_pattern_folder_l.config(state='enabled')
			self.input_pattern_button.config(state='enabled')

			self.mh_dilation_value_ent.config(state='disabled')
			self.mh_dilation_value_l.config(state='disabled')
			self.mh_dilation_value_par.set('')
			self.mh_dilation_value_start_par.set('')
			self.mh_dilation_value_start_ent.config(state='disabled')
			self.mh_dilation_value_start_l.config(state='disabled')
			self.mh_dilation_value_stop_par.set('')
			self.mh_dilation_value_stop_ent.config(state='disabled')
			self.mh_dilation_value_stop_l.config(state='disabled')
			self.mh_dilation_step_par.set('')
			self.mh_dilation_step_ent.config(state='disabled')
			self.mh_dilation_step_l.config(state='disabled')
			self.mh_iteration_checkbox.config(state='disabled')
			self.mh_iteration_par.set(0)


	def transform_combo_fun(self, event):
		if self.transform_par.get() == 'Compute slopes' or \
				self.transform_par.get() == 'Compute slopes and perform local translation':
			self.size_of_cell_ent.config(state='enabled')
			self.size_of_cell_l.config(state='enabled')
			self.size_of_the_cell_par.set(1)
		else:
			self.size_of_cell_ent.config(state='disabled')
			self.size_of_cell_l.config(state='disabled')
			self.size_of_the_cell_par.set('')


	def open_raster_file(self):
		file = filedialog.askopenfilename(initialdir="/", title="Select raster", filetypes=(("tif files", "*.tif"),
																							("all files", "*.*")))
		if file:
			self.raster_par.set(file)

	def open_point_file(self):
		file = filedialog.askopenfilename(initialdir="/", title="Select point file", filetypes=(("Shp files", "*.shp"),
																								("All files", "*.*")))
		if file:
			self.input_point_vectors_par.set(file)

			self.mapping_field_combo_box['value'] = ([''])

			self.set_field_names(file)

	def open_directory(self):
		dir = filedialog.askdirectory(initialdir="/", title="Select directory...")
		if dir:
			self.input_pattern_folder_par.set(dir)

	def save_sim_matrix(self):
		file = filedialog.asksaveasfilename(initialdir="/", title="Select file", filetypes=(("csv files", "*.csv"),
																							("xlsx files", "*.xlsx"),
																							("all files", "*.*")))
		if file:
			self.out_sim_matrix_par.set(file)

	def save_table(self):
		file = filedialog.asksaveasfilename(initialdir="/", title="Select file", filetypes=(("csv files", "*.csv"),
																							("xlsx files", "*.xlsx"),
																							("all files", "*.*")))
		if file:
			self.out_table_par.set(file)

	def out_raster_wksp(self):
		dir = filedialog.askdirectory(initialdir="/", title="Select directory...")
		if dir:
			self.out_raster_wksp_par.set(dir)

	def set_field_names(self, shp_file):

		try:
			ds = ogr.Open(shp_file)
			lyr = ds.GetLayer()
			field_names = [field.name for field in lyr.schema]
			self.mapping_field_combo_box['value'] = field_names
			self.mapping_field_combo_box.current(0)

			self.mapping_field_l.configure(style='MY.TLabel')
			self.mapping_field_combo_box.configure(style='Def.TCombobox')

		except Exception as e:
			log_msg("Cannot find a field from the shapefile to use as mapping field. Please check the input shapefile!", 2)
			self.mapping_field_combo_box.config(style='Red.TCombobox')
			self.mapping_field_combo_box['value'] = '#####'
			self.mapping_field_par.set('#####')
			self.mapping_field_combo_box.current(0)

			self.mapping_field_l.configure(style='Red.TLabel')

	def run_tool(self):

		params_mapping ={
							"raster_par": ["input_raster", ""],
							"approach_par": ["approach", ""],
							"predefined_pattern_par": ["predefined_pattern", ""],
							"input_pattern_folder_par": ["pattern_workspace", ""],
							"point_matrix_size_par": ["point_matrix_size", ""],
							"input_point_vectors_par": ["input_vector_observations", ""],
							"mapping_field_par": ["mapping_field", ""],
							"move_to_maximum_par": ["move_to_max", ""],
							"move_to_maximum_dist_par": ["move_to_max_distance", ""],
							"mh_iteration_par": ["mh_iteration", ""],
							"mh_dilation_value_par": ["mh_dil_val", ""],
							"mh_dilation_value_start_par": ["mh_start_dil_val", ""],
							"mh_dilation_value_stop_par": ["mh_end_dil_val", ""],
							"mh_dilation_step_par": ["mh_dil_step", ""],
							"transform_par": ["transform", ""],
							"size_of_the_cell_par": ["size_of_cell", ""],
							"out_sim_matrix_par": ["output_sim_matrix", ""],
							"out_table_par": ["output_table", ""],
							"out_raster_wksp_par": ["output_raster_workspace", ""]
						}

		log_msg('Running tool...', 0)
		all_members = self.__dict__.keys()
		list_all = [(item, self.__dict__[item]) for item in all_members if not item.startswith("_")]
		list_pars = [i[0] for i in list_all if 'par' in i[0] and ('status' not in i[0])]
		list_labels = [i[0] for i in list_all if '_l' in i[0] and ('frame' not in i[0] and 'status' not in i[0])]  # all entries plus comboboxes
		list_tup = zip(list_labels, list_pars)

		for tuple in list_tup:
			label = tuple[0]
			par = tuple[1]
			val = 'self.' + label + "['state']"

			result = str(eval(val))

			if result == 'normal' or result == 'enabled':
				val_test = 'self.' + par + '.get()'

				try:
					result = str(eval(val_test))
				except Exception as e:
					result = ''

				params_mapping[par][1] = result


		# call main
		input_raster = params_mapping['raster_par'][1]
		approach = params_mapping['approach_par'][1]

		predefined_pattern = params_mapping['predefined_pattern_par'][1]
		pattern_workspace = params_mapping['input_pattern_folder_par'][1]
		point_matrix_size = params_mapping['point_matrix_size_par'][1]

		input_vector_observations = params_mapping['input_point_vectors_par'][1]

		mapping_field = params_mapping['mapping_field_par'][1]
		mh_dil_val = params_mapping['mh_dilation_value_par'][1]
		mh_iteration = params_mapping['mh_iteration_par'][1]
		mh_start_dil_val =  params_mapping['mh_dilation_value_start_par'][1]
		mh_end_dil_val = params_mapping['mh_dilation_value_stop_par'][1]
		mh_dil_step = params_mapping['mh_dilation_step_par'][1]

		move_to_max = params_mapping['move_to_maximum_par'][1]
		move_to_max_distance = params_mapping['move_to_maximum_dist_par'][1]
		size_of_cell = params_mapping['size_of_the_cell_par'][1]

		transform =  params_mapping['transform_par'][1]

		output_sim_matrix = params_mapping['out_sim_matrix_par'][1]
		output_table = params_mapping['out_table_par'][1]
		output_raster_workspace = params_mapping['out_raster_wksp_par'][1]
		#

		Main(input_raster, approach, predefined_pattern, pattern_workspace, point_matrix_size, input_vector_observations,
			 mapping_field, mh_dil_val, mh_iteration, mh_start_dil_val, mh_end_dil_val, mh_dil_step,
			 move_to_max, move_to_max_distance, size_of_cell, transform, output_sim_matrix, output_table, output_raster_workspace)


	def check_status(self):
		all_members = self.__dict__.keys()
		list_all = [(item, self.__dict__[item]) for item in all_members if not item.startswith("_")]
		list_pars = [i[0] for i in list_all if 'par' in i[0] and ('status' not in i[0])]
		list_labels = [i[0] for i in list_all if '_l' in i[0] and ('frame' not in i[0] and 'status' not in i[0])]  # all entries plus comboboxes
		list_tup = zip(list_labels, list_pars)

		for tuple in list_tup:
			label = tuple[0]
			par = tuple[1]
			val = 'self.' + label + "['state']"

			result = str(eval(val))

			if result == 'normal' or result == 'enabled':
				val_test = 'self.' + par + '.get()'

				try:
					result = str(eval(val_test))
				except Exception as e:
					result = ''

				if result != '' and len(str(result).strip()) > 0:
					self.status_checker.append(0)

					# Todo check bad mapping field here
					if self.mapping_field_par.get() == '#####':
						self.status_checker.append(1)

				else:
					self.status_checker.append(1)

		if 1 in self.status_checker:
			self.status_par.set(0)  # not ready to run
		else:
			self.status_par.set(2)  # ready to run

		if self.status_par.get() == 0:
			self.status_val_l.config(style='RedStatus.TLabel', text='Not ready to run!')
			self.run_but.config(state='disabled')
		elif self.status_par.get() == 1:
			self.status_val_l.config(style='YellowStatus.TLabel')
			self.run_but.config(state='enabled')
		elif self.status_par.get() == 2:
			self.status_val_l.config(style='GreenStatus.TLabel', text='Ready to run!')
			self.run_but.config(state='enabled')

		self.status_checker = []


class LoggingHandler(logging.StreamHandler):

	def __init__(self, textctrl):
		logging.StreamHandler.__init__(self)
		self.textctrl = textctrl

	def emit(self, record):
		msg = self.format(record)
		self.textctrl.config(state="normal")
		self.textctrl.insert("end", msg + "\n")
		self.flush()
		self.textctrl.config(state="disabled")



### Tool Backend ###
#Classes
class Reader(object):
	"""
	Class for creating numpy arrays from input rasters
	"""
	def __init__(self, raster_path):
		self.raster_path = raster_path

	def raster_to_numpy(self):
		# Create a raster object, whose extent properties are used in the matrix manipulations
		raster_obj = gdal.Open(self.raster_path)
		# Convert to numpy array
		matrix = np.array(raster_obj.GetRasterBand(1).ReadAsArray())

		return raster_obj, matrix


class Writer(object):
	"""
	Class for writing numpy arrays back to raster type
	"""
	def __init__(self, raster_obj, matrix, out_raster_path):
		"""
		Init function
		:param raster_obj: the raster that is returned from Reader class
		:param raster_convol: the array that results from the convolution operation (should have the same nr or rows/cols)
		:param out_raster_path: the path of the output raster
		:return: None
		"""

		self.raster_obj = raster_obj
		self.raster_matrix = matrix
		self.out_raster_path = out_raster_path

	def write_to_raster(self):

		# Get the info required
		geotransform = self.raster_obj.GetGeoTransform()
		projection = self.raster_obj.GetProjection()
		origin_x = geotransform[0]
		origin_y = geotransform[3]
		pixel_type = self.raster_obj.GetRasterBand(1).DataType
		pixelWidth = geotransform[1]
		pixelHeight = geotransform[5]

		# Write the raster
		cols = self.raster_matrix.shape[1]
		rows = self.raster_matrix.shape[0]

		driver = gdal.GetDriverByName('GTiff')
		outRaster = driver.Create(self.out_raster_path, cols, rows, 1, pixel_type)
		outRaster.SetGeoTransform((origin_x, pixelWidth, 0, origin_y, 0, pixelHeight))
		outband = outRaster.GetRasterBand(1)
		outband.WriteArray(self.raster_matrix)
		outRaster.SetProjection(projection)
		outband.FlushCache()


class MexicanHat(object):
	"""
	Generates a Mexican Hat wavelet filter based on input parameters
	"""

	def __init__(self, filter_size, ker_a, ker_bx=0, ker_by=0, ker_zs=False, ker_norm=False):
		self.d = filter_size
		self.ker_a = ker_a
		self.ker_bx = ker_bx
		self.ker_by = ker_by
		self.ker_zs = ker_zs
		self.ker_norm = ker_norm
		self.lx = 0
		self.ly = 0
		self.i = 0
		self.j = 0

	def generate_MH_filter(self):
		h = np.empty([self.d, self.d], dtype=np.float)
		# global lx
		# global ly
		lx_list = []
		ly_list = []
		x_list = []
		y_list = []

		b = np.array([self.ker_bx, self.ker_by], dtype=np.float)

		for line in range(h.shape[0]):
			for col in range(h.shape[1]):
				lx = col - (self.d - 1) / 2
				ly = (self.d - 1) / 2 - line
				lx_list.append(lx)
				ly_list.append(ly)

		for i_x in lx_list:
			val = (i_x - b[0]) / self.ker_a
			x_list.append(val)
		for j_x in ly_list:
			val = (j_x - b[1]) / self.ker_a
			y_list.append(val)
		xy = np.column_stack((x_list, y_list))

		self.i = 0
		for line in range(h.shape[0]):
			for col in range(h.shape[1]):
				val = (1 - xy[self.i][0] ** 2 - xy[self.i][1] ** 2) *\
					  math.exp(-((xy[self.i][0] ** 2) + (xy[self.i][1] ** 2)) / 2) / math.sqrt(self.ker_a)
				h[line, col] = val
				self.i += 1

		if self.ker_zs:
			sum = np.sum(h)
			v = (-1 * sum) / (h.shape[0] * h.shape[1])
			for line in range(h.shape[0]):
				for col in range(h.shape[1]):
					h[line, col] = v + h[line, col]

		if self.ker_norm:
			h = h /np.linalg.norm(h)

		return h


class InputTester(object):
	"""
	Class to test the input parameters
	"""

	def __init__(self, **kwargs):
		self.input_raster = kwargs.get('input_raster')
		self.input_vector_observations = kwargs.get('input_vector_observations')
		self.move_to_max_distance = (kwargs.get('move_to_max_distance'))
		self.point_matrix_size = (kwargs.get('point_matrix_size'))
		self.ker_dil_start = (kwargs.get('ker_dil_start'))
		self.ker_dil_end = (kwargs.get('ker_dil_end'))
		self.ker_dil_step = (kwargs.get('ker_dil_step'))
		self.raster_workspace = kwargs.get('raster_workspace')

	def test_input(self):
		def test_raster_sp_ref():
			raster_obj = gdal.Open(self.input_raster)
			raster_proj = raster_obj.GetProjection()
			srs_ras = osr.SpatialReference(wkt=raster_proj)

			proj_ras = srs_ras.GetAttrValue('projcs')
			geogsc_ras = srs_ras.GetAttrValue('geogcs')

			return [proj_ras, geogsc_ras]

		def test_vector_sp_ref():
			# get vector spatial ref
			driver = ogr.GetDriverByName('ESRI Shapefile')
			dataset = driver.Open(self.input_vector_observations)

			try:
				layer = dataset.GetLayer()
			except Exception as e:
				log_msg('Invalid Shapefile. Please check the input file!', 2)
				sys.exit(1)

			vector_proj = layer.GetSpatialRef()

			if vector_proj == None:
				log_msg('The input shapefile is not projected. Please project the shapefile!', 2)

			proj_vect = vector_proj.GetAttrValue('projcs')
			proj_geogcs = vector_proj.GetAttrValue('geogcs')

			return [proj_vect, proj_geogcs]

		def records(file):  # todo test
			# generator
			reader = ogr.Open(file)
			layer = reader.GetLayer(0)
			for i in range(layer.GetFeatureCount()):
				feature = layer.GetFeature(i)
				yield json.loads(feature.ExportToJson())

		try:
			rast = gdal.Open(self.input_raster)
			band_nr = rast.RasterCount

			if band_nr > 1:
				return 1, "The input raster must be single-band!"

			else:

				if self.input_vector_observations:
					## Test spatial references
					ras_prj, ras_geogcs = test_raster_sp_ref()
					vec_prj, vec_geogcs = test_vector_sp_ref()

					log_msg("ras prj: {0}, {1}, ras geogcs".format(ras_prj, ras_geogcs), 0)
					log_msg("vec prj: {0}, {1}, vec geogcs".format(vec_prj, vec_geogcs), 0)

					if ras_geogcs != vec_geogcs:  # todo test prj
						return 1, "The raster and vector inputs are not in the same coordinate system! " \
								  "Please project them in the same coordinate system!"
					else:
						return 0, ''
				else:
					return 0, ''
		except:
			return 1, "Cannot read input raster. Please try another file!"


class MetadataElement(object):

	def __init__(self):
		self.metadata_text = 'The tool was run with the following parameters:'

	def add_text(self, text):
		self.metadata_text = self.metadata_text + '\n' + text

	def save_metadata(self, out_folder):
		with open(os.path.join(out_folder, 'Tool_Metadata_OS.txt'), 'w') as outfile:
			outfile.write(self.metadata_text)


class CustomPattern(object):

	def __init__(self, workspace):
		self.workspace = workspace

	def get_patterns(self):

		numpy_filters = []

		pattern_files = [os.path.join(self.workspace, i) for i in os.listdir(self.workspace) if i.endswith('.csv')]

		if len(pattern_files) == 0:
			return []
		else:
			for pat_file in pattern_files:
				np_filter = np.genfromtxt(pat_file, delimiter=',')
				numpy_filters.append((os.path.basename(pat_file).replace('.csv', ''), np_filter))
			return numpy_filters



# Run toolbox class
class RunToolbox(object):
	"""
	Class used to run the process
	"""

	def __init__(self, **kwargs):
		self.input_raster = kwargs.get('input_raster')
		self.point_observations = kwargs.get('point_observations')

	def transform_raster_and_identify_points(self, mapping_field, move_to_max=False, move_to_max_distance=3):
		"""
		Function to transform the input raster into a numpy array
		 and to process the input points to obtain the spatial parameters
		:param mapping_field: the layer field used to map the values
		:param move_to_max: move the points positions to the maximum value found in the raster,
		 within a search distance
		:param move_to_max_distance: the search distance in cell units, that is used to search the maximum cell value
		"""

		self.dict_points = get_points_values(self.point_observations, self.input_raster, mapping_field=mapping_field,
											 move_to_max_value=move_to_max, search_distance=move_to_max_distance)
		self.raster_obj, self.raster_matrix = Reader(self.input_raster).raster_to_numpy()

		return self.dict_points, self.raster_obj, self.raster_matrix

	def process_pattern_vs_point_obs(self, point_id, point_matrix_size, ker_a=1.0, ker_bx=0, ker_by=0,
								ker_zero_sum=False, ker_normalization=False, find_tangents=False, find_tangents_l_value=1.0,
								normalization=False, local_minima=False, existing_filter=None):
		"""
		Function to compare a Mexican Hat wavelet filter with a point matrix
		:param point_id: the id (mapping_field) of the input point
		:param point_matrix_size: the size of the point matrix that is extracted from the raster
		:param ker_a: the dilation parameter of the Mexican Hat wavelet filter
		:param ker_bx: the x translation parameter of the Mexican Hat wavelet filter
		:param ker_by: the y translation parameter of the Mexican Hat wavelet filter
		:param ker_zero_sum: to zero sum the process
		:param ker_normalization: to normalize the convolution process
		:param find_tangents: used to apply the findTangents function to the input point matrix
		:param find_tangents_l_value: the l value of the findTangents function
		:param normalization: to apply normalization on the input matrices
		:param local_minima: to subtract the local minima of the input matrix cells
		:param existing_filter: a filter wich is already created
		:return: the cosine similarity value between the two matrices
		"""

		pattern_filter = existing_filter

		# Generate Mexican Hat filter
		try:
			test = pattern_filter.shape[0]
		except:
			test = -1

		if test == -1:
			log_msg("\nCreating MexicanHat wavelet with the following parameters: "
						  "filter size: {0}, dilation value: {1}, zero summing: {2}, filter normalization: {3}"
						  .format(point_matrix_size, ker_a, ker_zero_sum, ker_normalization), 0)
			pattern_filter = MexicanHat(point_matrix_size, ker_a, ker_bx, ker_by, ker_zero_sum, ker_normalization).generate_MH_filter()
			log_msg("\nCreated filter: {0}".format(pattern_filter), 0)

		# Get Point locations and the point matrix
		row, col = get_row_col_from_dict(self.dict_points, point_id)
		mat_point = slice_point_matrix(self.raster_matrix , row, col, point_matrix_size)
		if find_tangents:
			mh_filter = findTangents(pattern_filter, find_tangents_l_value)
			mat_point = findTangents(mat_point, find_tangents_l_value)

		if local_minima:
			mh_filter = apply_local_minima(pattern_filter)
			mat_point = apply_local_minima(mat_point)

		if normalization:
			mh_filter = normalize_matrix(pattern_filter)
			mat_point = normalize_matrix(mat_point)

		# Perform cosine similarity
		res = cosine_similarity_matrices(pattern_filter, mat_point)
		return res

	def process_point_matrix_vs_point_matrix(self, point_id_1, point_id_2, point_matrix_size,
											 find_tangents=False, find_tangents_l_value=1.0,
											 local_minima=False, normalization=False):
		"""
		:param point_id_1: the id (mapping_field) of the input point 1
		:param point_id_2: the id (mapping_field) of the input point 2
		:param point_matrix_size:  the size of the point matrix that is extracted from the raster
		:param find_tangents: used to apply the findTangents function to the input point matrix
		:param find_tangents_l_value: the l value of the findTangents function
		:param local_minima: to subtract the local minima of the input matrix cells
		:param normalization: to apply normalization on the input matrices
		:return: the cosine similarity value between the two matrices
		"""
		# Get Point locations and the point matrix
		row1, col1 = get_row_col_from_dict(self.dict_points, point_id_1)
		row2, col2 = get_row_col_from_dict(self.dict_points, point_id_2)

		mat_point1 = slice_point_matrix(self.raster_matrix, row1, col1, point_matrix_size)
		mat_point2 = slice_point_matrix(self.raster_matrix, row2, col2, point_matrix_size)

		# if find_tangents and not local_minima and not normalization:
		if find_tangents:
			mat_point1 = findTangents(mat_point1, find_tangents_l_value)
			mat_point2 = findTangents(mat_point2, find_tangents_l_value)

		if local_minima:
			mat_point1 = apply_local_minima(mat_point1)
			mat_point2 = apply_local_minima(mat_point2)

		if normalization:
			mat_point1 = normalize_matrix(mat_point1)
			mat_point2 = normalize_matrix(mat_point2)

		res = cosine_similarity_matrices(mat_point1, mat_point2)

		return res

# Functions
def log_msg(text, mode):

	if mode == 0:
		module_logger.info('INFO: {0}'.format(text))
	elif mode == 1:
		module_logger.warning('WARNING: {0}'.format(text))
	elif mode == 2:
		module_logger.error('ERROR: {0}'.format(text))
	else:
		raise(Exception("Invalid logging mode!"))

def perform_convolution_2D(matrix, kernel, normalization=True, find_tangents=False, size_of_cell=None, init_rast=None):
	"""
	Performs a 2D convolution between two matrices
	# matrix - indexed with (v, w)
	# kernel - indexed with  (s, t),
	# h - output matrix indexed with (x, y),
	# smid and tmid are the kernel centre (floor division)
	# The output size is calculated by adding smid, tmid to each side of the dimensions of the input image.
	:param matrix: the input matrix
	:param kernel: the kernel
	:param normalization: to normalize the operation
	:return: convolution result
	"""

	norm_filter = np.linalg.norm(kernel)

	vmax = matrix.shape[0]
	wmax = matrix.shape[1]
	smax = kernel.shape[0]
	tmax = kernel.shape[1]
	smid = smax // 2
	tmid = tmax // 2
	xmax = vmax + 2 * smid
	ymax = wmax + 2 * tmid
	window_list = []

	# Value to slice the borders that get appended during convolution
	slice_val = int(kernel.shape[0]) // 2

	# Creating the shape of the output matrix
	h = np.zeros([xmax, ymax], dtype=np.float)

	# Applying the convolution
	for x in range(xmax):
		for y in range(ymax):
			# Calculating the pixel value for h[x,y]. Summing the components for each pixel (s,t) from filter (kernel)
			s_from = max(smid - x, -smid)
			s_to = min((xmax - x) - smid, smid + 1)
			t_from = max(tmid - y, -tmid)
			t_to = min((ymax - y) - tmid, tmid + 1)
			value = 0

			for s in range(s_from, s_to):
				for t in range(t_from, t_to):
					v = x - smid + s
					w = y - tmid + t
					value += kernel[smid - s, tmid - t] * matrix[v, w]
					window_list.append(matrix[v, w])

			if normalization:
				window_array = np.asarray(window_list, dtype=float)
				norm_window = np.linalg.norm(window_array)
				calc_value = value / (norm_window * norm_filter)

				h[x, y] = calc_value
				window_list = []

			else:
				h[x, y] = value
				window_list = []

	# Slice the resulted raster to correct borders
	h_slice = h[slice_val:-slice_val, slice_val:-slice_val]

	return h_slice


def calculate_slopes(raster_matrix, filter, size_of_cell, local_translation=False):
	"""
	Function that applies transforms for slope and local translation on the entire raster matrix
	:param raster_matrix: the matrix of the raster
	:param filter: the filter matrix
	:param size_of_cell: the cell size for findTangents()
	:param local_translation: to perform the local translation
	:return: the raster matrix
	"""

	h = np.zeros([raster_matrix.shape[0], raster_matrix.shape[1]], dtype=np.float)

	border_rem_value = (filter.shape[0]-1)//2
	raster_matrix_remove_border = raster_matrix[border_rem_value:-border_rem_value, border_rem_value:-border_rem_value]

	tg_filter = findTangents(filter, size_of_cell)

	for i in np.arange(0, raster_matrix_remove_border.shape[0]):
		for j in np.arange(0, raster_matrix_remove_border.shape[1]):
			sl_mat = slice_point_matrix(raster_matrix, i+border_rem_value, j+border_rem_value, filter.shape[0])  # adapt the slicing to the padded array

			# calculate slopes
			tg_raster_matrix = findTangents(sl_mat, size_of_cell)

			if local_translation:
				out_raster_mat = apply_local_minima(tg_raster_matrix)

				val_res = np.dot(out_raster_mat.flatten(), tg_filter.flatten()) / (np.linalg.norm(out_raster_mat)*np.linalg.norm(tg_filter))
				h[i+border_rem_value, j+border_rem_value] = val_res

			else:
				val_res = np.dot(tg_raster_matrix.flatten(), tg_filter.flatten()) / (np.linalg.norm(tg_raster_matrix)*np.linalg.norm(tg_filter))
				h[i+border_rem_value, j+border_rem_value] = val_res

	return h


def cosine_similarity_matrices(mat1, mat2):
	"""
	Performs the cosine similarity operation between two matrices
	:return: cosine similarity result
	"""
	if (mat1.shape[0] != mat2.shape[0]) or (mat1.shape[1] != mat2.shape[1]):
		raise Exception('The matrices are of different size!')
	else:
		return 1 - spatial.distance.cosine(mat1.flatten(), mat2.flatten())


def findTangents(mat_in, l):
	"""
	Calculates the slopes of an input matrix
	:returns the same matrix shape, with slopes calculated based on tangents
	"""

	def findCircularProxies(row, column, d):

		# center in (0,0)
		r = (row - (d + 1) / 2) + 1
		c = (column - (d + 1) / 2) + 1

		# Compute increments (establish quadrant/ axes)
		incrementr = - np.sign(r)
		incrementc = - np.sign(c)

		# one neighbor is clear
		r1 = r + incrementr
		c1 = c + incrementc
		r2 = r + incrementr
		c2 = c + incrementc

		# determine position wrt to the 1st bisector
		if (c + r) * (c - r) < 0:
			c2 = c2 - incrementc

		elif (c + r) * (c - r) > 0:
			r2 = r2 - incrementr

		# translation back
		r1 = r1 + (d + 1) / 2
		c1 = c1 + (d + 1) / 2
		r2 = r2 + (d + 1) / 2
		c2 = c2 + (d + 1) / 2
		return int(r1 - 1), int(c1 - 1), int(r2 - 1), int(c2 - 1)

	nr = mat_in.shape[0]
	nc = mat_in.shape[1]

	if nr == nc:
		mat_out = np.zeros((nr, nc))

		for ii in range(nc):
			for jj in range(nc):
				ii1, jj1, ii2, jj2 = findCircularProxies(ii, jj, nr)

				vec1 = mat_in[int(ii1), int(jj1)]
				vec2 = mat_in[int(ii2), int(jj2)]
				aux = np.mean([vec1, vec2]) - mat_in[ii, jj]
				mat_out[ii, jj] = aux / l

		return mat_out
	else:
		sys.exit('The input matrix must be square!')


def get_points_values(projectedPointFC, raster, mapping_field, move_to_max_value=False, search_distance=3):
	"""
	Get the points values and row/col from raster
	:param projectedPointFC: point shp/feature class
	:param raster: input raster
	:param mapping_field: the field from the vector layer used to map the trees
	:param move_to_max_value: moves the point to the maximum value found in raster, based on a search distance
	:param search_distance: the search distance to search the maximum value around the point, expressed in cell numbers
	:return: returns all the values in the input shp/ fc, and also the row/ col and value from the raster
	"""
	output_dict = {}

	def map_to_pixel(point_x, point_y, cellx, celly, xmin, ymax, raster_array):
		"""
		Returns the row/column of point values in a shp based on extent properties of a raster
		:param point_x: x coord of the point
		:param point_y: y coord of the point
		:param cellx: x resolution of raster
		:param celly: y resolution of raster
		:param xmin: xmin (top left cell)
		:param ymax: ymax (top left cell)
		:param raster_array: the raster numpy array
		:return: (row, col) of input point
		"""

		row = int((point_y - ymax) / -celly)
		col = int((point_x - xmin) / -cellx)

		if move_to_max_value == True:
			row_max, col_max = move_to_max(row, col, raster_array, search_distance)
			return row_max, col_max
		else:
			return row, col  # to convert to correct value

	def move_to_max(row, col, raster_matrix, search_size):
		"""
		Function to obtain the (row, column) values of a cell, that has the the maximum value
		found within a predefined square buffer (3x3, 5x5, ...).
		The row, col tuple represents the input cell's location in the raster matrix.
		:param row: row value
		:param col: col value
		:param raster_matrix: the numpy matrix of the input raster
		:param search_size: the neighbourhood search distance for the max value
		:return: tuple (row, col) of the cell with the maximum value in the search neighbourhood
		"""

		def get_cell_neighbour_locations(row_cell, col_cell, size):
			"""
			Function to obtain all the neighbour (row, column) locations of a matrix around an input (row, col values),
			with a predefined size.
			The row, col tuple represents the center of the matrix
			:param row_cell: row value
			:param col_cell: col value
			:param size: size of the matrix in cells (3x3, 5x5)
			:return: list of tuples with (row, col) values of each cell in the search neighbourhood
			"""

			# Build matrix of zeros
			matrix_zeros = np.zeros((size, size))

			# Get the center index of the zero matrix
			center = int(matrix_zeros.shape[0]/2)

			list_out = []  # to store the neighbour row, col tuples

			# Iterate the zero matrix
			for row in range(matrix_zeros.shape[0]):
				for col in range(matrix_zeros.shape[1]):

					if row < center:
						if col < center:
							oper_row = (center - row) * -1
							oper_col = (center - col) * -1
							rowval = row_cell + oper_row
							colval = col_cell + oper_col
							list_out.append((rowval, colval))

						elif col == center:
							oper_row = (center - row) * -1
							oper_col = 0
							rowval = row_cell + oper_row
							colval = col_cell + oper_col
							list_out.append((rowval, colval))

						else:
							oper_row = (center - row) * -1
							oper_col = col - center
							rowval = row_cell + oper_row
							colval = col_cell + oper_col
							list_out.append((rowval, colval))

					elif row == center:
						if col < center:
							oper_row = 0
							oper_col = (center - col) * -1
							rowval = row_cell + oper_row
							colval = col_cell + oper_col
							list_out.append((rowval, colval))

						elif col == center:
							oper_row = 0
							oper_col = 0
							rowval = row_cell + oper_row
							colval = col_cell + oper_col
							list_out.append((rowval, colval))

						else:
							oper_row = 0
							oper_col = col - center
							rowval = row_cell + oper_row
							colval = col_cell + oper_col
							list_out.append((rowval, colval))

					else:
						if col < center:
							oper_row = row - center
							oper_col = (center - col) * -1
							rowval = row_cell + oper_row
							colval = col_cell + oper_col
							list_out.append((rowval, colval))

						elif col == center:
							oper_row = row - center
							oper_col = 0
							rowval = row_cell + oper_row
							colval = col_cell + oper_col
							list_out.append((rowval, colval))

						else:
							oper_row = row - center
							oper_col = col - center
							rowval = row_cell + oper_row
							colval = col_cell + oper_col
							list_out.append((rowval, colval))

			return list_out

		# List to store the values of the matrix
		list_matrix_max_values = []

		# Get the (row, col) values of all the cells in the search neighbourhood
		list_vals = get_cell_neighbour_locations(row, col, search_size)

		for loc in list_vals:
			# Get the raster value of the input row, col cell position
			val = raster_matrix[loc[0]][loc[1]]
			list_matrix_max_values.append((loc, val))

		sublist = []

		for i in list_matrix_max_values:
			sublist.append(i[1])

		max_val = max(sublist)
		max_point = []

		# Find the cell with the maximum value
		for i in list_matrix_max_values:
			if max_val == i[1]:
				max_point = i

		return max_point[0][0], max_point[0][1]  # return row, col of max value

	##
	# process vector
	# wgs = 'GEOGCS["GCS_WGS_1984",' + \
	# 	  'DATUM["D_WGS_1984",' + \
	# 	  'SPHEROID["WGS_1984",6378137,298.257223563]],' + \
	# 	  'PRIMEM["Greenwich",0],' + \
	# 	  'UNIT["Degree",0.017453292519943295]]'  # can be used to convert to WGS point coordinates, unused
	# fields = ['OID@', 'SHAPE@XY']  # get OID (index) and coordinates fields, unused

	# Get the fields from SHP
	dataSource = ogr.Open(projectedPointFC, 0)
	dataLyr = dataSource.GetLayer(0)
	layerDefinition = dataLyr.GetLayerDefn()

	field_names = [layerDefinition.GetFieldDefn(i).GetName() for i in range(layerDefinition.GetFieldCount())]

	raster_obj = gdal.Open(raster)
	raster_arr = np.array(raster_obj.GetRasterBand(1).ReadAsArray())
	geotransform = raster_obj.GetGeoTransform()

	# process raster
	cellx = geotransform[5]
	celly = geotransform[1]

	xmin = geotransform[0]
	ymax = geotransform[3]

	# iterate point fc and get row, col and value

	for feature in dataLyr:
		geometry = feature.GetGeometryRef()
		x_val = geometry.GetX()  # x coord
		y_val = geometry.GetY()  # y coord

		shape_all_values = [(feature.GetField(field), field) for field in field_names]

		row, col = map_to_pixel(x_val, y_val, cellx, celly, xmin, ymax, raster_array=raster_arr)
		raster_val = raster_arr[row, col]

		# build the output dictionary
		output_dict[feature.GetField(mapping_field)] = [shape_all_values, ('Row', row), ('Col', col),
														('RasterVal', raster_val)]

	return output_dict


def slice_point_matrix(raster_matrix, row, col, size):
	"""
	Slices a square matrix from a raster, based on row/col from points,
	and size of the submatrix that is sliced (3x3, 5x5...)
	:param raster_matrix: the numpy matrix of a raster
	:param row: the row value of the point (cell)
	:param col: the column value of the point (cell)
	:param size: the size of the matrix (3x3, 5x5, ...), window "radius"; S=3 gives a 5x5 submatrix
	:return: a matrix extracted from the numpy array
	"""

	size = int(size)  # to conver the str from toolbox

	if size % 2 == 0:
		exit("\nThe neighbourhood search distance must be an odd value!")
	else:
		size = int((size//2)+1)

		mat_slice = raster_matrix[row - size + 1:row + size, col - size + 1:col + size]

		return mat_slice


def get_row_col_from_dict(points_dict, dict_key):
	"""
	Function to get the row column from a dict object obtained from a feature class/ vector (function get_points_values)
	:param points_dict: the dict of points obtained from 'get_points_values' function
	:param dict_key: the key of the point, that is used to extract the matrix, type int
	:return: tuple of row and column (row, col) corresponding to the ID of the feature
	"""

	# Get row, value from dict
	# val = points_dict.get(int(dict_key))  #old
	val = points_dict.get(dict_key)
	row = val[1][1]
	col = val[2][1]

	return (row, col)


def apply_local_minima(input_matrix):
	"""
	Function to return the local minima values of a matrix.
	For each cell in the input matrix, the minimum value of the whole matrix is subtracted
	:param input_matrix: the input numpy matrix
	:return: the output numpy matrix, with minimum values subtracted
	"""

	return input_matrix - np.min(input_matrix)


def normalize_matrix(input_matrix):
	"""
	Function to normalize an input matrix
	:param input_matrix: the input numpy matrix
	:return: the output normalized numpy matrix
	"""

	norm_value = np.linalg.norm(input_matrix)

	return input_matrix / norm_value



def get_mapping_values(input_vector, field):
	"""
	Function to return the values of the mapping column
	:param input_vector: feature class
	:param field: field of the input vector file
	:return: list of values used to map the output file
	"""

	dataSource = ogr.Open(input_vector, 0)
	daLayer = dataSource.GetLayer(0)

	list_oid_init = [feature.GetField(field) for feature in daLayer]

	if len(list_oid_init) != len(set(list_oid_init)):
		log_msg("\nThere are duplicate values in the mapping column. "
						 "This may result in confusion when interpreting the results!", 1)

	return list_oid_init


def Main(input_raster, approach, predefined_pattern, pattern_workspace, point_matrix_size, input_vector_observations,
			 mapping_field, mh_dil_val, mh_iteration, mh_start_dil_val, mh_end_dil_val, mh_dil_step,
			 move_to_max, move_to_max_distance, size_of_cell, transform, output_sim_matrix, output_table, output_raster_workspace):

	input_raster = input_raster
	approach = approach

	predefined_pattern = predefined_pattern
	pattern_workspace = pattern_workspace
	point_matrix_size = int(point_matrix_size)

	input_vector_observations = input_vector_observations

	mapping_field = mapping_field
	if mh_dil_val != '':
		mh_dil_val = float(mh_dil_val)

	if mh_iteration == '':
		mh_iteration = False
	else:
		mh_iteration = True

	if mh_start_dil_val != '':
		mh_start_dil_val = float(mh_start_dil_val)

	if mh_end_dil_val != '':
		mh_end_dil_val = float(mh_end_dil_val)
	if mh_dil_step != '':
		mh_dil_step = float(mh_dil_step)

	if move_to_max == '':
		move_to_max = False
	else:
		move_to_max = True

	if move_to_max_distance != '':
		move_to_max_distance = int(move_to_max_distance)

	if size_of_cell != '':
		size_of_cell = float(size_of_cell)

	output_sim_matrix = output_sim_matrix
	output_table = output_table
	output_raster_workspace = output_raster_workspace

	## Instantiate metadata element
	metadata = MetadataElement()

	# Get raster paths
	raster_source = input_raster

	metadata.add_text('Input raster: {0}'.format(raster_source))
	metadata.add_text('Approach: {0}'.format(approach))
	metadata.add_text('Transform: {0}'.format(transform))
	metadata.add_text('Move to maximum: {0}'.format(move_to_max))

	# Test parameters
	test_result, test_message = InputTester(input_raster=input_raster, input_vector_observations=input_vector_observations,
				move_to_max_distance=move_to_max_distance,
				point_matrix_size=point_matrix_size,
				ker_dil_start=mh_start_dil_val, ker_dil_end=mh_end_dil_val, ker_dil_step=mh_dil_step,
				raster_workspace=output_raster_workspace).test_input()

	if test_result == 1:
		log_msg(test_message, 2)
	else:
		dict_rename = {}  # to rename the rows of the pandas data frame
		pandas_data_result = {}

		## Run tool
		# Parse transform
		if transform == 'Perform a local translation':
			find_tangents = False
			local_minima = True
		elif transform == 'Compute slopes':
			find_tangents = True
			local_minima = False
		elif transform == 'Compute slopes' or transform == 'Compute slopes and perform local translation':
			find_tangents = True
			local_minima = True
		else:
			find_tangents = False
			local_minima = False

		if approach == 'Locations in the DEM generated from field observations':
			log_msg("\nLocations in the DEM generated from field observations", 0)

			tool = RunToolbox(input_raster=input_raster, point_observations=input_vector_observations)

			mapping_list = get_mapping_values(input_vector_observations, mapping_field)
			for index, objid_1 in enumerate(mapping_list):
				list_vals_result = []
				dict_rename[index] = objid_1

				for objid_2 in mapping_list:
					result_cossim = tool.process_point_matrix_vs_point_matrix(point_id_1=objid_1, point_id_2=objid_2,
																			  point_matrix_size=point_matrix_size,
																			  find_tangents=find_tangents,
																			  find_tangents_l_value=size_of_cell,
																			  local_minima=local_minima,
																			  normalization=True)
					list_vals_result.append(result_cossim)

				pandas_data_result[objid_1] = list_vals_result

			df1 = pd.DataFrame(pandas_data_result, columns=mapping_list)

			df1_ren = df1.rename(index=dict_rename)

			if output_sim_matrix.endswith('.csv'):
				df1_ren.to_csv(output_sim_matrix)
			elif output_sim_matrix.endswith('.xls') or output_sim_matrix.endswith('.xlsx'):
				df1_ren.to_excel(output_sim_matrix)

			# Get points path
			points_source = input_vector_observations

			metadata.add_text('Input Point Vectors: {0}'.format(points_source))
			metadata.add_text('Mapping field: {0}'.format(mapping_field))
			metadata.add_text('Output similarity matrix: {0}'.format(output_sim_matrix))

			RUN_RESULT = 0

		elif approach == 'Locations in the DEM versus pre-defined pattern':  # MexicanHat
			log_msg("\n Locations in the DEM versus pre-defined pattern", 0)
			# Get points path
			points_source = input_vector_observations

			metadata.add_text('Input Point Vectors: {0}'.format(points_source))
			metadata.add_text('Mapping field: {0}'.format(mapping_field))
			metadata.add_text('Predefined pattern: {0}'.format(predefined_pattern))
			metadata.add_text('Point matrix size: {0}'.format(point_matrix_size))

			tool = RunToolbox(input_raster=input_raster, point_observations=input_vector_observations)

			mapping_list = get_mapping_values(input_vector_observations, mapping_field)

			if predefined_pattern == 'Mexican Hat wavelet':
				metadata.add_text('Mexican Hat wavelet iteration: {0}'.format(mh_iteration))

				if not mh_iteration:
					metadata.add_text('Mexican Hat dilation value: {0}'.format(mh_dil_val))

					pattern_filter = MexicanHat(filter_size=int(point_matrix_size), ker_a=mh_dil_val, ker_zs=False,
												ker_norm=True).generate_MH_filter()

					for index, obs_id in enumerate(mapping_list):
						list_vals_result = []
						dict_rename[index] = mh_dil_val

						result_cossim = tool.process_pattern_vs_point_obs(point_id=obs_id,
																		  point_matrix_size=point_matrix_size,
																		  ker_a=mh_dil_val,
																		  ker_zero_sum=False, ker_normalization=True,
																		  find_tangents=find_tangents,
																		  find_tangents_l_value=size_of_cell,
																		  local_minima=local_minima, normalization=True,
																		  existing_filter=pattern_filter)

						list_vals_result.append(result_cossim)
						pandas_data_result[obs_id] = list_vals_result
						df1 = pd.DataFrame(pandas_data_result, columns=mapping_list)
						df1_ren = df1.rename(index=dict_rename)
						df1_ren = df1_ren.T  # transpose matrix

						df1_ren.to_csv(output_table)

					RUN_RESULT = 0


				else:
					# MH iteration
					metadata.add_text('Iteration start: {0}'.format(str(mh_start_dil_val)))
					metadata.add_text('Iteration stop: {0}'.format(str(mh_end_dil_val)))
					metadata.add_text('Iteration step: {0}'.format(str(mh_dil_step)))

					dil_interval = np.arange(float(mh_start_dil_val), float(mh_end_dil_val) + float(mh_dil_step),
											 float(mh_dil_step))

					frames = []
					df1_ren = {}

					for dil_val in dil_interval:
						for index, obs_id in enumerate(mapping_list):
							pattern_filter = MexicanHat(filter_size=int(point_matrix_size), ker_a=dil_val,
														ker_zs=False,
														ker_norm=True).generate_MH_filter()

							list_vals_result = []
							dict_rename[index] = dil_val

							result_cossim = tool.process_pattern_vs_point_obs(point_id=obs_id,
																			  point_matrix_size=point_matrix_size,
																			  ker_a=dil_val,
																			  ker_zero_sum=False, ker_normalization=True,
																			  find_tangents=find_tangents,
																			  find_tangents_l_value=size_of_cell,
																			  local_minima=local_minima, normalization=True,
																			  existing_filter=pattern_filter)

							list_vals_result.append(result_cossim)

							pandas_data_result[obs_id] = list_vals_result
							df1 = pd.DataFrame(pandas_data_result, columns=mapping_list)

							df1_ren = df1.rename(index=dict_rename)
						# df1_ren.to_csv(output_result_file)
						frames.append(df1_ren)

					d_out = pd.concat(frames)
					d_out = d_out.T

					d_out.to_csv(output_table)

					if output_table.endswith('.csv'):
						d_out.to_csv(output_table)
					elif output_table.endswith('.xls') or output_table.endswith('.xlsx'):
						d_out.to_excel(output_table)

					RUN_RESULT = 0

			else:
				patterns_list = CustomPattern(pattern_workspace).get_patterns()
				metadata.add_text('Custom patterns workspace: {0}'.format(pattern_workspace))

				if len(patterns_list) > 0:

					frames = []
					df1_ren = {}

					for iter_id, pattern_filter_tuple in enumerate(patterns_list):
						pattern_filter_file, pattern_filter = pattern_filter_tuple

						for index, obs_id in enumerate(mapping_list):
							list_vals_result = []
							dict_rename[index] = pattern_filter_file

							result_cossim = tool.process_pattern_vs_point_obs(point_id=obs_id,
																			  point_matrix_size=int(point_matrix_size),
																			  ker_a=float(1),
																			  ker_zero_sum=False, ker_normalization=True,
																			  find_tangents=find_tangents,
																			  find_tangents_l_value=size_of_cell,
																			  local_minima=local_minima, normalization=True,
																			  existing_filter=pattern_filter)
							list_vals_result.append(result_cossim)

							pandas_data_result[obs_id] = list_vals_result
							df1 = pd.DataFrame(pandas_data_result, columns=mapping_list)

							df1_ren = df1.rename(index=dict_rename)
						frames.append(df1_ren)

						d_out = pd.concat(frames)
						d_out = d_out.T

						if output_table.endswith('.csv'):
							d_out.to_csv(output_table)
						elif output_table.endswith('.xls') or output_table.endswith('.xlsx'):
							d_out.to_excel(output_table)

					RUN_RESULT = 0

				else:
					log_msg("The input patterns workspace is empty! No .csv files found!", 2)
					RUN_RESULT = 2

			metadata.add_text('Output table: {0}'.format(output_table))

		else:  # Seek occurrence of pre-defined pattern in the DEM
			log_msg("Seek occurrence of pre-defined pattern in the DEM", 0)
			raster_obj, raster_matrix = Reader(input_raster).raster_to_numpy()

			metadata.add_text('Predefined pattern: {0}'.format(predefined_pattern))
			metadata.add_text('Point matrix size: {0}'.format(point_matrix_size))

			if predefined_pattern == 'Mexican Hat wavelet':
				metadata.add_text('Mexican Hat wavelet iteration: {0}'.format(mh_iteration))

				if not mh_iteration:
					# No MH iteration
					metadata.add_text('Mexican Hat wavelet dilation value: {0}'.format(mh_dil_val))
					pattern_filter = MexicanHat(filter_size=int(point_matrix_size), ker_a=mh_dil_val, ker_zs=False,
												ker_norm=True).generate_MH_filter()

					if transform == 'Compute slopes' or transform == 'Compute slopes and perform local translation':
						raster_conv = calculate_slopes(raster_matrix, pattern_filter, int(size_of_cell),
													   local_translation=local_minima)

					else:
						raster_conv = perform_convolution_2D(raster_matrix, pattern_filter, normalization=True,
															 find_tangents=find_tangents)

					out_raster_file = os.path.join(output_raster_workspace,
													   'Convolution_Raster_d' + str(point_matrix_size)
													   + '_s' + str(mh_dil_val) + '.tif')
					Writer(raster_obj=raster_obj, matrix=raster_conv, out_raster_path=out_raster_file).write_to_raster()

					RUN_RESULT = 0

				else:
					dil_interval = np.arange(float(mh_start_dil_val), float(mh_end_dil_val) + float(mh_dil_step),
											 float(mh_dil_step))

					list_a = []

					metadata.add_text('Iteration start: {0}'.format(str(mh_start_dil_val)))
					metadata.add_text('Iteration stop: {0}'.format(str(mh_end_dil_val)))
					metadata.add_text('Iteration step: {0}'.format(str(mh_dil_step)))

					for dil_val in dil_interval:
						list_a.append(dil_val)
						pattern_filter = MexicanHat(filter_size=int(point_matrix_size), ker_a=dil_val, ker_zs=False,
													ker_norm=True).generate_MH_filter()

						if transform == 'Compute slopes' or transform == 'Compute slopes and perform local translation':
							raster_conv = calculate_slopes(raster_matrix, pattern_filter, int(size_of_cell),
														   local_translation=local_minima)

						else:
							raster_conv = perform_convolution_2D(raster_matrix, pattern_filter, normalization=True,
																 find_tangents=find_tangents)

						out_raster_file = os.path.join(output_raster_workspace,
														   'Convolution_Raster_d' + str(point_matrix_size) + '_s' + \
														   str(dil_val) + '.tif')

						Writer(raster_obj=raster_obj, matrix=raster_conv, out_raster_path=out_raster_file).write_to_raster()

					RUN_RESULT = 0

			else:
				# One or more filters
				patterns_list = CustomPattern(pattern_workspace).get_patterns()

				for iter_id, pattern_filter_tuple in enumerate(patterns_list):
					pattern_filter_file, pattern_filter = pattern_filter_tuple

					if transform == 'Compute slopes' or transform == 'Compute slopes and perform local translation':
						raster_conv = calculate_slopes(raster_matrix, pattern_filter, size_of_cell,
													   local_translation=local_minima)

					else:
						raster_conv = perform_convolution_2D(raster_matrix, pattern_filter, normalization=True,
															 find_tangents=find_tangents)

					if output_raster_workspace.endswith('.gdb'):
						out_raster_file = os.path.join(output_raster_workspace,
													   'Convolution_Raster_' + str(pattern_filter_file))
					else:
						out_raster_file = os.path.join(output_raster_workspace,
													   'Convolution_Raster_' + str(pattern_filter_file) + '.tif')

					Writer(raster_obj=raster_obj, matrix=raster_conv,
						   out_raster_path=out_raster_file).write_to_raster()

				RUN_RESULT = 0

			metadata.add_text('Raster workspace: {0}'.format(output_raster_workspace))

		metadata.save_metadata(tempfile.gettempdir())
		log_msg('Saved metadata to: {0}'.format(os.path.join(tempfile.gettempdir(), 'Tool_Metadata_OS.txt')), 0)

		# Finished
		if RUN_RESULT == 2:
			tkinter.messagebox.showerror("Error", "Run completed with errors. Please check the log messages and metadata file!")
		elif RUN_RESULT == 1:
			tkinter.messagebox.showwarning("Warning", "Run completed with warnings. Please check the log messages and metadata file!")
		else:
			tkinter.messagebox.showinfo("Info", "Run completed successfully. Please check the log messages and metadata file for details!")


# --------------------------------------------------------------------- #
if __name__ == "__main__":

	root = Tk()

	gui_style = ttk.Style()
	gui_style.theme_use('clam')

	gui_style.configure('My.TFrame', background='#F2E9DC')

	gui_style.map('Default.TButton',
		foreground=[('disabled', 'gray'),
                    ('active', 'black')],
        background=[
                    ('pressed', '!focus', 'cyan'),
                    ('active', '#91C499')],
        highlightcolor=[('focus', '#91C499'),
                        ('!focus', 'red')],
        relief=[('pressed', 'groove'),
                ('!pressed', 'ridge')])

	gui_style.configure('GreenStatus.TLabel', background='#00ff00')
	gui_style.configure('YellowStatus.TLabel', background='#ffff00')
	gui_style.configure('RedStatus.TLabel', background='#ff0000')

	gui_style.configure('Def.TCombobox')
	gui_style.configure('Red.TCombobox', foreground='#ff0000', background='#ff0000', font=('arial', 12))
	gui_style.configure('Red.TLabel', foreground='#ff0000', font=('arial', 12))

	root.configure(bg='#F2E9DC')

	interface_obj = GUI(root)

	stderr_handler = logging.StreamHandler()
	module_logger.addHandler(stderr_handler)
	gui_handler = LoggingHandler(interface_obj.mytext)
	module_logger.addHandler(gui_handler)
	module_logger.setLevel(logging.INFO)

	# Check the status of the GUI
	def task_check_status():
		interface_obj.check_status()
		root.after(500, task_check_status)

	root.after(500, task_check_status)

	root.mainloop()
