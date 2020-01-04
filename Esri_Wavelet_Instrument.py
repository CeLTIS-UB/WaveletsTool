"""
Tool pentru detectarea de pattern-uri prin metoda cosinus similarity
Python 2, 3
Esri ArcGIS
Last update: 22.12.2019
"""

# --------------------------------------------------------------------- #
from __future__ import division  # floor division from Python 3
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import with_statement
import arcpy
import os
import numpy as np
import math
import sys
import pandas as pd
from scipy import spatial
import tempfile
arcpy.env.overwriteOutput = True
# import warnings
# warnings.filterwarnings("error")

# --------------------------------------------------------------------- #

#Classes
class Reader(object):
	"""
	Class for creating numpy arrays from input rasters
	"""
	def __init__(self, raster_path):
		self.raster_path = raster_path

	def raster_to_numpy(self):
		# Create a raster object, whose extent properties are used in the matrix manipulations
		raster_obj = arcpy.Raster(self.raster_path)
		# Convert to numpy array
		matrix = arcpy.RasterToNumPyArray(raster_obj)

		return raster_obj, matrix


class Writer(object):
	"""
	Class for writing numpy arrays back to raster type
	"""
	def __init__(self, raster_obj, matrix, out_raster_path):
		"""
		Init function
		:param raster_obj: the raster that is returned from Reader class
		:param matrix: the array that results from the convolution operation (should have the same nr or rows/cols)
		:param out_raster_path: the path of the output raster
		:return: None
		"""

		self.raster_obj = raster_obj
		self.raster_matrix = matrix
		self.out_raster_path = out_raster_path

	def write_to_raster(self):
		out_raster= arcpy.NumPyArrayToRaster(self.raster_matrix.reshape(self.raster_obj.height, self.raster_obj.width),
													arcpy.Point(self.raster_obj.extent.XMin, self.raster_obj.extent.YMin),
													self.raster_obj.meanCellWidth, self.raster_obj.meanCellHeight)

		sp_ref = arcpy.Describe(self.raster_obj).spatialReference
		arcpy.DefineProjection_management(out_raster, sp_ref)
		out_raster.save(self.out_raster_path)



class MexicanHat(object):
	"""
	Generates a Mexican Hat wavelet filter based on input parameters
	"""

	def __init__(self, filter_size, ker_a, ker_bx=0, ker_by=0, ker_zs=False, ker_norm=False): # todo remove
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
		print("recieved: size: ", self.d, " sigma= ", self.ker_a)

	def generate_MH_filter(self):
		h = np.empty([self.d, self.d], dtype=np.float)
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

		if self.ker_zs is True:
			sum = np.sum(h)
			v = (-1 * sum) / (h.shape[0] * h.shape[1])
			for line in range(h.shape[0]):
				for col in range(h.shape[1]):
					h[line, col] = v + h[line, col]

		if self.ker_norm is True:
			h = h /np.linalg.norm(h)

		return h


class InputTester(object):
	"""
	Class to test the input parameters
	"""

	def __init__(self, **kwargs):
		self.input_raster = kwargs.get('input_raster')
		self.input_vector_observations = kwargs.get('input_vector_observations')
		self.move_to_max_distance = int(kwargs.get('move_to_max_distance'))
		self.point_matrix_size = int(kwargs.get('point_matrix_size'))
		self.ker_dil_start = float(kwargs.get('ker_dil_start'))
		self.ker_dil_end = float(kwargs.get('ker_dil_end'))
		self.ker_dil_step = float(kwargs.get('ker_dil_step'))
		self.raster_workspace = kwargs.get('raster_workspace')
		self.approach = kwargs.get('approach')

	def test_input(self):
		## Test spatial references
		sp_ref_raster = arcpy.Describe(self.input_raster).spatialReference

		if self.input_vector_observations:
			sp_ref_vect = arcpy.Describe(self.input_vector_observations).spatialReference

			if sp_ref_raster.PCSCode != sp_ref_vect.PCSCode:
				arcpy.AddError("\nThe raster and vector inputs are not in the same coordinate system!"
							   " Please project them in the same coordinate system!")
				exit(-1)

			## Test if points are inside the raster
			arcpy.CheckOutExtension("Spatial")
			out_const_raster = arcpy.sa.Int(arcpy.sa.Times(self.input_raster, 0))
			arcpy.RasterToPolygon_conversion(out_const_raster, 'in_memory\out_polygon', simplify='NO_SIMPLIFY')

			# get point count
			nr_points = int(arcpy.GetCount_management(input_vector_observations).getOutput(0))

			arcpy.MakeFeatureLayer_management(input_vector_observations, 'in_memory\points_lyr')
			arcpy.SelectLayerByLocation_management('in_memory\points_lyr', 'WITHIN', 'in_memory\out_polygon')
			nr_points_inside = int(arcpy.GetCount_management('in_memory\points_lyr').getOutput(0))

			arcpy.CheckInExtension("Spatial")

			if nr_points != nr_points_inside:
				arcpy.AddError("\nNot all points fall inside the input raster. Please check the raster"
							   " and/or the input point observations.")
				exit(-1)

		## Test number of raster bands
		desc_raster = arcpy.Describe(self.input_raster)
		nr_bands = len(desc_raster.children)

		if nr_bands > 1:
			arcpy.AddError("\nThe input raster must be one band!")
			exit(-1)

		## Test search neighbourhood distance and matrix size
		if move_to_max_distance:
			if self.move_to_max_distance % 2 == 0 or self.move_to_max_distance <= 0:
				arcpy.AddError("\nThe 'Move to maximum distance parameter' must be an odd value greater than 0!")
				exit(-1)

		if self.point_matrix_size % 2 == 0 or self.point_matrix_size <= 0:
			arcpy.AddError("\nThe 'matrix size parameter' must be an odd value greater than 0!")
			exit(-1)

		## Test the kernel dilation interval
		if self.ker_dil_start:
			dil_vals = np.arange(float(self.ker_dil_start), float(self.ker_dil_end),
								 float(self.ker_dil_step))
			if len(dil_vals) > 30 and self.approach == 'Seek occurrence of pre-defined pattern in the DEM':
				print_message("\nWarning: More than 30 rasters will be generated. This will take some"
							  " time depending on the machine performance!", general_message_level,
							  'Warning')


class MetadataElement(object):

	def __init__(self):
		self.metadata_text = 'The tool was run with the following parameters:'

	def add_text(self, text):
		self.metadata_text = self.metadata_text + '\n' + text

	def save_metadata(self, out_folder):
		with open(os.path.join(out_folder, 'Tool_Metadata.txt'), 'w') as outfile:
			outfile.write(self.metadata_text)


class CustomPattern(object):

	def __init__(self, workspace):
		self.workspace = workspace

	def get_patterns(self):

		numpy_filters = []

		pattern_files = [os.path.join(self.workspace, i) for i in os.listdir(self.workspace) if i.endswith('.csv')]

		if len(pattern_files) == 0:
			arcpy.AddError("The input patterns workspace is empty! No .csv files found!")
		else:
			for pat_file in pattern_files:
				np_filter = np.genfromtxt(pat_file, delimiter=',')
				numpy_filters.append((os.path.basename(pat_file).replace('.csv', ''), np_filter))
			return numpy_filters


# Functions
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


def calculate_approach3_slopes(raster_matrix, filter, size_of_cell, local_translation=False):
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
		print_message("The input matrix must be square. Exiting...", general_message_level, 'Error')
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
		col = int((point_x - xmin) / cellx)

		if move_to_max_value == True:
			row_max, col_max = move_to_max(row, col, raster_array, search_distance)
			return row_max, col_max
		else:
			return row, col

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

	# Create a search cursor on the input shp/ feature class
	scursor = arcpy.da.SearchCursor(projectedPointFC, '*')  # * = all fields
	raster_arr = arcpy.RasterToNumPyArray(raster)   # save raster as array

	# process raster
	dsc = arcpy.Describe(raster)  # describe raster dataset

	cellx = dsc.MeanCellHeight
	celly = dsc.MeanCellWidth

	xmin = dsc.extent.XMin
	ymax = dsc.extent.YMax

	# iterate point fc and get row, col and value
	field_names = scursor.fields
	idx = scursor.fields.index(mapping_field)  # get the index of the mapping field

	for row in scursor:
		field_map = row[idx]  # the field used to index the dictionary of the shp/ fc -->
		x_val = row[1][0]  # x coord
		y_val = row[1][1]  # y coord
		shape_all_values = [(i, field_names[int(index)]) for index, i in enumerate(row)]  # get all the values
		# feat = row.getValue(0)  # unused
		# pnt = feat.getPart()    # unused

		row, col = map_to_pixel(x_val, y_val, cellx, celly, xmin, ymax, raster_array=raster_arr)
		raster_val = raster_arr[row, col]
		# Insert all the row's values into a list
		output_dict[field_map] = [shape_all_values, ('Row', row), ('Col', col), ('RasterVal', raster_val)]

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
	Function to get the row column from a dict object obtained from a feature class/ vector (function get_points_values())
	:param points_dict: the dict of points obtained from 'get_points_values' function
	:param dict_key: the key of the point, that is used to extract the matrix, type int
	:return: tuple of row and column (row, col) corresponding to the ID of the feature
	"""

	# Get row, value from dict
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

	def process_pattern_vs_point_obs(self, point_id, point_matrix_size, mh_filter_size, ker_a=1.0, ker_bx=0, ker_by=0,
								ker_zero_sum=False, ker_normalization=False, find_tangents=False, find_tangents_l_value=1.0,
								normalization=True, local_minima=False, existing_filter=None):
		"""
		Function to compare a Mexican Hat wavelet filter with a point matrix
		:param point_id: the id (mapping_field) of the input point
		:param point_matrix_size: the size of the point matrix that is extracted from the raster
		:param mh_filter_size: the size of the Mexican Hat wavelet filter
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
			pattern_filter = MexicanHat(mh_filter_size, ker_a, ker_bx, ker_by, ker_zero_sum, False).generate_MH_filter()

		print("\nInit pattern: ", pattern_filter)
		# Get Point locations and the point matrix
		row, col = get_row_col_from_dict(self.dict_points, point_id)
		print("\nrow: ", row, " col: ", col)
		mat_point = slice_point_matrix(self.raster_matrix, row, col, point_matrix_size)
		print("\nmat point \n ")  # todo delete
		print(mat_point)

		if find_tangents:
			pattern_filter = findTangents(pattern_filter, find_tangents_l_value)
			mat_point = findTangents(mat_point, find_tangents_l_value)

			print("#################")
			print("\nmh filter tangents: ")
			print(pattern_filter)
			print("\nmat point tangents: ", mat_point, "\n")

		# if local_minima:
		#
		# 	mh_filter = apply_local_minima(pattern_filter)
		# 	mat_point = apply_local_minima(mat_point)
		# 	print("LOCAL MINIMA APPLIED")

		if normalization:
			pattern_filter = normalize_matrix(pattern_filter)
			mat_point = normalize_matrix(mat_point)

			print("\nmh filter tangents norm: ", pattern_filter)
			print("\nmh point tangents norm: ", mat_point)
			print("\nmh point tangents norm round: \n", np.array(mat_point).round(6))


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


def get_mapping_values(input_vector, field):
	"""
	Function to return the values of the mapping column
	:param input_vector: feature class
	:param field: field of the input vector file
	:return: list of values used to map the output file
	"""

	scursor = arcpy.da.SearchCursor(input_vector, '*')
	idx = scursor.fields.index(field)
	list_oid_init = [row[idx] for row in scursor]  # row[0] is OBJECTID/ FID
	# list_oid = sorted(list_oid_init)  # no need

	if len(list_oid_init) != len(set(list_oid_init)):
		arcpy.AddWarning("\nThere are duplicate values in the mapping column. "
						 "This may result in confusion when interpreting the results!")

	return list_oid_init


def print_message(message, gen_level, msg_level='Debug'):
	"""
	Prints messages to output. Used for debugging
	:param message: message string
	:param gen_level: the general message level (Info or Debug)
	:param msg_level: message type (Debug, Info, Warning, Error)
	"""

	if gen_level == 'Debug':
		arcpy.AddMessage(message)

	elif gen_level == "Info" and msg_level == 'Info':
		arcpy.AddMessage(message)

	elif msg_level == 'Warning':
		arcpy.AddWarning(message)

	elif msg_level == 'Error':
		arcpy.AddError(message)


# --------------------------------------------------------------------- #
if __name__ == "__main__":
	# Debug (local) mode vs ArcGIS toolbox mode
	_debug = False
	general_message_level = 'Info'

	if _debug:
		# local mode
		input_raster = r"E:\Proiecte\Doc\Date\Revised\Fundata_Data_Updated\DSM_Plot_1_Ras.tif"
		size_of_cell = 1  # for findTangents()
		approach = 'Locations in the DEM versus pre-defined pattern'
		# Approach vals
		# 'Locations in the DEM generated from field observations'
		# 'Locations in the DEM versus pre-defined pattern'
		# 'Seek occurrence of pre-defined pattern in the DEM'

		predefined_pattern = 'Mexican Hat wavelet'  # 'Mexican Hat wavelet' or 'Custom pattern'
		pattern_workspace = r'E:\Proiecte\Doc\Date\Output\Teste_Toolbox_v2\Pattern_Workspace'

		input_vector_observations = r"E:\Proiecte\Doc\Date\Dec_2019\In_Tree_21_X42.shp"
		input_vector_observations = r"E:\Proiecte\Doc\Date\Dec_2019\In_Tree_X29_Location7.shp"
		input_vector_observations = r"E:\Proiecte\Doc\Date\Dec_2019\In_Tree_Location12_X02_C13.shp"
		input_vector_observations = r"E:\Proiecte\Doc\Date\Dec_2019\In_Tree_Location34_X36_C41.shp"
		input_vector_observations = r"E:\Proiecte\Doc\Date\Dec_2019\In_Tree_32_X77_C68.shp"

		mapping_field = 'COD_Tree'  # or FID for shapefile
		mh_filter_size = 3
		mh_ker_a = 2.1
		mh_iteration = False
		mh_start_dil_val = 0.1
		mh_end_dil_val = 10
		mh_dil_step = 1

		move_to_max = False
		move_to_max_distance = 3
		point_matrix_size = 3
		point_1_id = ''
		point_2_id = ''
		mh_zs = False
		transform = 'Compute slopes'

		output_file = r'E:\Proiecte\Doc\Date\Dec_2019\out_test_toolbox\out_debug_test.xlsx'
		output_raster_workspace = 'E:\Proiecte\Doc\Date\Dec_2019\out_test_toolbox'
		output_sim_matrix = r'E:\Proiecte\Doc\Date\Dec_2019\out_test_toolbox\out_sim_matrix.xlsx'
		output_table = 'E:\Proiecte\Doc\Date\Dec_2019\out_test_toolbox\out_table_32.xlsx'

	else:
		# Main parameters
		input_raster = arcpy.GetParameterAsText(0)   # required
		approach = arcpy.GetParameterAsText(1) # List of possible values: 'Locations in the DEM generated from field observations',
												#  'Locations in the DEM versus pre-defined pattern',
											   	# 'Seek occurrence of pre-defined pattern in the DEM'
		predefined_pattern = arcpy.GetParameterAsText(2)
		pattern_workspace = arcpy.GetParameterAsText(3)
		point_matrix_size = arcpy.GetParameterAsText(4)  # required (default 3)
		input_vector_observations = arcpy.GetParameterAsText(5)  # optional
		mapping_field = arcpy.GetParameterAsText(6)   # required (default OBJECTID)
		move_to_max = arcpy.GetParameter(7)   # optional
		move_to_max_distance = arcpy.GetParameterAsText(8)  # optional (default 3)

		# MexicanHat parameters
		mh_iteration = arcpy.GetParameter(9)  # Optional: Boolean
		mh_ker_a = arcpy.GetParameterAsText(10)  # Only if iteration not chosen: Default 1
		mh_start_dil_val = arcpy.GetParameterAsText(11)  # Only if iteration not chosen: Default 1
		mh_end_dil_val = arcpy.GetParameterAsText(12)  # Only if iteration not chosen: Default 1
		mh_dil_step = arcpy.GetParameterAsText(13)  # Only if iteration not chosen: Default 1
		mh_zs = False  # unused
		# transform parameters
		transform = arcpy.GetParameterAsText(14)  # Optional: Boolean
		size_of_cell = arcpy.GetParameterAsText(15)
		# Output
		output_sim_matrix = arcpy.GetParameterAsText(16)
		output_table = arcpy.GetParameterAsText(17)
		output_raster_workspace = arcpy.GetParameterAsText(18)

	## Instantiate metadata element
	metadata = MetadataElement()

	# Get raster paths
	desc_raster = arcpy.Describe(input_raster)
	path_raster = desc_raster.path
	raster_source = str(path_raster) + "/" + input_raster

	metadata.add_text('Input raster: {0}'.format(raster_source))
	metadata.add_text('Approach: {0}'.format(approach))
	metadata.add_text('Transform: {0}'.format(transform))

	# Resolve boolean parameters
	if move_to_max is False:
		metadata.add_text('Move to maximum: {0}'.format(move_to_max))
	else:
		metadata.add_text('Move to maximum: {0}'.format(move_to_max))
		metadata.add_text('Move to maximum distance: {0}'.format(move_to_max_distance))

	# Test parameters
	InputTester(input_raster=input_raster, input_vector_observations=input_vector_observations,
				move_to_max_distance=int(move_to_max_distance),
				point_matrix_size=point_matrix_size, mh_filter_size=point_matrix_size,
				ker_dil_start=mh_start_dil_val, ker_dil_end=mh_end_dil_val, ker_dil_step=mh_dil_step,
				raster_workspace=output_raster_workspace, approach=approach).test_input()

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
	elif transform == 'Compute slopes and perform local translation':
		find_tangents = True
		local_minima = True
	else:  # directly on elevation matrix
		find_tangents = False
		local_minima = False

	if approach == 'Locations in the DEM generated from field observations':
		print_message("\nLocations in the DEM generated from field observations", general_message_level, 'Info')

		tool = RunToolbox(input_raster=input_raster, point_observations=input_vector_observations)
		dict_points, raster_obj, raster_matrix = tool.transform_raster_and_identify_points(
			move_to_max=move_to_max, move_to_max_distance=int(move_to_max_distance),
			mapping_field=mapping_field)

		mapping_list = get_mapping_values(input_vector_observations, mapping_field)

		for index, objid_1 in enumerate(mapping_list):
			list_vals_result = []
			dict_rename[index] = objid_1

			for objid_2 in mapping_list:

				result_cossim = tool.process_point_matrix_vs_point_matrix(point_id_1=objid_1, point_id_2=objid_2,
																		point_matrix_size=point_matrix_size,
																		find_tangents=find_tangents,
																		find_tangents_l_value=float(size_of_cell),
																		local_minima=local_minima,
																		normalization=True)
				list_vals_result.append(result_cossim)

			pandas_data_result[objid_1] = list_vals_result

		df1 = pd.DataFrame(pandas_data_result, columns=mapping_list)

		df1_ren = df1.rename(index=dict_rename)

		if output_sim_matrix.endswith('.xlsx'):
			df1_ren.to_excel(output_sim_matrix)
		elif output_sim_matrix.endswith('.csv'):
			df1_ren.to_csv(output_sim_matrix)

		# Get points path
		desc_points = arcpy.Describe(input_vector_observations)
		path_points = desc_points.path
		points_source = str(path_points) + "/" + input_vector_observations

		metadata.add_text('Input Point Vectors: {0}'.format(points_source))
		metadata.add_text('Mapping field: {0}'.format(mapping_field))
		metadata.add_text('Output similarity matrix: {0}'.format(output_sim_matrix))

	elif approach == 'Locations in the DEM versus pre-defined pattern':   # MexicanHat
		print_message("\n Locations in the DEM versus pre-defined pattern", general_message_level, 'Info')
		# Get points path
		desc_points = arcpy.Describe(input_vector_observations)
		path_points = desc_points.path
		points_source = str(path_points) + "/" + input_vector_observations

		metadata.add_text('Input Point Vectors: {0}'.format(points_source))
		metadata.add_text('Mapping field: {0}'.format(mapping_field))
		metadata.add_text('Predefined pattern: {0}'.format(predefined_pattern))
		metadata.add_text('Point matrix size: {0}'.format(point_matrix_size))

		tool = RunToolbox(input_raster=input_raster, point_observations=input_vector_observations)
		dict_points, raster_obj, raster_matrix = tool.transform_raster_and_identify_points(
			move_to_max=move_to_max, move_to_max_distance=int(move_to_max_distance),
			mapping_field=mapping_field)

		mapping_list = get_mapping_values(input_vector_observations, mapping_field)

		if predefined_pattern == 'Mexican Hat wavelet':
			metadata.add_text('Mexican Hat wavelet iteration: {0}'.format(mh_iteration))

			if not mh_iteration:
				metadata.add_text('Mexican Hat dilation value: {0}'.format(mh_ker_a))

				pattern_filter = MexicanHat(filter_size=int(point_matrix_size), ker_a=float(mh_ker_a), ker_zs=mh_zs,
								   ker_norm=True).generate_MH_filter()

				for index, obs_id in enumerate(mapping_list):
					list_vals_result = []
					dict_rename[index] = mh_ker_a

					result_cossim = tool.process_pattern_vs_point_obs(point_id=obs_id, point_matrix_size=int(point_matrix_size),
																 mh_filter_size=int(point_matrix_size), ker_a=float(mh_ker_a),
																 ker_zero_sum=mh_zs, ker_normalization=True,
																 find_tangents=find_tangents,
																 find_tangents_l_value=float(size_of_cell),
																 local_minima=local_minima, normalization=True,
																 existing_filter=pattern_filter)

					list_vals_result.append(result_cossim)
					pandas_data_result[obs_id] = list_vals_result
					df1 = pd.DataFrame(pandas_data_result, columns=mapping_list)
					df1_ren = df1.rename(index=dict_rename)
					df1_ren = df1_ren.T  # transpose matrix

					if output_table.endswith('.xlsx'):
						df1_ren.to_excel(output_table)
					elif output_table.endswith('.csv'):
						df1_ren.to_csv(output_table)

			else:
				# MH iteration
				metadata.add_text('Iteration start: {0}'.format(str(mh_start_dil_val)))
				metadata.add_text('Iteration stop: {0}'.format(str(mh_end_dil_val)))
				metadata.add_text('Iteration step: {0}'.format(str(mh_dil_step)))

				dil_interval = np.arange(float(mh_start_dil_val), float(mh_end_dil_val) + float(mh_dil_step), float(mh_dil_step))
				frames = []
				df1_ren = {}

				for dil_val in dil_interval:
					for index, obs_id in enumerate(mapping_list):

						pattern_filter = MexicanHat(filter_size=int(point_matrix_size), ker_a=float(dil_val), ker_zs=mh_zs,
											   ker_norm=True).generate_MH_filter()

						list_vals_result = []
						dict_rename[index] = dil_val

						result_cossim = tool.process_pattern_vs_point_obs(point_id=obs_id, point_matrix_size=int(point_matrix_size),
																	 mh_filter_size=int(point_matrix_size), ker_a=float(dil_val),
																	 ker_zero_sum=mh_zs, ker_normalization=True,
																	 find_tangents=find_tangents,
																	 find_tangents_l_value=float(size_of_cell),
																	 local_minima=local_minima, normalization=True,
																	 existing_filter=pattern_filter)

						list_vals_result.append(result_cossim)

						pandas_data_result[obs_id] = list_vals_result
						df1 = pd.DataFrame(pandas_data_result, columns=mapping_list)

						df1_ren = df1.rename(index=dict_rename)
					frames.append(df1_ren)

				d_out = pd.concat(frames)
				d_out = d_out.T

				if output_table.endswith('.xlsx'):
					d_out.to_excel(output_table)
				elif output_table.endswith('.csv'):
					d_out.to_csv(output_table)

		else:
			patterns_list = CustomPattern(pattern_workspace).get_patterns()
			metadata.add_text('Custom patterns workspace: {0}'.format(pattern_workspace))

			frames = []
			df1_ren = {}

			for iter_id, pattern_filter_tuple in enumerate(patterns_list):
				pattern_filter_file, pattern_filter = pattern_filter_tuple

				for index, obs_id in enumerate(mapping_list):

					list_vals_result = []
					dict_rename[index] = pattern_filter_file

					result_cossim = tool.process_pattern_vs_point_obs(point_id=obs_id,
																	  point_matrix_size=int(point_matrix_size),
																	  mh_filter_size=int(point_matrix_size),
																	  ker_a=float(1),
																	  ker_zero_sum=mh_zs, ker_normalization=True,
																	  find_tangents=find_tangents,
																	  find_tangents_l_value=float(size_of_cell),
																	  local_minima=local_minima, normalization=True,
																	  existing_filter=pattern_filter)
					list_vals_result.append(result_cossim)

					pandas_data_result[obs_id] = list_vals_result
					df1 = pd.DataFrame(pandas_data_result, columns=mapping_list)

					df1_ren = df1.rename(index=dict_rename)
				frames.append(df1_ren)

				d_out = pd.concat(frames)
				d_out = d_out.T

				if output_table.endswith('.xlsx'):
					d_out.to_excel(output_table)
				elif output_table.endswith('.csv'):
					d_out.to_csv(output_table)

		metadata.add_text('Output table: {0}'.format(output_table))

	else:  # Seek occurrence of pre-defined pattern in the DEM
		print_message("\nSeek occurrence of pre-defined pattern in the DEM", general_message_level, 'Info')
		raster_obj, raster_matrix = Reader(input_raster).raster_to_numpy()

		metadata.add_text('Predefined pattern: {0}'.format(predefined_pattern))
		metadata.add_text('Point matrix size: {0}'.format(point_matrix_size))

		if predefined_pattern == 'Mexican Hat wavelet':
			metadata.add_text('Mexican Hat wavelet iteration: {0}'.format(mh_iteration))

			if mh_iteration is False:

				# No MH iteration
				metadata.add_text('Mexican Hat wavelet dilation value: {0}'.format(mh_ker_a))

				pattern_filter = MexicanHat(filter_size=int(point_matrix_size), ker_a=float(mh_ker_a)).generate_MH_filter()

				if transform == 'Compute slopes' or transform == 'Compute slopes and perform local translation':
					raster_conv = calculate_approach3_slopes(raster_matrix, pattern_filter, int(size_of_cell), local_translation=local_minima)

				else:
					raster_conv = perform_convolution_2D(raster_matrix, pattern_filter, normalization=True, find_tangents=find_tangents)

				if output_raster_workspace.endswith('.gdb'):

					out_raster_file = os.path.join(output_raster_workspace, 'Convolution_Raster_d' + str(point_matrix_size)
												   + '_s' + str(mh_ker_a))
					print_message("is gdb: {0}".format(out_raster_file), general_message_level, 'Info')

				else:
					out_raster_file = os.path.join(output_raster_workspace, 'Convolution_Raster_d' + str(point_matrix_size)
												   + '_s' + str(mh_ker_a) + '.tif')

				print_message("Output raster file: {0}".format(out_raster_file), general_message_level, 'Info')

				Writer(raster_obj=raster_obj, matrix=raster_conv, out_raster_path=out_raster_file).write_to_raster()

			else:
				dil_interval = np.arange(float(mh_start_dil_val), float(mh_end_dil_val) + float(mh_dil_step), float(mh_dil_step))
				list_a = []

				metadata.add_text('Iteration start: {0}'.format(str(mh_start_dil_val)))
				metadata.add_text('Iteration stop: {0}'.format(str(mh_end_dil_val)))
				metadata.add_text('Iteration step: {0}'.format(str(mh_dil_step)))

				for dil_val in dil_interval:
					list_a.append(dil_val)
					pattern_filter = MexicanHat(filter_size=int(point_matrix_size),
												ker_a=float(mh_ker_a)).generate_MH_filter()

					if transform == 'Compute slopes' or transform == 'Compute slopes and perform local translation':
						raster_conv = calculate_approach3_slopes(raster_matrix, pattern_filter, int(size_of_cell),
																 local_translation=local_minima)

					else:
						raster_conv = perform_convolution_2D(raster_matrix, pattern_filter, normalization=True,
															 find_tangents=find_tangents)

					if output_raster_workspace.endswith('.gdb'):
						out_raster_file = os.path.join(output_raster_workspace,
												  'Convolution_Raster_d' + str(point_matrix_size) + '_s' + \
												  str(dil_val))
					else:
						out_raster_file = os.path.join(output_raster_workspace,
												  'Convolution_Raster_d' + str(point_matrix_size) + '_s' + \
												  str(dil_val) + '.tif')
					Writer(raster_obj=raster_obj, matrix=raster_conv,
										out_raster_path=out_raster_file).write_to_raster()
		else:
			# One or more filters
			patterns_list = CustomPattern(pattern_workspace).get_patterns()

			for iter_id, pattern_filter_tuple in enumerate(patterns_list):
				pattern_filter_file, pattern_filter = pattern_filter_tuple

				if transform == 'Compute slopes' or transform == 'Compute slopes and perform local translation':
					raster_conv = calculate_approach3_slopes(raster_matrix, pattern_filter, int(size_of_cell),
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

				print_message("Output raster file: {0}".format(out_raster_file), general_message_level, 'Info')

				Writer(raster_obj=raster_obj, matrix=raster_conv,
									out_raster_path=out_raster_file).write_to_raster()

		metadata.add_text('Raster workspace: {0}'.format(output_raster_workspace))

	metadata.save_metadata(tempfile.gettempdir())
	print_message('Saved metadata to: {0}'.format(os.path.join(tempfile.gettempdir(), 'Tool_Metadata.txt')),
														 general_message_level, 'Info')
