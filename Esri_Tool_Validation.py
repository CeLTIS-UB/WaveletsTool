import arcpy


class ToolValidator(object):
	"""Class for validating a tool's parameter values and controlling
	the behavior of the tool's dialog."""

	def __init__(self):
		"""Setup arcpy and the list of tool parameters."""
		self.params = arcpy.GetParameterInfo()

		self.input_raster = self.params[0]
		self.approach = self.params[1]
		self.predefined_pattern = self.params[2]
		self.pattern_workspace = self.params[3]
		self.point_matrix_size = self.params[4]
		self.point_vectors = self.params[5]
		self.mapping_field = self.params[6]
		self.move_to_max = self.params[7]
		self.move_to_max_distance = self.params[8]
		self.mh_iteration = self.params[9]
		self.mh_dil_val = self.params[10]
		self.mh_dil_start = self.params[11]
		self.mh_dil_stop = self.params[12]
		self.mh_dil_step = self.params[13]
		self.transform = self.params[14]
		self.size_of_the_cell = self.params[15]
		self.output_sim_matrix = self.params[16]
		self.output_table = self.params[17]
		self.output_raster_workspace = self.params[18]

	def initializeParameters(self):
		"""Refine the properties of a tool's parameters. This method is
		called when the tool is opened."""
		self.input_raster.enabled = True
		self.approach.enabled = True
		self.predefined_pattern.enabled = False
		self.predefined_pattern.value = 'Mexican Hat wavelet'
		self.pattern_workspace.enabled = False
		self.point_matrix_size.enabled = False
		self.point_matrix_size.value = 3
		self.point_vectors.enabled = False
		self.mapping_field.enabled = False
		self.move_to_max.enabled = False
		self.move_to_max_distance.enabled = False
		self.move_to_max_distance.value = 3
		self.mh_iteration.enabled = False
		self.mh_dil_val.enabled = False
		self.mh_dil_val.value = 1
		self.mh_dil_start.value = 0.01
		self.mh_dil_stop.value = 1
		self.mh_dil_step.value = 0.1
		self.mh_dil_start.enabled = False
		self.mh_dil_stop.enabled = False
		self.mh_dil_step.enabled = False
		self.transform.enabled = False
		self.size_of_the_cell.enabled = False
		self.size_of_the_cell.value = 1
		self.output_sim_matrix.enabled = False
		self.output_table.enabled = False
		self.output_raster_workspace.enabled = False

	def updateParameters(self):
		"""Modify the values and properties of parameters before internal
		validation is performed. This method is called whenever a parameter
		has been changed."""

		if self.approach.altered:
			self.transform.enabled = True

			if self.approach.value == 'Locations in the DEM generated from field observations':
				self.predefined_pattern.enabled = False
				self.pattern_workspace.enabled = False
				self.point_matrix_size.enabled = True
				self.point_vectors.enabled = True
				self.mapping_field.enabled = True
				self.move_to_max.enabled = True
				self.output_sim_matrix.enabled = True
				self.mh_dil_val.enabled = False

				self.mh_iteration.enabled = False
				self.mh_iteration.value = False
				self.output_table.enabled = False
				self.output_raster_workspace.enabled = False
				self.output_raster_workspace.value = ''

			elif self.approach.value == 'Locations in the DEM versus pre-defined pattern':
				self.predefined_pattern.enabled = True
				self.point_matrix_size.enabled = True
				self.point_vectors.enabled = True
				self.mapping_field.enabled = True
				self.move_to_max.enabled = True
				self.mh_dil_val.enabled = True
				self.mh_iteration.enabled = True
				self.output_table.enabled = True
				self.output_sim_matrix.enabled = False
				self.output_sim_matrix.value = ''
				self.output_raster_workspace.enabled = False
				self.output_raster_workspace.value = ''

			else:  # seek pre-defined pattern in DEM
				self.predefined_pattern.enabled = True
				self.point_matrix_size.enabled = True
				self.mh_iteration.enabled = True
				self.output_raster_workspace.enabled = True
				self.point_vectors.enabled = False
				self.point_vectors.value = ''
				self.mapping_field.enabled = False
				self.move_to_max.enabled = False
				self.move_to_max.value = False
				self.mh_dil_val.enabled = True
				self.output_sim_matrix.enabled = False
				self.output_sim_matrix.value = ''
				self.output_table.enabled = False
				self.output_table.value = ''

		if self.mh_iteration.altered:

			if self.mh_iteration.value is True:
				self.mh_dil_start.enabled = True
				self.mh_dil_stop.enabled = True
				self.mh_dil_step.enabled = True
				self.mh_dil_val.enabled = False
				self.mh_dil_val.value = 1

			else:
				if self.approach.value == 'Locations in the DEM generated from field observations':
					self.mh_dil_val.enabled = False
					self.mh_dil_val.value = 1
				else:
					self.mh_dil_val.enabled = True

				self.mh_dil_start.enabled = False
				self.mh_dil_stop.enabled = False
				self.mh_dil_step.enabled = False
				self.mh_dil_start.value = 0.01
				self.mh_dil_stop.value = 1
				self.mh_dil_step.value = 0.1

		if self.move_to_max.altered:
			if self.move_to_max.value is True:
				self.move_to_max_distance.enabled = True
			else:
				self.move_to_max_distance.enabled = False
				self.move_to_max_distance.value = 3

		if self.transform.altered:
			if self.transform.value == 'Work directly on the elevation matrix':
				self.size_of_the_cell.enabled = False
			elif self.transform.value == 'Perform a local translation':
				self.size_of_the_cell.enabled = False
			elif self.transform.value == 'Compute slopes' or self.transform.value == \
					'Compute slopes and perform local translation':
				self.size_of_the_cell.enabled = True

		if self.predefined_pattern.altered:
			if self.predefined_pattern.value == 'Custom pattern':
				self.pattern_workspace.enabled = True

				self.mh_iteration.value = False
				self.mh_iteration.enabled = False
				self.mh_dil_start.enabled = False
				self.mh_dil_stop.enabled = False
				self.mh_dil_step.enabled = False
				self.mh_dil_start.value = 0.01
				self.mh_dil_stop.value = 1
				self.mh_dil_step.value = 0.1
				self.mh_dil_val.enabled = False
				self.mh_dil_val.value = 1
			else:
				self.pattern_workspace.enabled = False

	def updateMessages(self):
		"""Modify the messages created by internal validation for each tool
		parameter. This method is called after internal validation."""
		if self.approach.altered:
			if self.approach.value == "Locations in the DEM generated from field observations":
				if not self.point_vectors.altered:
					self.point_vectors.setIDMessage("ERROR", 735, self.point_vectors.displayName)
				if not self.mapping_field.altered:
					self.mapping_field.setIDMessage("ERROR", 735, self.mapping_field.displayName)
				if self.mapping_field.value == '' or self.mapping_field.value == ' ' or \
						self.mapping_field.value is None or self.mapping_field.value is False:
					self.mapping_field.setIDMessage("ERROR", 735, self.mapping_field.displayName)

			elif self.approach.value == "Locations in the DEM versus pre-defined pattern":
				if not self.point_vectors.altered:
					self.point_vectors.setIDMessage("ERROR", 735, self.point_vectors.displayName)
				if not self.mapping_field.altered:
					self.mapping_field.setIDMessage("ERROR", 735, self.mapping_field.displayName)
				if self.mapping_field.value == '' or self.mapping_field.value == ' ' or \
						self.mapping_field.value is None or self.mapping_field.value is False:
					self.mapping_field.setIDMessage("ERROR", 735, self.mapping_field.displayName)
				if self.predefined_pattern.value == '' or self.predefined_pattern.value == ' ' or \
						self.predefined_pattern.value is None or self.predefined_pattern.value is False:
					self.predefined_pattern.setIDMessage("ERROR", 735, self.predefined_pattern.displayName)

		if not self.mh_dil_val.altered:
			self.mh_dil_val.setIDMessage("ERROR", 735, self.mh_dil_val.displayName)
		if not self.point_matrix_size.altered:
			self.point_matrix_size.setIDMessage("ERROR", 735, self.point_matrix_size.displayName)
		if not self.move_to_max_distance.altered:
			self.move_to_max_distance.setIDMessage("ERROR", 735, self.move_to_max_distance.displayName)
		if not self.mh_dil_start.altered:
			self.mh_dil_start.setIDMessage("ERROR", 735, self.mh_dil_start.displayName)
		if not self.mh_dil_stop.altered:
			self.mh_dil_stop.setIDMessage("ERROR", 735, self.mh_dil_stop.displayName)
		if not self.mh_dil_step.altered:
			self.mh_dil_step.setIDMessage("ERROR", 735, self.mh_dil_step.displayName)

		if self.input_raster.altered:
			# Test number of raster bands
			desc_raster = arcpy.Describe(self.input_raster)
			nr_bands = len(desc_raster.children)

			if nr_bands > 1:
				self.input_raster.setErrorMessage("The input raster must be one band!")

		if self.point_vectors.altered:
			# Test if observations layer is point
			try:
				desc = arcpy.Describe(self.point_vectors).shapeType
				if desc != 'Point':
					self.point_vectors.setErrorMessage("The input observations vector layer is not of point type!")
			except:
				pass

		if self.point_vectors.altered:
			if not self.mapping_field.altered:
				self.mapping_field.setIDMessage("ERROR", 735, self.mapping_field.displayName)

		if self.input_raster.altered and self.point_vectors.altered:

			try:
				sp_ref_raster = arcpy.Describe(self.input_raster).spatialReference
				sp_ref_vector = arcpy.Describe(self.point_vectors).spatialReference

				if sp_ref_raster.PCSCode != sp_ref_vector.PCSCode:
					self.input_raster.setErrorMessage(
						"The raster and vector inputs are not in the same coordinate system!"
						" Please project them in the same coordinate system!")
					self.point_vectors.setErrorMessage(
						"The raster and vector inputs are not in the same coordinate system!"
						" Please project them in the same coordinate system!")
			except:
				pass

		if self.point_matrix_size.altered:
			if self.point_matrix_size.value % 2 == 0 or self.point_matrix_size.value <= 0:
				self.point_matrix_size.setErrorMessage("The value must be an odd number greater than 0!")

		if self.move_to_max_distance.altered:
			if self.move_to_max_distance.value <= 0:
				self.move_to_max_distance.setErrorMessage("The value must be greater than zero!")

		if self.mh_dil_val.altered:
			if not float(self.mh_dil_val.value) > 0.0:
				self.mh_dil_val.setErrorMessage("The dilation value must be greater than 0!")

		if self.mh_dil_start.altered:
			if not float(self.mh_dil_start.value) >= 0.0:
				self.mh_dil_start.setErrorMessage("The dilation value must be greater than 0!")

		if self.mh_dil_stop.altered:
			if not float(self.mh_dil_stop.value) > 0.0:
				self.mh_dil_stop.setErrorMessage("The dilation value must be greater than 0!")

		if self.mh_dil_step.altered:
			if not float(self.mh_dil_step.value) > 0.0:
				self.mh_dil_step.setErrorMessage("The dilation step value must be greater than 0!")

		if self.mh_dil_start.altered and self.mh_dil_stop.altered:
			if not float(self.mh_dil_stop.value) > float(self.mh_dil_start.value):
				self.mh_dil_start.setErrorMessage("The dilation start value must be greater than the "
												  "dilation stop value!")
				self.mh_dil_stop.setErrorMessage("The dilation stop value must be greater than the "
												 "dilation start value!")

		if self.transform.altered:
			if self.transform.value == 'Compute slopes' or self.transform.value == \
					'Compute slopes and perform local translation':
				if not self.size_of_the_cell.altered:
					self.size_of_the_cell.setIDMessage("ERROR", 735, self.mapping_field.displayName)

		if self.size_of_the_cell.altered:
			if not self.size_of_the_cell.value > 0:
				self.size_of_the_cell.setErrorMessage("The value must be greater than 0!")

		if self.predefined_pattern.altered:
			if self.predefined_pattern.value == 'Custom pattern':
				if not self.pattern_workspace.altered:
					self.pattern_workspace.setIDMessage("ERROR", 735, self.pattern_workspace.displayName)

		if self.output_sim_matrix.enabled:
			if self.output_sim_matrix.value == '' or self.output_sim_matrix.value == ' ' \
					or self.output_sim_matrix.value == None:
				self.output_sim_matrix.setIDMessage("ERROR", 735, self.output_sim_matrix.displayName)
			else:
				if not (str(self.output_sim_matrix.value).endswith('.xlsx') or str(
						self.output_sim_matrix.value).endswith('.csv')):
					self.output_sim_matrix.setErrorMessage(
						"The output similarity matrix must be an Excel (.xlsx) or .csv file!")

		if self.output_table.enabled:
			if self.output_table.value == '' or self.output_table.value == ' ' \
					or self.output_table.value == None:
				self.output_table.setIDMessage("ERROR", 735, self.output_table.displayName)
			else:
				if not (str(self.output_table.value).endswith('.xlsx') or str(self.output_table.value).endswith(
						'.csv')):
					self.output_table.setErrorMessage("The output table must be an Excel (.xlsx) or .csv file!")

		if self.output_raster_workspace.enabled:
			if self.output_raster_workspace.value == '' or self.output_raster_workspace.value == ' ' \
					or self.output_raster_workspace.value == None:
				self.output_raster_workspace.setIDMessage("ERROR", 735, self.output_raster_workspace.displayName)
