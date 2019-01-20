import numpy as np
import inspect
import sys

class QSO:
	def __init__(self, name):
		self._path = "/Users/rcooke/Work/Research/Cosmo/SandageTest/Optical/Data/"
		try:
			return eval("self."+name+"()")
		except:
			print("No such DLA! Try one of these:\n")
			members = [x for x, y in inspect.getmembers(self, predicate=inspect.ismethod)]
			for m in members:
				print(m)

	def HS1700p6416(self):
		self._zem = 2.7348
		self._path += "HS1700p6416/"
		self._filename = "HS1700p6416_1p7241_HIRES_C1.dat"

