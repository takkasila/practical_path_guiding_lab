import pathlib


class FileNameManager:
	"""
		File name manager/generator in a class method fashion.
		Always call 'setSceneName' before use.
	"""

	SCENE_NAME = 'scene'
	DEBUG_FOLDER_PATH = 'debug/' + SCENE_NAME + '/'
	TREE_DATA_FOLDER_PATH = DEBUG_FOLDER_PATH + 'tree-data/'
	IMAGE_FOLDER_PATH = DEBUG_FOLDER_PATH + 'image/'
	PLOT_FOLDER_PATH = DEBUG_FOLDER_PATH + 'plot/'
	OBJ_FOLDER_PATH = DEBUG_FOLDER_PATH + 'obj/'
	PERFORMANCE_FOLDER_PATH = DEBUG_FOLDER_PATH + 'performance'


	@classmethod
	def setSceneName( cls, sceneName: str ) -> None:
		cls.SCENE_NAME = sceneName
		cls.DEBUG_FOLDER_PATH = 'debug/' + cls.SCENE_NAME + '/'
		cls.TREE_DATA_FOLDER_PATH = cls.DEBUG_FOLDER_PATH + 'tree-data/'
		cls.IMAGE_FOLDER_PATH = cls.DEBUG_FOLDER_PATH + 'image/'
		cls.PLOT_FOLDER_PATH = cls.DEBUG_FOLDER_PATH + 'plot/'
		cls.OBJ_FOLDER_PATH = cls.DEBUG_FOLDER_PATH + 'obj/'
		cls.PERFORMANCE_FOLDER_PATH = cls.DEBUG_FOLDER_PATH + 'performance/'


	@classmethod
	def createDebugFolder( cls ) -> None:
		"""
			If the debug folder path does not exist yet then create it
		"""
		pathlib.Path( cls.DEBUG_FOLDER_PATH ).mkdir( parents= True, exist_ok= True )
		pathlib.Path( cls.TREE_DATA_FOLDER_PATH ).mkdir( parents= True, exist_ok= True )
		pathlib.Path( cls.IMAGE_FOLDER_PATH ).mkdir( parents= True, exist_ok= True )
		pathlib.Path( cls.PLOT_FOLDER_PATH ).mkdir( parents= True, exist_ok= True )
		pathlib.Path( cls.OBJ_FOLDER_PATH ).mkdir( parents= True, exist_ok= True )
		pathlib.Path( cls.PERFORMANCE_FOLDER_PATH ).mkdir( parents= True, exist_ok= True )


	@classmethod
	def generateFileNameFormat( cls, path: str ) -> str:
		fileNameFormat = path + cls.SCENE_NAME + '_iter-{0}'
		return fileNameFormat


	@classmethod
	def generateTreeDataFileName( cls, iteration: int, withNpzEnding: bool = True ) -> str:
		fileName = cls.generateFileNameFormat( cls.TREE_DATA_FOLDER_PATH ).format( iteration )

		if withNpzEnding:
			fileName += '.npz'
		
		return fileName


	@classmethod
	def generateImageFileName( cls, iteration: int, spp: int ) -> str:
		fileName = cls.generateFileNameFormat( cls.IMAGE_FOLDER_PATH ).format( iteration ) + f'_spp-{spp}'
		return fileName

	
	@classmethod
	def generateOBJFileName( cls, iteration: int ) -> str:
		fileName = cls.generateFileNameFormat( cls.OBJ_FOLDER_PATH ).format( iteration ) + '.obj'
		return fileName



# Test
if __name__ == '__main__':

	from src.file_name_manager import FileNameManager
	FileNameManager.setSceneName( 'teapot-Au' )

	print( FileNameManager.SCENE_NAME )

	FileNameManager.createDebugFolder()

	print( FileNameManager.generateTreeDataFileName( 2 ) )
	print( FileNameManager.generateTreeDataFileName( 2, True ) )
	print( FileNameManager.generateImageFileName( 3 ) )